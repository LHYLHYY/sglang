"""Probe Sparse Flash Attention statistics support in an NPU graph.

This file contains isolated Ascend hardware capability tests. The first test
captures one ``npu_sparse_flash_attention`` call with
``return_softmax_lse=True``. The second captures two fixed-capacity hit/miss SFA
calls followed by a numerically stable FP32 merge on the same stream. Both tests
replay with different fixed-shape inputs and compare graph results with eager
references, including ``attention_out``, ``softmax_max``, ``softmax_sum``, and
the reconstructed LSE.

The test deliberately fails (rather than xfails) when an installed
torch_npu/CANN combination cannot capture or replay the statistics outputs. A
failure means the split-attention path must continue to use the eager-only path
or the graph-mode combined fallback on that target.

Example (run on the target Ascend host):

    python -m pytest -v -s \
        test/manual/ascend/test_sparse_flash_attention_npugraph_stats.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import pytest
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


@dataclass
class SFAInputs:
    query: torch.Tensor
    key: torch.Tensor
    query_rope: torch.Tensor
    key_rope: torch.Tensor
    sparse_indices: torch.Tensor
    actual_query_lengths: torch.Tensor
    actual_kv_lengths: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.query,
            self.key,
            self.query_rope,
            self.key_rope,
            self.sparse_indices,
            self.actual_query_lengths,
            self.actual_kv_lengths,
        )


@dataclass
class AttentionState:
    output: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor


@dataclass
class SFAPartitionInputs:
    key: torch.Tensor
    key_rope: torch.Tensor
    sparse_indices: torch.Tensor
    actual_kv_lengths: torch.Tensor
    true_counts: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.key,
            self.key_rope,
            self.sparse_indices,
            self.actual_kv_lengths,
            self.true_counts,
        )


@dataclass
class SplitSFAInputs:
    query: torch.Tensor
    query_rope: torch.Tensor
    actual_query_lengths: torch.Tensor
    hit: SFAPartitionInputs
    miss: SFAPartitionInputs

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.query,
            self.query_rope,
            self.actual_query_lengths,
            *self.hit.tensors(),
            *self.miss.tensors(),
        )


@dataclass
class SplitSFACase:
    union: SFAInputs
    split: SplitSFAInputs


def _make_case(
    *,
    valid_lengths: tuple[int, ...],
    capacity: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> SFAInputs:
    if not valid_lengths:
        raise ValueError("valid_lengths must contain at least one request")
    if any(length <= 0 or length > capacity for length in valid_lengths):
        raise ValueError(
            f"each valid length must be in [1, {capacity}], got {valid_lengths}"
        )

    batch_size = len(valid_lengths)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def make_random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

    query = make_random((batch_size, 1, num_heads, 512))
    key = make_random((batch_size, capacity, 1, 512))
    query_rope = make_random((batch_size, 1, num_heads, 64))
    key_rope = make_random((batch_size, capacity, 1, 64))

    actual_query_lengths = torch.ones(batch_size, dtype=torch.int32, device=device)
    actual_kv_lengths = torch.tensor(valid_lengths, dtype=torch.int32, device=device)
    positions = torch.arange(capacity, dtype=torch.int32, device=device).view(
        1, 1, 1, capacity
    )
    positions = positions.expand(batch_size, 1, 1, capacity)
    sparse_indices = torch.where(
        positions < actual_kv_lengths.view(batch_size, 1, 1, 1),
        positions,
        torch.full_like(positions, -1),
    ).contiguous()

    return SFAInputs(
        query=query,
        key=key,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_indices=sparse_indices,
        actual_query_lengths=actual_query_lengths,
        actual_kv_lengths=actual_kv_lengths,
    )


def _make_partition_metadata(
    counts: tuple[int, ...], *, capacity: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(count < 0 or count > capacity for count in counts):
        raise ValueError(f"partition counts must be in [0, {capacity}], got {counts}")

    true_counts = torch.tensor(counts, dtype=torch.int32, device=device)
    batch_size = len(counts)
    positions = torch.arange(capacity, dtype=torch.int32, device=device).view(
        1, capacity
    )
    valid = positions < true_counts.view(batch_size, 1)
    indices = torch.where(
        valid,
        positions.expand(batch_size, capacity),
        torch.full((batch_size, capacity), -1, dtype=torch.int32, device=device),
    )
    # SFA does not accept an empty KV sequence. Empty partitions run against a
    # zero-valued dummy token and are neutralized by true_counts during merge.
    indices[:, 0] = torch.where(
        true_counts > 0,
        indices[:, 0],
        torch.zeros(batch_size, dtype=torch.int32, device=device),
    )
    actual_lengths = true_counts.clamp(min=1, max=capacity).contiguous()
    return (
        indices.view(batch_size, 1, 1, capacity).contiguous(),
        actual_lengths,
        true_counts,
    )


def _make_split_case(
    *,
    hit_counts: tuple[int, ...],
    capacity: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> SplitSFACase:
    if not hit_counts:
        raise ValueError("hit_counts must contain at least one request")
    if any(count < 0 or count > capacity for count in hit_counts):
        raise ValueError(f"hit counts must be in [0, {capacity}], got {hit_counts}")

    batch_size = len(hit_counts)
    miss_counts = tuple(capacity - count for count in hit_counts)
    union = _make_case(
        valid_lengths=(capacity,) * batch_size,
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=seed,
    )

    hit_key = torch.zeros_like(union.key)
    miss_key = torch.zeros_like(union.key)
    hit_key_rope = torch.zeros_like(union.key_rope)
    miss_key_rope = torch.zeros_like(union.key_rope)
    partition_generator = torch.Generator(device="cpu")
    partition_generator.manual_seed(seed + 100_000)

    for batch_index, hit_count in enumerate(hit_counts):
        permutation = torch.randperm(capacity, generator=partition_generator)
        hit_token_indices = permutation[:hit_count].to(device=device)
        miss_token_indices = permutation[hit_count:].to(device=device)
        if hit_count > 0:
            hit_key[batch_index, :hit_count].copy_(
                union.key[batch_index].index_select(0, hit_token_indices)
            )
            hit_key_rope[batch_index, :hit_count].copy_(
                union.key_rope[batch_index].index_select(0, hit_token_indices)
            )
        miss_count = capacity - hit_count
        if miss_count > 0:
            miss_key[batch_index, :miss_count].copy_(
                union.key[batch_index].index_select(0, miss_token_indices)
            )
            miss_key_rope[batch_index, :miss_count].copy_(
                union.key_rope[batch_index].index_select(0, miss_token_indices)
            )

    hit_indices, hit_actual_lengths, hit_true_counts = _make_partition_metadata(
        hit_counts, capacity=capacity, device=device
    )
    miss_indices, miss_actual_lengths, miss_true_counts = _make_partition_metadata(
        miss_counts, capacity=capacity, device=device
    )
    split = SplitSFAInputs(
        query=union.query.clone(),
        query_rope=union.query_rope.clone(),
        actual_query_lengths=union.actual_query_lengths.clone(),
        hit=SFAPartitionInputs(
            key=hit_key,
            key_rope=hit_key_rope,
            sparse_indices=hit_indices,
            actual_kv_lengths=hit_actual_lengths,
            true_counts=hit_true_counts,
        ),
        miss=SFAPartitionInputs(
            key=miss_key,
            key_rope=miss_key_rope,
            sparse_indices=miss_indices,
            actual_kv_lengths=miss_actual_lengths,
            true_counts=miss_true_counts,
        ),
    )
    return SplitSFACase(union=union, split=split)


def _clone_inputs(inputs: SFAInputs) -> SFAInputs:
    return SFAInputs(*(tensor.clone() for tensor in inputs.tensors()))


def _copy_inputs(destination: SFAInputs, source: SFAInputs) -> None:
    for destination_tensor, source_tensor in zip(
        destination.tensors(), source.tensors(), strict=True
    ):
        destination_tensor.copy_(source_tensor)


def _clone_partition_inputs(inputs: SFAPartitionInputs) -> SFAPartitionInputs:
    return SFAPartitionInputs(*(tensor.clone() for tensor in inputs.tensors()))


def _clone_split_inputs(inputs: SplitSFAInputs) -> SplitSFAInputs:
    return SplitSFAInputs(
        query=inputs.query.clone(),
        query_rope=inputs.query_rope.clone(),
        actual_query_lengths=inputs.actual_query_lengths.clone(),
        hit=_clone_partition_inputs(inputs.hit),
        miss=_clone_partition_inputs(inputs.miss),
    )


def _copy_split_inputs(destination: SplitSFAInputs, source: SplitSFAInputs) -> None:
    for destination_tensor, source_tensor in zip(
        destination.tensors(), source.tensors(), strict=True
    ):
        destination_tensor.copy_(source_tensor)


def _run_sfa(inputs: SFAInputs, scale: float) -> AttentionState:
    if torch_npu is None:
        raise RuntimeError("torch_npu is required")

    output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        inputs.query,
        inputs.key,
        inputs.key,
        inputs.sparse_indices,
        scale,
        actual_seq_lengths_query=inputs.actual_query_lengths,
        actual_seq_lengths_kv=inputs.actual_kv_lengths,
        query_rope=inputs.query_rope,
        key_rope=inputs.key_rope,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return AttentionState(output, softmax_max, softmax_sum)


def _run_partition_sfa(
    inputs: SplitSFAInputs, partition: SFAPartitionInputs, scale: float
) -> AttentionState:
    return _run_sfa(
        SFAInputs(
            query=inputs.query,
            key=partition.key,
            query_rope=inputs.query_rope,
            key_rope=partition.key_rope,
            sparse_indices=partition.sparse_indices,
            actual_query_lengths=inputs.actual_query_lengths,
            actual_kv_lengths=partition.actual_kv_lengths,
        ),
        scale,
    )


def _merge_partition_states(
    hit: AttentionState,
    miss: AttentionState,
    hit_true_counts: torch.Tensor,
    miss_true_counts: torch.Tensor,
) -> AttentionState:
    if hit.output.shape != miss.output.shape:
        raise ValueError(
            f"partition output shapes differ: {hit.output.shape} vs {miss.output.shape}"
        )
    if hit.softmax_max.shape != miss.softmax_max.shape:
        raise ValueError(
            "partition statistics shapes differ: "
            f"{hit.softmax_max.shape} vs {miss.softmax_max.shape}"
        )

    batch_size = hit.output.shape[0]
    stats_mask_shape = (batch_size, 1, 1, 1)
    hit_nonempty = (hit_true_counts > 0).view(stats_mask_shape)
    miss_nonempty = (miss_true_counts > 0).view(stats_mask_shape)
    any_nonempty = hit_nonempty | miss_nonempty

    negative_infinity = torch.full_like(hit.softmax_max, float("-inf"))
    hit_max = torch.where(hit_nonempty, hit.softmax_max.float(), negative_infinity)
    miss_max = torch.where(miss_nonempty, miss.softmax_max.float(), negative_infinity)
    global_max = torch.maximum(hit_max, miss_max)
    safe_global_max = torch.where(
        any_nonempty, global_max, torch.zeros_like(global_max)
    )

    hit_delta = torch.where(
        hit_nonempty,
        hit.softmax_max.float() - safe_global_max,
        torch.zeros_like(hit.softmax_max),
    )
    miss_delta = torch.where(
        miss_nonempty,
        miss.softmax_max.float() - safe_global_max,
        torch.zeros_like(miss.softmax_max),
    )
    hit_mass = torch.where(
        hit_nonempty,
        hit.softmax_sum.float() * torch.exp(hit_delta),
        torch.zeros_like(hit.softmax_sum),
    )
    miss_mass = torch.where(
        miss_nonempty,
        miss.softmax_sum.float() * torch.exp(miss_delta),
        torch.zeros_like(miss.softmax_sum),
    )
    merged_sum = hit_mass + miss_mass
    safe_sum = merged_sum.clamp_min(torch.finfo(torch.float32).tiny)
    hit_weight = (hit_mass / safe_sum).permute(0, 2, 3, 1)
    miss_weight = (miss_mass / safe_sum).permute(0, 2, 3, 1)
    merged_output = hit.output.float() * hit_weight + miss.output.float() * miss_weight
    merged_output = torch.where(
        any_nonempty.permute(0, 2, 3, 1),
        merged_output,
        torch.zeros_like(merged_output),
    ).to(hit.output.dtype)
    merged_max = torch.where(any_nonempty, global_max, negative_infinity)
    merged_sum = torch.where(any_nonempty, merged_sum, torch.zeros_like(merged_sum))
    return AttentionState(merged_output, merged_max, merged_sum)


def _run_split_sfa(inputs: SplitSFAInputs, scale: float) -> AttentionState:
    # Deliberately run both calls sequentially on the caller's current stream.
    hit_state = _run_partition_sfa(inputs, inputs.hit, scale)
    miss_state = _run_partition_sfa(inputs, inputs.miss, scale)
    return _merge_partition_states(
        hit_state, miss_state, inputs.hit.true_counts, inputs.miss.true_counts,
    )


def _clone_state(state: AttentionState) -> AttentionState:
    return AttentionState(
        state.output.detach().clone(),
        state.softmax_max.detach().clone(),
        state.softmax_sum.detach().clone(),
    )


def _expected_stats_shape(inputs: SFAInputs) -> tuple[int, int, int, int]:
    batch_size = inputs.query.shape[0]
    num_query_heads = inputs.query.shape[2]
    num_kv_heads = inputs.key.shape[2]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads ({num_query_heads}) must be divisible by "
            f"KV heads ({num_kv_heads})"
        )
    return (
        batch_size,
        num_kv_heads,
        inputs.query.shape[1],
        num_query_heads // num_kv_heads,
    )


def _validate_contract(state: AttentionState, inputs: SFAInputs, *, stage: str) -> None:
    expected_stats_shape = _expected_stats_shape(inputs)

    assert state.output.shape == inputs.query.shape, (
        f"{stage}: output shape={tuple(state.output.shape)}, "
        f"expected={tuple(inputs.query.shape)}"
    )
    assert state.output.dtype == inputs.query.dtype, (
        f"{stage}: output dtype={state.output.dtype}, " f"expected={inputs.query.dtype}"
    )
    for name, value in (
        ("softmax_max", state.softmax_max),
        ("softmax_sum", state.softmax_sum),
    ):
        assert value.numel() > 0, f"{stage}: {name} is empty"
        assert tuple(value.shape) == expected_stats_shape, (
            f"{stage}: {name} shape={tuple(value.shape)}, "
            f"expected={expected_stats_shape}"
        )
        assert (
            value.dtype == torch.float32
        ), f"{stage}: {name} dtype={value.dtype}, expected=torch.float32"
        assert (
            torch.isfinite(value).all().item()
        ), f"{stage}: {name} contains non-finite values"

    assert (
        torch.isfinite(state.output).all().item()
    ), f"{stage}: output contains non-finite values"
    assert (
        (state.softmax_sum > 0).all().item()
    ), f"{stage}: softmax_sum must be positive"


def _state_lse(state: AttentionState) -> torch.Tensor:
    return state.softmax_max + torch.log(state.softmax_sum)


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _assert_matches(
    actual: AttentionState,
    expected: AttentionState,
    *,
    output_atol: float,
    output_rtol: float,
    stage: str,
) -> None:
    torch.testing.assert_close(
        actual.output.float(),
        expected.output.float(),
        atol=output_atol,
        rtol=output_rtol,
        msg=f"{stage}: attention output differs from eager",
    )
    for name, actual_stats, expected_stats in (
        ("softmax_max", actual.softmax_max, expected.softmax_max),
        ("softmax_sum", actual.softmax_sum, expected.softmax_sum),
    ):
        torch.testing.assert_close(
            actual_stats,
            expected_stats,
            atol=5e-3,
            rtol=5e-3,
            msg=f"{stage}: {name} differs from eager",
        )
    torch.testing.assert_close(
        _state_lse(actual),
        _state_lse(expected),
        atol=2e-3,
        rtol=2e-3,
        msg=f"{stage}: reconstructed LSE differs from eager",
    )


def _runtime_description(device_index: int) -> str:
    return (
        f"torch={torch.__version__}, "
        f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}, "
        f"device={torch.npu.get_device_name(device_index)}"
    )


@pytest.mark.skipif(
    torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu and an Ascend NPU are required",
)
def test_sfa_statistics_survive_npugraph_dynamic_replay() -> None:
    device_index = 0
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    dtype = torch.bfloat16
    batch_size = 2
    # A non-unit head dimension makes the test sensitive to the documented
    # BSND statistics layout: [B, KV_N, query_seq_len, query_heads / KV_N].
    num_heads = 16
    capacity = 2048
    scale = 1.0 / math.sqrt(128 + 64)
    output_atol = 2e-2
    output_rtol = 2e-2
    runtime = _runtime_description(device_index)

    print(runtime)
    print(
        f"dtype={dtype} batch={batch_size} heads={num_heads} "
        f"capacity={capacity} scale={scale:.8f}"
    )

    capture_case = _make_case(
        valid_lengths=(1024, 2048),
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081400,
    )
    replay_case_a = _make_case(
        valid_lengths=(2048, 1024),
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081401,
    )
    replay_case_b = _make_case(
        valid_lengths=(512, 1536),
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081402,
    )

    try:
        eager_a = _run_sfa(replay_case_a, scale)
        eager_b = _run_sfa(replay_case_b, scale)
        torch.npu.synchronize()
    except Exception as error:
        raise AssertionError(
            "SFA statistics failed in eager mode before NPUGraph capture; "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error

    _validate_contract(eager_a, replay_case_a, stage="eager replay A")
    _validate_contract(eager_b, replay_case_b, stage="eager replay B")
    eager_a = _clone_state(eager_a)
    eager_b = _clone_state(eager_b)

    minimum_case_deltas = {
        "output": 1e-2,
        "softmax_max": 1e-2,
        "softmax_sum": 1e-1,
        "lse": 1e-2,
    }
    eager_case_deltas = {
        "output": _max_abs_error(eager_a.output, eager_b.output),
        "softmax_max": _max_abs_error(eager_a.softmax_max, eager_b.softmax_max),
        "softmax_sum": _max_abs_error(eager_a.softmax_sum, eager_b.softmax_sum),
        "lse": _max_abs_error(_state_lse(eager_a), _state_lse(eager_b)),
    }
    for name, minimum_delta in minimum_case_deltas.items():
        assert eager_case_deltas[name] > minimum_delta, (
            f"the eager {name} values differ by only "
            f"{eager_case_deltas[name]:.4e}; replay cases are not sufficiently "
            "distinct to detect a frozen graph output"
        )

    static_inputs = _clone_inputs(capture_case)
    graph_output = torch.empty_like(static_inputs.query)
    stats_shape = _expected_stats_shape(static_inputs)
    graph_softmax_max = torch.empty(stats_shape, dtype=torch.float32, device=device)
    graph_softmax_sum = torch.empty_like(graph_softmax_max)

    def run_once() -> None:
        state = _run_sfa(static_inputs, scale)
        graph_output.copy_(state.output)
        graph_softmax_max.copy_(state.softmax_max)
        graph_softmax_sum.copy_(state.softmax_sum)

    input_pointers = tuple(tensor.data_ptr() for tensor in static_inputs.tensors())
    output_pointers = (
        graph_output.data_ptr(),
        graph_softmax_max.data_ptr(),
        graph_softmax_sum.data_ptr(),
    )
    try:
        capture_stream = torch.npu.Stream()
        graph_pool = torch.npu.graph_pool_handle()
    except Exception as error:
        raise AssertionError(
            "NPUGraph runtime setup failed before SFA capture; "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error

    try:
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            for _ in range(2):
                run_once()
        torch.npu.synchronize()
        _validate_contract(
            AttentionState(graph_output, graph_softmax_max, graph_softmax_sum),
            capture_case,
            stage="graph warmup",
        )
    except Exception as error:
        raise AssertionError(
            "SFA statistics failed during pre-capture graph-stream warmup; "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error

    try:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            graph, pool=graph_pool, stream=capture_stream, auto_dispatch_capture=True,
        ):
            run_once()
        torch.npu.synchronize()
    except Exception as error:
        raise AssertionError(
            "NPUGraph could not capture SFA with return_softmax_lse=True; "
            "keep graph-mode sparse KV attention on the combined fallback (or "
            "provide a graph-compatible statistics/custom attention op); "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error

    assert tuple(tensor.data_ptr() for tensor in static_inputs.tensors()) == (
        input_pointers
    ), "capture changed a static input buffer address"
    assert (
        graph_output.data_ptr(),
        graph_softmax_max.data_ptr(),
        graph_softmax_sum.data_ptr(),
    ) == output_pointers, "capture changed a static output buffer address"

    def replay(case: SFAInputs, stage: str) -> AttentionState:
        _copy_inputs(static_inputs, case)
        graph_output.fill_(float("nan"))
        graph_softmax_max.fill_(float("nan"))
        graph_softmax_sum.fill_(float("nan"))
        torch.npu.synchronize()

        try:
            graph.replay()
            torch.npu.synchronize()
        except Exception as error:
            raise AssertionError(
                f"{stage}: NPUGraph replay failed for SFA statistics; {runtime}; "
                f"original error: {type(error).__name__}: {error}"
            ) from error

        assert tuple(tensor.data_ptr() for tensor in static_inputs.tensors()) == (
            input_pointers
        ), f"{stage}: a static input buffer address changed"
        assert (
            graph_output.data_ptr(),
            graph_softmax_max.data_ptr(),
            graph_softmax_sum.data_ptr(),
        ) == output_pointers, f"{stage}: a static output buffer address changed"

        state = _clone_state(
            AttentionState(graph_output, graph_softmax_max, graph_softmax_sum)
        )
        _validate_contract(state, case, stage=stage)
        return state

    graph_a = replay(replay_case_a, "graph replay A")
    graph_b = replay(replay_case_b, "graph replay B")

    graph_case_deltas = {
        "output": _max_abs_error(graph_a.output, graph_b.output),
        "softmax_max": _max_abs_error(graph_a.softmax_max, graph_b.softmax_max),
        "softmax_sum": _max_abs_error(graph_a.softmax_sum, graph_b.softmax_sum),
        "lse": _max_abs_error(_state_lse(graph_a), _state_lse(graph_b)),
    }
    for name, minimum_delta in minimum_case_deltas.items():
        assert graph_case_deltas[name] > minimum_delta, (
            f"NPUGraph replay did not refresh {name}: replay A/B delta="
            f"{graph_case_deltas[name]:.4e}"
        )

    _assert_matches(
        graph_a,
        eager_a,
        output_atol=output_atol,
        output_rtol=output_rtol,
        stage="graph replay A",
    )
    _assert_matches(
        graph_b,
        eager_b,
        output_atol=output_atol,
        output_rtol=output_rtol,
        stage="graph replay B",
    )

    print(
        "replay A errors: "
        f"out={_max_abs_error(graph_a.output, eager_a.output):.4e} "
        f"max={_max_abs_error(graph_a.softmax_max, eager_a.softmax_max):.4e} "
        f"sum={_max_abs_error(graph_a.softmax_sum, eager_a.softmax_sum):.4e} "
        f"lse={_max_abs_error(_state_lse(graph_a), _state_lse(eager_a)):.4e}"
    )
    print(
        "replay B errors: "
        f"out={_max_abs_error(graph_b.output, eager_b.output):.4e} "
        f"max={_max_abs_error(graph_b.softmax_max, eager_b.softmax_max):.4e} "
        f"sum={_max_abs_error(graph_b.softmax_sum, eager_b.softmax_sum):.4e} "
        f"lse={_max_abs_error(_state_lse(graph_b), _state_lse(eager_b)):.4e}"
    )
    print("PASSED: SFA output/max/sum update correctly across NPUGraph replay")


@pytest.mark.skipif(
    torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu and an Ascend NPU are required",
)
def test_two_sfa_fp32_merge_survives_npugraph_dynamic_replay() -> None:
    device_index = 0
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    dtype = torch.bfloat16
    batch_size = 2
    num_heads = 16
    capacity = 2048
    scale = 1.0 / math.sqrt(128 + 64)
    output_atol = 2e-2
    output_rtol = 2e-2
    runtime = _runtime_description(device_index)

    print(runtime)
    print(
        "single-stream split graph: "
        f"dtype={dtype} batch={batch_size} heads={num_heads} "
        f"capacity={capacity} scale={scale:.8f}"
    )

    capture_case = _make_split_case(
        hit_counts=(1024, 1536),
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081410,
    )
    replay_case_a = _make_split_case(
        hit_counts=(2048, 0),
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081411,
    )
    replay_case_b = _make_split_case(
        hit_counts=(256, 1792),
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081412,
    )

    def eager_reference(
        case: SplitSFACase, stage: str
    ) -> tuple[AttentionState, AttentionState, AttentionState, AttentionState]:
        try:
            union_state = _run_sfa(case.union, scale)
            hit_state = _run_partition_sfa(case.split, case.split.hit, scale)
            miss_state = _run_partition_sfa(case.split, case.split.miss, scale)
            merged_state = _merge_partition_states(
                hit_state,
                miss_state,
                case.split.hit.true_counts,
                case.split.miss.true_counts,
            )
            torch.npu.synchronize()
        except Exception as error:
            raise AssertionError(
                f"{stage}: eager two-SFA merge failed; {runtime}; "
                f"original error: {type(error).__name__}: {error}"
            ) from error

        _validate_contract(union_state, case.union, stage=f"{stage} union")
        _validate_contract(hit_state, case.union, stage=f"{stage} hit")
        _validate_contract(miss_state, case.union, stage=f"{stage} miss")
        _validate_contract(merged_state, case.union, stage=f"{stage} merged")
        union_state = _clone_state(union_state)
        hit_state = _clone_state(hit_state)
        miss_state = _clone_state(miss_state)
        merged_state = _clone_state(merged_state)
        _assert_matches(
            merged_state,
            union_state,
            output_atol=output_atol,
            output_rtol=output_rtol,
            stage=f"{stage} eager split vs union",
        )
        return union_state, merged_state, hit_state, miss_state

    try:
        capture_union = _run_sfa(capture_case.union, scale)
        torch.npu.synchronize()
    except Exception as error:
        raise AssertionError(
            "capture case union SFA failed before split graph setup; "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error
    _validate_contract(capture_union, capture_case.union, stage="capture union")
    capture_union = _clone_state(capture_union)

    union_a, eager_merged_a, eager_hit_a, eager_miss_a = eager_reference(
        replay_case_a, "replay A"
    )
    union_b, eager_merged_b, _, _ = eager_reference(replay_case_b, "replay B")

    minimum_case_deltas = {
        "output": 1e-2,
        "softmax_max": 1e-2,
        "softmax_sum": 1e-1,
        "lse": 1e-2,
    }
    eager_case_deltas = {
        "output": _max_abs_error(eager_merged_a.output, eager_merged_b.output),
        "softmax_max": _max_abs_error(
            eager_merged_a.softmax_max, eager_merged_b.softmax_max
        ),
        "softmax_sum": _max_abs_error(
            eager_merged_a.softmax_sum, eager_merged_b.softmax_sum
        ),
        "lse": _max_abs_error(_state_lse(eager_merged_a), _state_lse(eager_merged_b)),
    }
    for name, minimum_delta in minimum_case_deltas.items():
        assert eager_case_deltas[name] > minimum_delta, (
            f"the eager split {name} values differ by only "
            f"{eager_case_deltas[name]:.4e}; replay cases are not sufficiently "
            "distinct to detect a frozen graph output"
        )

    # Replay A has an all-hit first row and an all-miss second row. The empty
    # dummy partition must contribute exactly zero mass after masking.
    torch.testing.assert_close(
        eager_merged_a.output[0].float(),
        eager_hit_a.output[0].float(),
        atol=output_atol,
        rtol=output_rtol,
        msg="all-hit row was changed by the empty miss dummy",
    )
    torch.testing.assert_close(
        eager_merged_a.output[1].float(),
        eager_miss_a.output[1].float(),
        atol=output_atol,
        rtol=output_rtol,
        msg="all-miss row was changed by the empty hit dummy",
    )
    torch.testing.assert_close(
        _state_lse(eager_merged_a)[0],
        _state_lse(eager_hit_a)[0],
        atol=2e-3,
        rtol=2e-3,
        msg="all-hit row LSE was changed by the empty miss dummy",
    )
    torch.testing.assert_close(
        _state_lse(eager_merged_a)[1],
        _state_lse(eager_miss_a)[1],
        atol=2e-3,
        rtol=2e-3,
        msg="all-miss row LSE was changed by the empty hit dummy",
    )

    static_inputs = _clone_split_inputs(capture_case.split)
    graph_output = torch.empty_like(static_inputs.query)
    stats_shape = _expected_stats_shape(capture_case.union)
    graph_softmax_max = torch.empty(stats_shape, dtype=torch.float32, device=device)
    graph_softmax_sum = torch.empty_like(graph_softmax_max)

    def run_once() -> None:
        # No stream switch or Event is allowed in this test. Both SFA calls and
        # every FP32 merge operation are captured on the current graph stream.
        merged_state = _run_split_sfa(static_inputs, scale)
        graph_output.copy_(merged_state.output)
        graph_softmax_max.copy_(merged_state.softmax_max)
        graph_softmax_sum.copy_(merged_state.softmax_sum)

    input_pointers = tuple(tensor.data_ptr() for tensor in static_inputs.tensors())
    output_pointers = (
        graph_output.data_ptr(),
        graph_softmax_max.data_ptr(),
        graph_softmax_sum.data_ptr(),
    )
    try:
        capture_stream = torch.npu.Stream()
        graph_pool = torch.npu.graph_pool_handle()
    except Exception as error:
        raise AssertionError(
            "NPUGraph runtime setup failed before two-SFA capture; "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error

    try:
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            for _ in range(2):
                run_once()
        torch.npu.synchronize()
        warmup_state = AttentionState(
            graph_output, graph_softmax_max, graph_softmax_sum
        )
        _validate_contract(warmup_state, capture_case.union, stage="split graph warmup")
        _assert_matches(
            warmup_state,
            capture_union,
            output_atol=output_atol,
            output_rtol=output_rtol,
            stage="split graph warmup vs union",
        )
    except Exception as error:
        raise AssertionError(
            "two-SFA FP32 merge failed during graph-stream warmup; "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error

    try:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            graph, pool=graph_pool, stream=capture_stream, auto_dispatch_capture=True,
        ):
            run_once()
        torch.npu.synchronize()
    except Exception as error:
        raise AssertionError(
            "NPUGraph could not capture single-stream hit SFA, miss SFA, and "
            "FP32 merge; keep graph mode on the combined fallback; "
            f"{runtime}; original error: {type(error).__name__}: {error}"
        ) from error

    assert tuple(tensor.data_ptr() for tensor in static_inputs.tensors()) == (
        input_pointers
    ), "two-SFA capture changed a static input buffer address"
    assert (
        graph_output.data_ptr(),
        graph_softmax_max.data_ptr(),
        graph_softmax_sum.data_ptr(),
    ) == output_pointers, "two-SFA capture changed a static output buffer address"

    def replay(case: SplitSFACase, stage: str) -> AttentionState:
        _copy_split_inputs(static_inputs, case.split)
        graph_output.fill_(float("nan"))
        graph_softmax_max.fill_(float("nan"))
        graph_softmax_sum.fill_(float("nan"))
        torch.npu.synchronize()

        try:
            graph.replay()
            torch.npu.synchronize()
        except Exception as error:
            raise AssertionError(
                f"{stage}: two-SFA NPUGraph replay failed; {runtime}; "
                f"original error: {type(error).__name__}: {error}"
            ) from error

        assert tuple(tensor.data_ptr() for tensor in static_inputs.tensors()) == (
            input_pointers
        ), f"{stage}: a split graph input buffer address changed"
        assert (
            graph_output.data_ptr(),
            graph_softmax_max.data_ptr(),
            graph_softmax_sum.data_ptr(),
        ) == output_pointers, f"{stage}: a split graph output buffer address changed"

        state = _clone_state(
            AttentionState(graph_output, graph_softmax_max, graph_softmax_sum)
        )
        _validate_contract(state, case.union, stage=stage)
        return state

    graph_a = replay(replay_case_a, "two-SFA graph replay A")
    graph_b = replay(replay_case_b, "two-SFA graph replay B")

    for graph_state, eager_state, union_state, stage in (
        (graph_a, eager_merged_a, union_a, "two-SFA graph replay A"),
        (graph_b, eager_merged_b, union_b, "two-SFA graph replay B"),
    ):
        _assert_matches(
            graph_state,
            eager_state,
            output_atol=output_atol,
            output_rtol=output_rtol,
            stage=f"{stage} vs eager split",
        )
        _assert_matches(
            graph_state,
            union_state,
            output_atol=output_atol,
            output_rtol=output_rtol,
            stage=f"{stage} vs eager union",
        )

    graph_case_deltas = {
        "output": _max_abs_error(graph_a.output, graph_b.output),
        "softmax_max": _max_abs_error(graph_a.softmax_max, graph_b.softmax_max),
        "softmax_sum": _max_abs_error(graph_a.softmax_sum, graph_b.softmax_sum),
        "lse": _max_abs_error(_state_lse(graph_a), _state_lse(graph_b)),
    }
    for name, minimum_delta in minimum_case_deltas.items():
        assert graph_case_deltas[name] > minimum_delta, (
            f"two-SFA graph replay did not refresh {name}: replay A/B delta="
            f"{graph_case_deltas[name]:.4e}"
        )

    print(
        "replay A graph-vs-union errors: "
        f"out={_max_abs_error(graph_a.output, union_a.output):.4e} "
        f"max={_max_abs_error(graph_a.softmax_max, union_a.softmax_max):.4e} "
        f"sum={_max_abs_error(graph_a.softmax_sum, union_a.softmax_sum):.4e} "
        f"lse={_max_abs_error(_state_lse(graph_a), _state_lse(union_a)):.4e}"
    )
    print(
        "replay B graph-vs-union errors: "
        f"out={_max_abs_error(graph_b.output, union_b.output):.4e} "
        f"max={_max_abs_error(graph_b.softmax_max, union_b.softmax_max):.4e} "
        f"sum={_max_abs_error(graph_b.softmax_sum, union_b.softmax_sum):.4e} "
        f"lse={_max_abs_error(_state_lse(graph_b), _state_lse(union_b)):.4e}"
    )
    print("PASSED: single-stream hit/miss SFA and FP32 merge survive NPUGraph replay")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
