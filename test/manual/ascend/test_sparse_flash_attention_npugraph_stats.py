"""Probe Sparse Flash Attention statistics support in an NPU graph.

This is an isolated Ascend hardware capability test. It captures one
``npu_sparse_flash_attention`` call with ``return_softmax_lse=True`` and then
replays the graph with two different, fixed-shape input sets. Each replay is
compared with an eager result, including ``attention_out``, ``softmax_max``,
``softmax_sum``, and the reconstructed LSE.

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


def _clone_inputs(inputs: SFAInputs) -> SFAInputs:
    return SFAInputs(*(tensor.clone() for tensor in inputs.tensors()))


def _copy_inputs(destination: SFAInputs, source: SFAInputs) -> None:
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
