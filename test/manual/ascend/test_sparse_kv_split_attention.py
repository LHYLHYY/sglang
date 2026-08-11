"""Validate split hit/miss Sparse Flash Attention on an Ascend NPU.

This is a manual hardware test for the sparsity-driven KV-offload decode path.
It compares one Sparse Flash Attention (SFA) call over the full selected KV set
against two SFA calls over disjoint hit/miss partitions followed by a numerically
stable merge of ``attention_out``, ``softmax_max``, and ``softmax_sum``.

Example (run on the target Ascend host):

    python -m pytest -v -s \
        test/manual/ascend/test_sparse_kv_split_attention.py

The test runs in eager mode because some torch_npu/CANN releases do not support
``return_softmax_lse=True`` during NPU graph capture.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


@dataclass
class AttentionState:
    output: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor


def _make_inputs(
    *,
    batch_size: int,
    topk: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    query = torch.randn(
        batch_size, 1, num_heads, 512, generator=generator, dtype=torch.float32,
    ).to(dtype=dtype, device=device)
    key = torch.randn(
        batch_size, topk, 1, 512, generator=generator, dtype=torch.float32,
    ).to(dtype=dtype, device=device)
    query_rope = torch.randn(
        batch_size, 1, num_heads, 64, generator=generator, dtype=torch.float32,
    ).to(dtype=dtype, device=device)
    key_rope = torch.randn(
        batch_size, topk, 1, 64, generator=generator, dtype=torch.float32,
    ).to(dtype=dtype, device=device)
    return query, key, query_rope, key_rope


def _make_sparse_indices(
    *, batch_size: int, capacity: int, valid_length: int, device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(capacity, dtype=torch.int32, device=device)
    indices = torch.where(
        indices < valid_length, indices, torch.full_like(indices, -1),
    )
    return (
        indices.view(1, 1, 1, capacity).expand(batch_size, 1, 1, capacity).contiguous()
    )


def _run_sfa(
    torch_npu,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    valid_length: int,
    scale: float,
) -> AttentionState:
    if valid_length <= 0:
        raise ValueError("SFA cannot be called with an empty KV partition")

    batch_size = query.shape[0]
    capacity = key.shape[1]
    device = query.device
    sparse_indices = _make_sparse_indices(
        batch_size=batch_size,
        capacity=capacity,
        valid_length=valid_length,
        device=device,
    )
    actual_query_lengths = torch.ones(batch_size, dtype=torch.int32, device=device)
    actual_kv_lengths = torch.full(
        (batch_size,), valid_length, dtype=torch.int32, device=device
    )

    output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query,
        key,
        key,
        sparse_indices,
        scale,
        actual_seq_lengths_query=actual_query_lengths,
        actual_seq_lengths_kv=actual_kv_lengths,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )

    if softmax_max.numel() == 0 or softmax_sum.numel() == 0:
        raise RuntimeError(
            "npu_sparse_flash_attention returned empty softmax statistics; "
            "the installed torch_npu/CANN build may not support "
            "return_softmax_lse=True for this mode"
        )

    num_kv_heads = key.shape[2]
    num_query_heads = query.shape[2]
    if num_query_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"query heads ({num_query_heads}) must be divisible by "
            f"KV heads ({num_kv_heads})"
        )
    expected_stats_shape = (
        batch_size,
        num_kv_heads,
        query.shape[1],
        num_query_heads // num_kv_heads,
    )
    if output.shape != query.shape or output.dtype != query.dtype:
        raise RuntimeError(
            "unexpected attention output contract: "
            f"got shape={tuple(output.shape)}, dtype={output.dtype}; "
            f"expected shape={tuple(query.shape)}, dtype={query.dtype}"
        )
    for name, value in (("softmax_max", softmax_max), ("softmax_sum", softmax_sum)):
        if tuple(value.shape) != expected_stats_shape or value.dtype != torch.float32:
            raise RuntimeError(
                f"unexpected {name} contract: "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}; "
                f"expected shape={expected_stats_shape}, dtype=torch.float32"
            )
    return AttentionState(output, softmax_max, softmax_sum)


def _compact_and_pad(
    tensor: torch.Tensor, token_indices: torch.Tensor, capacity: int,
) -> torch.Tensor:
    """Gather per-request tokens and pad them back to a graph-like capacity."""

    batch_size, selected_length = token_indices.shape
    if selected_length == 0:
        return torch.zeros_like(tensor)

    gather_indices = token_indices.view(batch_size, selected_length, 1, 1).expand(
        batch_size, selected_length, tensor.shape[2], tensor.shape[3],
    )
    selected = torch.gather(tensor, dim=1, index=gather_indices)
    if selected_length == capacity:
        return selected.contiguous()

    padding = torch.zeros(
        batch_size,
        capacity - selected_length,
        tensor.shape[2],
        tensor.shape[3],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([selected, padding], dim=1).contiguous()


def _make_partition_indices(
    *, batch_size: int, topk: int, hit_count: int, seed: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(topk, generator=generator) for _ in range(batch_size)]
    )
    hit_indices = permutations[:, :hit_count].sort(dim=1).values.to(device)
    miss_indices = permutations[:, hit_count:].sort(dim=1).values.to(device)
    return hit_indices, miss_indices


def _merge_states(
    hit: Optional[AttentionState], miss: Optional[AttentionState],
) -> AttentionState:
    if hit is None:
        if miss is None:
            raise ValueError("both attention partitions are empty")
        return miss
    if miss is None:
        return hit

    # softmax_max/sum have shape [B, N2, S1, N1/N2]. For the supported
    # MLA path N2 == 1, while attention_out has shape [B, S1, N1, D].
    global_max = torch.maximum(hit.softmax_max, miss.softmax_max)
    hit_mass = hit.softmax_sum * torch.exp(hit.softmax_max - global_max)
    miss_mass = miss.softmax_sum * torch.exp(miss.softmax_max - global_max)
    global_sum = hit_mass + miss_mass

    hit_weight = hit_mass.permute(0, 2, 3, 1)
    miss_weight = miss_mass.permute(0, 2, 3, 1)
    denominator = global_sum.permute(0, 2, 3, 1)
    merged_output = (
        (hit.output.float() * hit_weight + miss.output.float() * miss_weight)
        / denominator
    ).to(hit.output.dtype)

    return AttentionState(merged_output, global_max, global_sum)


def _state_lse(state: AttentionState) -> torch.Tensor:
    return state.softmax_max + torch.log(state.softmax_sum)


def _run_split(
    torch_npu,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    hit_indices: torch.Tensor,
    miss_indices: torch.Tensor,
    scale: float,
    parallel: bool,
    hit_stream,
    miss_stream,
) -> AttentionState:
    capacity = key.shape[1]
    hit_count = hit_indices.shape[1]
    miss_count = miss_indices.shape[1]

    hit_key = _compact_and_pad(key, hit_indices, capacity)
    hit_key_rope = _compact_and_pad(key_rope, hit_indices, capacity)
    miss_key = _compact_and_pad(key, miss_indices, capacity)
    miss_key_rope = _compact_and_pad(key_rope, miss_indices, capacity)

    hit_state: Optional[AttentionState] = None
    miss_state: Optional[AttentionState] = None

    if parallel and hit_count > 0 and miss_count > 0:
        current_stream = torch.npu.current_stream()
        hit_stream.wait_stream(current_stream)
        miss_stream.wait_stream(current_stream)

        with torch.npu.stream(hit_stream):
            hit_state = _run_sfa(
                torch_npu,
                query=query,
                key=hit_key,
                query_rope=query_rope,
                key_rope=hit_key_rope,
                valid_length=hit_count,
                scale=scale,
            )
        with torch.npu.stream(miss_stream):
            miss_state = _run_sfa(
                torch_npu,
                query=query,
                key=miss_key,
                query_rope=query_rope,
                key_rope=miss_key_rope,
                valid_length=miss_count,
                scale=scale,
            )

        current_stream.wait_stream(hit_stream)
        current_stream.wait_stream(miss_stream)
    else:
        if hit_count > 0:
            hit_state = _run_sfa(
                torch_npu,
                query=query,
                key=hit_key,
                query_rope=query_rope,
                key_rope=hit_key_rope,
                valid_length=hit_count,
                scale=scale,
            )
        if miss_count > 0:
            miss_state = _run_sfa(
                torch_npu,
                query=query,
                key=miss_key,
                query_rope=query_rope,
                key_rope=miss_key_rope,
                valid_length=miss_count,
                scale=scale,
            )

    return _merge_states(hit_state, miss_state)


def _max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.abs().clamp_min(1e-5)
    return float(((actual - expected).abs() / denominator).max().item())


@pytest.mark.skipif(
    torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu and an Ascend NPU are required",
)
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        pytest.param(torch.bfloat16, 2e-2, 2e-2, id="bf16"),
        pytest.param(torch.float16, 5e-3, 5e-3, id="fp16"),
    ],
)
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_split_hit_miss_attention_matches_one_shot(
    dtype: torch.dtype, atol: float, rtol: float, parallel: bool,
) -> None:
    device_index = 0
    device = torch.device(f"npu:{device_index}")
    torch.npu.set_device(device_index)
    batch_size = 1
    num_heads = 1
    topk = 2048
    ratios = (0.0, 0.25, 0.5, 0.75, 1.0)
    trials = 3
    seed = 20260811
    # DeepSeek MLA keeps the model's original QK scaling after absorbing the
    # projection into the 512-dimensional latent query/key representation.
    scale = 1.0 / math.sqrt(128 + 64)
    lse_atol = 2e-3

    print(f"torch={torch.__version__}")
    print(f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}")
    print(f"device={torch.npu.get_device_name(device_index)}")
    print(
        f"dtype={dtype} batch={batch_size} heads={num_heads} "
        f"topk={topk} scale={scale:.8f} parallel={parallel}"
    )
    print(f"tolerances: atol={atol} rtol={rtol} lse_atol={lse_atol}")

    query, key, query_rope, key_rope = _make_inputs(
        batch_size=batch_size,
        topk=topk,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    baseline = _run_sfa(
        torch_npu,
        query=query,
        key=key,
        query_rope=query_rope,
        key_rope=key_rope,
        valid_length=topk,
        scale=scale,
    )
    torch.npu.synchronize()
    baseline_lse = _state_lse(baseline)

    hit_stream = torch.npu.Stream() if parallel else None
    miss_stream = torch.npu.Stream() if parallel else None

    print("\n ratio  trial   hit  miss     max_abs     max_rel     lse_abs  result")
    print("-" * 76)

    failures = []
    for ratio in ratios:
        hit_count = int(round(topk * ratio))
        hit_count = min(max(hit_count, 0), topk)
        miss_count = topk - hit_count

        for trial in range(trials):
            hit_indices, miss_indices = _make_partition_indices(
                batch_size=batch_size,
                topk=topk,
                hit_count=hit_count,
                seed=seed + 1000 * trial + hit_count,
                device=device,
            )
            merged = _run_split(
                torch_npu,
                query=query,
                key=key,
                query_rope=query_rope,
                key_rope=key_rope,
                hit_indices=hit_indices,
                miss_indices=miss_indices,
                scale=scale,
                parallel=parallel,
                hit_stream=hit_stream,
                miss_stream=miss_stream,
            )
            torch.npu.synchronize()

            actual = merged.output.float()
            expected = baseline.output.float()
            max_abs = float((actual - expected).abs().max().item())
            max_rel = _max_relative_error(actual, expected)
            lse_abs = float((_state_lse(merged) - baseline_lse).abs().max().item())
            output_close = torch.allclose(actual, expected, atol=atol, rtol=rtol)
            lse_close = lse_abs <= lse_atol
            passed = output_close and lse_close
            if not passed:
                failures.append(
                    f"ratio={ratio:.2f} trial={trial} max_abs={max_abs:.4e} "
                    f"max_rel={max_rel:.4e} lse_abs={lse_abs:.4e}"
                )

            print(
                f"{ratio:6.2f} {trial:6d} {hit_count:5d} {miss_count:5d} "
                f"{max_abs:11.4e} {max_rel:11.4e} {lse_abs:11.4e}  "
                f"{'PASS' if passed else 'FAIL'}"
            )

    assert not failures, "split-attention cases exceeded tolerance:\n" + "\n".join(
        failures
    )
    print("\nPASSED: split hit/miss attention matches one-shot attention")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
