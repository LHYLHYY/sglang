"""Validate direct Sparse Flash Attention reads from a paged hot KV cache.

This is the first capability test for removing the hit-KV D2D gather from the
sparsity-driven KV-offload path.  It compares two calls that attend to exactly
the same selected KV rows:

* the current reference shape: selected KV in a contiguous BSND tensor;
* the proposed shape: selected physical hot slots in a global PA_BSND cache,
  reached through a per-request block table.

The paged cache uses nontrivial request IDs, randomly scattered physical slots,
and unrelated data in every unselected slot.  The test therefore fails if the
operator ignores either the block table or the physical sparse indices.
It also splits those physical indices into hit/miss partitions and verifies
that PA_BSND softmax statistics reconstruct the single-call union output.

Run on the target Ascend host with:

    python -m pytest -v -s \
        test/manual/ascend/test_sparse_flash_attention_pa_hot_cache.py
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

try:
    import torch_npu
except (ImportError, OSError):
    torch_npu = None


def _runtime_description(device_index: int) -> str:
    torch_npu_version = getattr(torch_npu, "__version__", "unknown")
    try:
        device_name = torch.npu.get_device_name(device_index)
    except Exception:
        device_name = "unknown"
    return (
        f"torch={torch.__version__}, torch_npu={torch_npu_version}, "
        f"device={device_name}"
    )


def _run_sfa(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    sparse_indices: torch.Tensor,
    actual_query_lengths: torch.Tensor,
    actual_kv_lengths: torch.Tensor,
    scale: float,
    layout_kv: str,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    output, _, _ = _run_sfa_state(
        query=query,
        key=key,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_indices=sparse_indices,
        actual_query_lengths=actual_query_lengths,
        actual_kv_lengths=actual_kv_lengths,
        scale=scale,
        layout_kv=layout_kv,
        block_table=block_table,
        return_softmax_lse=False,
    )
    return output


def _run_sfa_state(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    sparse_indices: torch.Tensor,
    actual_query_lengths: torch.Tensor,
    actual_kv_lengths: torch.Tensor,
    scale: float,
    layout_kv: str,
    block_table: torch.Tensor | None = None,
    return_softmax_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return torch_npu.npu_sparse_flash_attention(
        query=query,
        key=key,
        value=key,
        sparse_indices=sparse_indices,
        scale_value=scale,
        actual_seq_lengths_query=actual_query_lengths,
        actual_seq_lengths_kv=actual_kv_lengths,
        query_rope=query_rope,
        key_rope=key_rope,
        block_table=block_table,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv=layout_kv,
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=return_softmax_lse,
    )


def _merge_sfa_states(
    hit_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    miss_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    hit_output, hit_max, hit_sum = hit_state
    miss_output, miss_max, miss_sum = miss_state
    global_max = torch.maximum(hit_max.float(), miss_max.float())
    hit_mass = hit_sum.float() * torch.exp(hit_max.float() - global_max)
    miss_mass = miss_sum.float() * torch.exp(miss_max.float() - global_max)
    denominator = (hit_mass + miss_mass).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    hit_weight = (hit_mass / denominator).permute(0, 2, 3, 1)
    miss_weight = (miss_mass / denominator).permute(0, 2, 3, 1)
    return (
        hit_output.float() * hit_weight + miss_output.float() * miss_weight
    ).to(hit_output.dtype)


@pytest.mark.skipif(
    torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu and an Ascend NPU are required",
)
def test_pa_bsnd_hot_cache_matches_bsnd_selected_kv() -> None:
    """PA_BSND should select scattered hot slots without a D2D gather."""

    device_index = 0
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    dtype = torch.bfloat16
    batch_size = 2
    request_pool_size = 4
    request_ids = (3, 1)
    selected_counts = (1025, 1537)
    num_heads = 16
    hot_capacity = 2048
    block_size = 1024
    blocks_per_request = hot_capacity // block_size
    total_blocks = request_pool_size * blocks_per_request
    scale = 1.0 / math.sqrt(128 + 64)
    output_atol = 2e-2
    output_rtol = 2e-2

    assert hot_capacity % block_size == 0
    assert len(request_ids) == batch_size
    assert len(set(request_ids)) == batch_size
    assert all(0 <= request_id < request_pool_size for request_id in request_ids)
    assert all(0 < count <= hot_capacity for count in selected_counts)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026083101)

    def random_bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            dtype=dtype
        )

    query_cpu = random_bf16((batch_size, 1, num_heads, 512))
    query_rope_cpu = random_bf16((batch_size, 1, num_heads, 64))

    # BSND reference: the selected rows are physically compacted at the front.
    compact_key_cpu = random_bf16((batch_size, hot_capacity, 1, 512))
    compact_key_rope_cpu = random_bf16((batch_size, hot_capacity, 1, 64))
    reference_indices_cpu = torch.full(
        (batch_size, 1, 1, hot_capacity), -1, dtype=torch.int32
    )

    # PA cache: start with unrelated decoy data, then scatter the selected rows
    # into random physical hot slots belonging to nontrivial request IDs.
    paged_key_by_request_cpu = random_bf16((request_pool_size, hot_capacity, 1, 512))
    paged_key_rope_by_request_cpu = random_bf16(
        (request_pool_size, hot_capacity, 1, 64)
    )
    physical_indices_cpu = torch.full_like(reference_indices_cpu, -1)

    for batch_index, (request_id, selected_count) in enumerate(
        zip(request_ids, selected_counts, strict=True)
    ):
        physical_slots = torch.randperm(
            hot_capacity, generator=generator, dtype=torch.int64
        )[:selected_count]
        selected_positions = torch.arange(selected_count, dtype=torch.int64)

        paged_key_by_request_cpu[request_id].index_copy_(
            0,
            physical_slots,
            compact_key_cpu[batch_index].index_select(0, selected_positions),
        )
        paged_key_rope_by_request_cpu[request_id].index_copy_(
            0,
            physical_slots,
            compact_key_rope_cpu[batch_index].index_select(0, selected_positions),
        )
        reference_indices_cpu[batch_index, 0, 0, :selected_count] = torch.arange(
            selected_count, dtype=torch.int32
        )
        physical_indices_cpu[batch_index, 0, 0, :selected_count] = physical_slots.to(
            torch.int32
        )

        # Ensure this case actually exercises both PA blocks and a nonidentity
        # physical mapping instead of accidentally degenerating to BSND order.
        assert torch.any(physical_slots < block_size).item()
        assert torch.any(physical_slots >= block_size).item()
        assert not torch.equal(physical_slots, selected_positions)

    paged_key_cpu = paged_key_by_request_cpu.view(total_blocks, block_size, 1, 512)
    paged_key_rope_cpu = paged_key_rope_by_request_cpu.view(
        total_blocks, block_size, 1, 64
    )
    block_table_cpu = torch.tensor(
        [
            [
                request_id * blocks_per_request + block_offset
                for block_offset in range(blocks_per_request)
            ]
            for request_id in request_ids
        ],
        dtype=torch.int32,
    )

    query = query_cpu.to(device=device)
    query_rope = query_rope_cpu.to(device=device)
    compact_key = compact_key_cpu.to(device=device)
    compact_key_rope = compact_key_rope_cpu.to(device=device)
    paged_key = paged_key_cpu.to(device=device)
    paged_key_rope = paged_key_rope_cpu.to(device=device)
    reference_indices = reference_indices_cpu.to(device=device)
    physical_indices = physical_indices_cpu.to(device=device)
    block_table = block_table_cpu.to(device=device)
    actual_query_lengths = torch.ones(batch_size, dtype=torch.int32, device=device)
    # This describes the addressable hot-slot sequence. The sparse indices,
    # rather than this length, determine which selected rows participate.
    actual_kv_lengths = torch.full(
        (batch_size,), hot_capacity, dtype=torch.int32, device=device
    )

    try:
        reference_output = _run_sfa(
            query=query,
            key=compact_key,
            query_rope=query_rope,
            key_rope=compact_key_rope,
            sparse_indices=reference_indices,
            actual_query_lengths=actual_query_lengths,
            actual_kv_lengths=actual_kv_lengths,
            scale=scale,
            layout_kv="BSND",
        )
        paged_output = _run_sfa(
            query=query,
            key=paged_key,
            query_rope=query_rope,
            key_rope=paged_key_rope,
            sparse_indices=physical_indices,
            actual_query_lengths=actual_query_lengths,
            actual_kv_lengths=actual_kv_lengths,
            scale=scale,
            layout_kv="PA_BSND",
            block_table=block_table,
        )
        torch.npu.synchronize()
    except Exception as error:
        raise AssertionError(
            "PA_BSND SFA could not read selected physical hot-cache slots; "
            f"{_runtime_description(device_index)}; original error: "
            f"{type(error).__name__}: {error}"
        ) from error

    assert reference_output.shape == query.shape
    assert paged_output.shape == query.shape
    assert reference_output.dtype == dtype
    assert paged_output.dtype == dtype
    assert torch.isfinite(reference_output).all().item()
    assert torch.isfinite(paged_output).all().item()

    max_abs_error = (paged_output.float() - reference_output.float()).abs().max().item()
    print(_runtime_description(device_index))
    print(
        "PA_BSND direct hot-cache SFA: "
        f"requests={request_ids} selected_counts={selected_counts} "
        f"block_size={block_size} max_abs_error={max_abs_error:.4e}"
    )
    torch.testing.assert_close(
        paged_output.float().cpu(),
        reference_output.float().cpu(),
        atol=output_atol,
        rtol=output_rtol,
        msg=(
            "PA_BSND direct hot-cache output differs from the BSND selected-KV "
            "reference; block-table or physical sparse-index semantics may not "
            "match the proposed no-D2D design"
        ),
    )
    print("PASSED: PA_BSND SFA reads scattered physical hot slots without D2D")

    hit_indices = torch.full_like(physical_indices, -1)
    miss_indices = torch.full_like(physical_indices, -1)
    for batch_index, selected_count in enumerate(selected_counts):
        hit_count = selected_count * 3 // 4
        miss_count = selected_count - hit_count
        hit_indices[batch_index, 0, 0, :hit_count].copy_(
            physical_indices[batch_index, 0, 0, :hit_count]
        )
        miss_indices[batch_index, 0, 0, :miss_count].copy_(
            physical_indices[
                batch_index, 0, 0, hit_count : hit_count + miss_count
            ]
        )

    hit_state = _run_sfa_state(
        query=query,
        key=paged_key,
        query_rope=query_rope,
        key_rope=paged_key_rope,
        sparse_indices=hit_indices,
        actual_query_lengths=actual_query_lengths,
        actual_kv_lengths=actual_kv_lengths,
        scale=scale,
        layout_kv="PA_BSND",
        block_table=block_table,
    )
    miss_state = _run_sfa_state(
        query=query,
        key=paged_key,
        query_rope=query_rope,
        key_rope=paged_key_rope,
        sparse_indices=miss_indices,
        actual_query_lengths=actual_query_lengths,
        actual_kv_lengths=actual_kv_lengths,
        scale=scale,
        layout_kv="PA_BSND",
        block_table=block_table,
    )
    assert all(value is not None for value in hit_state[1:])
    assert all(value is not None for value in miss_state[1:])
    split_output = _merge_sfa_states(hit_state, miss_state)
    split_max_abs_error = (
        split_output.float() - paged_output.float()
    ).abs().max().item()
    print(
        "PA_BSND split hit/miss merge: "
        f"max_abs_error={split_max_abs_error:.4e}"
    )
    torch.testing.assert_close(
        split_output.float().cpu(),
        paged_output.float().cpu(),
        atol=output_atol,
        rtol=output_rtol,
        msg=(
            "two PA_BSND SFA partitions plus online-softmax merge differ from "
            "the union PA_BSND result"
        ),
    )
    print("PASSED: split PA-SFA statistics merge matches union PA-SFA")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
