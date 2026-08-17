"""Probe Sparse Flash Attention statistics support in an NPU graph.

This file contains isolated Ascend hardware capability tests. The first test
captures one ``npu_sparse_flash_attention`` call with
``return_softmax_lse=True``. The second captures two fixed-capacity hit/miss SFA
calls followed by a numerically stable FP32 merge on the same stream. The third
adds the real sparse-KV lookup and compact-copy kernels in front of the same two
SFA calls and merge. All tests replay with different fixed-shape inputs and
compare graph results with eager references, including ``attention_out``,
``softmax_max``, ``softmax_sum``, and the reconstructed LSE.

The test deliberately fails (rather than xfails) when an installed
torch_npu/CANN combination cannot capture or replay the statistics outputs. A
failure means the split-attention path must continue to use the eager-only path
or the graph-mode combined fallback on that target.

Example (run on the target Ascend host):

    python -m pytest -v -s \
        test/manual/ascend/test_sparse_flash_attention_npugraph_stats.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import uuid
from dataclasses import dataclass

import pytest
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    import sgl_kernel_npu.sparsity_driven_kv_offload as sparse_kv_ops
except (ImportError, OSError) as error:
    sparse_kv_ops = None
    _SPARSE_KV_OPS_IMPORT_ERROR = error
else:
    _SPARSE_KV_OPS_IMPORT_ERROR = None


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


@dataclass
class PrefetchCompactCase:
    union: SFAInputs
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    topk_indices: torch.Tensor
    slot_map: torch.Tensor
    device_kv: torch.Tensor
    host_kv: torch.Tensor
    expected_token_on_device: torch.Tensor
    expected_device_token_pos: torch.Tensor
    expected_hit_kv: torch.Tensor
    expected_miss_kv: torch.Tensor
    hit_counts: tuple[int, ...]
    miss_counts: tuple[int, ...]


@dataclass
class StaticPrefetchInputs:
    query: torch.Tensor
    query_rope: torch.Tensor
    actual_query_lengths: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    topk_indices: torch.Tensor
    slot_map: torch.Tensor
    device_kv: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.query,
            self.query_rope,
            self.actual_query_lengths,
            self.req_pool_indices,
            self.seq_lens,
            self.topk_indices,
            self.slot_map,
            self.device_kv,
        )


@dataclass
class PrefetchGraphBuffers:
    hit_kv: torch.Tensor
    miss_kv: torch.Tensor
    token_on_device: torch.Tensor
    device_token_pos: torch.Tensor
    hit_counts: torch.Tensor
    miss_counts: torch.Tensor
    output: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hit_kv,
            self.miss_kv,
            self.token_on_device,
            self.device_token_pos,
            self.hit_counts,
            self.miss_counts,
            self.output,
            self.softmax_max,
            self.softmax_sum,
        )


@dataclass
class PrefetchSnapshot:
    state: AttentionState
    hit_kv: torch.Tensor
    miss_kv: torch.Tensor
    token_on_device: torch.Tensor
    device_token_pos: torch.Tensor
    hit_counts: torch.Tensor
    miss_counts: torch.Tensor


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


def _make_prefetch_compact_case(
    *,
    hit_counts: tuple[int, ...],
    req_pool_indices: tuple[int, ...],
    capacity: int,
    max_context_len: int,
    request_pool_size: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> PrefetchCompactCase:
    if not hit_counts:
        raise ValueError("hit_counts must contain at least one request")
    if len(req_pool_indices) != len(hit_counts):
        raise ValueError(
            "req_pool_indices and hit_counts must have the same length, got "
            f"{len(req_pool_indices)} and {len(hit_counts)}"
        )
    if len(hit_counts) > request_pool_size:
        raise ValueError(
            f"batch size {len(hit_counts)} exceeds request pool size "
            f"{request_pool_size}"
        )
    if max_context_len < capacity:
        raise ValueError(
            f"max_context_len={max_context_len} must be >= capacity={capacity}"
        )
    if any(count < 0 or count > capacity for count in hit_counts):
        raise ValueError(f"hit counts must be in [0, {capacity}], got {hit_counts}")
    if any(req < 0 or req >= request_pool_size for req in req_pool_indices):
        raise ValueError(
            f"request IDs must be in [0, {request_pool_size}), got "
            f"{req_pool_indices}"
        )
    if len(set(req_pool_indices)) != len(req_pool_indices):
        raise ValueError(f"request IDs must be unique, got {req_pool_indices}")

    batch_size = len(hit_counts)
    miss_counts = tuple(capacity - count for count in hit_counts)
    head_dim = 512 + 64
    slot_map_width = (max_context_len // 8 + 1) * 8
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def make_cpu_random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)

    query = make_cpu_random((batch_size, 1, num_heads, 512)).to(device)
    query_rope = make_cpu_random((batch_size, 1, num_heads, 64)).to(device)
    union_kv = torch.empty(
        (batch_size, capacity, 1, head_dim), dtype=dtype, device="cpu"
    )
    host_kv = torch.zeros(
        (request_pool_size, max_context_len, 1, head_dim), dtype=dtype, device="cpu",
    )
    device_kv = torch.zeros(
        (request_pool_size, capacity, 1, head_dim), dtype=dtype, device="cpu"
    )
    slot_map = torch.full(
        (request_pool_size + 1, slot_map_width), -1, dtype=torch.int32, device="cpu",
    )
    topk_indices = torch.empty((batch_size, capacity), dtype=torch.int32, device="cpu")
    expected_token_on_device = torch.zeros_like(topk_indices)
    expected_device_token_pos = torch.full_like(topk_indices, -1)
    expected_hit_kv = torch.zeros_like(union_kv)
    expected_miss_kv = torch.zeros_like(union_kv)

    for batch_index, hit_count in enumerate(hit_counts):
        request_id = req_pool_indices[batch_index]
        logical_tokens = torch.randperm(
            max_context_len, generator=generator, dtype=torch.int64
        )[:capacity]
        selected_kv = make_cpu_random((capacity, 1, head_dim))
        union_kv[batch_index].copy_(selected_kv)
        topk_indices[batch_index].copy_(logical_tokens.to(torch.int32))
        host_kv[request_id].index_copy_(0, logical_tokens, selected_kv)

        partition_order = torch.randperm(capacity, generator=generator)
        hit_positions = partition_order[:hit_count]
        hit_mask = torch.zeros(capacity, dtype=torch.bool)
        hit_mask[hit_positions] = True
        stable_hit_positions = torch.nonzero(hit_mask, as_tuple=False).squeeze(1)
        stable_miss_positions = torch.nonzero(~hit_mask, as_tuple=False).squeeze(1)

        if hit_count > 0:
            hot_slots = torch.randperm(capacity, generator=generator)[:hit_count]
            logical_hit_tokens = logical_tokens.index_select(0, hit_positions)
            selected_hit_kv = selected_kv.index_select(0, hit_positions)
            slot_map[request_id].index_copy_(
                0, logical_hit_tokens, hot_slots.to(torch.int32)
            )
            device_kv[request_id].index_copy_(0, hot_slots, selected_hit_kv)
            expected_token_on_device[batch_index, hit_positions] = 1
            expected_device_token_pos[batch_index, hit_positions] = hot_slots.to(
                torch.int32
            )
            expected_hit_kv[batch_index, :hit_count].copy_(
                selected_kv.index_select(0, stable_hit_positions)
            )

        miss_count = capacity - hit_count
        if miss_count > 0:
            expected_miss_kv[batch_index, :miss_count].copy_(
                selected_kv.index_select(0, stable_miss_positions)
            )

    positions = torch.arange(capacity, dtype=torch.int32, device=device)
    sparse_indices = positions.view(1, 1, 1, capacity).expand(
        batch_size, 1, 1, capacity
    )
    union = SFAInputs(
        query=query,
        key=union_kv[..., :512].contiguous().to(device),
        query_rope=query_rope,
        key_rope=union_kv[..., 512:].contiguous().to(device),
        sparse_indices=sparse_indices.contiguous(),
        actual_query_lengths=torch.ones(batch_size, dtype=torch.int32, device=device),
        actual_kv_lengths=torch.full(
            (batch_size,), capacity, dtype=torch.int32, device=device
        ),
    )
    return PrefetchCompactCase(
        union=union,
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        seq_lens=torch.full(
            (batch_size,), max_context_len, dtype=torch.int32, device="cpu"
        ),
        topk_indices=topk_indices,
        slot_map=slot_map,
        device_kv=device_kv,
        host_kv=host_kv,
        expected_token_on_device=expected_token_on_device,
        expected_device_token_pos=expected_device_token_pos,
        expected_hit_kv=expected_hit_kv,
        expected_miss_kv=expected_miss_kv,
        hit_counts=hit_counts,
        miss_counts=miss_counts,
    )


def _copy_prefetch_case(
    static_inputs: StaticPrefetchInputs,
    host_kv: torch.Tensor,
    case: PrefetchCompactCase,
) -> None:
    static_inputs.query.copy_(case.union.query)
    static_inputs.query_rope.copy_(case.union.query_rope)
    static_inputs.actual_query_lengths.copy_(case.union.actual_query_lengths)
    static_inputs.req_pool_indices.copy_(case.req_pool_indices)
    static_inputs.seq_lens.copy_(case.seq_lens)
    static_inputs.topk_indices.copy_(case.topk_indices)
    static_inputs.slot_map.copy_(case.slot_map)
    static_inputs.device_kv.copy_(case.device_kv)
    # This is the same registered host allocation for warmup, capture, and
    # every replay. The raw device-visible address is intentionally immutable.
    host_kv.copy_(case.host_kv)


def _build_dynamic_partition_metadata(
    counts: torch.Tensor, capacity: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = counts.shape[0]
    positions = torch.arange(capacity, dtype=torch.int32, device=counts.device).view(
        1, capacity
    )
    valid = positions < counts.view(batch_size, 1)
    indices = torch.where(
        valid,
        positions.expand(batch_size, capacity),
        torch.full((batch_size, capacity), -1, dtype=torch.int32, device=counts.device),
    )
    indices[:, 0] = torch.where(
        counts > 0,
        indices[:, 0],
        torch.zeros(batch_size, dtype=torch.int32, device=counts.device),
    )
    actual_lengths = counts.clamp(min=1, max=capacity).to(torch.int32).contiguous()
    return (
        indices.view(batch_size, 1, 1, capacity).contiguous(),
        actual_lengths,
    )


def _run_prefetch_compact_split(
    static_inputs: StaticPrefetchInputs,
    buffers: PrefetchGraphBuffers,
    host_kv: torch.Tensor,
    host_kv_dev_ptr: int,
    *,
    request_pool_size: int,
    max_context_len: int,
    capacity: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sparse_kv_ops is None:
        raise RuntimeError("sgl_kernel_npu sparse-KV operators are required")

    req_pool_indices = static_inputs.req_pool_indices.to(torch.long).contiguous()
    valid_req_mask = (req_pool_indices >= 0) & (req_pool_indices < request_pool_size)
    slot_map_row_indices = torch.where(
        valid_req_mask,
        req_pool_indices,
        torch.full_like(req_pool_indices, request_pool_size),
    )
    device_cache_row_indices = torch.where(
        valid_req_mask, req_pool_indices, torch.zeros_like(req_pool_indices),
    )
    topk_indices = static_inputs.topk_indices
    batch_size, topk_len = topk_indices.shape
    if topk_len != capacity:
        raise RuntimeError(f"expected fixed top-k={capacity}, got {topk_len}")

    valid_topk_mask = (
        (topk_indices >= 0)
        & (topk_indices < max_context_len)
        & valid_req_mask.unsqueeze(1)
        & (static_inputs.seq_lens[:batch_size] > 0).view(batch_size, 1)
    )
    token_on_device, device_token_pos = sparse_kv_ops.slot_map_lookup(
        static_inputs.slot_map,
        slot_map_row_indices.to(torch.int32).contiguous(),
        topk_indices.to(torch.int32).contiguous(),
    )
    hit_valid = token_on_device.to(torch.bool) & valid_topk_mask
    miss_valid = (~hit_valid) & valid_topk_mask
    hit_counts = hit_valid.sum(dim=1).to(torch.int32).contiguous()
    miss_counts = miss_valid.sum(dim=1).to(torch.int32).contiguous()
    hit_ranks = torch.cumsum(hit_valid.to(torch.int32), dim=1).to(torch.int64) - 1
    miss_ranks = torch.cumsum(miss_valid.to(torch.int32), dim=1).to(torch.int64) - 1

    batch_offsets = (
        torch.arange(batch_size, dtype=torch.int64, device=topk_indices.device)
        .unsqueeze(1)
        .mul(capacity)
    )
    hit_compact_indices = (
        (batch_offsets + hit_ranks.clamp(min=0, max=capacity - 1))
        .reshape(-1)
        .contiguous()
    )
    miss_compact_indices = (
        (batch_offsets + miss_ranks.clamp(min=0, max=capacity - 1))
        .reshape(-1)
        .contiguous()
    )

    request_device_offsets = device_cache_row_indices.unsqueeze(1) * capacity
    hit_src_indices = (
        (
            request_device_offsets
            + device_token_pos.to(torch.int64).clamp(min=0, max=capacity - 1)
        )
        .reshape(-1)
        .contiguous()
    )
    request_host_offsets = device_cache_row_indices.unsqueeze(1) * max_context_len
    miss_src_indices = (
        (
            request_host_offsets
            + topk_indices.to(torch.int64).clamp(min=0, max=max_context_len - 1)
        )
        .reshape(-1)
        .contiguous()
    )
    hit_valid_flat = hit_valid.reshape(-1).contiguous()
    miss_valid_flat = miss_valid.reshape(-1).contiguous()

    # Empty partitions use a zero-valued dummy token. All other valid compact
    # rows overwrite slot zero through the copies below.
    buffers.hit_kv[:, :1].zero_()
    buffers.miss_kv[:, :1].zero_()
    sparse_kv_ops.unidex_copy_inplace(
        static_inputs.device_kv,
        buffers.hit_kv,
        hit_src_indices,
        hit_compact_indices,
        hit_valid_flat,
        2,
        2,
        block_dim=24,
    )
    sparse_kv_ops.unidex_copy_inplace(
        host_kv,
        buffers.miss_kv,
        miss_src_indices,
        miss_compact_indices,
        miss_valid_flat,
        2,
        2,
        block_dim=24,
        src_ptr=host_kv_dev_ptr,
    )

    hit_sparse_indices, hit_actual_lengths = _build_dynamic_partition_metadata(
        hit_counts, capacity
    )
    miss_sparse_indices, miss_actual_lengths = _build_dynamic_partition_metadata(
        miss_counts, capacity
    )
    hit_key, hit_key_rope = torch.split(buffers.hit_kv, (512, 64), dim=-1)
    miss_key, miss_key_rope = torch.split(buffers.miss_kv, (512, 64), dim=-1)
    split_inputs = SplitSFAInputs(
        query=static_inputs.query,
        query_rope=static_inputs.query_rope,
        actual_query_lengths=static_inputs.actual_query_lengths,
        hit=SFAPartitionInputs(
            key=hit_key.contiguous(),
            key_rope=hit_key_rope.contiguous(),
            sparse_indices=hit_sparse_indices,
            actual_kv_lengths=hit_actual_lengths,
            true_counts=hit_counts,
        ),
        miss=SFAPartitionInputs(
            key=miss_key.contiguous(),
            key_rope=miss_key_rope.contiguous(),
            sparse_indices=miss_sparse_indices,
            actual_kv_lengths=miss_actual_lengths,
            true_counts=miss_counts,
        ),
    )
    merged_state = _run_split_sfa(split_inputs, scale)

    buffers.token_on_device.copy_(token_on_device)
    buffers.device_token_pos.copy_(device_token_pos)
    buffers.hit_counts.copy_(hit_counts)
    buffers.miss_counts.copy_(miss_counts)
    buffers.output.copy_(merged_state.output)
    buffers.softmax_max.copy_(merged_state.softmax_max)
    buffers.softmax_sum.copy_(merged_state.softmax_sum)
    return token_on_device, device_token_pos


def _run_prefetch_compact_split_dual_stream_refill(
    static_inputs: StaticPrefetchInputs,
    buffers: PrefetchGraphBuffers,
    host_kv: torch.Tensor,
    host_kv_dev_ptr: int,
    miss_stream,
    inputs_ready,
    hit_copy_done,
    miss_attention_done,
    refill_done,
    *,
    request_pool_size: int,
    max_context_len: int,
    capacity: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a two-stream compact/SFA path and publish the refill in-graph.

    The caller's current stream is the graph's main/hit stream.  All lookup and
    compact-index tensors are produced there, then ``inputs_ready`` releases the
    registered-host-memory copy and miss SFA on ``miss_stream``.  That worker
    refills the fixed hot-cache slots after ``hit_copy_done`` and publishes the
    corresponding slot-map entries.  The main stream joins miss attention for
    merge, then joins ``refill_done`` before returning.  Streams and Events are
    supplied by the caller so capture never creates runtime objects whose
    lifetime would end before replay.
    """

    if sparse_kv_ops is None:
        raise RuntimeError("sgl_kernel_npu sparse-KV operators are required")

    main_stream = torch.npu.current_stream()
    req_pool_indices = static_inputs.req_pool_indices.to(torch.long).contiguous()
    valid_req_mask = (req_pool_indices >= 0) & (req_pool_indices < request_pool_size)
    slot_map_row_indices = torch.where(
        valid_req_mask,
        req_pool_indices,
        torch.full_like(req_pool_indices, request_pool_size),
    )
    device_cache_row_indices = torch.where(
        valid_req_mask, req_pool_indices, torch.zeros_like(req_pool_indices)
    )
    topk_indices = static_inputs.topk_indices
    batch_size, topk_len = topk_indices.shape
    if topk_len != capacity:
        raise RuntimeError(f"expected fixed top-k={capacity}, got {topk_len}")

    valid_topk_mask = (
        (topk_indices >= 0)
        & (topk_indices < max_context_len)
        & valid_req_mask.unsqueeze(1)
        & (static_inputs.seq_lens[:batch_size] > 0).view(batch_size, 1)
    )
    token_on_device, device_token_pos = sparse_kv_ops.slot_map_lookup(
        static_inputs.slot_map,
        slot_map_row_indices.to(torch.int32).contiguous(),
        topk_indices.to(torch.int32).contiguous(),
    )
    hit_valid = token_on_device.to(torch.bool) & valid_topk_mask
    miss_valid = (~hit_valid) & valid_topk_mask
    hit_counts = hit_valid.sum(dim=1).to(torch.int32).contiguous()
    miss_counts = miss_valid.sum(dim=1).to(torch.int32).contiguous()
    hit_ranks = torch.cumsum(hit_valid.to(torch.int32), dim=1).to(torch.int64) - 1
    miss_ranks = torch.cumsum(miss_valid.to(torch.int32), dim=1).to(torch.int64) - 1

    batch_offsets = (
        torch.arange(batch_size, dtype=torch.int64, device=topk_indices.device)
        .unsqueeze(1)
        .mul(capacity)
    )
    hit_compact_indices = (
        (batch_offsets + hit_ranks.clamp(min=0, max=capacity - 1))
        .reshape(-1)
        .contiguous()
    )
    miss_compact_indices = (
        (batch_offsets + miss_ranks.clamp(min=0, max=capacity - 1))
        .reshape(-1)
        .contiguous()
    )
    request_device_offsets = device_cache_row_indices.unsqueeze(1) * capacity
    hit_src_indices = (
        (
            request_device_offsets
            + device_token_pos.to(torch.int64).clamp(min=0, max=capacity - 1)
        )
        .reshape(-1)
        .contiguous()
    )
    request_host_offsets = device_cache_row_indices.unsqueeze(1) * max_context_len
    miss_src_indices = (
        (
            request_host_offsets
            + topk_indices.to(torch.int64).clamp(min=0, max=max_context_len - 1)
        )
        .reshape(-1)
        .contiguous()
    )
    hit_valid_flat = hit_valid.reshape(-1).contiguous()
    miss_valid_flat = miss_valid.reshape(-1).contiguous()

    # Both SFA calls retain their fixed capacity.  Empty partitions consume the
    # zero dummy at slot zero and are neutralized by true_counts during merge.
    buffers.hit_kv[:, :1].zero_()
    buffers.miss_kv[:, :1].zero_()
    hit_sparse_indices, hit_actual_lengths = _build_dynamic_partition_metadata(
        hit_counts, capacity
    )
    miss_sparse_indices, miss_actual_lengths = _build_dynamic_partition_metadata(
        miss_counts, capacity
    )
    inputs_ready.record(main_stream)

    # Main/hit path. Record copy completion separately so refill may overlap the
    # hit SFA without reading hit_kv before its compact copy has completed.
    sparse_kv_ops.unidex_copy_inplace(
        static_inputs.device_kv,
        buffers.hit_kv,
        hit_src_indices,
        hit_compact_indices,
        hit_valid_flat,
        2,
        2,
        block_dim=24,
    )
    hit_copy_done.record(main_stream)
    hit_key, hit_key_rope = torch.split(buffers.hit_kv, (512, 64), dim=-1)
    hit_partition = SFAPartitionInputs(
        key=hit_key.contiguous(),
        key_rope=hit_key_rope.contiguous(),
        sparse_indices=hit_sparse_indices,
        actual_kv_lengths=hit_actual_lengths,
        true_counts=hit_counts,
    )
    hit_inputs = SplitSFAInputs(
        query=static_inputs.query,
        query_rope=static_inputs.query_rope,
        actual_query_lengths=static_inputs.actual_query_lengths,
        hit=hit_partition,
        miss=hit_partition,
    )
    hit_state = _run_partition_sfa(hit_inputs, hit_partition, scale)

    # Worker/miss path, followed by stateful refill. The miss-attention Event is
    # deliberately recorded before refill so main-stream merge can overlap the
    # cache update, matching the production graph-dual ordering.
    with torch.npu.stream(miss_stream):
        miss_stream.wait_event(inputs_ready)
        sparse_kv_ops.unidex_copy_inplace(
            host_kv,
            buffers.miss_kv,
            miss_src_indices,
            miss_compact_indices,
            miss_valid_flat,
            2,
            2,
            block_dim=24,
            src_ptr=host_kv_dev_ptr,
        )
        miss_key, miss_key_rope = torch.split(buffers.miss_kv, (512, 64), dim=-1)
        miss_partition = SFAPartitionInputs(
            key=miss_key.contiguous(),
            key_rope=miss_key_rope.contiguous(),
            sparse_indices=miss_sparse_indices,
            actual_kv_lengths=miss_actual_lengths,
            true_counts=miss_counts,
        )
        miss_inputs = SplitSFAInputs(
            query=static_inputs.query,
            query_rope=static_inputs.query_rope,
            actual_query_lengths=static_inputs.actual_query_lengths,
            hit=miss_partition,
            miss=miss_partition,
        )
        miss_state = _run_partition_sfa(miss_inputs, miss_partition, scale)
        miss_attention_done.record(miss_stream)

        miss_stream.wait_event(hit_copy_done)
        topk_positions = torch.arange(
            capacity, dtype=torch.int64, device=topk_indices.device
        ).unsqueeze(0)
        refill_dst_indices = (
            (request_device_offsets + topk_positions).reshape(-1).contiguous()
        )
        sparse_kv_ops.unidex_copy_inplace(
            buffers.hit_kv,
            static_inputs.device_kv,
            hit_compact_indices,
            refill_dst_indices,
            hit_valid_flat,
            2,
            2,
            block_dim=24,
        )
        sparse_kv_ops.unidex_copy_inplace(
            buffers.miss_kv,
            static_inputs.device_kv,
            miss_compact_indices,
            refill_dst_indices,
            miss_valid_flat,
            2,
            2,
            block_dim=24,
        )
        static_inputs.slot_map.index_fill_(0, slot_map_row_indices, -1)
        cache_slot_ids = torch.arange(
            capacity, dtype=torch.int32, device=topk_indices.device
        ).unsqueeze(0)
        slot_map_token_indices = torch.where(
            valid_topk_mask,
            topk_indices.to(torch.long),
            torch.full_like(topk_indices, max_context_len, dtype=torch.long),
        )
        slot_map_slot_values = torch.where(
            valid_topk_mask, cache_slot_ids, torch.full_like(cache_slot_ids, -1),
        )
        slot_map_flat_indices = (
            slot_map_row_indices.unsqueeze(1) * static_inputs.slot_map.shape[1]
            + slot_map_token_indices
        ).reshape(-1)
        static_inputs.slot_map.view(-1).scatter_(
            0, slot_map_flat_indices, slot_map_slot_values.reshape(-1)
        )
        refill_done.record(miss_stream)

    main_stream.wait_event(miss_attention_done)
    merged_state = _merge_partition_states(
        hit_state, miss_state, hit_counts, miss_counts
    )
    main_stream.wait_event(refill_done)

    buffers.token_on_device.copy_(token_on_device)
    buffers.device_token_pos.copy_(device_token_pos)
    buffers.hit_counts.copy_(hit_counts)
    buffers.miss_counts.copy_(miss_counts)
    buffers.output.copy_(merged_state.output)
    buffers.softmax_max.copy_(merged_state.softmax_max)
    buffers.softmax_sum.copy_(merged_state.softmax_sum)
    return token_on_device, device_token_pos


def _snapshot_prefetch_buffers(buffers: PrefetchGraphBuffers) -> PrefetchSnapshot:
    return PrefetchSnapshot(
        state=_clone_state(
            AttentionState(buffers.output, buffers.softmax_max, buffers.softmax_sum)
        ),
        hit_kv=buffers.hit_kv.detach().clone(),
        miss_kv=buffers.miss_kv.detach().clone(),
        token_on_device=buffers.token_on_device.detach().clone(),
        device_token_pos=buffers.device_token_pos.detach().clone(),
        hit_counts=buffers.hit_counts.detach().clone(),
        miss_counts=buffers.miss_counts.detach().clone(),
    )


def _assert_prefetch_compaction(
    snapshot: PrefetchSnapshot, case: PrefetchCompactCase, *, stage: str
) -> None:
    torch.testing.assert_close(
        snapshot.token_on_device.cpu(),
        case.expected_token_on_device,
        atol=0,
        rtol=0,
        msg=f"{stage}: slot-map hit flags differ from the CPU oracle",
    )
    torch.testing.assert_close(
        snapshot.device_token_pos.cpu(),
        case.expected_device_token_pos,
        atol=0,
        rtol=0,
        msg=f"{stage}: slot-map device positions differ from the CPU oracle",
    )
    expected_hit_counts = torch.tensor(case.hit_counts, dtype=torch.int32)
    expected_miss_counts = torch.tensor(case.miss_counts, dtype=torch.int32)
    torch.testing.assert_close(
        snapshot.hit_counts.cpu(), expected_hit_counts, atol=0, rtol=0
    )
    torch.testing.assert_close(
        snapshot.miss_counts.cpu(), expected_miss_counts, atol=0, rtol=0
    )

    for batch_index, (hit_count, miss_count) in enumerate(
        zip(case.hit_counts, case.miss_counts, strict=True)
    ):
        if hit_count > 0:
            torch.testing.assert_close(
                snapshot.hit_kv[batch_index, :hit_count].cpu(),
                case.expected_hit_kv[batch_index, :hit_count],
                atol=0,
                rtol=0,
                msg=f"{stage}: row {batch_index} hit compact prefix is incorrect",
            )
        else:
            assert torch.count_nonzero(snapshot.hit_kv[batch_index, 0]).item() == 0, (
                f"{stage}: row {batch_index} empty hit partition did not get a "
                "zero dummy"
            )
        if miss_count > 0:
            torch.testing.assert_close(
                snapshot.miss_kv[batch_index, :miss_count].cpu(),
                case.expected_miss_kv[batch_index, :miss_count],
                atol=0,
                rtol=0,
                msg=f"{stage}: row {batch_index} miss compact prefix is incorrect",
            )
        else:
            assert torch.count_nonzero(snapshot.miss_kv[batch_index, 0]).item() == 0, (
                f"{stage}: row {batch_index} empty miss partition did not get a "
                "zero dummy"
            )


def _assert_refill_and_slot_map_published(
    static_inputs: StaticPrefetchInputs,
    case: PrefetchCompactCase,
    *,
    capacity: int,
    stage: str,
) -> None:
    expected_kv = torch.cat((case.union.key, case.union.key_rope), dim=-1)
    expected_slots = torch.arange(capacity, dtype=torch.int32)
    for batch_index, request_id_tensor in enumerate(case.req_pool_indices):
        request_id = int(request_id_tensor.item())
        torch.testing.assert_close(
            static_inputs.device_kv[request_id, :capacity].cpu(),
            expected_kv[batch_index].cpu(),
            atol=0,
            rtol=0,
            msg=f"{stage}: request {request_id} hot-cache refill is incorrect",
        )
        logical_tokens = case.topk_indices[batch_index].to(torch.long)
        torch.testing.assert_close(
            static_inputs.slot_map[request_id]
            .index_select(0, logical_tokens.to(static_inputs.slot_map.device))
            .cpu(),
            expected_slots,
            atol=0,
            rtol=0,
            msg=f"{stage}: request {request_id} slot-map publish is incorrect",
        )


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


@pytest.mark.skipif(
    torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu and an Ascend NPU are required",
)
def test_real_prefetch_compact_two_sfa_merge_survives_npugraph() -> None:
    """Capture lookup, D2D/H2D compact, two SFA calls, and merge on one stream.

    The test intentionally excludes refill, slot-map mutation, auxiliary
    streams, and Events. ``free_shm`` frees every SHM allocation registered by
    this process, so run this manual capability test in an isolated pytest
    process rather than inside a live SGLang server. Requests and top-k entries
    remain valid here; padded/sentinel rows are a separate follow-up boundary.
    """

    if sparse_kv_ops is None:
        pytest.fail(
            "the matching sgl_kernel_npu sparse-KV package is required for the "
            "real prefetch NPUGraph test; import failed with "
            f"{type(_SPARSE_KV_OPS_IMPORT_ERROR).__name__}: "
            f"{_SPARSE_KV_OPS_IMPORT_ERROR}",
            pytrace=False,
        )

    device_index = 0
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    dtype = torch.bfloat16
    batch_size = 2
    # Request IDs deliberately differ from batch rows so the test catches an
    # incorrect b*K/b*C source offset in place of req_id*K/req_id*C.
    request_pool_size = 4
    num_heads = 16
    capacity = 2048
    max_context_len = 4096
    head_dim = 512 + 64
    slot_map_width = (max_context_len // 8 + 1) * 8
    scale = 1.0 / math.sqrt(128 + 64)
    output_atol = 2e-2
    output_rtol = 2e-2
    runtime = _runtime_description(device_index)

    print(runtime)
    print(
        "single-stream real prefetch graph: "
        f"dtype={dtype} batch={batch_size} heads={num_heads} "
        f"capacity={capacity} context={max_context_len} scale={scale:.8f}"
    )

    capture_case = _make_prefetch_compact_case(
        hit_counts=(1024, 1536),
        req_pool_indices=(2, 0),
        capacity=capacity,
        max_context_len=max_context_len,
        request_pool_size=request_pool_size,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081420,
    )
    replay_case_a = _make_prefetch_compact_case(
        hit_counts=(2048, 0),
        req_pool_indices=(3, 1),
        capacity=capacity,
        max_context_len=max_context_len,
        request_pool_size=request_pool_size,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081421,
    )
    replay_case_b = _make_prefetch_compact_case(
        hit_counts=(256, 1792),
        req_pool_indices=(1, 3),
        capacity=capacity,
        max_context_len=max_context_len,
        request_pool_size=request_pool_size,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081422,
    )

    shm_name = f"sglang_prefetch_graph_{os.getpid()}_{device_index}_{uuid.uuid4().hex}"
    host_kv = None
    host_kv_host_ptr = None
    host_kv_dev_ptr = None
    graph = None
    shm_created = False
    lookup_refs: list[torch.Tensor] = []

    try:
        try:
            (
                host_kv,
                host_kv_host_ptr,
                host_kv_dev_ptr,
            ) = sparse_kv_ops.create_shm_tensor(
                (request_pool_size, max_context_len, 1, head_dim),
                dtype,
                device_index,
                shm_name,
            )
            shm_created = True
        except Exception as error:
            raise AssertionError(
                "could not allocate/register the host KV SHM used by the "
                f"prefetch graph; {runtime}; original error: "
                f"{type(error).__name__}: {error}"
            ) from error

        assert host_kv is not None
        assert host_kv_host_ptr is not None
        assert host_kv_dev_ptr is not None
        assert host_kv.data_ptr() == host_kv_host_ptr

        static_inputs = StaticPrefetchInputs(
            query=torch.empty_like(capture_case.union.query),
            query_rope=torch.empty_like(capture_case.union.query_rope),
            actual_query_lengths=torch.empty_like(
                capture_case.union.actual_query_lengths
            ),
            req_pool_indices=torch.empty(batch_size, dtype=torch.int64, device=device),
            seq_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            topk_indices=torch.empty(
                (batch_size, capacity), dtype=torch.int32, device=device
            ),
            slot_map=torch.empty(
                (request_pool_size + 1, slot_map_width),
                dtype=torch.int32,
                device=device,
            ),
            device_kv=torch.empty(
                (request_pool_size, capacity, 1, head_dim), dtype=dtype, device=device,
            ),
        )
        stats_shape = _expected_stats_shape(capture_case.union)
        buffers = PrefetchGraphBuffers(
            hit_kv=torch.empty(
                (batch_size, capacity, 1, head_dim), dtype=dtype, device=device
            ),
            miss_kv=torch.empty(
                (batch_size, capacity, 1, head_dim), dtype=dtype, device=device
            ),
            token_on_device=torch.empty(
                (batch_size, capacity), dtype=torch.int32, device=device
            ),
            device_token_pos=torch.empty(
                (batch_size, capacity), dtype=torch.int32, device=device
            ),
            hit_counts=torch.empty(batch_size, dtype=torch.int32, device=device),
            miss_counts=torch.empty(batch_size, dtype=torch.int32, device=device),
            output=torch.empty_like(capture_case.union.query),
            softmax_max=torch.empty(stats_shape, dtype=torch.float32, device=device),
            softmax_sum=torch.empty(stats_shape, dtype=torch.float32, device=device),
        )

        def load_case(case: PrefetchCompactCase) -> None:
            # The mapped host cache may still be read by the previous replay.
            # Synchronize before overwriting it, then stage every fixed-address
            # input in place.
            torch.npu.synchronize()
            _copy_prefetch_case(static_inputs, host_kv, case)
            buffers.hit_kv.fill_(float("nan"))
            buffers.miss_kv.fill_(float("nan"))
            buffers.token_on_device.fill_(-777)
            buffers.device_token_pos.fill_(-777)
            buffers.hit_counts.fill_(-777)
            buffers.miss_counts.fill_(-777)
            buffers.output.fill_(float("nan"))
            buffers.softmax_max.fill_(float("nan"))
            buffers.softmax_sum.fill_(float("nan"))
            torch.npu.synchronize()

        def run_once() -> None:
            # This is deliberately a single-stream sequence. Do not add an
            # auxiliary stream or Event here; those are tested in a later step.
            token_on_device, device_token_pos = _run_prefetch_compact_split(
                static_inputs,
                buffers,
                host_kv,
                host_kv_dev_ptr,
                request_pool_size=request_pool_size,
                max_context_len=max_context_len,
                capacity=capacity,
                scale=scale,
            )
            lookup_refs[:] = [token_on_device, device_token_pos]

        def validate_snapshot(
            snapshot: PrefetchSnapshot,
            case: PrefetchCompactCase,
            union_state: AttentionState,
            *,
            stage: str,
        ) -> None:
            _assert_prefetch_compaction(snapshot, case, stage=stage)
            _validate_contract(snapshot.state, case.union, stage=stage)
            _assert_matches(
                snapshot.state,
                union_state,
                output_atol=output_atol,
                output_rtol=output_rtol,
                stage=f"{stage} vs union SFA",
            )

        def eager_reference(
            case: PrefetchCompactCase, stage: str
        ) -> tuple[AttentionState, PrefetchSnapshot]:
            load_case(case)
            try:
                run_once()
                union_state = _run_sfa(case.union, scale)
                torch.npu.synchronize()
            except Exception as error:
                raise AssertionError(
                    f"{stage}: eager prefetch/compact + two SFA + merge failed; "
                    f"{runtime}; original error: {type(error).__name__}: {error}"
                ) from error

            union_state = _clone_state(union_state)
            snapshot = _snapshot_prefetch_buffers(buffers)
            _validate_contract(union_state, case.union, stage=f"{stage} union")
            validate_snapshot(snapshot, case, union_state, stage=stage)
            return union_state, snapshot

        capture_union, _ = eager_reference(capture_case, "capture eager")
        union_a, eager_a = eager_reference(replay_case_a, "replay A eager")
        union_b, eager_b = eager_reference(replay_case_b, "replay B eager")

        minimum_case_deltas = {
            "output": 1e-2,
            "softmax_max": 1e-2,
            "softmax_sum": 1e-1,
            "lse": 1e-2,
        }
        eager_case_deltas = {
            "output": _max_abs_error(eager_a.state.output, eager_b.state.output),
            "softmax_max": _max_abs_error(
                eager_a.state.softmax_max, eager_b.state.softmax_max
            ),
            "softmax_sum": _max_abs_error(
                eager_a.state.softmax_sum, eager_b.state.softmax_sum
            ),
            "lse": _max_abs_error(_state_lse(eager_a.state), _state_lse(eager_b.state)),
        }
        for name, minimum_delta in minimum_case_deltas.items():
            assert eager_case_deltas[name] > minimum_delta, (
                f"the eager real-prefetch {name} values differ by only "
                f"{eager_case_deltas[name]:.4e}; replay cases cannot detect a "
                "frozen graph output"
            )

        input_pointers = tuple(tensor.data_ptr() for tensor in static_inputs.tensors())
        buffer_pointers = tuple(tensor.data_ptr() for tensor in buffers.tensors())
        fixed_host_ptrs = (host_kv.data_ptr(), host_kv_host_ptr, host_kv_dev_ptr)
        try:
            capture_stream = torch.npu.Stream()
            graph_pool = torch.npu.graph_pool_handle()
        except Exception as error:
            raise AssertionError(
                "NPUGraph runtime setup failed before real-prefetch capture; "
                f"{runtime}; original error: {type(error).__name__}: {error}"
            ) from error

        load_case(capture_case)
        try:
            capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(capture_stream):
                for _ in range(2):
                    run_once()
            torch.npu.synchronize()
            warmup_snapshot = _snapshot_prefetch_buffers(buffers)
            validate_snapshot(
                warmup_snapshot,
                capture_case,
                capture_union,
                stage="real-prefetch graph warmup",
            )
        except Exception as error:
            raise AssertionError(
                "real prefetch/compact + two SFA + merge failed during graph "
                f"stream warmup; {runtime}; original error: "
                f"{type(error).__name__}: {error}"
            ) from error

        try:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                graph,
                pool=graph_pool,
                stream=capture_stream,
                auto_dispatch_capture=True,
            ):
                run_once()
            torch.npu.synchronize()
        except Exception as error:
            raise AssertionError(
                "NPUGraph could not capture the single-stream slot lookup, "
                "device/host compact copies, two SFA calls, and FP32 merge; "
                "keep graph mode on the combined fallback; "
                f"{runtime}; original error: {type(error).__name__}: {error}"
            ) from error

        assert len(lookup_refs) == 2, "capture did not retain slot-lookup outputs"
        lookup_pointers = tuple(tensor.data_ptr() for tensor in lookup_refs)
        assert (
            tuple(tensor.data_ptr() for tensor in static_inputs.tensors())
            == input_pointers
        ), "capture changed a real-prefetch static input address"
        assert (
            tuple(tensor.data_ptr() for tensor in buffers.tensors()) == buffer_pointers
        ), "capture changed a real-prefetch output/scratch address"
        assert (
            host_kv.data_ptr(),
            host_kv_host_ptr,
            host_kv_dev_ptr,
        ) == fixed_host_ptrs, "capture changed the registered host KV address"
        capture_snapshot = _snapshot_prefetch_buffers(buffers)
        validate_snapshot(
            capture_snapshot,
            capture_case,
            capture_union,
            stage="real-prefetch graph capture",
        )

        def replay(case: PrefetchCompactCase, stage: str) -> PrefetchSnapshot:
            load_case(case)
            try:
                graph.replay()
                torch.npu.synchronize()
            except Exception as error:
                raise AssertionError(
                    f"{stage}: real-prefetch NPUGraph replay failed; {runtime}; "
                    f"original error: {type(error).__name__}: {error}"
                ) from error

            assert (
                tuple(tensor.data_ptr() for tensor in static_inputs.tensors())
                == input_pointers
            ), f"{stage}: a static input address changed"
            assert (
                tuple(tensor.data_ptr() for tensor in buffers.tensors())
                == buffer_pointers
            ), f"{stage}: an output/scratch address changed"
            assert tuple(tensor.data_ptr() for tensor in lookup_refs) == (
                lookup_pointers
            ), f"{stage}: a captured slot-lookup output address changed"
            assert (
                host_kv.data_ptr(),
                host_kv_host_ptr,
                host_kv_dev_ptr,
            ) == fixed_host_ptrs, f"{stage}: the registered host KV address changed"
            return _snapshot_prefetch_buffers(buffers)

        graph_a = replay(replay_case_a, "real-prefetch graph replay A")
        graph_b = replay(replay_case_b, "real-prefetch graph replay B")

        for graph_snapshot, eager_snapshot, union_state, case, stage in (
            (graph_a, eager_a, union_a, replay_case_a, "real-prefetch graph replay A",),
            (graph_b, eager_b, union_b, replay_case_b, "real-prefetch graph replay B",),
        ):
            validate_snapshot(graph_snapshot, case, union_state, stage=stage)
            _assert_matches(
                graph_snapshot.state,
                eager_snapshot.state,
                output_atol=output_atol,
                output_rtol=output_rtol,
                stage=f"{stage} vs eager real-prefetch",
            )

        graph_case_deltas = {
            "output": _max_abs_error(graph_a.state.output, graph_b.state.output),
            "softmax_max": _max_abs_error(
                graph_a.state.softmax_max, graph_b.state.softmax_max
            ),
            "softmax_sum": _max_abs_error(
                graph_a.state.softmax_sum, graph_b.state.softmax_sum
            ),
            "lse": _max_abs_error(_state_lse(graph_a.state), _state_lse(graph_b.state)),
        }
        for name, minimum_delta in minimum_case_deltas.items():
            assert graph_case_deltas[name] > minimum_delta, (
                f"real-prefetch graph replay did not refresh {name}: replay "
                f"A/B delta={graph_case_deltas[name]:.4e}"
            )

        print(
            "replay A graph-vs-union errors: "
            f"out={_max_abs_error(graph_a.state.output, union_a.output):.4e} "
            f"max={_max_abs_error(graph_a.state.softmax_max, union_a.softmax_max):.4e} "
            f"sum={_max_abs_error(graph_a.state.softmax_sum, union_a.softmax_sum):.4e} "
            f"lse={_max_abs_error(_state_lse(graph_a.state), _state_lse(union_a)):.4e}"
        )
        print(
            "replay B graph-vs-union errors: "
            f"out={_max_abs_error(graph_b.state.output, union_b.output):.4e} "
            f"max={_max_abs_error(graph_b.state.softmax_max, union_b.softmax_max):.4e} "
            f"sum={_max_abs_error(graph_b.state.softmax_sum, union_b.softmax_sum):.4e} "
            f"lse={_max_abs_error(_state_lse(graph_b.state), _state_lse(union_b)):.4e}"
        )
        print(
            "PASSED: real slot lookup + device/host compact + two SFA + FP32 "
            "merge survive single-stream NPUGraph replay"
        )
    finally:
        if shm_created:
            active_error = sys.exc_info()[1]
            try:
                torch.npu.synchronize()
            except Exception as cleanup_error:
                # A failed synchronization means a graph/kernel may still be
                # reading the raw SHM address. Do not unregister that memory;
                # let the isolated test process reclaim it on exit instead.
                if active_error is None:
                    raise
                print(
                    "WARNING: leaving sparse-KV SHM registered because final "
                    f"NPU synchronization failed: {type(cleanup_error).__name__}: "
                    f"{cleanup_error}",
                    file=sys.stderr,
                )
            else:
                lookup_refs.clear()
                graph_released = True
                if graph is not None and hasattr(graph, "reset"):
                    try:
                        graph.reset()
                    except Exception as cleanup_error:
                        graph_released = False
                        if active_error is None:
                            raise
                        print(
                            "WARNING: leaving sparse-KV SHM registered because "
                            "NPUGraph reset failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}",
                            file=sys.stderr,
                        )
                if graph_released:
                    graph = None
                    gc.collect()
                    try:
                        sparse_kv_ops.free_shm(device_index)
                    except Exception as cleanup_error:
                        if active_error is None:
                            raise
                        print(
                            "WARNING: sparse-KV SHM cleanup failed while "
                            "propagating the test error: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}",
                            file=sys.stderr,
                        )


@pytest.mark.skipif(
    torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu and an Ascend NPU are required",
)
def test_real_prefetch_dual_stream_refill_slot_map_survives_npugraph() -> None:
    """Capture two streams and prove refill affects the following replay.

    Capture starts from a mixed hit/miss cache.  Replay 1 resets that same mixed
    state, refills every selected token into deterministic hot-cache slots, and
    publishes the new slot map.  Replay 2 changes only query data, so observing
    larger (all-hit) counts proves graph state written by replay 1 is visible to
    replay 2. Run the test in an isolated process because ``free_shm`` releases
    all sparse-KV SHM allocations registered by this process.
    """

    if sparse_kv_ops is None:
        pytest.fail(
            "the matching sgl_kernel_npu sparse-KV package is required for the "
            "dual-stream refill NPUGraph test; import failed with "
            f"{type(_SPARSE_KV_OPS_IMPORT_ERROR).__name__}: "
            f"{_SPARSE_KV_OPS_IMPORT_ERROR}",
            pytrace=False,
        )

    device_index = 0
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    dtype = torch.bfloat16
    batch_size = 2
    request_pool_size = 4
    num_heads = 16
    capacity = 2048
    max_context_len = 4096
    head_dim = 512 + 64
    slot_map_width = (max_context_len // 8 + 1) * 8
    scale = 1.0 / math.sqrt(128 + 64)
    runtime = _runtime_description(device_index)

    print(runtime)
    print(
        "dual-stream stateful prefetch graph: "
        f"dtype={dtype} batch={batch_size} heads={num_heads} "
        f"capacity={capacity} context={max_context_len} scale={scale:.8f}"
    )

    capture_case = _make_prefetch_compact_case(
        hit_counts=(1024, 1536),
        req_pool_indices=(2, 1),
        capacity=capacity,
        max_context_len=max_context_len,
        request_pool_size=request_pool_size,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081701,
    )
    query_variant = _make_case(
        valid_lengths=(capacity, capacity),
        capacity=capacity,
        num_heads=num_heads,
        dtype=dtype,
        device=device,
        seed=2026081702,
    )
    replay_union_inputs = SFAInputs(
        query=query_variant.query,
        key=capture_case.union.key,
        query_rope=query_variant.query_rope,
        key_rope=capture_case.union.key_rope,
        sparse_indices=capture_case.union.sparse_indices,
        actual_query_lengths=capture_case.union.actual_query_lengths,
        actual_kv_lengths=capture_case.union.actual_kv_lengths,
    )
    capture_union = _clone_state(_run_sfa(capture_case.union, scale))
    replay_union = _clone_state(_run_sfa(replay_union_inputs, scale))
    torch.npu.synchronize()

    shm_name = (
        f"sglang_prefetch_dual_graph_{os.getpid()}_{device_index}_{uuid.uuid4().hex}"
    )
    host_kv = None
    host_kv_host_ptr = None
    host_kv_dev_ptr = None
    graph = None
    shm_created = False
    lookup_refs: list[torch.Tensor] = []
    miss_stream = None
    inputs_ready = None
    hit_copy_done = None
    miss_attention_done = None
    refill_done = None

    try:
        (host_kv, host_kv_host_ptr, host_kv_dev_ptr,) = sparse_kv_ops.create_shm_tensor(
            (request_pool_size, max_context_len, 1, head_dim),
            dtype,
            device_index,
            shm_name,
        )
        shm_created = True
        assert host_kv.data_ptr() == host_kv_host_ptr

        static_inputs = StaticPrefetchInputs(
            query=torch.empty_like(capture_case.union.query),
            query_rope=torch.empty_like(capture_case.union.query_rope),
            actual_query_lengths=torch.empty_like(
                capture_case.union.actual_query_lengths
            ),
            req_pool_indices=torch.empty(batch_size, dtype=torch.int64, device=device),
            seq_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            topk_indices=torch.empty(
                (batch_size, capacity), dtype=torch.int32, device=device
            ),
            slot_map=torch.empty(
                (request_pool_size + 1, slot_map_width),
                dtype=torch.int32,
                device=device,
            ),
            device_kv=torch.empty(
                (request_pool_size, capacity, 1, head_dim), dtype=dtype, device=device,
            ),
        )
        stats_shape = _expected_stats_shape(capture_case.union)
        buffers = PrefetchGraphBuffers(
            hit_kv=torch.empty(
                (batch_size, capacity, 1, head_dim), dtype=dtype, device=device
            ),
            miss_kv=torch.empty(
                (batch_size, capacity, 1, head_dim), dtype=dtype, device=device
            ),
            token_on_device=torch.empty(
                (batch_size, capacity), dtype=torch.int32, device=device
            ),
            device_token_pos=torch.empty(
                (batch_size, capacity), dtype=torch.int32, device=device
            ),
            hit_counts=torch.empty(batch_size, dtype=torch.int32, device=device),
            miss_counts=torch.empty(batch_size, dtype=torch.int32, device=device),
            output=torch.empty_like(capture_case.union.query),
            softmax_max=torch.empty(stats_shape, dtype=torch.float32, device=device),
            softmax_sum=torch.empty(stats_shape, dtype=torch.float32, device=device),
        )

        def poison_outputs() -> None:
            buffers.hit_kv.fill_(float("nan"))
            buffers.miss_kv.fill_(float("nan"))
            buffers.token_on_device.fill_(-777)
            buffers.device_token_pos.fill_(-777)
            buffers.hit_counts.fill_(-777)
            buffers.miss_counts.fill_(-777)
            buffers.output.fill_(float("nan"))
            buffers.softmax_max.fill_(float("nan"))
            buffers.softmax_sum.fill_(float("nan"))

        def reset_initial_state() -> None:
            torch.npu.synchronize()
            _copy_prefetch_case(static_inputs, host_kv, capture_case)
            poison_outputs()
            torch.npu.synchronize()

        def load_replay_query_only() -> None:
            # Preserve device_kv and slot_map: those are the state produced by
            # capture whose visibility to replay this test is designed to prove.
            torch.npu.synchronize()
            static_inputs.query.copy_(query_variant.query)
            static_inputs.query_rope.copy_(query_variant.query_rope)
            poison_outputs()
            torch.npu.synchronize()

        capture_stream = torch.npu.Stream()
        miss_stream = torch.npu.Stream()
        inputs_ready = torch.npu.Event()
        hit_copy_done = torch.npu.Event()
        miss_attention_done = torch.npu.Event()
        refill_done = torch.npu.Event()
        graph_pool = torch.npu.graph_pool_handle()

        def run_once() -> None:
            (
                token_on_device,
                device_token_pos,
            ) = _run_prefetch_compact_split_dual_stream_refill(
                static_inputs,
                buffers,
                host_kv,
                host_kv_dev_ptr,
                miss_stream,
                inputs_ready,
                hit_copy_done,
                miss_attention_done,
                refill_done,
                request_pool_size=request_pool_size,
                max_context_len=max_context_len,
                capacity=capacity,
                scale=scale,
            )
            lookup_refs[:] = [token_on_device, device_token_pos]

        reset_initial_state()
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            # First warmup starts mixed; the second observes the first refill.
            run_once()
            run_once()
        torch.npu.synchronize()
        _assert_refill_and_slot_map_published(
            static_inputs,
            capture_case,
            capacity=capacity,
            stage="dual-stream graph warmup",
        )

        reset_initial_state()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            graph, pool=graph_pool, stream=capture_stream, auto_dispatch_capture=True,
        ):
            run_once()
        torch.npu.synchronize()

        input_pointers = tuple(tensor.data_ptr() for tensor in static_inputs.tensors())
        buffer_pointers = tuple(tensor.data_ptr() for tensor in buffers.tensors())
        lookup_pointers = tuple(tensor.data_ptr() for tensor in lookup_refs)
        fixed_host_ptrs = (host_kv.data_ptr(), host_kv_host_ptr, host_kv_dev_ptr)

        # NPUGraph capture is record-only on some torch_npu/CANN versions, so
        # captured output buffers are not a valid correctness oracle. Replay 1
        # starts from a freshly restored mixed state and must publish the
        # refill. Replay 2 deliberately preserves that state and must see a
        # larger hit count (all selected tokens are now resident).
        reset_initial_state()
        graph.replay()
        torch.npu.synchronize()

        replay_one = _snapshot_prefetch_buffers(buffers)
        _assert_prefetch_compaction(
            replay_one, capture_case, stage="dual-stream graph replay 1"
        )
        _assert_matches(
            replay_one.state,
            capture_union,
            output_atol=2e-2,
            output_rtol=2e-2,
            stage="dual-stream graph replay 1 vs union SFA",
        )
        _assert_refill_and_slot_map_published(
            static_inputs,
            capture_case,
            capacity=capacity,
            stage="dual-stream graph replay 1",
        )

        load_replay_query_only()
        graph.replay()
        torch.npu.synchronize()

        assert tuple(tensor.data_ptr() for tensor in static_inputs.tensors()) == (
            input_pointers
        ), "dual-stream replay changed a static input address"
        assert tuple(tensor.data_ptr() for tensor in buffers.tensors()) == (
            buffer_pointers
        ), "dual-stream replay changed an output/scratch address"
        assert tuple(tensor.data_ptr() for tensor in lookup_refs) == lookup_pointers
        assert (
            host_kv.data_ptr(),
            host_kv_host_ptr,
            host_kv_dev_ptr,
        ) == fixed_host_ptrs

        replay_snapshot = _snapshot_prefetch_buffers(buffers)
        torch.testing.assert_close(
            replay_snapshot.token_on_device.cpu(),
            torch.ones((batch_size, capacity), dtype=torch.int32),
            atol=0,
            rtol=0,
            msg="replay 2 did not observe the slot map published by replay 1",
        )
        expected_positions = torch.arange(capacity, dtype=torch.int32).expand(
            batch_size, capacity
        )
        torch.testing.assert_close(
            replay_snapshot.device_token_pos.cpu(),
            expected_positions,
            atol=0,
            rtol=0,
            msg="replay did not resolve refilled tokens to deterministic slots",
        )
        torch.testing.assert_close(
            replay_snapshot.hit_counts.cpu(),
            torch.full((batch_size,), capacity, dtype=torch.int32),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            replay_snapshot.miss_counts.cpu(),
            torch.zeros(batch_size, dtype=torch.int32),
            atol=0,
            rtol=0,
        )
        assert torch.all(replay_snapshot.hit_counts > replay_one.hit_counts).item(), (
            "replay 2 hit_counts did not increase after replay 1 refill: "
            f"before={replay_one.hit_counts.cpu().tolist()}, "
            f"after={replay_snapshot.hit_counts.cpu().tolist()}"
        )
        expected_kv = torch.cat(
            (capture_case.union.key, capture_case.union.key_rope), dim=-1
        )
        torch.testing.assert_close(
            replay_snapshot.hit_kv.cpu(), expected_kv.cpu(), atol=0, rtol=0
        )
        assert torch.count_nonzero(replay_snapshot.miss_kv[:, 0]).item() == 0
        _assert_matches(
            replay_snapshot.state,
            replay_union,
            output_atol=2e-2,
            output_rtol=2e-2,
            stage="dual-stream stateful replay vs union SFA",
        )
        _assert_refill_and_slot_map_published(
            static_inputs,
            capture_case,
            capacity=capacity,
            stage="dual-stream stateful replay",
        )
        assert (
            _max_abs_error(replay_one.state.output, replay_snapshot.state.output) > 1e-2
        ), "changed replay-2 query did not refresh the graph output"

        print(
            "PASSED: dual-stream D2D/H2D + two SFA + merge + refill + slot-map "
            "publication survive stateful NPUGraph replay"
        )
    finally:
        if shm_created:
            active_error = sys.exc_info()[1]
            try:
                torch.npu.synchronize()
            except Exception as cleanup_error:
                if active_error is None:
                    raise
                print(
                    "WARNING: leaving dual-stream sparse-KV SHM registered "
                    "because final NPU synchronization failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                    file=sys.stderr,
                )
            else:
                lookup_refs.clear()
                graph_released = True
                if graph is not None and hasattr(graph, "reset"):
                    try:
                        graph.reset()
                    except Exception as cleanup_error:
                        graph_released = False
                        if active_error is None:
                            raise
                        print(
                            "WARNING: leaving dual-stream sparse-KV SHM "
                            "registered because NPUGraph reset failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}",
                            file=sys.stderr,
                        )
                if graph_released:
                    graph = None
                    miss_stream = None
                    inputs_ready = None
                    hit_copy_done = None
                    miss_attention_done = None
                    refill_done = None
                    gc.collect()
                    try:
                        sparse_kv_ops.free_shm(device_index)
                    except Exception as cleanup_error:
                        if active_error is None:
                            raise
                        print(
                            "WARNING: dual-stream sparse-KV SHM cleanup failed "
                            "while propagating the test error: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}",
                            file=sys.stderr,
                        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
