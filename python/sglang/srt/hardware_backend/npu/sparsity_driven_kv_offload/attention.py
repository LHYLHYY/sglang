"""Ascend attention path backed by sparsity-driven KV offload."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

try:
    import sgl_kernel_npu.sparsity_driven_kv_offload as _sparse_kv_ops
except ModuleNotFoundError as error:
    if error.name not in (
        "sgl_kernel_npu",
        "sgl_kernel_npu.sparsity_driven_kv_offload",
    ):
        raise
    _fused_sfa_state_merge_inplace = None
else:
    _fused_sfa_state_merge_inplace = getattr(
        _sparse_kv_ops, "sfa_state_merge_inplace", None
    )

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    SPARSE_KV_ATTN_IMPL_SPLIT_EAGER,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
    SPARSE_KV_MERGE_IMPL_AUTO,
    SPARSE_KV_MERGE_IMPL_FUSED,
    SPARSE_KV_MERGE_IMPL_PYTHON,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
        SparseKVPartition,
        SparseKVPrefetchTicket,
        SparseKVGraphDualPrefetch,
        SparseKVCacheManager,
        SparseKVSingleStreamPrefetch,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

_SPLIT_MODE_PARALLEL = "parallel"
_SPLIT_MODE_SINGLE_STREAM = "single_stream"
_SPLIT_MODE_DUAL_STREAM = "dual_stream"
_FUSED_MERGE_FALLBACK_LOGGED = False
_FUSED_MERGE_SELECTION_LOGGED = False


def _is_fused_sfa_state_merge_available() -> bool:
    """Return whether both the Python wrapper and native schema are installed."""

    return _fused_sfa_state_merge_inplace is not None and hasattr(
        torch.ops.npu, "sfa_state_merge"
    )


def _select_split_decode_mode(attn_impl: str, graph_mode: bool) -> Optional[str]:
    if graph_mode:
        if attn_impl == SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH:
            return _SPLIT_MODE_SINGLE_STREAM
        if attn_impl == SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL:
            return _SPLIT_MODE_DUAL_STREAM
        return None
    if attn_impl in (
        SPARSE_KV_ATTN_IMPL_SPLIT_EAGER,
        SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH,
        SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
    ):
        return _SPLIT_MODE_PARALLEL
    return None


@dataclass
class _SfaPartitionState:
    output: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor
    true_counts: torch.Tensor


def _get_sparse_kv_manager(backend: AscendAttnBackend):
    if backend.sparse_kv_manager is None:
        raise RuntimeError(
            "Sparsity-driven KV offload is disabled or was not initialized."
        )
    return backend.sparse_kv_manager


def _expand_dsa_sparse_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    """Expand [T, K] to [T, 1, K] for NPU sparse attention."""
    if topk_indices.dim() == 2:
        return topk_indices.unsqueeze(-2)
    return topk_indices


def _record_stream_event(stream, event) -> None:
    if hasattr(stream, "record_event"):
        stream.record_event(event)
    else:
        event.record(stream)


def _wait_stream_event(stream, event) -> None:
    if hasattr(stream, "wait_event"):
        stream.wait_event(event)
    else:
        event.wait(stream)


def _run_decode_sfa_partition(
    partition: SparseKVPartition,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    nope_head_dim: int,
    rope_head_dim: int,
    scale_value: float,
    record_stream: bool = True,
) -> _SfaPartitionState:
    if record_stream:
        for tensor in (
            partition.kv,
            partition.sparse_indices,
            partition.actual_seq_lengths_kv,
            partition.true_counts,
            query,
            query_rope,
        ):
            tensor.record_stream(partition.stream)
    key, key_rope = partition.kv.split([nope_head_dim, rope_head_dim], dim=-1)
    key = key.contiguous()
    key_rope = key_rope.contiguous()
    batch_size, query_length, padded_heads, value_dim = query.shape
    actual_query_lengths = torch.ones(
        batch_size, dtype=torch.int32, device=query.device
    ).contiguous()

    output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query=query,
        key=key,
        value=key,
        sparse_indices=partition.sparse_indices,
        scale_value=scale_value,
        actual_seq_lengths_query=actual_query_lengths,
        actual_seq_lengths_kv=partition.actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )

    expected_output_shape = (
        batch_size,
        query_length,
        padded_heads,
        value_dim,
    )
    expected_stats_shape = (batch_size, 1, query_length, padded_heads)
    if tuple(output.shape) != expected_output_shape or output.dtype != query.dtype:
        raise RuntimeError(
            "Unexpected split SFA output contract: "
            f"got shape={tuple(output.shape)}, dtype={output.dtype}; "
            f"expected shape={expected_output_shape}, dtype={query.dtype}."
        )
    for name, value in (("softmax_max", softmax_max), ("softmax_sum", softmax_sum)):
        if tuple(value.shape) != expected_stats_shape or value.dtype != torch.float32:
            raise RuntimeError(
                f"Unexpected split SFA {name} contract: "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}; "
                f"expected shape={expected_stats_shape}, dtype=torch.float32."
            )

    return _SfaPartitionState(
        output=output,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        true_counts=partition.true_counts,
    )


def _merge_decode_sfa_partitions_python(
    hit: _SfaPartitionState, miss: _SfaPartitionState
) -> torch.Tensor:
    if hit.output.shape != miss.output.shape:
        raise RuntimeError(
            "Split SFA output shape mismatch: "
            f"hit={tuple(hit.output.shape)}, miss={tuple(miss.output.shape)}."
        )
    if hit.softmax_max.shape != miss.softmax_max.shape:
        raise RuntimeError(
            "Split SFA statistics shape mismatch: "
            f"hit={tuple(hit.softmax_max.shape)}, "
            f"miss={tuple(miss.softmax_max.shape)}."
        )

    batch_size = hit.output.shape[0]
    stats_mask_shape = (batch_size, 1, 1, 1)
    hit_nonempty = (hit.true_counts > 0).view(stats_mask_shape)
    miss_nonempty = (miss.true_counts > 0).view(stats_mask_shape)
    any_nonempty = hit_nonempty | miss_nonempty

    neg_inf = torch.full_like(hit.softmax_max, float("-inf"))
    hit_max = torch.where(hit_nonempty, hit.softmax_max.float(), neg_inf)
    miss_max = torch.where(miss_nonempty, miss.softmax_max.float(), neg_inf)
    global_max = torch.maximum(hit_max, miss_max)
    safe_global_max = torch.where(
        any_nonempty, global_max, torch.zeros_like(global_max)
    )

    hit_delta = torch.where(
        hit_nonempty, hit.softmax_max.float() - safe_global_max, 0.0
    )
    miss_delta = torch.where(
        miss_nonempty, miss.softmax_max.float() - safe_global_max, 0.0
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
    denominator = hit_mass + miss_mass
    safe_denominator = denominator.clamp_min(torch.finfo(torch.float32).tiny)
    hit_weight = (hit_mass / safe_denominator).permute(0, 2, 3, 1)
    miss_weight = (miss_mass / safe_denominator).permute(0, 2, 3, 1)

    merged = hit.output.float() * hit_weight + miss.output.float() * miss_weight
    output_mask = any_nonempty.permute(0, 2, 3, 1)
    merged = torch.where(output_mask, merged, torch.zeros_like(merged))
    return merged.to(hit.output.dtype)


def _merge_decode_sfa_partitions(
    hit: _SfaPartitionState,
    miss: _SfaPartitionState,
    merge_impl: str = SPARSE_KV_MERGE_IMPL_PYTHON,
) -> torch.Tensor:
    """Merge independently normalized hit/miss attention states.

    The fused path keeps the same device-side empty-partition semantics as the
    PyTorch reference while collapsing its pointwise graph into one AIV task.
    """

    global _FUSED_MERGE_FALLBACK_LOGGED, _FUSED_MERGE_SELECTION_LOGGED

    if merge_impl not in (
        SPARSE_KV_MERGE_IMPL_AUTO,
        SPARSE_KV_MERGE_IMPL_PYTHON,
        SPARSE_KV_MERGE_IMPL_FUSED,
    ):
        raise ValueError(f"Unsupported sparse KV merge implementation: {merge_impl!r}")

    fused_available = _is_fused_sfa_state_merge_available()
    use_fused = merge_impl == SPARSE_KV_MERGE_IMPL_FUSED or (
        merge_impl == SPARSE_KV_MERGE_IMPL_AUTO and fused_available
    )
    if not use_fused:
        if (
            merge_impl == SPARSE_KV_MERGE_IMPL_AUTO
            and not fused_available
            and not _FUSED_MERGE_FALLBACK_LOGGED
        ):
            logger.warning(
                "Fused sparse KV attention merge is unavailable in the installed "
                "sgl-kernel-npu; using the PyTorch merge implementation."
            )
            _FUSED_MERGE_FALLBACK_LOGGED = True
        return _merge_decode_sfa_partitions_python(hit, miss)

    if not fused_available:
        raise RuntimeError(
            "SGLANG_NPU_SPARSE_KV_MERGE_IMPL=fused requires an "
            "sgl-kernel-npu build that exports sfa_state_merge_inplace and "
            "registers torch.ops.npu.sfa_state_merge."
        )

    if not _FUSED_MERGE_SELECTION_LOGGED:
        logger.info("Using fused sgl-kernel-npu sparse KV attention state merge.")
        _FUSED_MERGE_SELECTION_LOGGED = True

    output = torch.empty_like(hit.output)
    return _fused_sfa_state_merge_inplace(
        hit.output,
        hit.softmax_max,
        hit.softmax_sum,
        miss.output,
        miss.softmax_max,
        miss.softmax_sum,
        hit.true_counts,
        miss.true_counts,
        output,
    )


def _run_split_decode_attention(
    ticket: SparseKVPrefetchTicket,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    nope_head_dim: int,
    rope_head_dim: int,
    scale_value: float,
    merge_impl: str,
    stream,
) -> torch.Tensor:
    hit_attention_done = torch.npu.Event()
    miss_attention_done = torch.npu.Event()

    with torch.profiler.record_function("sparse_kv_split.hit_attention"):
        with torch.npu.stream(ticket.hit.stream):
            hit_state = _run_decode_sfa_partition(
                ticket.hit,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
            )
            _record_stream_event(ticket.hit.stream, hit_attention_done)

    with torch.profiler.record_function("sparse_kv_split.miss_attention"):
        with torch.npu.stream(ticket.miss.stream):
            miss_state = _run_decode_sfa_partition(
                ticket.miss,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
            )
            _record_stream_event(ticket.miss.stream, miss_attention_done)

    with torch.npu.stream(stream):
        _wait_stream_event(stream, hit_attention_done)
        _wait_stream_event(stream, miss_attention_done)
        _wait_stream_event(stream, ticket.refill_done)
        # These tensors were allocated on the two producer streams but are
        # consumed by the merge stream.  Record that ownership transfer so the
        # caching allocator cannot recycle them before the merge completes.
        for state in (hit_state, miss_state):
            for tensor in (
                state.output,
                state.softmax_max,
                state.softmax_sum,
                state.true_counts,
            ):
                tensor.record_stream(stream)
        with torch.profiler.record_function("sparse_kv_split.merge"):
            return _merge_decode_sfa_partitions(hit_state, miss_state, merge_impl)


def _run_split_decode_attention_single_stream(
    partitions: SparseKVSingleStreamPrefetch,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    nope_head_dim: int,
    rope_head_dim: int,
    scale_value: float,
    merge_impl: str,
) -> torch.Tensor:
    """Run both partition attentions and merge on the current graph stream."""

    with torch.profiler.record_function("sparse_kv_split_graph.hit_attention"):
        hit_state = _run_decode_sfa_partition(
            partitions.hit,
            query=query,
            query_rope=query_rope,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            scale_value=scale_value,
            record_stream=False,
        )
    with torch.profiler.record_function("sparse_kv_split_graph.miss_attention"):
        miss_state = _run_decode_sfa_partition(
            partitions.miss,
            query=query,
            query_rope=query_rope,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            scale_value=scale_value,
            record_stream=False,
        )
    with torch.profiler.record_function("sparse_kv_split_graph.merge"):
        return _merge_decode_sfa_partitions(hit_state, miss_state, merge_impl)


def _run_split_decode_attention_graph_dual(
    ticket: SparseKVGraphDualPrefetch,
    manager: SparseKVCacheManager,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    nope_head_dim: int,
    rope_head_dim: int,
    scale_value: float,
    stream,
) -> torch.Tensor:
    """Run hit attention on the graph stream and miss attention/refill in parallel."""

    with torch.profiler.record_function("sparse_kv_split_graph_dual.hit_attention"):
        with torch.npu.stream(stream):
            hit_state = _run_decode_sfa_partition(
                ticket.hit,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
                record_stream=False,
            )

    with torch.profiler.record_function("sparse_kv_split_graph_dual.miss_attention"):
        with torch.npu.stream(ticket.miss.stream):
            miss_state = _run_decode_sfa_partition(
                ticket.miss,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
                record_stream=False,
            )
            _record_stream_event(ticket.miss.stream, ticket.events.miss_attention_done)
            # Refill runs after miss attention on the same worker stream.  The
            # main stream may merge concurrently, but joins refill before the
            # shared graph workspace can be reused by the next layer.
            manager.commit_graph_dual_refill(ticket)

    with torch.npu.stream(stream):
        _wait_stream_event(stream, ticket.events.miss_attention_done)
        for tensor in (
            miss_state.output,
            miss_state.softmax_max,
            miss_state.softmax_sum,
            miss_state.true_counts,
        ):
            tensor.record_stream(stream)
        with torch.profiler.record_function("sparse_kv_split_graph_dual.merge"):
            merged = _merge_decode_sfa_partitions(
                hit_state, miss_state, manager.merge_impl
            )
        _wait_stream_event(stream, ticket.events.refill_done)
        return merged


def forward_sparsity_driven_kv_offload(
    backend: AscendAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache: bool = True,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
):
    """Run sparse attention using host-offloaded compact MLA KV."""
    del v
    if q_rope is None or k_rope is None or topk_indices is None:
        raise ValueError(
            "Sparsity-driven KV offload requires q_rope, k_rope, and topk_indices."
        )

    is_prefill = forward_batch.forward_mode.is_extend_without_speculative()

    q_nope, q_pe = q, q_rope
    k_nope = k.view(-1, layer.tp_k_head_num, backend.kv_lora_rank).contiguous()
    k_pe = k_rope.view(-1, layer.tp_k_head_num, backend.qk_rope_head_dim).contiguous()
    sparse_kv_manager = _get_sparse_kv_manager(backend)
    stream = torch.npu.current_stream(backend.device)

    if save_kv_cache:
        sparse_kv_manager.offload_v2(k_nope, k_pe, layer, forward_batch, stream)

    if is_prefill:
        if backend.forward_metadata.actual_seq_lengths_q is not None:
            actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q
        else:
            actual_seq_qlen = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
    elif backend.forward_metadata.actual_seq_lengths_q is None:
        if (
            forward_batch.forward_mode.is_draft_extend_v2()
            or forward_batch.forward_mode.is_target_verify()
        ):
            actual_seq_qlen = (
                torch.arange(
                    backend.speculative_num_draft_tokens,
                    backend.speculative_num_draft_tokens + q.shape[0],
                    backend.speculative_num_draft_tokens,
                    dtype=torch.int32,
                )
                .to(q.device)
                .to(torch.int32)
            )
        else:
            actual_seq_qlen = (
                torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
            )
    else:
        actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q

    if backend.forward_metadata.actual_seq_lengths_kv is not None:
        actual_seq_lengths_kv = backend.forward_metadata.actual_seq_lengths_kv
    elif backend.forward_metadata.seq_lens_cpu_int is not None:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens_cpu_int
    else:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens

    if (
        is_prefill
        and is_dsa_enable_prefill_cp()
        and forward_batch.attn_cp_metadata is not None
    ):
        attn_out = backend.do_cp_balance_attn(
            q_nope,
            k_nope,
            q_pe,
            k_pe,
            topk_indices,
            layer,
            actual_seq_qlen,
            actual_seq_lengths_kv,
        )
    elif forward_batch.forward_mode.is_decode():
        batch_size = forward_batch.batch_size
        selected_kv_length = 2048
        num_kv_heads = layer.tp_k_head_num
        num_query_heads = layer.tp_q_head_num
        nope_head_dim = backend.kv_lora_rank
        rope_head_dim = backend.qk_rope_head_dim

        assert num_kv_heads == 1, (
            "FIA_v2 MLA selected KV path expects KV_N == 1, "
            f"got num_kv_heads={num_kv_heads}"
        )

        padded_query_heads = q_nope.numel() // (batch_size * nope_head_dim)
        assert padded_query_heads >= num_query_heads, (
            "query head count mismatch: "
            f"padded_query_heads={padded_query_heads}, "
            f"num_query_heads={num_query_heads}"
        )

        split_mode = _select_split_decode_mode(
            sparse_kv_manager.attn_impl, backend.graph_mode
        )
        if (
            sparse_kv_manager.attn_impl == SPARSE_KV_ATTN_IMPL_SPLIT_EAGER
            and backend.graph_mode
            and not sparse_kv_manager._split_graph_fallback_logged
        ):
            logger.warning(
                "Sparse KV split attention is eager-only; using combined "
                "attention for NPU graph capture and replay."
            )
            sparse_kv_manager._split_graph_fallback_logged = True
        if (
            split_mode == _SPLIT_MODE_SINGLE_STREAM
            and not sparse_kv_manager._split_graph_phase_one_logged
        ):
            logger.warning(
                "Sparse KV split_graph phase 1 uses the graph-safe single-stream "
                "read path without hot-cache refill; outputs remain correct from "
                "the authoritative host cache, but graph-mode hit rate may be low."
            )
            sparse_kv_manager._split_graph_phase_one_logged = True
        if (
            split_mode == _SPLIT_MODE_DUAL_STREAM
            and not sparse_kv_manager._split_graph_dual_logged
        ):
            logger.warning(
                "Sparse KV split_graph_dual is experimental: hit attention runs "
                "on the graph stream while host misses, miss attention, and "
                "hot-cache refill run on a persistent worker stream."
            )
            sparse_kv_manager._split_graph_dual_logged = True

        if split_mode is not None:
            q_nope_sfa = q_nope.view(
                batch_size, 1, padded_query_heads, nope_head_dim
            ).contiguous()
            q_rope_sfa = q_pe.view(
                batch_size, 1, padded_query_heads, rope_head_dim
            ).contiguous()
            if split_mode == _SPLIT_MODE_SINGLE_STREAM:
                partitions = sparse_kv_manager.prefetch_partitions_single_stream(
                    layer,
                    forward_batch,
                    topk_indices,
                    stream,
                    dtype=k.dtype,
                )
                decode_output = _run_split_decode_attention_single_stream(
                    partitions,
                    query=q_nope_sfa,
                    query_rope=q_rope_sfa,
                    nope_head_dim=nope_head_dim,
                    rope_head_dim=rope_head_dim,
                    scale_value=layer.scaling,
                    merge_impl=sparse_kv_manager.merge_impl,
                )
            elif split_mode == _SPLIT_MODE_DUAL_STREAM:
                ticket = sparse_kv_manager.prefetch_partitions_graph_dual(
                    layer,
                    forward_batch,
                    topk_indices,
                    stream,
                    dtype=k.dtype,
                )
                decode_output = _run_split_decode_attention_graph_dual(
                    ticket,
                    sparse_kv_manager,
                    query=q_nope_sfa,
                    query_rope=q_rope_sfa,
                    nope_head_dim=nope_head_dim,
                    rope_head_dim=rope_head_dim,
                    scale_value=layer.scaling,
                    stream=stream,
                )
            else:
                ticket = sparse_kv_manager.prefetch_partitions(
                    layer,
                    forward_batch,
                    topk_indices,
                    stream,
                    dtype=k.dtype,
                )
                decode_output = _run_split_decode_attention(
                    ticket,
                    query=q_nope_sfa,
                    query_rope=q_rope_sfa,
                    nope_head_dim=nope_head_dim,
                    rope_head_dim=rope_head_dim,
                    scale_value=layer.scaling,
                    merge_impl=sparse_kv_manager.merge_impl,
                    stream=stream,
                )
            return decode_output[:, :, :num_query_heads, :].reshape(
                batch_size, num_query_heads * nope_head_dim
            )

        selected_kv = torch.zeros(
            (
                batch_size,
                selected_kv_length,
                num_kv_heads,
                nope_head_dim + rope_head_dim,
            ),
            dtype=k.dtype,
            device=backend.device,
        )
        sparse_kv_manager.prefetch(
            layer, forward_batch, topk_indices, selected_kv, stream
        )
        selected_k_nope, selected_k_rope = selected_kv.split(
            [nope_head_dim, rope_head_dim], dim=-1
        )

        topk_2d = topk_indices
        if topk_2d.dim() == 3:
            topk_2d = topk_2d[:, 0, :]
        elif topk_2d.dim() == 4:
            topk_2d = topk_2d[:, 0, 0, :]
        elif topk_2d.dim() != 2:
            raise RuntimeError(
                "SFA BSND compact path expects topk rank 2/3/4, " f"got {topk_2d.dim()}"
            )
        topk_2d = topk_2d[:, :selected_kv_length].contiguous()
        topk_length = topk_2d.shape[1]

        topk_valid = topk_2d >= 0
        if forward_batch.seq_lens is not None:
            valid_rows = (forward_batch.seq_lens[:batch_size] > 0).view(batch_size, 1)
            topk_valid = topk_valid & valid_rows

        actual_seq_lengths_kv = (
            topk_valid.sum(dim=1)
            .clamp(min=1, max=topk_length)
            .to(device=q_nope.device, dtype=torch.int32)
            .contiguous()
        )
        actual_seq_lengths_query = torch.ones(
            batch_size, dtype=torch.int32, device=q_nope.device
        ).contiguous()

        compact_indices = (
            torch.arange(topk_length, device=q_nope.device, dtype=torch.int32)
            .view(1, 1, 1, topk_length)
            .expand(batch_size, 1, num_kv_heads, topk_length)
            .clone()
        )
        compact_valid = topk_valid.view(batch_size, 1, 1, topk_length).expand(
            batch_size, 1, num_kv_heads, topk_length
        )
        sparse_indices = torch.where(
            compact_valid,
            compact_indices,
            torch.full_like(compact_indices, -1),
        ).contiguous()

        empty_rows = (topk_valid.sum(dim=1) == 0).view(batch_size, 1, 1)
        sparse_indices[:, :, :, 0] = torch.where(
            empty_rows.expand(batch_size, 1, num_kv_heads),
            torch.zeros(
                (batch_size, 1, num_kv_heads),
                dtype=torch.int32,
                device=q_nope.device,
            ),
            sparse_indices[:, :, :, 0],
        )

        q_nope_sfa = q_nope.view(
            batch_size, 1, padded_query_heads, nope_head_dim
        ).contiguous()
        q_rope_sfa = q_pe.view(
            batch_size, 1, padded_query_heads, rope_head_dim
        ).contiguous()
        k_nope_sfa = selected_k_nope.contiguous()
        k_rope_sfa = selected_k_rope.contiguous()

        assert q_nope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            nope_head_dim,
        )
        assert q_rope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            rope_head_dim,
        )
        assert k_nope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            nope_head_dim,
        )
        assert k_rope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            rope_head_dim,
        )

        ret = torch_npu.npu_sparse_flash_attention(
            q_nope_sfa,
            k_nope_sfa,
            k_nope_sfa,
            sparse_indices,
            layer.scaling,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=q_rope_sfa,
            key_rope=k_rope_sfa,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=False,
        )

        attn_out = ret[0] if isinstance(ret, tuple) else ret
        attn_out = attn_out[:, :, :num_query_heads, :].reshape(
            batch_size, num_query_heads * nope_head_dim
        )
    else:
        if is_prefill:
            k_nope_sfa, k_pe_sfa = sparse_kv_manager.get_forward_kv(
                layer, forward_batch, stream
            )
            forward_actual_seq_lengths_kv = torch.cumsum(forward_batch.seq_lens, dim=0)
        else:
            k_nope_sfa, k_pe_sfa = k_nope, k_pe
            forward_actual_seq_lengths_kv = actual_seq_lengths_kv

        topk_indices = _expand_dsa_sparse_indices(topk_indices)
        attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope,
            key=k_nope_sfa,
            value=k_nope_sfa,
            query_rope=q_pe,
            key_rope=k_pe_sfa,
            sparse_indices=topk_indices,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=forward_actual_seq_lengths_kv.to(
                device=q_nope.device, dtype=torch.int32
            ),
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )

    return attn_out
