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
    SPARSE_KV_ATTN_IMPL_NATIVE_FIA,
    SPARSE_KV_ATTN_IMPL_PA_GRAPH,
    SPARSE_KV_ATTN_IMPL_SPLIT_EAGER,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2,
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
        SparseKVCacheManager,
        SparseKVFiaPartition,
        SparseKVGraphDualPrefetch,
        SparseKVGraphDualV2Prefetch,
        SparseKVIndexedPartition,
        SparseKVPAHotCachePrefetch,
        SparseKVPAPartition,
        SparseKVPartition,
        SparseKVPrefetchTicket,
        SparseKVSingleStreamPrefetch,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

_SPLIT_MODE_PARALLEL = "parallel"
_SPLIT_MODE_SINGLE_STREAM = "single_stream"
_SPLIT_MODE_DUAL_STREAM = "dual_stream"
_SPLIT_MODE_DUAL_STREAM_V2 = "dual_stream_v2"
_SPLIT_MODE_PA_HOT_CACHE = "pa_hot_cache"
_FUSED_MERGE_FALLBACK_LOGGED = False
_FUSED_MERGE_SELECTION_LOGGED = False


def _is_fused_sfa_state_merge_available() -> bool:
    """Return whether both the Python wrapper and native schema are installed."""

    return _fused_sfa_state_merge_inplace is not None and hasattr(
        torch.ops.npu, "sfa_state_merge"
    )


def _select_split_decode_mode(attn_impl: str, graph_mode: bool) -> Optional[str]:
    if attn_impl == SPARSE_KV_ATTN_IMPL_PA_GRAPH:
        return _SPLIT_MODE_PA_HOT_CACHE
    if graph_mode:
        if attn_impl == SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH:
            return _SPLIT_MODE_SINGLE_STREAM
        if attn_impl in (
            SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
            SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA,
        ):
            return _SPLIT_MODE_DUAL_STREAM
        if attn_impl == SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2:
            return _SPLIT_MODE_DUAL_STREAM_V2
        return None
    if attn_impl in (
        SPARSE_KV_ATTN_IMPL_SPLIT_EAGER,
        SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH,
        SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
        SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA,
        SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2,
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


def validate_split_fia_support(*, graph_enabled: bool) -> None:
    """Reject runtimes missing the MLA LSE / graph API before allocating KV."""
    fia = getattr(torch_npu, "npu_fused_infer_attention_score_v2", None)
    workspace = getattr(
        torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace", None
    )
    if fia is None or not hasattr(fia, "out") or not callable(workspace):
        raise RuntimeError(
            "split_graph_dual_fia requires FIA v2 .out and its max-workspace API. "
            "Install matching torch_npu/CANN with MLA D=512 mask and LSE support; "
            "the legacy FIA interface cannot supply MLA partition LSE."
        )
    device_name = torch.npu.get_device_name()
    if "950" in device_name:
        raise RuntimeError(
            "split_graph_dual_fia uses a dynamic MLA decode mask supported on "
            "Atlas A2/A3; the Ascend 950 MLA decode mask is not supported."
        )
    if graph_enabled:
        try:
            from torch_npu.npu._npugraph_handlers.npugraph_handler import (
                _NPU_GRAPH_OP_HANDLERS,
            )
        except ImportError as exc:
            raise RuntimeError(
                "split_graph_dual_fia requires torch_npu's FIA v2 NPUGraph "
                "auto-dispatch handler registry. Upgrade the matching NPU runtime."
            ) from exc
        if "npu_fused_infer_attention_score_v2.out" not in _NPU_GRAPH_OP_HANDLERS:
            raise RuntimeError(
                "split_graph_dual_fia requires the FIA v2 .out graph-update handler."
            )
    logger.info(
        "Sparse KV dual FIA: FIA v2 with device partition masks and FP32 LSE; "
        "full-model warmup will validate MLA mask/LSE support in the installed CANN."
    )


def _run_decode_fia_partition(
    partition: SparseKVFiaPartition,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    nope_head_dim: int,
    rope_head_dim: int,
    scale_value: float,
    page_size: int,
    record_stream: bool = True,
) -> _SfaPartitionState:
    """Run FIA v2 on an independent compact partition without host count reads.

    Per-layer hit/miss counts are computed inside the graph. Keep CPU lengths
    at physical capacity and use a device mask for the actual partition. LSE
    represents the same softmax mass as the existing max/sum merge with sum=1.
    """
    if query.dim() != 4 or partition.key.dim() != 4:
        raise ValueError(
            "Dual FIA requires MLA decode Q=[B,1,N,512], RoPE=64, KV_N=1."
        )
    batch_size, query_length, num_heads, value_dim = query.shape
    capacity = partition.key.shape[1]
    if (
        query_length != 1
        or nope_head_dim != 512
        or rope_head_dim != 64
        or value_dim != nope_head_dim
        or num_heads not in (1, 2, 4, 8, 16, 32, 64, 128)
        or tuple(query_rope.shape) != (batch_size, 1, num_heads, rope_head_dim)
        or capacity <= 0
        or tuple(partition.key.shape) != (batch_size, capacity, 1, nope_head_dim)
        or tuple(partition.key_rope.shape)
        != (batch_size, capacity, 1, rope_head_dim)
        or tuple(partition.true_counts.shape) != (batch_size,)
    ):
        raise ValueError(
            "Dual FIA requires MLA decode Q=[B,1,N,512], RoPE=64, KV_N=1."
        )
    if page_size <= 0 or page_size % 16 or page_size > 1024 or capacity % page_size:
        raise ValueError(
            "Dual FIA capacity must be divisible by a 16-aligned page <=1024."
        )
    if (
        partition.buffer.dim() != 1
        or partition.buffer.numel()
        != batch_size * capacity * (nope_head_dim + rope_head_dim)
        or not partition.buffer.is_contiguous()
        or not partition.key.is_contiguous()
        or not partition.key_rope.is_contiguous()
    ):
        raise ValueError("Dual FIA requires prepared contiguous NoPE/RoPE buffers.")
    if any(
        tensor.dtype != query.dtype
        for tensor in (partition.buffer, partition.key, partition.key_rope, query_rope)
    ):
        raise ValueError("Dual FIA query and unquantized KV must have the same dtype.")
    if any(
        tensor.device != query.device
        for tensor in (
            partition.buffer,
            partition.key,
            partition.key_rope,
            partition.true_counts,
            query_rope,
        )
    ):
        raise ValueError("Dual FIA query and prepared KV must be on the same device.")
    if record_stream:
        for tensor in (
            partition.buffer,
            partition.key,
            partition.key_rope,
            partition.true_counts,
            query,
            query_rope,
        ):
            tensor.record_stream(partition.stream)

    positions = torch.arange(capacity, device=query.device, dtype=torch.int32)
    counts = partition.true_counts.view(batch_size, 1)
    # The producer clears these buffers before the fused gather/split copy.
    # Fixed-capacity FIA can read masked V rows, so zero tails and the empty
    # partition dummy must already be finite. Paging only reinterprets storage.
    key = partition.key.view(-1, page_size, nope_head_dim)
    key_rope = partition.key_rope.view(-1, page_size, rope_head_dim)
    # Empty partitions attend to one finite zero dummy; neutralize their output
    # below. This avoids relying on all-masked-row softmax behavior.
    atten_mask = (
        positions.view(1, capacity) >= counts.clamp(min=1)
    ).view(batch_size, 1, 1, capacity).contiguous()
    blocks_per_req = capacity // page_size
    block_table = torch.arange(
        batch_size * blocks_per_req, device=query.device, dtype=torch.int32
    ).view(batch_size, blocks_per_req)
    kwargs = dict(
        query_rope=query_rope,
        key_rope=key_rope,
        atten_mask=atten_mask,
        actual_seq_kvlen=[capacity] * batch_size,
        block_table=block_table,
        block_size=page_size,
        num_query_heads=num_heads,
        num_key_value_heads=1,
        softmax_scale=scale_value,
        input_layout="BSND",
        sparse_mode=0,
        return_softmax_lse=True,
    )
    output = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, num_heads, query_length, 1),
        dtype=torch.float32,
        device=query.device,
    )
    try:
        workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            query, key, key, **kwargs
        )
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query, key, key, **kwargs, workspace=workspace, out=[output, lse]
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Dual FIA requires CANN support for FIA v2 BSND MLA D=512 with "
            "a device attention mask and return_softmax_lse=True."
        ) from exc
    if (
        tuple(output.shape) != tuple(query.shape)
        or output.dtype != query.dtype
        or tuple(lse.shape) != (batch_size, num_heads, query_length, 1)
        or lse.dtype != torch.float32
    ):
        raise RuntimeError("Unexpected dual FIA output or FP32 [B,N,1,1] LSE contract.")
    nonempty = (partition.true_counts > 0).view(batch_size, 1, 1, 1)
    output = torch.where(nonempty, output, 0.0)
    lse = torch.where(nonempty, lse, 0.0).permute(0, 3, 2, 1).contiguous()
    return _SfaPartitionState(
        output=output,
        softmax_max=lse,
        softmax_sum=torch.ones_like(lse),
        true_counts=partition.true_counts,
    )


def _run_combined_decode_fia(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    key: torch.Tensor,
    key_rope: torch.Tensor,
    *,
    page_size: int,
    scale_value: float,
) -> torch.Tensor:
    """FIA smoke test matching AscendAttnBackend.forward_decode_graph's MLA ABI.

    The selected KV comes from sparse prefetch, or is zero-filled in diagnostic
    experiment A. Treat its entire capacity (including padding) as valid;
    this is NOT accuracy-preserving DSA.
    Only reinterpret its contiguous storage as pages -- no additional KV copy.
    """
    batch_size, _, num_heads, _ = query.shape
    _, capacity, num_kv_heads, value_dim = key.shape
    if page_size <= 0 or capacity % page_size:
        raise ValueError(
            f"Combined FIA KV capacity {capacity} must be divisible by "
            f"the positive page size {page_size}."
        )
    blocks_per_req = capacity // page_size
    block_table = torch.arange(
        batch_size * blocks_per_req, dtype=torch.int32, device=query.device
    ).view(batch_size, blocks_per_req)
    kv_cache = key.view(-1, page_size, num_kv_heads * value_dim)
    rope_cache = key_rope.view(-1, page_size, num_kv_heads * key_rope.shape[-1])
    # Like ordinary MLA, pass CPU lengths, not a device Tensor or None.
    # These lengths describe the selected buffer, NOT the full host KV context.
    fia_kwargs = dict(
        query_rope=query_rope,
        key_rope=rope_cache,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        block_table=block_table,
        block_size=page_size,
        input_layout="BSND",
        scale=scale_value,
        actual_seq_lengths_kv=[capacity] * batch_size,
        antiquant_mode=0,
        antiquant_scale=None,
        sparse_mode=0,
    )
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query, kv_cache, kv_cache, **fia_kwargs
    )
    output = torch.empty_like(query, dtype=query.dtype, device=query.device)
    softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
    torch_npu.npu_fused_infer_attention_score.out(
        query,
        kv_cache,
        kv_cache,
        **fia_kwargs,
        workspace=workspace,
        out=[output, softmax_lse],
    )
    return output


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


def _run_decode_sfa_indexed_partition(
    partition: SparseKVIndexedPartition,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    scale_value: float,
) -> _SfaPartitionState:
    """Run SFA over scattered indices in an already split KV workspace."""

    batch_size, query_length, padded_heads, value_dim = query.shape
    actual_query_lengths = torch.ones(
        batch_size, dtype=torch.int32, device=query.device
    ).contiguous()
    output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query=query,
        key=partition.key,
        value=partition.key,
        sparse_indices=partition.sparse_indices,
        scale_value=scale_value,
        actual_seq_lengths_query=actual_query_lengths,
        actual_seq_lengths_kv=partition.actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=partition.key_rope,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )

    expected_output_shape = (batch_size, query_length, padded_heads, value_dim)
    expected_stats_shape = (batch_size, 1, query_length, padded_heads)
    if tuple(output.shape) != expected_output_shape or output.dtype != query.dtype:
        raise RuntimeError(
            "Unexpected indexed split SFA output contract: "
            f"got shape={tuple(output.shape)}, dtype={output.dtype}; "
            f"expected shape={expected_output_shape}, dtype={query.dtype}."
        )
    for name, value in (("softmax_max", softmax_max), ("softmax_sum", softmax_sum)):
        if tuple(value.shape) != expected_stats_shape or value.dtype != torch.float32:
            raise RuntimeError(
                f"Unexpected indexed split SFA {name} contract: "
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
    use_fia: bool = False,
    page_size: int = 128,
) -> torch.Tensor:
    partition_attention = (
        _run_decode_fia_partition if use_fia else _run_decode_sfa_partition
    )
    extra_kwargs = {"page_size": page_size} if use_fia else {}
    hit_attention_done = torch.npu.Event()
    miss_attention_done = torch.npu.Event()

    with torch.profiler.record_function("sparse_kv_split.hit_attention"):
        with torch.npu.stream(ticket.hit.stream):
            hit_state = partition_attention(
                ticket.hit,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
                **extra_kwargs,
            )
            _record_stream_event(ticket.hit.stream, hit_attention_done)

    with torch.profiler.record_function("sparse_kv_split.miss_attention"):
        with torch.npu.stream(ticket.miss.stream):
            miss_state = partition_attention(
                ticket.miss,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
                **extra_kwargs,
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
    use_fia: bool = False,
    page_size: int = 128,
) -> torch.Tensor:
    """Run hit attention on the graph stream and miss attention/refill in parallel."""

    partition_attention = (
        _run_decode_fia_partition if use_fia else _run_decode_sfa_partition
    )
    extra_kwargs = {"page_size": page_size} if use_fia else {}
    with torch.profiler.record_function("sparse_kv_split_graph_dual.hit_attention"):
        with torch.npu.stream(stream):
            hit_state = partition_attention(
                ticket.hit,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
                record_stream=False,
                **extra_kwargs,
            )

    with torch.profiler.record_function("sparse_kv_split_graph_dual.miss_attention"):
        with torch.npu.stream(ticket.miss.stream):
            miss_state = partition_attention(
                ticket.miss,
                query=query,
                query_rope=query_rope,
                nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim,
                scale_value=scale_value,
                record_stream=False,
                **extra_kwargs,
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


def _run_split_decode_attention_graph_dual_v2(
    ticket: SparseKVGraphDualV2Prefetch,
    manager: SparseKVCacheManager,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    scale_value: float,
    stream,
) -> torch.Tensor:
    """Overlap hit SFA with Host miss copy/SFA and avoid a refill copy."""

    with torch.profiler.record_function("sparse_kv_split_graph_dual_v2.hit_attention"):
        with torch.npu.stream(stream):
            hit_state = _run_decode_sfa_indexed_partition(
                ticket.hit,
                query=query,
                query_rope=query_rope,
                scale_value=scale_value,
            )

    with torch.profiler.record_function(
        "sparse_kv_split_graph_dual_v2.miss_attention"
    ):
        with torch.npu.stream(ticket.miss.stream):
            miss_state = _run_decode_sfa_indexed_partition(
                ticket.miss,
                query=query,
                query_rope=query_rope,
                scale_value=scale_value,
            )
            _record_stream_event(ticket.miss.stream, ticket.events.miss_attention_done)

    with torch.npu.stream(stream):
        # The miss copy has already promoted every missing row into a free hot
        # slot. Publish the new map only after those writes are complete.
        manager.publish_graph_dual_v2_slot_map(ticket, stream)
        _wait_stream_event(stream, ticket.events.miss_attention_done)
        for tensor in (
            miss_state.output,
            miss_state.softmax_max,
            miss_state.softmax_sum,
            miss_state.true_counts,
        ):
            tensor.record_stream(stream)
        with torch.profiler.record_function("sparse_kv_split_graph_dual_v2.merge"):
            return _merge_decode_sfa_partitions(
                hit_state, miss_state, manager.merge_impl
            )


def _run_decode_sfa_pa_partition(
    partition: SparseKVPAPartition,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    scale_value: float,
) -> _SfaPartitionState:
    """Run one online-softmax partition directly over the PA hot cache."""

    batch_size, query_length, padded_heads, value_dim = query.shape
    output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query=query,
        key=partition.key,
        value=partition.key,
        sparse_indices=partition.sparse_indices,
        scale_value=scale_value,
        actual_seq_lengths_query=partition.actual_seq_lengths_query,
        actual_seq_lengths_kv=partition.actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=partition.key_rope,
        block_table=partition.block_table,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )

    expected_output_shape = (batch_size, query_length, padded_heads, value_dim)
    expected_stats_shape = (batch_size, 1, query_length, padded_heads)
    if tuple(output.shape) != expected_output_shape or output.dtype != query.dtype:
        raise RuntimeError(
            "Unexpected PA partition SFA output contract: "
            f"got shape={tuple(output.shape)}, dtype={output.dtype}; "
            f"expected shape={expected_output_shape}, dtype={query.dtype}."
        )
    for name, value in (("softmax_max", softmax_max), ("softmax_sum", softmax_sum)):
        if tuple(value.shape) != expected_stats_shape or value.dtype != torch.float32:
            raise RuntimeError(
                f"Unexpected PA partition SFA {name} contract: "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}; "
                f"expected shape={expected_stats_shape}, dtype=torch.float32."
            )

    return _SfaPartitionState(
        output=output,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        true_counts=partition.true_counts,
    )


def _run_split_decode_attention_pa_hot_cache(
    ticket: SparseKVPAHotCachePrefetch,
    manager: SparseKVCacheManager,
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    scale_value: float,
    stream,
) -> torch.Tensor:
    """Overlap hit PA-SFA with H2D, then run miss PA-SFA and merge."""

    with torch.npu.stream(stream):
        with torch.profiler.record_function(
            "sparse_kv_pa_hot_cache.hit_attention"
        ):
            hit_state = _run_decode_sfa_pa_partition(
                ticket.hit,
                query=query,
                query_rope=query_rope,
                scale_value=scale_value,
            )

        # The wait is sequenced after hit attention on this same stream. The
        # auxiliary stream only writes slots not referenced by hit indices.
        manager.publish_pa_slot_map(ticket, stream)

        with torch.profiler.record_function(
            "sparse_kv_pa_hot_cache.miss_attention"
        ):
            miss_state = _run_decode_sfa_pa_partition(
                ticket.miss,
                query=query,
                query_rope=query_rope,
                scale_value=scale_value,
            )

        with torch.profiler.record_function("sparse_kv_pa_hot_cache.merge"):
            return _merge_decode_sfa_partitions(
                hit_state, miss_state, manager.merge_impl
            )


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

    # The startup-only diagnostic must cover both warmup and capture. Never
    # bypass prefill: it still needs the authoritative host KV for attention.
    skip_decode_kv_io = (
        forward_batch.forward_mode.is_decode() and sparse_kv_manager.fia_skip_kv_io
    )
    if save_kv_cache and not skip_decode_kv_io:
        sparse_kv_manager.offload_v2(k_nope, k_pe, layer, forward_batch, stream)

    if (
        sparse_kv_manager.attn_impl == SPARSE_KV_ATTN_IMPL_NATIVE_FIA
        and forward_batch.forward_mode.is_decode()
    ):
        if layer.tp_k_head_num != 1:
            raise ValueError("Native FIA offload requires one MLA KV head.")
        c_kv_cache, k_rope_cache, block_table = (
            sparse_kv_manager.prefetch_native_fia(layer, forward_batch, topk_indices)
        )
        # Capture uses the already prepared CPU list; replay updates this FIA
        # input explicitly. The indexer keeps its original full-context lengths.
        metadata = backend.forward_metadata
        seq_lens = (
            metadata.seq_lens_cpu_list
            if metadata.seq_lens_cpu_int is None
            else metadata.seq_lens_cpu_int.tolist()
        )
        selected_lengths = [
            min(max(int(length), 0), sparse_kv_manager.sparse_context_len)
            for length in seq_lens
        ]
        # Eager metadata is prepared before DP pads the query/request rows.
        # Preserve local lengths and explicitly mask the extra FIA batch rows.
        batch_size = block_table.shape[0]
        if len(selected_lengths) > batch_size:
            raise ValueError("Native FIA CPU lengths exceed the selected-KV batch.")
        selected_lengths += [0] * (batch_size - len(selected_lengths))
        return backend.forward_mla_fia(
            q_nope,
            q_pe,
            layer,
            c_kv_cache,
            k_rope_cache,
            block_table,
            selected_lengths,
        )

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
        selected_kv_length = sparse_kv_manager.sparse_context_len
        num_kv_heads = layer.tp_k_head_num
        num_query_heads = layer.tp_q_head_num
        nope_head_dim = backend.kv_lora_rank
        rope_head_dim = backend.qk_rope_head_dim

        assert num_kv_heads == 1, (
            "MLA selected KV path expects KV_N == 1, "
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
        use_split_fia = (
            sparse_kv_manager.attn_impl == SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA
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
                "Sparse KV %s is experimental: hit attention runs "
                "on the graph stream while host misses, miss attention, and "
                "hot-cache refill run on a persistent worker stream. "
                "Partition attention: %s.",
                sparse_kv_manager.attn_impl,
                "FIA v2 with device masks and LSE merge" if use_split_fia else "SFA",
            )
            sparse_kv_manager._split_graph_dual_logged = True
        if (
            split_mode == _SPLIT_MODE_DUAL_STREAM_V2
            and not sparse_kv_manager._split_graph_dual_v2_logged
        ):
            logger.warning(
                "Sparse KV split_graph_dual_v2 is experimental: KV remains "
                "noncompact, hit rows keep their HBM slots, and each miss is "
                "copied once into the SFA workspace and hot cache."
            )
            sparse_kv_manager._split_graph_dual_v2_logged = True
        if (
            split_mode == _SPLIT_MODE_PA_HOT_CACHE
            and not sparse_kv_manager._pa_hot_cache_logged
        ):
            logger.warning(
                "Sparse KV pa_graph is experimental: hit PA-SFA overlaps one "
                "Host miss promotion, then miss PA-SFA and state merge run on "
                "the graph stream without a hit-KV D2D gather."
            )
            sparse_kv_manager._pa_hot_cache_logged = True

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
                    use_fia=use_split_fia,
                    page_size=backend.page_size,
                )
            elif split_mode == _SPLIT_MODE_DUAL_STREAM_V2:
                ticket = sparse_kv_manager.prefetch_partitions_graph_dual_v2(
                    layer,
                    forward_batch,
                    topk_indices,
                    stream,
                    dtype=k.dtype,
                )
                decode_output = _run_split_decode_attention_graph_dual_v2(
                    ticket,
                    sparse_kv_manager,
                    query=q_nope_sfa,
                    query_rope=q_rope_sfa,
                    scale_value=layer.scaling,
                    stream=stream,
                )
            elif split_mode == _SPLIT_MODE_PA_HOT_CACHE:
                ticket = sparse_kv_manager.prefetch_pa_hot_cache(
                    layer,
                    forward_batch,
                    topk_indices,
                    stream,
                    dtype=k.dtype,
                )
                decode_output = _run_split_decode_attention_pa_hot_cache(
                    ticket,
                    sparse_kv_manager,
                    query=q_nope_sfa,
                    query_rope=q_rope_sfa,
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
                    use_fia=use_split_fia,
                    page_size=backend.page_size,
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
        # Experiment A keeps the same initialized buffer, splits, contiguous
        # conversions and FIA call, but removes real KV traffic and its events.
        if not skip_decode_kv_io:
            sparse_kv_manager.prefetch(
                layer, forward_batch, topk_indices, selected_kv, stream
            )
        selected_k_nope, selected_k_rope = selected_kv.split(
            [nope_head_dim, rope_head_dim], dim=-1
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

        attn_out = _run_combined_decode_fia(
            q_nope_sfa,
            q_rope_sfa,
            k_nope_sfa,
            k_rope_sfa,
            page_size=backend.page_size,
            scale_value=layer.scaling,
        )
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
