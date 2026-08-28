"""Sparsity-driven KV offload manager for the Ascend backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, AnyStr, List, Optional, Union

import sgl_kernel_npu.sparsity_driven_kv_offload as sparse_kv_ops
import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    slot_map_lookup,
    unidex_copy_inplace,
)

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparse_kv_attn_impl,
    get_sparse_kv_merge_impl,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    import torch.npu

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
_LAYER_TO_LOG = [0, 1]


def _log_debug(device: torch.device, layer: RadixAttention, msg: AnyStr):
    if device == torch.device("npu:0") and layer.layer_id in _LAYER_TO_LOG:
        logger.info(msg)


def _profile_push(name: str):
    profiler_range = torch.profiler.record_function(name)
    profiler_range.__enter__()
    return profiler_range


def _profile_pop(profiler_range):
    profiler_range.__exit__(None, None, None)


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


@dataclass
class SparseKVPartition:
    kv: torch.Tensor
    sparse_indices: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    true_counts: torch.Tensor
    stream: torch.npu.Stream


@dataclass
class SparseKVPrefetchTicket:
    hit: SparseKVPartition
    miss: SparseKVPartition
    refill_done: torch.npu.Event


@dataclass
class SparseKVSingleStreamPrefetch:
    hit: SparseKVPartition
    miss: SparseKVPartition


@dataclass
class SparseKVGraphDualLayerEvents:
    inputs_ready: torch.npu.Event
    hit_copy_done: torch.npu.Event
    miss_attention_done: torch.npu.Event
    refill_done: torch.npu.Event


@dataclass
class SparseKVGraphDualState:
    max_batch_size: int
    hit_kv: torch.Tensor
    miss_kv: torch.Tensor
    miss_stream: torch.npu.Stream
    layer_events: list[SparseKVGraphDualLayerEvents]


@dataclass
class SparseKVGraphDualPrefetch:
    hit: SparseKVPartition
    miss: SparseKVPartition
    layer_idx: int
    hit_compact_indices: torch.Tensor
    miss_compact_indices: torch.Tensor
    refill_dst_indices: torch.Tensor
    hit_valid_flat: torch.Tensor
    miss_valid_flat: torch.Tensor
    slot_map_row_indices: torch.Tensor
    slot_map_flat_indices: torch.Tensor
    slot_map_slot_values: torch.Tensor
    events: SparseKVGraphDualLayerEvents


@dataclass
class SparseKVIndexedPartition:
    key: torch.Tensor
    key_rope: torch.Tensor
    sparse_indices: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    true_counts: torch.Tensor
    stream: torch.npu.Stream


@dataclass
class SparseKVGraphDualV2LayerEvents:
    inputs_ready: torch.npu.Event
    miss_copy_done: torch.npu.Event
    miss_attention_done: torch.npu.Event


@dataclass
class SparseKVGraphDualV2State:
    max_batch_size: int
    selected_k_nope: torch.Tensor
    selected_k_rope: torch.Tensor
    attention_dst_indices: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    hit_sparse_indices: torch.Tensor
    miss_sparse_indices: torch.Tensor
    hit_counts: torch.Tensor
    miss_counts: torch.Tensor
    hit_src_indices: torch.Tensor
    miss_src_indices: torch.Tensor
    miss_hot_dst_indices: torch.Tensor
    hit_valid_mask: torch.Tensor
    miss_valid_mask: torch.Tensor
    slot_map_flat_indices: torch.Tensor
    slot_map_slot_values: torch.Tensor
    tile_hit_counts: torch.Tensor
    tile_miss_counts: torch.Tensor
    tile_hit_offsets: torch.Tensor
    tile_miss_offsets: torch.Tensor
    selected_slots: torch.Tensor
    tile_occupied_bitmaps: torch.Tensor
    occupied_bitmaps: torch.Tensor
    free_slot_prefixes: torch.Tensor
    miss_stream: torch.npu.Stream
    layer_events: list[SparseKVGraphDualV2LayerEvents]


@dataclass
class SparseKVGraphDualV2Prefetch:
    hit: SparseKVIndexedPartition
    miss: SparseKVIndexedPartition
    layer_idx: int
    slot_map_row_indices: torch.Tensor
    slot_map_flat_indices: torch.Tensor
    slot_map_slot_values: torch.Tensor
    events: SparseKVGraphDualV2LayerEvents


class SparseKVCacheManager:
    copy_stream = None
    miss_shm_cpu_tensor: list = []
    miss_shm_dev_ptr: Optional[int] = None
    miss_shm_shape: list = []
    miss_shm_dtype: list = []

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        # tp_group: torch.distributed.ProcessGroup,
        # server_args: ServerArgs,
    ) -> None:
        enable_memory_saver = False
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        # Include the padding row because real request IDs can equal the
        # configured capacity when row 0 is reserved for graph padding.
        self.size = int(req_to_token_pool.req_to_token.shape[0])
        self.max_context_len = req_to_token_pool.max_context_len
        self.sparse_context_len = 2048
        self.device = req_to_token_pool.device
        paged_kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(paged_kv_cache, MLATokenToKVPool):
            raise TypeError(
                "SparseKVCacheManager requires an MLATokenToKVPool, "
                f"got {type(paged_kv_cache).__name__}"
            )
        self.paged_kv_cache = paged_kv_cache
        self.start_layer = paged_kv_cache.start_layer
        # MLA params
        self.head_num = 1
        # Maybe useful?
        self.kv_lora_rank = self.paged_kv_cache.kv_lora_rank
        self.qk_rope_head_dim = self.paged_kv_cache.qk_rope_head_dim
        # kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.head_dim = (
            self.paged_kv_cache.kv_lora_rank + self.paged_kv_cache.qk_rope_head_dim
        )
        self.store_dtype = self.paged_kv_cache.store_dtype
        self.layer_num = self.paged_kv_cache.layer_num
        self.attn_impl = get_sparse_kv_attn_impl()
        self.merge_impl = get_sparse_kv_merge_impl()
        logger.info(
            "Sparse KV attention implementation: %s; merge implementation: %s.",
            self.attn_impl,
            self.merge_impl,
        )
        self._split_graph_fallback_logged = False
        self._split_graph_phase_one_logged = False
        self._split_graph_dual_logged = False
        self._split_graph_dual_v2_logged = False
        self._graph_dual_state: Optional[SparseKVGraphDualState] = None
        self._graph_dual_v2_state: Optional[SparseKVGraphDualV2State] = None
        self._prefetch_d2d_hit_stream = torch.npu.Stream()
        self._prefetch_h2d_miss_stream = torch.npu.Stream()
        self._prefetch_refill_stream = torch.npu.Stream()
        # device KV buffer
        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                # [bs, ctx_len, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.device_kv_buffer: list[torch.Tensor] = [
                    torch.full(
                        (
                            self.size,
                            self.sparse_context_len,
                            self.head_num,
                            self.head_dim,
                        ),
                        1111.11,
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
        except Exception as e:
            self._raise_buffer_allocation_error("device_kv_buffer", e)

        try:
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                # Reserve the last row for padded requests and ensure token index
                # `max_context_len` is a valid sentinel column for masked writes.
                # The row width is also aligned to eight int32 values (32 bytes).
                self.device_slot_map: list[torch.Tensor] = [
                    torch.full(
                        (
                            self.size + 1,
                            (self.max_context_len // 8 + 1) * 8,
                        ),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
        except Exception as e:
            self._raise_buffer_allocation_error("device_slot_map", e)

        # Host KV buffer
        # [bs, ctx_len, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        # self.host_kv_buffer = [
        #     torch.full(
        #         (self.size, self.max_context_len, self.head_num, self.head_dim),
        #         2222.22,
        #         dtype=self.store_dtype,
        #         device="cpu",
        #         pin_memory=True
        #     )
        #     for _ in range(self.layer_num)
        # ]

        self.host_kv_buffer: list[torch.Tensor] = []
        self.host_ptr_list: list[int] = []
        self.dev_ptr_list: list[int] = []

        host_kv_shape = (
            self.size,
            self.max_context_len,
            self.head_num,
            self.head_dim,
        )
        logger.info("Sparse KV host buffer shape: %s", host_kv_shape)
        device_id = torch.npu.current_device()

        try:
            for layer_idx in range(self.layer_num):
                shm_cpu_tensor, host_ptr, dev_ptr = create_shm_tensor(
                    shape=host_kv_shape,
                    dtype=self.store_dtype,
                    device_id=device_id,
                    name=f"host_kv_layer_{layer_idx}_rank_{device_id}",
                )
                shm_cpu_tensor.fill_(2222.22)

                self.host_kv_buffer.append(shm_cpu_tensor)
                self.host_ptr_list.append(host_ptr)
                self.dev_ptr_list.append(dev_ptr)
        except Exception as e:
            self._raise_buffer_allocation_error("host_kv_buffer", e)
        # Maybe useless
        self.host_kv_ctx_len = torch.zeros(
            (self.size, self.max_context_len), dtype=torch.int32, device="cpu"
        )
        self.topk_indices_cpu = None
        self.token_on_device_cpu = None
        self.device_token_pos_cpu = None
        self.current_req_indices_cpu = None
        self._debug_log_cnt = 10

        self._device_cache_slot_ids = torch.arange(
            self.sparse_context_len, dtype=torch.long, device=self.device
        )
        self._slot_map_width = (self.max_context_len // 8 + 1) * 8

        self._install_req_alloc_hook(req_to_token_pool)

    def _raise_buffer_allocation_error(
        self,
        buffer_name: str,
        exc: Exception,
    ) -> None:
        raise RuntimeError(
            "Failed to allocate sparse KV buffer "
            f"{buffer_name}: req_capacity={self.size}, "
            f"max_context_len={self.max_context_len}, "
            f"sparse_context_len={self.sparse_context_len}. "
            "The sparse KV request capacity may be too large; set a smaller "
            "--max-running-requests for sparse KV offload."
        ) from exc

    def prepare_graph_dual_state(self, max_batch_size: int) -> None:
        """Allocate fixed-address workspace and synchronization for graph dual-stream.

        A single hit/miss workspace pair is reused across layers.  The graph path
        joins ``refill_done`` before returning from each layer, so the next layer
        cannot overwrite a partition while the miss stream is still consuming it.
        Graph replays are expected to be serialized by the scheduler.
        """

        max_batch_size = int(max_batch_size)
        if max_batch_size <= 0:
            raise ValueError(
                "Sparse KV graph-dual workspace requires max_batch_size > 0, "
                f"got {max_batch_size}."
            )

        state = self._graph_dual_state
        if state is not None:
            if max_batch_size > state.max_batch_size:
                raise RuntimeError(
                    "Sparse KV graph-dual workspace cannot grow after it has been "
                    f"initialized: requested={max_batch_size}, "
                    f"allocated={state.max_batch_size}."
                )
            return

        partition_shape = (
            max_batch_size,
            self.sparse_context_len,
            self.head_num,
            self.head_dim,
        )
        try:
            hit_kv = torch.empty(
                partition_shape, dtype=self.store_dtype, device=self.device
            )
            miss_kv = torch.empty_like(hit_kv)
            miss_stream = torch.npu.Stream()
            layer_events = [
                SparseKVGraphDualLayerEvents(
                    inputs_ready=torch.npu.Event(),
                    hit_copy_done=torch.npu.Event(),
                    miss_attention_done=torch.npu.Event(),
                    refill_done=torch.npu.Event(),
                )
                for _ in range(self.layer_num)
            ]
        except Exception as exc:
            raise RuntimeError(
                "Failed to allocate sparse KV graph-dual workspace: "
                f"max_batch_size={max_batch_size}, shape={partition_shape}."
            ) from exc

        self._graph_dual_state = SparseKVGraphDualState(
            max_batch_size=max_batch_size,
            hit_kv=hit_kv,
            miss_kv=miss_kv,
            miss_stream=miss_stream,
            layer_events=layer_events,
        )
        workspace_gib = (
            (hit_kv.numel() + miss_kv.numel()) * hit_kv.element_size() / GB
        )
        logger.info(
            "Prepared sparse KV graph-dual workspace with shape %s (%.3f GiB).",
            partition_shape,
            workspace_gib,
        )

    def prepare_graph_dual_v2_state(self, max_batch_size: int) -> None:
        """Allocate shared noncompact SFA buffers for the two-copy graph path."""

        required_ops = (
            ("unidex_split_copy_inplace", "unidex_split_copy"),
            ("unidex_split_copy_promote_inplace", "unidex_split_copy_promote"),
            (
                "sparse_kv_partition_plan_parallel_inplace",
                "sparse_kv_partition_plan_parallel",
            ),
        )
        for python_symbol, native_schema in required_ops:
            if not hasattr(sparse_kv_ops, python_symbol) or not hasattr(
                torch.ops.npu, native_schema
            ):
                raise RuntimeError(
                    "Sparse KV split_graph_dual_v2 requires a matching "
                    "sgl-kernel-npu wheel exporting both the Python wrapper "
                    f"{python_symbol} and native schema npu::{native_schema}."
                )

        max_batch_size = int(max_batch_size)
        if max_batch_size <= 0:
            raise ValueError(
                "Sparse KV graph-dual-v2 workspace requires max_batch_size > 0, "
                f"got {max_batch_size}."
            )
        state = self._graph_dual_v2_state
        if state is not None:
            if max_batch_size > state.max_batch_size:
                raise RuntimeError(
                    "Sparse KV graph-dual-v2 workspace cannot grow after "
                    f"initialization: requested={max_batch_size}, "
                    f"allocated={state.max_batch_size}."
                )
            return

        nope_shape = (
            max_batch_size,
            self.sparse_context_len,
            self.head_num,
            self.kv_lora_rank,
        )
        rope_shape = (
            max_batch_size,
            self.sparse_context_len,
            self.head_num,
            self.qk_rope_head_dim,
        )
        try:
            selected_k_nope = torch.zeros(
                nope_shape, dtype=self.store_dtype, device=self.device
            )
            selected_k_rope = torch.zeros(
                rope_shape, dtype=self.store_dtype, device=self.device
            )
            plan_shape = (max_batch_size, self.sparse_context_len)
            plan_size = max_batch_size * self.sparse_context_len
            attention_dst_indices = torch.arange(
                plan_size, dtype=torch.int64, device=self.device
            )
            actual_seq_lengths_kv = torch.full(
                (max_batch_size,),
                self.sparse_context_len,
                dtype=torch.int32,
                device=self.device,
            )
            hit_sparse_indices = torch.empty(
                plan_shape, dtype=torch.int32, device=self.device
            )
            miss_sparse_indices = torch.empty_like(hit_sparse_indices)
            hit_counts = torch.empty(
                (max_batch_size,), dtype=torch.int32, device=self.device
            )
            miss_counts = torch.empty_like(hit_counts)
            hit_src_indices = torch.empty(
                (plan_size,), dtype=torch.int64, device=self.device
            )
            miss_src_indices = torch.empty_like(hit_src_indices)
            miss_hot_dst_indices = torch.empty_like(hit_src_indices)
            hit_valid_mask = torch.empty(
                (plan_size,), dtype=torch.bool, device=self.device
            )
            miss_valid_mask = torch.empty_like(hit_valid_mask)
            slot_map_flat_indices = torch.empty_like(hit_src_indices)
            slot_map_slot_values = torch.empty(
                plan_shape, dtype=torch.int32, device=self.device
            )
            tile_shape = (max_batch_size, self.sparse_context_len // 64)
            tile_hit_counts = torch.empty(
                tile_shape, dtype=torch.int32, device=self.device
            )
            tile_miss_counts = torch.empty_like(tile_hit_counts)
            tile_hit_offsets = torch.empty_like(tile_hit_counts)
            tile_miss_offsets = torch.empty_like(tile_hit_counts)
            selected_slots = torch.empty(
                plan_shape, dtype=torch.int32, device=self.device
            )
            bitmap_words = self.sparse_context_len // 32
            tile_occupied_bitmaps = torch.empty(
                (
                    max_batch_size,
                    self.sparse_context_len // 64,
                    bitmap_words,
                ),
                dtype=torch.int32,
                device=self.device,
            )
            occupied_bitmaps = torch.empty(
                (max_batch_size, bitmap_words),
                dtype=torch.int32,
                device=self.device,
            )
            free_slot_prefixes = torch.empty_like(occupied_bitmaps)
            miss_stream = torch.npu.Stream()
            layer_events = [
                SparseKVGraphDualV2LayerEvents(
                    inputs_ready=torch.npu.Event(),
                    miss_copy_done=torch.npu.Event(),
                    miss_attention_done=torch.npu.Event(),
                )
                for _ in range(self.layer_num)
            ]
        except Exception as exc:
            raise RuntimeError(
                "Failed to allocate sparse KV graph-dual-v2 workspace: "
                f"max_batch_size={max_batch_size}, nope_shape={nope_shape}, "
                f"rope_shape={rope_shape}."
            ) from exc

        self._graph_dual_v2_state = SparseKVGraphDualV2State(
            max_batch_size=max_batch_size,
            selected_k_nope=selected_k_nope,
            selected_k_rope=selected_k_rope,
            attention_dst_indices=attention_dst_indices,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            hit_sparse_indices=hit_sparse_indices,
            miss_sparse_indices=miss_sparse_indices,
            hit_counts=hit_counts,
            miss_counts=miss_counts,
            hit_src_indices=hit_src_indices,
            miss_src_indices=miss_src_indices,
            miss_hot_dst_indices=miss_hot_dst_indices,
            hit_valid_mask=hit_valid_mask,
            miss_valid_mask=miss_valid_mask,
            slot_map_flat_indices=slot_map_flat_indices,
            slot_map_slot_values=slot_map_slot_values,
            tile_hit_counts=tile_hit_counts,
            tile_miss_counts=tile_miss_counts,
            tile_hit_offsets=tile_hit_offsets,
            tile_miss_offsets=tile_miss_offsets,
            selected_slots=selected_slots,
            tile_occupied_bitmaps=tile_occupied_bitmaps,
            occupied_bitmaps=occupied_bitmaps,
            free_slot_prefixes=free_slot_prefixes,
            miss_stream=miss_stream,
            layer_events=layer_events,
        )
        workspace_bytes = (
            (selected_k_nope.numel() + selected_k_rope.numel())
            * selected_k_nope.element_size()
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    attention_dst_indices,
                    actual_seq_lengths_kv,
                    hit_sparse_indices,
                    miss_sparse_indices,
                    hit_counts,
                    miss_counts,
                    hit_src_indices,
                    miss_src_indices,
                    miss_hot_dst_indices,
                    hit_valid_mask,
                    miss_valid_mask,
                    slot_map_flat_indices,
                    slot_map_slot_values,
                    tile_hit_counts,
                    tile_miss_counts,
                    tile_hit_offsets,
                    tile_miss_offsets,
                    selected_slots,
                    tile_occupied_bitmaps,
                    occupied_bitmaps,
                    free_slot_prefixes,
                )
            )
        )
        workspace_gib = workspace_bytes / GB
        logger.info(
            "Prepared sparse KV graph-dual-v2 shared workspace: nope=%s, "
            "rope=%s (%.3f GiB).",
            nope_shape,
            rope_shape,
            workspace_gib,
        )

    def init_req(self, req: Req) -> None:
        if req.is_chunked > 0:
            return
        rid = req.req_pool_idx
        if rid is None:
            raise RuntimeError(
                "Cannot initialize sparse KV state before allocating a request pool slot"
            )
        current_len = len(req.origin_input_ids)
        self.host_kv_ctx_len[rid] = current_len
        self.reset_requests([rid])

    def reset_requests(self, req_ids: List[int]) -> None:
        if not req_ids:
            return

        req_ids_tensor = torch.tensor(
            req_ids, dtype=torch.long, device=self.device
        ).contiguous()
        for layer_idx in range(self.layer_num):
            self.device_slot_map[layer_idx].index_fill_(0, req_ids_tensor, -1)

    def _install_req_alloc_hook(self, req_to_token_pool: ReqToTokenPool) -> None:
        original_alloc = getattr(
            req_to_token_pool, "_sparse_kv_original_alloc", req_to_token_pool.alloc
        )
        setattr(req_to_token_pool, "_sparse_kv_original_alloc", original_alloc)

        def alloc_with_sparse_reset(reqs: list[Req]) -> Optional[List[int]]:
            newly_allocated = [req.req_pool_idx is None for req in reqs]
            req_pool_indices = original_alloc(reqs)
            if req_pool_indices is not None:
                self.reset_requests(
                    [
                        req_pool_indices[i]
                        for i, is_new in enumerate(newly_allocated)
                        if is_new
                    ]
                )
            return req_pool_indices

        setattr(req_to_token_pool, "alloc", alloc_with_sparse_reset)

    def offload(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
    ):
        layer_idx = layer.layer_id - self.start_layer
        device = k.device

        # k:        [total_token_slots, nhead, dim]
        # k_rope:   [total_token_slots, nhead, dim]
        # kv_device: [total_token_slots, nhead, 2*dim]
        kv_device = torch.cat([k, k_rope], dim=-1)

        # Source row indices into kv_device.
        # In graph mode this tensor is expected to have a static shape.
        src_token_indices = forward_batch.out_cache_loc.to(torch.long).contiguous()
        static_token_slots = int(src_token_indices.numel())

        if forward_batch.forward_mode.is_decode():
            # decode graph mode:
            # one token slot per request, padded requests are masked out
            req_ids = forward_batch.req_pool_indices.to(torch.long)
            token_pos = (forward_batch.seq_lens - 1).to(torch.long)

            dst_token_indices = (
                req_ids * self.max_context_len + token_pos
            ).contiguous()

            # Existing graph decode convention:
            # padded decode requests carry seq_len == 1.
            valid_mask = (
                (forward_batch.seq_lens != 1)
                & (src_token_indices >= 0)
                & (req_ids >= 0)
            ).contiguous()

        else:
            # prefill graph mode:
            # assume out_cache_loc is laid out as [B, TOKENS_PER_REQ] flattened row-major
            if (
                forward_batch.extend_seq_lens is None
                or forward_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "Sparse graph prefill offload requires extend_seq_lens and "
                    "extend_prefix_lens in ForwardBatch."
                )

            batch_size = int(forward_batch.req_pool_indices.shape[0])
            if batch_size <= 0:
                return

            if static_token_slots % batch_size != 0:
                raise RuntimeError(
                    f"out_cache_loc length {static_token_slots} is not divisible by "
                    f"batch size {batch_size}. Cannot infer graph static token layout."
                )

            tokens_per_req = static_token_slots // batch_size

            req_ids = forward_batch.req_pool_indices.to(torch.long)
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            extend_prefix_lens = forward_batch.extend_prefix_lens.to(torch.long)

            local_offsets = (
                torch.arange(
                    tokens_per_req,
                    device=device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, tokens_per_req)
            )

            req_ids_2d = req_ids.unsqueeze(1).expand(batch_size, tokens_per_req)
            dst_pos_2d = extend_prefix_lens.unsqueeze(1) + local_offsets

            dst_token_indices = (
                (req_ids_2d * self.max_context_len + dst_pos_2d)
                .reshape(-1)
                .contiguous()
            )

            valid_mask = (
                (local_offsets < extend_seq_lens.unsqueeze(1)).reshape(-1)
                & (src_token_indices >= 0)
                & (req_ids_2d.reshape(-1) >= 0)
            ).contiguous()

        # Layout check:
        # kv_device rows:                 [token_slot]
        # host_kv_buffer[layer] rows:     [req_id, seq_pos]
        # block dims must match on [nhead, 2*dim]
        assert kv_device.shape[1:] == self.host_kv_buffer[layer_idx].shape[2:]
        # torch.npu.synchronize()
        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                kv_device,
                self.host_kv_buffer[layer_idx],
                src_token_indices,
                dst_token_indices,
                valid_mask,
                1,  # kv_device: [token_slot, nhead, 2*dim]
                2,  # host_kv_buffer: [num_req, max_context_len, nhead, 2*dim]
                block_dim=48,
                dst_ptr=self.dev_ptr_list[layer_idx],
            )
        # torch.npu.synchronize()

    def offload_v2(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
    ):
        """Offload compact per-forward KV rows into the sparse host KV buffer.

        v1 expects k/k_rope to be full native KV-cache views and therefore uses
        forward_batch.out_cache_loc as source rows. v2 expects k/k_rope to be
        compact rows produced by the current forward pass, so source rows are
        simply [0, num_new_tokens). The native cache slot is kept only as
        validity metadata.

        Keep src_tensor/dst_tensor/src_index/dst_index/valid_mask explicit so
        the final copy can be swapped to a custom kernel without changing the
        graph-friendly index construction.
        """
        layer_idx = layer.layer_id - self.start_layer
        device = k.device

        # k:         [num_new_tokens, nhead, kv_lora_rank]
        # k_rope:    [num_new_tokens, nhead, qk_rope_head_dim]
        # kv_device: [num_new_tokens, nhead, kv_lora_rank + qk_rope_head_dim]
        src_tensor = torch.cat([k, k_rope], dim=-1).contiguous()
        dst_tensor = self.host_kv_buffer[layer_idx]
        src_index = torch.arange(src_tensor.shape[0], device=device, dtype=torch.long)
        num_src_rows = int(src_tensor.shape[0])

        if forward_batch.forward_mode.is_decode():
            req_ids = forward_batch.req_pool_indices.to(torch.long)
            token_pos = (forward_batch.seq_lens - 1).to(torch.long)
            cache_loc = forward_batch.out_cache_loc.to(torch.long)

            if int(req_ids.shape[0]) != num_src_rows:
                raise RuntimeError(
                    "Sparse v2 decode offload expects compact KV rows to match "
                    f"batch size, got {num_src_rows} and {int(req_ids.shape[0])}."
                )
            if int(cache_loc.shape[0]) != num_src_rows:
                raise RuntimeError(
                    "Sparse v2 decode offload expects out_cache_loc rows to match "
                    f"compact KV rows, got {int(cache_loc.shape[0])} and "
                    f"{num_src_rows}."
                )

            dst_index = (req_ids * self.max_context_len + token_pos).contiguous()
            valid_mask = (
                (forward_batch.seq_lens != 1)
                & (cache_loc >= 0)
                & (req_ids >= 0)
                & (token_pos >= 0)
                & (token_pos < self.max_context_len)
            ).contiguous()
        else:
            if (
                forward_batch.extend_seq_lens is None
                or forward_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "Sparse v2 prefill offload requires extend_seq_lens and "
                    "extend_prefix_lens in ForwardBatch."
                )

            req_ids = forward_batch.req_pool_indices.to(torch.long)
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            extend_prefix_lens = forward_batch.extend_prefix_lens.to(torch.long)

            batch_size = int(req_ids.shape[0])
            if batch_size <= 0:
                return

            if int(extend_seq_lens.shape[0]) != batch_size:
                raise RuntimeError(
                    "Sparse v2 prefill offload expects extend_seq_lens to be "
                    f"padded to batch size {batch_size}, got "
                    f"{int(extend_seq_lens.shape[0])}."
                )

            prefix_len_size = int(extend_prefix_lens.shape[0])
            if prefix_len_size < batch_size:
                extend_prefix_lens = torch.cat(
                    [
                        extend_prefix_lens,
                        torch.zeros(
                            batch_size - prefix_len_size,
                            device=device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
            elif prefix_len_size > batch_size:
                raise RuntimeError(
                    "Sparse v2 prefill offload expects extend_prefix_lens length "
                    f"<= batch size {batch_size}, got {prefix_len_size}."
                )

            if forward_batch.extend_seq_lens_cpu is not None:
                extend_seq_lens_sum = int(
                    sum(forward_batch.extend_seq_lens_cpu[:batch_size])
                )
            else:
                extend_seq_lens_sum = int(extend_seq_lens.sum().item())
            if extend_seq_lens_sum == num_src_rows:
                # Chunk prefill emits compact rows as [req0 tokens][req1 tokens]...
                # instead of graph-captured padded [B, tokens_per_req] rows.
                seq_starts = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens
                flat_req_ids = torch.repeat_interleave(
                    req_ids, extend_seq_lens, output_size=num_src_rows
                )
                flat_seq_starts = torch.repeat_interleave(
                    seq_starts, extend_seq_lens, output_size=num_src_rows
                )
                flat_prefix_lens = torch.repeat_interleave(
                    extend_prefix_lens, extend_seq_lens, output_size=num_src_rows
                )
                token_pos = (
                    flat_prefix_lens
                    + torch.arange(num_src_rows, device=device, dtype=torch.long)
                    - flat_seq_starts
                )
                dst_index = (
                    flat_req_ids * self.max_context_len + token_pos
                ).contiguous()
                valid_mask = (
                    (flat_req_ids >= 0)
                    & (token_pos >= 0)
                    & (token_pos < self.max_context_len)
                )
            else:
                if num_src_rows % batch_size != 0:
                    raise RuntimeError(
                        "Sparse v2 prefill offload expects either compact ragged "
                        "layout with rows=sum(extend_seq_lens) or graph-style "
                        f"row-major layout [B, tokens_per_req], got rows={num_src_rows}, "
                        f"batch={batch_size}, extend_seq_lens_sum={extend_seq_lens_sum}."
                    )

                # Graph-friendly static layout:
                # compact rows are interpreted as [batch_size, tokens_per_req].
                # Invalid padded columns are masked by local_offsets < extend_seq_lens.
                tokens_per_req = num_src_rows // batch_size

                local_offsets = (
                    torch.arange(tokens_per_req, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, tokens_per_req)
                )
                req_ids_2d = req_ids.unsqueeze(1).expand(batch_size, tokens_per_req)
                token_pos_2d = extend_prefix_lens.unsqueeze(1) + local_offsets

                dst_index = (
                    (req_ids_2d * self.max_context_len + token_pos_2d)
                    .reshape(-1)
                    .contiguous()
                )
                valid_mask = (
                    (local_offsets < extend_seq_lens.unsqueeze(1))
                    & (req_ids_2d >= 0)
                    & (token_pos_2d >= 0)
                    & (token_pos_2d < self.max_context_len)
                ).reshape(-1)

            if (
                forward_batch.out_cache_loc is not None
                and int(forward_batch.out_cache_loc.numel()) == num_src_rows
            ):
                valid_mask = valid_mask & (
                    forward_batch.out_cache_loc.to(torch.long) >= 0
                )
            valid_mask = valid_mask.contiguous()

        assert src_tensor.shape[1:] == dst_tensor.shape[2:]
        assert src_index.shape == dst_index.shape == valid_mask.shape

        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                src_tensor,
                dst_tensor,
                src_index,
                dst_index,
                valid_mask,
                1,  # src_tensor: [num_new_tokens, nhead, head_dim]
                2,  # dst_tensor: [num_req, max_context_len, nhead, head_dim]
                block_dim=48,
                dst_ptr=self.dev_ptr_list[layer_idx],
            )

    def get_forward_kv(
        self,
        layer: Union[RadixAttention, int],
        forward_batch: ForwardBatch,
        stream: Optional[torch.npu.Stream] = None,
    ):
        """Gather full request KV from sparse storage as compact TND tensors.

        This helper is intended for the prefill/extend sparse path. It returns
        KV in the same request order as forward_batch.req_pool_indices:
        [req0 tokens][req1 tokens]... . The SFA caller should pair this with
        actual_seq_lengths_kv = cumsum(forward_batch.seq_lens).

        TODO: Add a prefill-resident device KV cache shaped like
        [max_prefill_parallel_reqs, max_prefill_len, nhead, head_dim]. Reuse a
        slot while the same req_id continues chunked prefill, and fully
        overwrite it when a new req_id takes that slot. This mirrors the decode
        device cache idea and avoids repeatedly copying the full prefix KV from
        host for every chunk.
        """
        layer_id = layer.layer_id if hasattr(layer, "layer_id") else int(layer)
        layer_idx = layer_id - self.start_layer
        if layer_idx < 0 or layer_idx >= self.layer_num:
            raise RuntimeError(
                f"Invalid sparse KV layer id {layer_id}; start_layer="
                f"{self.start_layer}, layer_num={self.layer_num}."
            )

        if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
            raise RuntimeError(
                "get_forward_kv requires req_pool_indices and seq_lens in ForwardBatch."
            )

        device = (
            forward_batch.req_pool_indices.device
            if forward_batch.req_pool_indices.device.type == "npu"
            else self.device
        )
        req_ids = forward_batch.req_pool_indices.to(device=device, dtype=torch.long)
        seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)

        if int(req_ids.numel()) != int(seq_lens.numel()):
            raise RuntimeError(
                "get_forward_kv expects req_pool_indices and seq_lens to have the "
                f"same length, got {int(req_ids.numel())} and "
                f"{int(seq_lens.numel())}."
            )

        valid_reqs = (seq_lens > 0) & (req_ids >= 0)
        req_ids = req_ids[valid_reqs].contiguous()
        seq_lens = seq_lens[valid_reqs].contiguous()

        if int(seq_lens.numel()) == 0:
            empty_nope = torch.empty(
                (0, self.head_num, self.kv_lora_rank),
                dtype=self.store_dtype,
                device=device,
            )
            empty_pe = torch.empty(
                (0, self.head_num, self.qk_rope_head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            return empty_nope, empty_pe

        if bool((req_ids >= self.size).any().item()):
            raise RuntimeError(
                f"get_forward_kv got req id outside sparse pool size {self.size}."
            )
        if bool((seq_lens > self.max_context_len).any().item()):
            raise RuntimeError(
                "get_forward_kv got seq_len larger than max_context_len "
                f"{self.max_context_len}."
            )

        total_tokens = int(seq_lens.sum().item())
        kv_cat = torch.empty(
            (total_tokens, self.head_num, self.head_dim),
            dtype=self.store_dtype,
            device=device,
        )

        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        src_req_ids = torch.repeat_interleave(
            req_ids, seq_lens, output_size=total_tokens
        )
        src_seq_starts = torch.repeat_interleave(
            seq_starts, seq_lens, output_size=total_tokens
        )
        token_pos = (
            torch.arange(total_tokens, device=device, dtype=torch.long) - src_seq_starts
        )
        src_index = (src_req_ids * self.max_context_len + token_pos).contiguous()
        dst_index = torch.arange(total_tokens, device=device, dtype=torch.long)
        valid_mask = torch.ones(total_tokens, device=device, dtype=torch.bool)

        actual_stream = stream if stream is not None else torch.npu.current_stream()
        with torch.npu.stream(actual_stream):
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                kv_cat,
                src_index,
                dst_index,
                valid_mask,
                2,  # host_kv_buffer: [num_req, max_context_len, nhead, head_dim]
                1,  # kv_cat: [total_tokens, nhead, head_dim]
                block_dim=48,
                src_ptr=self.dev_ptr_list[layer_idx],
            )

        k_nope, k_pe = kv_cat.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return k_nope.contiguous(), k_pe.contiguous()

    def prefetch_partitions_single_stream(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        stream: torch.npu.Stream,
        dtype: torch.dtype,
    ) -> SparseKVSingleStreamPrefetch:
        """Compact hit/miss KV on one NPU stream for NPUGraph capture.

        This path is intended for NPUGraph capture.  It keeps every tensor
        shape fixed, creates no auxiliary stream or Event, and serializes
        lookup plus D2D/H2D compact copies on the caller's stream.  It does not
        mutate the hot cache or slot map; stateful refill is enabled only after
        its graph behavior is validated independently.
        """

        prefetch_profile_range = _profile_push(
            "sparse_kv_prefetch_partitions_single_stream"
        )
        layer_idx = layer.layer_id - self.start_layer
        stream = stream if stream is not None else torch.npu.current_stream()
        if dtype != self.store_dtype:
            raise RuntimeError(
                "Sparse KV split prefetch requires the query KV dtype to match "
                f"the cache dtype, got dtype={dtype}, cache_dtype={self.store_dtype}."
            )

        with torch.npu.stream(stream):
            req_pool_indices = forward_batch.req_pool_indices
            req_pool_indices = req_pool_indices.to(torch.long).contiguous()
            if req_pool_indices.dim() != 1:
                raise RuntimeError(
                    "Sparse KV split prefetch expects 1-D request indices, "
                    f"got shape={tuple(req_pool_indices.shape)}."
                )
            # ReqToTokenPool reserves row 0 for graph padding.  Route it to
            # the read-only sentinel row instead of treating it as a request.
            valid_req_mask = (req_pool_indices > 0) & (req_pool_indices < self.size)
            slot_map_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.full_like(req_pool_indices, self.size),
            )
            device_cache_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.zeros_like(req_pool_indices),
            )

            topk_indices = _normalize_topk_indices_2d(topk_indices)
            batch_size, topk_len = topk_indices.shape
            if batch_size != req_pool_indices.shape[0]:
                raise RuntimeError(
                    "Sparse KV split prefetch batch mismatch: "
                    f"topk batch={batch_size}, request batch="
                    f"{req_pool_indices.shape[0]}."
                )
            if topk_len != self.sparse_context_len:
                raise RuntimeError(
                    "Sparse KV split prefetch requires the fixed slot-lookup "
                    f"top-k length {self.sparse_context_len}, got {topk_len}."
                )

            valid_topk_mask = (
                (topk_indices >= 0)
                & (topk_indices < self.max_context_len)
                & valid_req_mask.unsqueeze(1)
            )
            if forward_batch.seq_lens is not None:
                valid_topk_mask = valid_topk_mask & (
                    forward_batch.seq_lens[:batch_size] > 0
                ).view(batch_size, 1)

            profile_range = _profile_push(
                "sparse_kv_prefetch_partitions_single_stream.slot_lookup"
            )
            token_on_device, device_token_pos = slot_map_lookup(
                self.device_slot_map[layer_idx],
                slot_map_row_indices.to(dtype=torch.int32).contiguous(),
                topk_indices.to(dtype=torch.int32).contiguous(),
            )
            hit_valid = token_on_device.to(torch.bool) & valid_topk_mask
            miss_valid = (~hit_valid) & valid_topk_mask
            _profile_pop(profile_range)

            hit_counts = hit_valid.sum(dim=1).to(torch.int32).contiguous()
            miss_counts = miss_valid.sum(dim=1).to(torch.int32).contiguous()
            hit_ranks = (
                torch.cumsum(hit_valid.to(torch.int32), dim=1).to(torch.int64) - 1
            )
            miss_ranks = (
                torch.cumsum(miss_valid.to(torch.int32), dim=1).to(torch.int64) - 1
            )

            batch_offsets = (
                torch.arange(batch_size, dtype=torch.int64, device=topk_indices.device)
                .unsqueeze(1)
                .mul(topk_len)
            )
            hit_compact_indices = (
                (batch_offsets + hit_ranks.clamp(min=0, max=topk_len - 1))
                .reshape(-1)
                .contiguous()
            )
            miss_compact_indices = (
                (batch_offsets + miss_ranks.clamp(min=0, max=topk_len - 1))
                .reshape(-1)
                .contiguous()
            )

            request_device_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.sparse_context_len
            )
            hit_src_indices = (
                (
                    request_device_offsets
                    + device_token_pos.to(torch.int64).clamp(
                        min=0, max=self.sparse_context_len - 1
                    )
                )
                .reshape(-1)
                .contiguous()
            )
            request_host_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.max_context_len
            )
            miss_src_indices = (
                (
                    request_host_offsets
                    + topk_indices.to(torch.int64).clamp(
                        min=0, max=self.max_context_len - 1
                    )
                )
                .reshape(-1)
                .contiguous()
            )
            hit_valid_flat = hit_valid.reshape(-1).contiguous()
            miss_valid_flat = miss_valid.reshape(-1).contiguous()

            partition_shape = (
                batch_size,
                topk_len,
                self.head_num,
                self.head_dim,
            )
            hit_kv = torch.empty(partition_shape, dtype=dtype, device=self.device)
            miss_kv = torch.empty(partition_shape, dtype=dtype, device=self.device)
            hit_kv[:, :1].zero_()
            miss_kv[:, :1].zero_()
            hit_sparse_indices, hit_actual_lengths = _build_partition_sparse_indices(
                hit_counts, topk_len
            )
            miss_sparse_indices, miss_actual_lengths = _build_partition_sparse_indices(
                miss_counts, topk_len
            )

            profile_range = _profile_push(
                "sparse_kv_prefetch_partitions_single_stream.d2d_hit_copy"
            )
            unidex_copy_inplace(
                self.device_kv_buffer[layer_idx],
                hit_kv,
                hit_src_indices,
                hit_compact_indices,
                hit_valid_flat,
                2,
                2,
                block_dim=24,
            )
            _profile_pop(profile_range)

            profile_range = _profile_push(
                "sparse_kv_prefetch_partitions_single_stream.h2d_miss_copy"
            )
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                miss_kv,
                miss_src_indices,
                miss_compact_indices,
                miss_valid_flat,
                2,
                2,
                block_dim=24,
                src_ptr=self.dev_ptr_list[layer_idx],
            )
            _profile_pop(profile_range)

        _profile_pop(prefetch_profile_range)
        return SparseKVSingleStreamPrefetch(
            hit=SparseKVPartition(
                kv=hit_kv,
                sparse_indices=hit_sparse_indices,
                actual_seq_lengths_kv=hit_actual_lengths,
                true_counts=hit_counts,
                stream=stream,
            ),
            miss=SparseKVPartition(
                kv=miss_kv,
                sparse_indices=miss_sparse_indices,
                actual_seq_lengths_kv=miss_actual_lengths,
                true_counts=miss_counts,
                stream=stream,
            ),
        )

    def prefetch_partitions_graph_dual(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        stream: torch.npu.Stream,
        dtype: torch.dtype,
    ) -> SparseKVGraphDualPrefetch:
        """Fork graph-safe hit and miss partitions onto two persistent streams.

        The caller/main graph stream owns lookup, metadata, the device-cache hit
        copy, and hit attention.  ``miss_stream`` waits for ``inputs_ready`` and
        performs the registered-host miss copy followed by miss attention.  A
        fixed workspace is shared across layers and therefore must be committed
        before this layer returns.
        """

        state = self._graph_dual_state
        if state is None:
            raise RuntimeError(
                "Sparse KV graph-dual state is not initialized. Call "
                "prepare_graph_dual_state() before graph capture."
            )

        layer_idx = layer.layer_id - self.start_layer
        if layer_idx < 0 or layer_idx >= self.layer_num:
            raise RuntimeError(
                f"Invalid sparse KV layer id {layer.layer_id}; start_layer="
                f"{self.start_layer}, layer_num={self.layer_num}."
            )
        if dtype != self.store_dtype:
            raise RuntimeError(
                "Sparse KV graph-dual prefetch requires the query KV dtype to "
                f"match the cache dtype, got dtype={dtype}, "
                f"cache_dtype={self.store_dtype}."
            )

        stream = stream if stream is not None else torch.npu.current_stream()
        events = state.layer_events[layer_idx]
        profile_range = _profile_push("sparse_kv_prefetch_partitions_graph_dual")

        with torch.npu.stream(stream):
            req_pool_indices = forward_batch.req_pool_indices.to(
                torch.long
            ).contiguous()
            if req_pool_indices.dim() != 1:
                raise RuntimeError(
                    "Sparse KV graph-dual prefetch expects 1-D request indices, "
                    f"got shape={tuple(req_pool_indices.shape)}."
                )

            valid_req_mask = (req_pool_indices > 0) & (req_pool_indices < self.size)
            if forward_batch.seq_lens is not None:
                active_req_mask = valid_req_mask & (
                    forward_batch.seq_lens[: req_pool_indices.shape[0]] > 0
                )
            else:
                active_req_mask = valid_req_mask

            slot_map_row_indices = torch.where(
                active_req_mask,
                req_pool_indices,
                torch.full_like(req_pool_indices, self.size),
            ).contiguous()
            device_cache_row_indices = torch.where(
                active_req_mask, req_pool_indices, torch.zeros_like(req_pool_indices),
            ).contiguous()

            topk_indices = _normalize_topk_indices_2d(topk_indices)
            batch_size, topk_len = topk_indices.shape
            if batch_size != req_pool_indices.shape[0]:
                raise RuntimeError(
                    "Sparse KV graph-dual batch mismatch: "
                    f"topk batch={batch_size}, request batch="
                    f"{req_pool_indices.shape[0]}."
                )
            if batch_size > state.max_batch_size:
                raise RuntimeError(
                    "Sparse KV graph-dual batch exceeds its fixed workspace: "
                    f"batch={batch_size}, max_batch={state.max_batch_size}."
                )
            if topk_len != self.sparse_context_len:
                raise RuntimeError(
                    "Sparse KV graph-dual prefetch requires the fixed slot-lookup "
                    f"top-k length {self.sparse_context_len}, got {topk_len}."
                )

            valid_topk_mask = (
                (topk_indices >= 0)
                & (topk_indices < self.max_context_len)
                & active_req_mask.unsqueeze(1)
            )

            lookup_range = _profile_push(
                "sparse_kv_prefetch_partitions_graph_dual.slot_lookup"
            )
            token_on_device, device_token_pos = slot_map_lookup(
                self.device_slot_map[layer_idx],
                slot_map_row_indices.to(dtype=torch.int32).contiguous(),
                topk_indices.to(dtype=torch.int32).contiguous(),
            )
            hit_valid = token_on_device.to(torch.bool) & valid_topk_mask
            miss_valid = (~hit_valid) & valid_topk_mask
            _profile_pop(lookup_range)

            hit_counts = hit_valid.sum(dim=1).to(torch.int32).contiguous()
            miss_counts = miss_valid.sum(dim=1).to(torch.int32).contiguous()
            hit_ranks = (
                torch.cumsum(hit_valid.to(torch.int32), dim=1).to(torch.int64) - 1
            )
            miss_ranks = (
                torch.cumsum(miss_valid.to(torch.int32), dim=1).to(torch.int64) - 1
            )

            batch_offsets = (
                torch.arange(batch_size, dtype=torch.int64, device=topk_indices.device)
                .unsqueeze(1)
                .mul(topk_len)
            )
            hit_compact_indices = (
                (batch_offsets + hit_ranks.clamp(min=0, max=topk_len - 1))
                .reshape(-1)
                .contiguous()
            )
            miss_compact_indices = (
                (batch_offsets + miss_ranks.clamp(min=0, max=topk_len - 1))
                .reshape(-1)
                .contiguous()
            )

            request_device_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.sparse_context_len
            )
            hit_src_indices = (
                (
                    request_device_offsets
                    + device_token_pos.to(torch.int64).clamp(
                        min=0, max=self.sparse_context_len - 1
                    )
                )
                .reshape(-1)
                .contiguous()
            )
            request_host_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.max_context_len
            )
            miss_src_indices = (
                (
                    request_host_offsets
                    + topk_indices.to(torch.int64).clamp(
                        min=0, max=self.max_context_len - 1
                    )
                )
                .reshape(-1)
                .contiguous()
            )

            hit_valid_flat = hit_valid.reshape(-1).contiguous()
            miss_valid_flat = miss_valid.reshape(-1).contiguous()
            hit_kv = state.hit_kv[:batch_size]
            miss_kv = state.miss_kv[:batch_size]
            hit_kv[:, :1].zero_()
            miss_kv[:, :1].zero_()
            hit_sparse_indices, hit_actual_lengths = _build_partition_sparse_indices(
                hit_counts, topk_len
            )
            miss_sparse_indices, miss_actual_lengths = _build_partition_sparse_indices(
                miss_counts, topk_len
            )

            cache_slot_ids = self._device_cache_slot_ids[:topk_len].view(1, -1)
            refill_dst_indices = (
                (request_device_offsets + cache_slot_ids).reshape(-1).contiguous()
            )
            slot_map_token_indices = torch.where(
                valid_topk_mask,
                topk_indices.to(torch.long),
                torch.full_like(topk_indices, self.max_context_len).to(torch.long),
            )
            slot_map_flat_indices = (
                (
                    slot_map_row_indices.unsqueeze(1) * self._slot_map_width
                    + slot_map_token_indices
                )
                .reshape(-1)
                .contiguous()
            )
            slot_map_slot_values = torch.where(
                valid_topk_mask,
                cache_slot_ids.to(torch.int32).expand(batch_size, -1),
                torch.full_like(topk_indices, -1, dtype=torch.int32),
            ).contiguous()

            _record_stream_event(stream, events.inputs_ready)

        miss_copy_range = _profile_push(
            "sparse_kv_prefetch_partitions_graph_dual.h2d_miss_copy"
        )
        with torch.npu.stream(state.miss_stream):
            _wait_stream_event(state.miss_stream, events.inputs_ready)
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                miss_kv,
                miss_src_indices,
                miss_compact_indices,
                miss_valid_flat,
                2,
                2,
                block_dim=24,
                src_ptr=self.dev_ptr_list[layer_idx],
            )
        _profile_pop(miss_copy_range)

        hit_copy_range = _profile_push(
            "sparse_kv_prefetch_partitions_graph_dual.d2d_hit_copy"
        )
        with torch.npu.stream(stream):
            unidex_copy_inplace(
                self.device_kv_buffer[layer_idx],
                hit_kv,
                hit_src_indices,
                hit_compact_indices,
                hit_valid_flat,
                2,
                2,
                block_dim=24,
            )
            _record_stream_event(stream, events.hit_copy_done)
        _profile_pop(hit_copy_range)
        _profile_pop(profile_range)

        return SparseKVGraphDualPrefetch(
            hit=SparseKVPartition(
                kv=hit_kv,
                sparse_indices=hit_sparse_indices,
                actual_seq_lengths_kv=hit_actual_lengths,
                true_counts=hit_counts,
                stream=stream,
            ),
            miss=SparseKVPartition(
                kv=miss_kv,
                sparse_indices=miss_sparse_indices,
                actual_seq_lengths_kv=miss_actual_lengths,
                true_counts=miss_counts,
                stream=state.miss_stream,
            ),
            layer_idx=layer_idx,
            hit_compact_indices=hit_compact_indices,
            miss_compact_indices=miss_compact_indices,
            refill_dst_indices=refill_dst_indices,
            hit_valid_flat=hit_valid_flat,
            miss_valid_flat=miss_valid_flat,
            slot_map_row_indices=slot_map_row_indices,
            slot_map_flat_indices=slot_map_flat_indices,
            slot_map_slot_values=slot_map_slot_values,
            events=events,
        )

    def prefetch_partitions_graph_dual_v2(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        stream: torch.npu.Stream,
        dtype: torch.dtype,
    ) -> SparseKVGraphDualV2Prefetch:
        """Prepare noncompact hit/miss SFA inputs with two KV copy kernels.

        Hit rows keep their existing hot-cache slots. Miss rows are copied once
        from registered Host memory into both the shared SFA buffers and free
        hot-cache slots. Only the sparse-index lists are compacted; the KV rows
        remain at their original Top-K positions in the shared buffers. A
        three-kernel planner classifies 64-entry tiles in parallel, scans the
        32 tile counts per request, and scatters stable compact prefixes before
        writing the fixed-address copy and publication descriptors.
        """

        state = self._graph_dual_v2_state
        if state is None:
            raise RuntimeError(
                "Sparse KV graph-dual-v2 state is not initialized. Call "
                "prepare_graph_dual_v2_state() before graph capture."
            )

        layer_idx = layer.layer_id - self.start_layer
        if layer_idx < 0 or layer_idx >= self.layer_num:
            raise RuntimeError(
                f"Invalid sparse KV layer id {layer.layer_id}; start_layer="
                f"{self.start_layer}, layer_num={self.layer_num}."
            )
        if dtype != self.store_dtype:
            raise RuntimeError(
                "Sparse KV graph-dual-v2 prefetch requires the query KV dtype "
                f"to match the cache dtype, got dtype={dtype}, "
                f"cache_dtype={self.store_dtype}."
            )

        stream = stream if stream is not None else torch.npu.current_stream()
        events = state.layer_events[layer_idx]
        profile_range = _profile_push("sparse_kv_prefetch_graph_dual_v2")

        with torch.npu.stream(stream):
            req_pool_indices = forward_batch.req_pool_indices.to(
                torch.long
            ).contiguous()
            if req_pool_indices.dim() != 1:
                raise RuntimeError(
                    "Sparse KV graph-dual-v2 prefetch expects 1-D request "
                    f"indices, got shape={tuple(req_pool_indices.shape)}."
                )

            valid_req_mask = (req_pool_indices > 0) & (req_pool_indices < self.size)
            if forward_batch.seq_lens is not None:
                active_req_mask = valid_req_mask & (
                    forward_batch.seq_lens[: req_pool_indices.shape[0]] > 0
                )
            else:
                active_req_mask = valid_req_mask

            slot_map_row_indices = torch.where(
                active_req_mask,
                req_pool_indices,
                torch.full_like(req_pool_indices, self.size),
            ).contiguous()
            device_cache_row_indices = torch.where(
                active_req_mask,
                req_pool_indices,
                torch.zeros_like(req_pool_indices),
            ).contiguous()

            topk_indices = _normalize_topk_indices_2d(topk_indices)
            batch_size, topk_len = topk_indices.shape
            if batch_size != req_pool_indices.shape[0]:
                raise RuntimeError(
                    "Sparse KV graph-dual-v2 batch mismatch: "
                    f"topk batch={batch_size}, request batch="
                    f"{req_pool_indices.shape[0]}."
                )
            if batch_size > state.max_batch_size:
                raise RuntimeError(
                    "Sparse KV graph-dual-v2 batch exceeds its fixed workspace: "
                    f"batch={batch_size}, max_batch={state.max_batch_size}."
                )
            if topk_len != self.sparse_context_len:
                raise RuntimeError(
                    "Sparse KV graph-dual-v2 prefetch requires the fixed "
                    f"slot-lookup top-k length {self.sparse_context_len}, "
                    f"got {topk_len}."
                )

            valid_topk_mask = (
                (topk_indices >= 0)
                & (topk_indices < self.max_context_len)
                & active_req_mask.unsqueeze(1)
            )

            lookup_range = _profile_push(
                "sparse_kv_prefetch_graph_dual_v2.slot_lookup"
            )
            topk_indices_int32 = topk_indices.to(dtype=torch.int32).contiguous()
            token_on_device, device_token_pos = slot_map_lookup(
                self.device_slot_map[layer_idx],
                slot_map_row_indices.to(dtype=torch.int32).contiguous(),
                topk_indices_int32,
            )
            _profile_pop(lookup_range)

            plan_size = batch_size * topk_len
            hit_sparse_indices_2d = state.hit_sparse_indices[:batch_size]
            miss_sparse_indices_2d = state.miss_sparse_indices[:batch_size]
            hit_counts = state.hit_counts[:batch_size]
            miss_counts = state.miss_counts[:batch_size]
            hit_src_indices = state.hit_src_indices[:plan_size]
            miss_src_indices = state.miss_src_indices[:plan_size]
            miss_hot_dst_indices = state.miss_hot_dst_indices[:plan_size]
            hit_valid_flat = state.hit_valid_mask[:plan_size]
            miss_valid_flat = state.miss_valid_mask[:plan_size]
            slot_map_flat_indices = state.slot_map_flat_indices[:plan_size]
            slot_map_slot_values = state.slot_map_slot_values[:batch_size]

            plan_range = _profile_push(
                "sparse_kv_prefetch_graph_dual_v2.partition_plan_parallel"
            )
            sparse_kv_ops.sparse_kv_partition_plan_parallel_inplace(
                token_on_device,
                device_token_pos,
                topk_indices_int32,
                device_cache_row_indices,
                slot_map_row_indices,
                valid_topk_mask.contiguous(),
                hit_sparse_indices_2d,
                miss_sparse_indices_2d,
                hit_counts,
                miss_counts,
                hit_src_indices,
                miss_src_indices,
                miss_hot_dst_indices,
                hit_valid_flat,
                miss_valid_flat,
                slot_map_flat_indices,
                slot_map_slot_values,
                state.tile_hit_counts[:batch_size],
                state.tile_miss_counts[:batch_size],
                state.tile_hit_offsets[:batch_size],
                state.tile_miss_offsets[:batch_size],
                state.selected_slots[:batch_size],
                state.tile_occupied_bitmaps[:batch_size],
                state.occupied_bitmaps[:batch_size],
                state.free_slot_prefixes[:batch_size],
                self.max_context_len,
                self._slot_map_width,
            )
            _profile_pop(plan_range)

            hit_sparse_indices = hit_sparse_indices_2d.view(
                batch_size, 1, 1, topk_len
            )
            miss_sparse_indices = miss_sparse_indices_2d.view(
                batch_size, 1, 1, topk_len
            )
            hit_actual_lengths = state.actual_seq_lengths_kv[:batch_size]
            miss_actual_lengths = hit_actual_lengths
            attention_dst_indices = state.attention_dst_indices[:plan_size]

            selected_k_nope = state.selected_k_nope[:batch_size]
            selected_k_rope = state.selected_k_rope[:batch_size]
            _record_stream_event(stream, events.inputs_ready)

        miss_copy_range = _profile_push(
            "sparse_kv_prefetch_graph_dual_v2.h2d_miss_copy_promote"
        )
        with torch.npu.stream(state.miss_stream):
            _wait_stream_event(state.miss_stream, events.inputs_ready)
            sparse_kv_ops.unidex_split_copy_promote_inplace(
                self.host_kv_buffer[layer_idx],
                selected_k_nope,
                selected_k_rope,
                self.device_kv_buffer[layer_idx],
                miss_src_indices,
                attention_dst_indices,
                miss_hot_dst_indices,
                miss_valid_flat,
                2,
                2,
                2,
                block_dim=24,
                src_ptr=self.dev_ptr_list[layer_idx],
            )
            _record_stream_event(state.miss_stream, events.miss_copy_done)
        _profile_pop(miss_copy_range)

        hit_copy_range = _profile_push(
            "sparse_kv_prefetch_graph_dual_v2.d2d_hit_split_copy"
        )
        with torch.npu.stream(stream):
            sparse_kv_ops.unidex_split_copy_inplace(
                self.device_kv_buffer[layer_idx],
                selected_k_nope,
                selected_k_rope,
                hit_src_indices,
                attention_dst_indices,
                hit_valid_flat,
                2,
                2,
                block_dim=24,
            )
        _profile_pop(hit_copy_range)
        _profile_pop(profile_range)

        return SparseKVGraphDualV2Prefetch(
            hit=SparseKVIndexedPartition(
                key=selected_k_nope,
                key_rope=selected_k_rope,
                sparse_indices=hit_sparse_indices,
                actual_seq_lengths_kv=hit_actual_lengths,
                true_counts=hit_counts,
                stream=stream,
            ),
            miss=SparseKVIndexedPartition(
                key=selected_k_nope,
                key_rope=selected_k_rope,
                sparse_indices=miss_sparse_indices,
                actual_seq_lengths_kv=miss_actual_lengths,
                true_counts=miss_counts,
                stream=state.miss_stream,
            ),
            layer_idx=layer_idx,
            slot_map_row_indices=slot_map_row_indices,
            slot_map_flat_indices=slot_map_flat_indices,
            slot_map_slot_values=slot_map_slot_values,
            events=events,
        )

    def publish_graph_dual_v2_slot_map(
        self, ticket: SparseKVGraphDualV2Prefetch, stream: torch.npu.Stream
    ) -> None:
        """Publish the new residency map after all miss KV rows are resident."""

        profile_range = _profile_push("sparse_kv_graph_dual_v2.slot_map_publish")
        with torch.npu.stream(stream):
            _wait_stream_event(stream, ticket.events.miss_copy_done)
            slot_map = self.device_slot_map[ticket.layer_idx]
            slot_map.index_fill_(0, ticket.slot_map_row_indices, -1)
            slot_map.view(-1).scatter_(
                0,
                ticket.slot_map_flat_indices,
                ticket.slot_map_slot_values.reshape(-1),
            )
        _profile_pop(profile_range)

    def commit_graph_dual_refill(self, ticket: SparseKVGraphDualPrefetch) -> None:
        """Refill the hot cache and publish its slot map on the miss stream."""

        miss_stream = ticket.miss.stream
        profile_range = _profile_push("sparse_kv_graph_dual.device_refill")
        with torch.npu.stream(miss_stream):
            _wait_stream_event(miss_stream, ticket.events.hit_copy_done)
            unidex_copy_inplace(
                ticket.hit.kv,
                self.device_kv_buffer[ticket.layer_idx],
                ticket.hit_compact_indices,
                ticket.refill_dst_indices,
                ticket.hit_valid_flat,
                2,
                2,
                block_dim=24,
            )
            unidex_copy_inplace(
                ticket.miss.kv,
                self.device_kv_buffer[ticket.layer_idx],
                ticket.miss_compact_indices,
                ticket.refill_dst_indices,
                ticket.miss_valid_flat,
                2,
                2,
                block_dim=24,
            )

            slot_map = self.device_slot_map[ticket.layer_idx]
            slot_map.index_fill_(0, ticket.slot_map_row_indices, -1)
            slot_map.view(-1).scatter_(
                0,
                ticket.slot_map_flat_indices,
                ticket.slot_map_slot_values.reshape(-1),
            )
            _record_stream_event(miss_stream, ticket.events.refill_done)
        _profile_pop(profile_range)

    def prefetch_partitions(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        stream: torch.npu.Stream,
        dtype: torch.dtype,
    ) -> SparseKVPrefetchTicket:
        """Prefetch compact hit/miss KV partitions without blocking attention.

        Device-cache hits and host-cache misses are compacted into independent
        fixed-capacity buffers.  The returned streams already contain the copy
        operations, so callers can append the corresponding attention call to
        each stream.  Refill and slot-map maintenance run concurrently on the
        refill stream and are represented by ``refill_done``.
        """

        prefetch_profile_range = _profile_push("sparse_kv_prefetch_partitions")
        layer_idx = layer.layer_id - self.start_layer
        stream = stream if stream is not None else torch.npu.current_stream()
        if dtype != self.store_dtype:
            raise RuntimeError(
                "Sparse KV split prefetch requires the query KV dtype to match "
                f"the cache dtype, got dtype={dtype}, cache_dtype={self.store_dtype}."
            )

        with torch.npu.stream(stream):
            req_pool_indices = forward_batch.req_pool_indices
            req_pool_indices = req_pool_indices.to(torch.long).contiguous()
            if req_pool_indices.dim() != 1:
                raise RuntimeError(
                    "Sparse KV split prefetch expects 1-D request indices, "
                    f"got shape={tuple(req_pool_indices.shape)}."
                )
            valid_req_mask = (req_pool_indices >= 0) & (req_pool_indices < self.size)
            slot_map_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.full_like(req_pool_indices, self.size),
            )
            device_cache_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.zeros_like(req_pool_indices),
            )

            topk_indices = _normalize_topk_indices_2d(topk_indices)
            batch_size, topk_len = topk_indices.shape
            if batch_size != req_pool_indices.shape[0]:
                raise RuntimeError(
                    "Sparse KV split prefetch batch mismatch: "
                    f"topk batch={batch_size}, request batch="
                    f"{req_pool_indices.shape[0]}."
                )
            if topk_len != self.sparse_context_len:
                raise RuntimeError(
                    "Sparse KV split prefetch requires the fixed slot-lookup "
                    f"top-k length {self.sparse_context_len}, got {topk_len}."
                )

            valid_topk_mask = (
                (topk_indices >= 0)
                & (topk_indices < self.max_context_len)
                & valid_req_mask.unsqueeze(1)
            )
            if forward_batch.seq_lens is not None:
                valid_topk_mask = valid_topk_mask & (
                    forward_batch.seq_lens[:batch_size] > 0
                ).view(batch_size, 1)

            profile_range = _profile_push("sparse_kv_prefetch_partitions.slot_lookup")
            token_on_device, device_token_pos = slot_map_lookup(
                self.device_slot_map[layer_idx],
                slot_map_row_indices.to(dtype=torch.int32).contiguous(),
                topk_indices.to(dtype=torch.int32).contiguous(),
            )
            hit_valid = token_on_device.to(torch.bool) & valid_topk_mask
            miss_valid = (~hit_valid) & valid_topk_mask
            _profile_pop(profile_range)

            hit_counts = hit_valid.sum(dim=1).to(torch.int32).contiguous()
            miss_counts = miss_valid.sum(dim=1).to(torch.int32).contiguous()
            hit_ranks = (
                torch.cumsum(hit_valid.to(torch.int32), dim=1).to(torch.int64) - 1
            )
            miss_ranks = (
                torch.cumsum(miss_valid.to(torch.int32), dim=1).to(torch.int64) - 1
            )

            batch_offsets = (
                torch.arange(batch_size, dtype=torch.int64, device=topk_indices.device)
                .unsqueeze(1)
                .mul(topk_len)
            )
            hit_compact_indices = (
                (batch_offsets + hit_ranks.clamp(min=0, max=topk_len - 1))
                .reshape(-1)
                .contiguous()
            )
            miss_compact_indices = (
                (batch_offsets + miss_ranks.clamp(min=0, max=topk_len - 1))
                .reshape(-1)
                .contiguous()
            )

            request_device_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.sparse_context_len
            )
            hit_src_indices = (
                (
                    request_device_offsets
                    + device_token_pos.to(torch.int64).clamp(
                        min=0, max=self.sparse_context_len - 1
                    )
                )
                .reshape(-1)
                .contiguous()
            )

            request_host_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.max_context_len
            )
            miss_src_indices = (
                (
                    request_host_offsets
                    + topk_indices.to(torch.int64).clamp(
                        min=0, max=self.max_context_len - 1
                    )
                )
                .reshape(-1)
                .contiguous()
            )

            hit_valid_flat = hit_valid.reshape(-1).contiguous()
            miss_valid_flat = miss_valid.reshape(-1).contiguous()

            partition_shape = (
                batch_size,
                topk_len,
                self.head_num,
                self.head_dim,
            )
            hit_kv = torch.empty(partition_shape, dtype=dtype, device=self.device)
            miss_kv = torch.empty(partition_shape, dtype=dtype, device=self.device)
            # Empty rows use slot zero as a dummy.  Valid rows overwrite it with
            # their first compacted token after copy_ready is recorded.
            hit_kv[:, :1].zero_()
            miss_kv[:, :1].zero_()

            hit_sparse_indices, hit_actual_lengths = _build_partition_sparse_indices(
                hit_counts, topk_len
            )
            miss_sparse_indices, miss_actual_lengths = _build_partition_sparse_indices(
                miss_counts, topk_len
            )

            topk_positions = torch.arange(
                topk_len, dtype=torch.int64, device=topk_indices.device
            ).unsqueeze(0)
            refill_dst_indices = (
                (request_device_offsets + topk_positions).reshape(-1).contiguous()
            )

            cache_slot_ids = self._device_cache_slot_ids[:topk_len]
            slot_map_token_indices = torch.where(
                valid_topk_mask,
                topk_indices.to(torch.long),
                torch.full_like(topk_indices, self.max_context_len, dtype=torch.long),
            )
            slot_map_slot_values = torch.where(
                valid_topk_mask,
                cache_slot_ids.to(torch.int32),
                torch.full_like(cache_slot_ids, -1, dtype=torch.int32),
            )
            slot_map_flat_indices = (
                slot_map_row_indices.unsqueeze(1) * self._slot_map_width
                + slot_map_token_indices
            ).reshape(-1)

            copy_ready = torch.npu.Event()
            hit_copy_done = torch.npu.Event()
            miss_copy_done = torch.npu.Event()
            refill_done = torch.npu.Event()
            _record_stream_event(stream, copy_ready)

        profile_range = _profile_push("sparse_kv_prefetch_partitions.d2d_hit_copy")
        with torch.npu.stream(self._prefetch_d2d_hit_stream):
            _wait_stream_event(self._prefetch_d2d_hit_stream, copy_ready)
            unidex_copy_inplace(
                self.device_kv_buffer[layer_idx],
                hit_kv,
                hit_src_indices,
                hit_compact_indices,
                hit_valid_flat,
                2,
                2,
                block_dim=24,
            )
            _record_stream_event(self._prefetch_d2d_hit_stream, hit_copy_done)
        _profile_pop(profile_range)

        profile_range = _profile_push("sparse_kv_prefetch_partitions.h2d_miss_copy")
        with torch.npu.stream(self._prefetch_h2d_miss_stream):
            _wait_stream_event(self._prefetch_h2d_miss_stream, copy_ready)
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                miss_kv,
                miss_src_indices,
                miss_compact_indices,
                miss_valid_flat,
                2,
                2,
                block_dim=24,
                src_ptr=self.dev_ptr_list[layer_idx],
            )
            _record_stream_event(self._prefetch_h2d_miss_stream, miss_copy_done)
        _profile_pop(profile_range)

        # Refill both compact partitions into their original top-k cache slots.
        # This is independent of the read-only attention calls appended to the
        # hit and miss streams by the caller.
        profile_range = _profile_push("sparse_kv_prefetch_partitions.device_refill")
        with torch.npu.stream(self._prefetch_refill_stream):
            _wait_stream_event(self._prefetch_refill_stream, hit_copy_done)
            _wait_stream_event(self._prefetch_refill_stream, miss_copy_done)
            unidex_copy_inplace(
                hit_kv,
                self.device_kv_buffer[layer_idx],
                hit_compact_indices,
                refill_dst_indices,
                hit_valid_flat,
                2,
                2,
                block_dim=24,
            )
            unidex_copy_inplace(
                miss_kv,
                self.device_kv_buffer[layer_idx],
                miss_compact_indices,
                refill_dst_indices,
                miss_valid_flat,
                2,
                2,
                block_dim=24,
            )

            self.device_slot_map[layer_idx].index_fill_(0, slot_map_row_indices, -1)
            self.device_slot_map[layer_idx].view(-1).scatter_(
                0, slot_map_flat_indices, slot_map_slot_values.reshape(-1)
            )
            _record_stream_event(self._prefetch_refill_stream, refill_done)
        _profile_pop(profile_range)
        _profile_pop(prefetch_profile_range)

        return SparseKVPrefetchTicket(
            hit=SparseKVPartition(
                kv=hit_kv,
                sparse_indices=hit_sparse_indices,
                actual_seq_lengths_kv=hit_actual_lengths,
                true_counts=hit_counts,
                stream=self._prefetch_d2d_hit_stream,
            ),
            miss=SparseKVPartition(
                kv=miss_kv,
                sparse_indices=miss_sparse_indices,
                actual_seq_lengths_kv=miss_actual_lengths,
                true_counts=miss_counts,
                stream=self._prefetch_h2d_miss_stream,
            ),
            refill_done=refill_done,
        )

    def prefetch(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        current_kv_buffer: torch.Tensor,
        stream: torch.npu.Stream,
    ):
        prefetch_profile_range = _profile_push("sparse_kv_prefetch")
        layer_idx = layer.layer_id - self.start_layer
        stream = stream if stream is not None else torch.npu.current_stream()

        with torch.npu.stream(stream):
            # Route invalid requests to sentinel rows without changing graph shape.
            # slot_map_row_indices: invalid -> self.size (reserved slot-map row)
            # device_cache_row_indices: invalid -> 0 (masked by valid_topk_mask)
            req_pool_indices = forward_batch.req_pool_indices
            req_pool_indices = req_pool_indices.to(torch.long).contiguous()
            valid_req_mask = (req_pool_indices >= 0) & (req_pool_indices < self.size)
            slot_map_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.full_like(req_pool_indices, self.size),
            )
            device_cache_row_indices = torch.where(
                valid_req_mask,
                req_pool_indices,
                torch.zeros_like(req_pool_indices),
            )

            # Normalize top-k indices and mask invalid requests and token IDs.
            topk_indices = topk_indices.squeeze(1)
            batch_size, topk_len = topk_indices.shape
            valid_topk_mask = (
                (topk_indices >= 0)
                & (topk_indices < self.max_context_len)
                & valid_req_mask.unsqueeze(1)
            )

            # Query the slot map for device-cache hits and their slot positions.
            profile_range = _profile_push("sparse_kv_prefetch.slot_lookup")
            slot_lookup_req_indices = slot_map_row_indices.to(
                dtype=torch.int32
            ).contiguous()
            slot_lookup_topk_indices = topk_indices.to(dtype=torch.int32).contiguous()
            token_on_device, device_token_pos = slot_map_lookup(
                self.device_slot_map[layer_idx],
                slot_lookup_req_indices,
                slot_lookup_topk_indices,
            )
            token_on_device = token_on_device.to(torch.bool) & valid_topk_mask
            _profile_pop(profile_range)

            # Build copy indices on the main stream, then protect their use on
            # the hit and miss streams with copy_ready.
            hit_src_index, hit_dst_index, hit_valid_mask = _build_hit_src_dst_index(
                token_on_device,
                device_token_pos,
                device_cache_row_indices,
                self.sparse_context_len,
            )

            host_miss_mask = (~token_on_device) & valid_topk_mask
            miss_src_index, miss_dst_index, miss_valid_mask = _build_miss_src_dst_index(
                host_miss_mask,
                topk_indices,
                device_cache_row_indices,
                self.max_context_len,
            )

            cache_slot_ids = self._device_cache_slot_ids[:topk_len]
            request_cache_offsets = (
                device_cache_row_indices.unsqueeze(1) * self.sparse_context_len
            )
            refill_src_index = torch.arange(
                batch_size * topk_len,
                dtype=torch.long,
                device=topk_indices.device,
            )
            refill_dst_index = (
                (request_cache_offsets + cache_slot_ids).reshape(-1).contiguous()
            )
            refill_valid_mask = valid_topk_mask.reshape(-1).contiguous()

            copy_ready = torch.npu.Event()
            hit_done = torch.npu.Event()
            miss_done = torch.npu.Event()
            refill_done = torch.npu.Event()
            _record_stream_event(stream, copy_ready)

        # Copy device-cache hits into the current KV buffer.
        profile_range = _profile_push("sparse_kv_prefetch.d2d_hit_copy")
        with torch.npu.stream(self._prefetch_d2d_hit_stream):
            _wait_stream_event(self._prefetch_d2d_hit_stream, copy_ready)
            unidex_copy_inplace(
                self.device_kv_buffer[layer_idx],
                current_kv_buffer,
                hit_src_index,
                hit_dst_index,
                hit_valid_mask,
                2,
                2,  #
                block_dim=24,
            )
            _record_stream_event(self._prefetch_d2d_hit_stream, hit_done)
        _profile_pop(profile_range)

        # Copy host shared-memory misses into the current KV buffer.
        profile_range = _profile_push("sparse_kv_prefetch.h2d_miss_copy")
        with torch.npu.stream(self._prefetch_h2d_miss_stream):
            _wait_stream_event(self._prefetch_h2d_miss_stream, copy_ready)
            unidex_copy_inplace(
                self.host_kv_buffer[layer_idx],
                current_kv_buffer,
                miss_src_index,
                miss_dst_index,
                miss_valid_mask,
                2,
                2,
                block_dim=24,
                src_ptr=self.dev_ptr_list[layer_idx],
            )
            _record_stream_event(self._prefetch_h2d_miss_stream, miss_done)
        _profile_pop(profile_range)

        # Refill the device cache with the current top-k after hit and miss
        # copies complete, so the next step can reuse these entries.
        profile_range = _profile_push("sparse_kv_prefetch.device_refill")
        with torch.npu.stream(self._prefetch_refill_stream):
            _wait_stream_event(self._prefetch_refill_stream, hit_done)
            _wait_stream_event(self._prefetch_refill_stream, miss_done)
            unidex_copy_inplace(
                current_kv_buffer,
                self.device_kv_buffer[layer_idx],
                refill_src_index,
                refill_dst_index,
                refill_valid_mask,
                2,
                2,
                block_dim=48,
            )
            _record_stream_event(self._prefetch_refill_stream, refill_done)
        _profile_pop(profile_range)

        _wait_stream_event(stream, refill_done)

        # Replace the slot-map row with the current top-k mapping. Invalid
        # entries use the max_context_len sentinel column to preserve shape.
        profile_range = _profile_push("sparse_kv_prefetch.metadata_update")
        with torch.npu.stream(stream):
            self.device_slot_map[layer_idx].index_fill_(0, slot_map_row_indices, -1)

            slot_map_token_indices = torch.where(
                valid_topk_mask,
                topk_indices.to(torch.long),
                torch.full_like(topk_indices, self.max_context_len, dtype=torch.long),
            )
            slot_map_slot_values = torch.where(
                valid_topk_mask,
                cache_slot_ids.to(torch.int32),
                torch.full_like(cache_slot_ids, -1, dtype=torch.int32),
            )
            slot_map_flat_indices = (
                slot_map_row_indices.unsqueeze(1) * self._slot_map_width
                + slot_map_token_indices
            ).reshape(-1)
            self.device_slot_map[layer_idx].view(-1).scatter_(
                0, slot_map_flat_indices, slot_map_slot_values.reshape(-1)
            )
        _profile_pop(profile_range)
        _profile_pop(prefetch_profile_range)


_global_sparse_kv_manager: Optional[SparseKVCacheManager] = None


def register_sparse_kv_manager(manager: SparseKVCacheManager) -> None:
    global _global_sparse_kv_manager
    _global_sparse_kv_manager = manager


def get_sparse_kv_manager() -> Optional[SparseKVCacheManager]:
    return _global_sparse_kv_manager


def _normalize_topk_indices_2d(topk_indices: torch.Tensor) -> torch.Tensor:
    if topk_indices.dim() == 2:
        normalized = topk_indices
    elif topk_indices.dim() == 3:
        if topk_indices.shape[1] != 1:
            raise RuntimeError(
                "Sparse KV top-k rank-3 input expects shape [B, 1, K], "
                f"got {tuple(topk_indices.shape)}."
            )
        normalized = topk_indices[:, 0, :]
    elif topk_indices.dim() == 4:
        if topk_indices.shape[1] != 1 or topk_indices.shape[2] != 1:
            raise RuntimeError(
                "Sparse KV top-k rank-4 input expects shape [B, 1, 1, K], "
                f"got {tuple(topk_indices.shape)}."
            )
        normalized = topk_indices[:, 0, 0, :]
    else:
        raise RuntimeError(
            "Sparse KV top-k input expects rank 2, 3, or 4, "
            f"got rank {topk_indices.dim()}."
        )
    return normalized.contiguous()


def _build_partition_sparse_indices(
    counts: torch.Tensor, topk_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if counts.dim() != 1:
        raise RuntimeError(
            f"Sparse KV partition counts must be 1-D, got {counts.dim()}."
        )
    if topk_len <= 0:
        raise RuntimeError(f"Sparse KV top-k length must be positive, got {topk_len}.")

    batch_size = counts.shape[0]
    positions = torch.arange(topk_len, dtype=torch.int32, device=counts.device).view(
        1, topk_len
    )
    valid = positions < counts.view(batch_size, 1)
    indices_2d = torch.where(
        valid,
        positions.expand(batch_size, topk_len),
        torch.full(
            (batch_size, topk_len), -1, dtype=torch.int32, device=counts.device,
        ),
    )

    # SFA does not accept an empty KV sequence.  Empty rows read a zeroed
    # dummy at slot zero and are converted to a neutral state during merge.
    indices_2d[:, 0] = torch.where(
        counts > 0,
        indices_2d[:, 0],
        torch.zeros(batch_size, dtype=torch.int32, device=counts.device),
    )
    actual_lengths = counts.clamp(min=1, max=topk_len).to(torch.int32).contiguous()
    return (
        indices_2d.view(batch_size, 1, 1, topk_len).contiguous(),
        actual_lengths,
    )


def _build_noncompact_partition_sparse_indices(
    valid_mask: torch.Tensor, topk_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack original Top-K positions while leaving their KV rows in place."""

    if valid_mask.dim() != 2 or valid_mask.shape[1] != topk_len:
        raise RuntimeError(
            "Sparse KV noncompact partition mask must have shape [B, K], got "
            f"{tuple(valid_mask.shape)} for K={topk_len}."
        )
    batch_size = valid_mask.shape[0]
    counts = valid_mask.sum(dim=1).to(torch.int32).contiguous()
    ranks = torch.cumsum(valid_mask.to(torch.int32), dim=1).to(torch.int64) - 1
    positions = torch.arange(
        topk_len, dtype=torch.int32, device=valid_mask.device
    ).view(1, topk_len)

    # All invalid items target a sentinel column, leaving the first K columns
    # as a stable compact list of original Top-K positions.
    packed = torch.full(
        (batch_size, topk_len + 1),
        -1,
        dtype=torch.int32,
        device=valid_mask.device,
    )
    packed_dst = torch.where(
        valid_mask,
        ranks,
        torch.full_like(ranks, topk_len),
    )
    packed_values = torch.where(
        valid_mask,
        positions.expand(batch_size, topk_len),
        torch.full_like(positions.expand(batch_size, topk_len), -1),
    )
    packed.scatter_(1, packed_dst, packed_values)
    indices_2d = packed[:, :topk_len]
    indices_2d[:, 0] = torch.where(
        counts > 0,
        indices_2d[:, 0],
        torch.zeros(batch_size, dtype=torch.int32, device=valid_mask.device),
    )

    # The physical SFA key has K rows even though each partition selects only
    # counts[b] scattered positions. true_counts still neutralizes empty rows.
    actual_lengths = torch.full_like(counts, topk_len, dtype=torch.int32)
    return (
        indices_2d.view(batch_size, 1, 1, topk_len).contiguous(),
        actual_lengths.contiguous(),
        counts,
        ranks,
    )


def _assign_miss_cache_slots(
    hit_valid: torch.Tensor,
    device_token_pos: torch.Tensor,
    miss_ranks: torch.Tensor,
    topk_len: int,
) -> torch.Tensor:
    """Assign misses to the complement of currently occupied hit slots."""

    if hit_valid.shape != device_token_pos.shape or hit_valid.shape != miss_ranks.shape:
        raise RuntimeError(
            "Sparse KV cache-slot planning expects matching [B, K] tensors, got "
            f"hit={tuple(hit_valid.shape)}, slots={tuple(device_token_pos.shape)}, "
            f"miss_ranks={tuple(miss_ranks.shape)}."
        )
    batch_size, width = hit_valid.shape
    if width != topk_len:
        raise RuntimeError(
            f"Sparse KV cache-slot planning expects K={topk_len}, got {width}."
        )

    # Preserve every hit in its current physical slot. The sentinel column
    # absorbs invalid scatter destinations, keeping all graph tensor shapes
    # fixed while the hit/miss distribution changes between replays.
    occupied = torch.zeros(
        (batch_size, topk_len + 1),
        dtype=torch.int32,
        device=hit_valid.device,
    )
    occupied_dst = torch.where(
        hit_valid,
        device_token_pos.to(torch.int64).clamp(min=0, max=topk_len - 1),
        torch.full_like(device_token_pos, topk_len, dtype=torch.int64),
    )
    occupied.scatter_add_(1, occupied_dst, hit_valid.to(torch.int32).contiguous())

    free_mask = occupied[:, :topk_len] == 0
    free_ranks = (
        torch.cumsum(free_mask.to(torch.int32), dim=1).to(torch.int64) - 1
    )
    positions = torch.arange(
        topk_len, dtype=torch.int64, device=hit_valid.device
    ).view(1, topk_len)
    free_slots = torch.full(
        (batch_size, topk_len + 1),
        topk_len,
        dtype=torch.int64,
        device=hit_valid.device,
    )
    free_dst = torch.where(
        free_mask,
        free_ranks,
        torch.full_like(free_ranks, topk_len),
    )
    free_values = torch.where(
        free_mask,
        positions.expand(batch_size, topk_len),
        torch.full_like(positions.expand(batch_size, topk_len), topk_len),
    )
    free_slots.scatter_(1, free_dst, free_values)
    return free_slots[:, :topk_len].gather(
        1, miss_ranks.clamp(min=0, max=topk_len - 1)
    )


def _build_hit_src_dst_index(
    token_on_device: torch.Tensor,
    device_token_pos: torch.Tensor,
    current_req_indices: torch.Tensor,
    sparse_context_len: int,
):
    """
    token_on_device: [bs, topk], bool
    device_token_pos: [bs, topk], int64 or int32
    current_req_indices: [bs], int64

    Return:
        src_index_full: [bs * topk], int64
        dst_index_full: [bs * topk], int64
        valid_mask: [bs * topk], bool

    Flattening rule:
        src row = req_id * sparse_context_len + device_token_pos
        dst row = batch_id * topk + topk_pos
    """
    if token_on_device.dim() != 2 or device_token_pos.dim() != 2:
        raise RuntimeError(
            f"token_on_device and device_token_pos must be 2-D, got "
            f"{token_on_device.dim()} and {device_token_pos.dim()}"
        )
    if token_on_device.shape != device_token_pos.shape:
        raise RuntimeError(
            f"token_on_device and device_token_pos must have the same shape, got "
            f"{tuple(token_on_device.shape)} and {tuple(device_token_pos.shape)}"
        )
    if current_req_indices.dim() != 1:
        raise RuntimeError(
            f"current_req_indices must be 1-D, got {current_req_indices.dim()}"
        )

    bs, topk = token_on_device.shape
    if current_req_indices.numel() != bs:
        raise RuntimeError(
            f"current_req_indices length mismatch: "
            f"{current_req_indices.numel()} vs batch {bs}"
        )
    if sparse_context_len <= 0:
        raise RuntimeError(
            f"sparse_context_len must be positive, got {sparse_context_len}"
        )

    device = token_on_device.device

    valid_mask = token_on_device.reshape(-1).contiguous()

    flat_dst_index_all = torch.arange(
        bs * topk,
        device=device,
        dtype=torch.int64,
    )

    req_offsets = current_req_indices.to(torch.int64).unsqueeze(1) * sparse_context_len
    src_index_2d = req_offsets + device_token_pos.to(torch.int64)
    flat_src_index_all = src_index_2d.reshape(-1).contiguous()

    return flat_src_index_all, flat_dst_index_all, valid_mask


def _build_miss_src_dst_index(
    token_from_host: torch.Tensor,
    topk_indices: torch.Tensor,
    current_req_indices: torch.Tensor,
    max_context_len: int,
):
    if token_from_host.dim() != 2 or topk_indices.dim() != 2:
        raise RuntimeError(
            f"token_from_host and topk_indices must be 2-D, got "
            f"{token_from_host.dim()} and {topk_indices.dim()}"
        )
    if token_from_host.shape != topk_indices.shape:
        raise RuntimeError(
            f"token_from_host and topk_indices must have the same shape, got "
            f"{tuple(token_from_host.shape)} and {tuple(topk_indices.shape)}"
        )
    if current_req_indices.dim() != 1:
        raise RuntimeError(
            f"current_req_indices must be 1-D, got {current_req_indices.dim()}"
        )
    if current_req_indices.numel() != token_from_host.shape[0]:
        raise RuntimeError(
            f"current_req_indices length mismatch: "
            f"{current_req_indices.numel()} vs batch {token_from_host.shape[0]}"
        )

    bs, topk = token_from_host.shape
    device = token_from_host.device

    valid_2d = token_from_host & (topk_indices >= 0) & (topk_indices < max_context_len)
    valid_mask = valid_2d.reshape(-1).contiguous()

    flat_dst_index_all = torch.arange(
        bs * topk,
        device=device,
        dtype=torch.int64,
    )

    req_offsets = current_req_indices.to(torch.int64).unsqueeze(1) * max_context_len
    src_index_2d = req_offsets + topk_indices.to(torch.int64)
    flat_src_index_all = src_index_2d.reshape(-1).contiguous()

    return flat_src_index_all, flat_dst_index_all, valid_mask
