"""Registered host KV and serial selected-KV copies for native paged FIA.

Request pool row 0 is reserved for graph padding. Real requests are rows
``1 <= req_id < size``; the first token of a real request is valid KV. Host
rows are overwritten by each new request's prefill, so this manager installs
no allocator hook and has no slot map or per-request cache state to reset.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import sgl_kernel_npu.sparsity_driven_kv_offload as sparse_kv_ops
import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)
_MAX_COPY_BYTES = (1 << 32) - 1


class NativeFIAOffloadManager:
    """A single current-stream staging buffer, shared by all local layers.

    The caller must enqueue offload, prefetch, and FIA in that order on the
    same stream. It must finish consuming one layer's staging buffer before
    the next prefetch. Decode uses only fixed-shape device operations; full
    prefill gathers are eager and may inspect sequence lengths on the CPU.
    """

    attn_impl = "native_fia"
    fia_skip_kv_io = False

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        *,
        topk: int,
        max_batch_size: int,
        page_size: int,
    ) -> None:
        if min(topk, max_batch_size, page_size) <= 0:
            raise ValueError(
                "Native FIA topk, max_batch_size and page_size must be positive."
            )
        for wrapper, native in (
            ("create_shm_tensor", "shm_allocator_create_and_register"),
            ("unidex_copy_inplace", "unidex_copy"),
            ("unidex_split_copy_inplace", "unidex_split_copy"),
        ):
            if not callable(getattr(sparse_kv_ops, wrapper, None)) or not hasattr(
                torch.ops.npu, native
            ):
                raise RuntimeError(
                    f"Native FIA offload requires sgl_kernel_npu {wrapper} and "
                    f"torch.ops.npu.{native}; rebuild/install the matching NPU kernels."
                )

        self.paged_kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(self.paged_kv_cache, MLATokenToKVPool):
            raise TypeError("Native FIA offload requires an MLATokenToKVPool.")
        self.size = int(req_to_token_pool.req_to_token.shape[0])
        self.max_context_len = int(req_to_token_pool.max_context_len)
        self.device = req_to_token_pool.device
        self.start_layer = self.paged_kv_cache.start_layer
        self.layer_num = self.paged_kv_cache.layer_num
        self.head_num = 1
        self.kv_lora_rank = self.paged_kv_cache.kv_lora_rank
        self.qk_rope_head_dim = self.paged_kv_cache.qk_rope_head_dim
        self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.store_dtype = self.paged_kv_cache.store_dtype
        if self.store_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Native FIA offload supports only float16/bfloat16 KV.")
        self.sparse_context_len = int(topk)
        self.max_batch_size = int(max_batch_size)
        self.page_size = int(page_size)
        self.blocks_per_request = (topk + page_size - 1) // page_size
        self.tokens_per_request = self.blocks_per_request * page_size
        self.selected_kv_capacity = max_batch_size * self.tokens_per_request
        row_bytes = self.head_dim * 2
        if (
            self.size <= 1
            or self.max_context_len <= 0
            or row_bytes > 32 * 1024
            or self.kv_lora_rank * 2 % 32
            or self.qk_rope_head_dim * 2 % 32
        ):
            raise ValueError(
                "Native FIA offload has invalid host capacity or KV row alignment."
            )
        if self.selected_kv_capacity * row_bytes > _MAX_COPY_BYTES:
            raise ValueError(
                "Native FIA selected KV exceeds the copy kernel's 4 GiB extent limit."
            )

        # These are the only persistent NPU allocations. PoolConfigurator
        # reserves both KV tensors, the int32 table, and this int64 arange.
        pages = max_batch_size * self.blocks_per_request
        self.selected_k_nope = torch.zeros(
            (pages, page_size, self.kv_lora_rank),
            dtype=self.store_dtype,
            device=self.device,
        )
        self.selected_k_rope = torch.zeros(
            (pages, page_size, self.qk_rope_head_dim),
            dtype=self.store_dtype,
            device=self.device,
        )
        self.block_table = torch.arange(
            pages, dtype=torch.int32, device=self.device
        ).view(max_batch_size, self.blocks_per_request)
        self._positions = torch.arange(
            self.tokens_per_request, dtype=torch.long, device=self.device
        )

        self.host_kv_buffer = []
        self.host_ptr_list = []
        self.dev_ptr_list = []
        self._host_shards = []
        host_shape = (self.size, self.max_context_len, self.head_num, self.head_dim)
        device_id = torch.npu.current_device()
        # The copy kernels use uint32 byte offsets. Keep the registered host
        # allocation whole, but pass bounded views and offset raw pointers.
        # Shard count is fixed at construction and introduces no decode sync.
        rows_per_shard = _MAX_COPY_BYTES // row_bytes
        host_rows = self.size * self.max_context_len
        for layer_idx in range(self.layer_num):
            host, host_ptr, dev_ptr = sparse_kv_ops.create_shm_tensor(
                shape=host_shape,
                dtype=self.store_dtype,
                device_id=device_id,
                name=f"native_fia_host_kv_layer_{layer_idx}_rank_{device_id}",
            )
            self.host_kv_buffer.append(host)
            self.host_ptr_list.append(host_ptr)
            self.dev_ptr_list.append(dev_ptr)
            flat_host = host.view(host_rows, self.head_num, self.head_dim)
            self._host_shards.append(
                [
                    (
                        flat_host[start : start + rows_per_shard],
                        dev_ptr + start * row_bytes,
                        start,
                        min(start + rows_per_shard, host_rows),
                    )
                    for start in range(0, host_rows, rows_per_shard)
                ]
            )
        logger.info(
            "Native FIA offload: host shape %s x %s layers; shared selected KV "
            "capacity=%s tokens, page_size=%s, max_batch_size=%s.",
            host_shape,
            self.layer_num,
            self.selected_kv_capacity,
            page_size,
            max_batch_size,
        )

    def _layer_index(self, layer):
        layer_id = layer.layer_id if hasattr(layer, "layer_id") else int(layer)
        index = layer_id - self.start_layer
        if not 0 <= index < self.layer_num:
            raise ValueError(f"Native FIA offload got invalid layer id {layer_id}.")
        return index

    @staticmethod
    def _check_stream(stream):
        if stream is not None and stream != torch.npu.current_stream():
            raise ValueError("Native FIA offload only accepts the current NPU stream.")

    def reset_requests(self, req_ids):
        """No-op: request reuse overwrites host KV; there is no hot-cache state."""

    def offload_v2(self, k, k_rope, layer, forward_batch, stream=None):
        """Write this forward's compact KV, including real requests of length 1."""
        if forward_batch.forward_mode.is_idle():
            return
        self._check_stream(stream)
        layer_idx = self._layer_index(layer)
        src = torch.cat([k, k_rope], dim=-1).contiguous()
        if tuple(src.shape[1:]) != (self.head_num, self.head_dim):
            raise ValueError("Native FIA offload expects compact [T, 1, D] KV.")
        rows = int(src.shape[0])
        if rows == 0:
            return
        if src.numel() * src.element_size() > _MAX_COPY_BYTES:
            raise ValueError(
                "Native FIA compact KV exceeds the copy kernel's 4 GiB limit."
            )
        src_index = torch.arange(rows, dtype=torch.long, device=src.device)
        req_ids = forward_batch.req_pool_indices.to(dtype=torch.long)
        if forward_batch.forward_mode.is_decode():
            seq_lens = forward_batch.seq_lens.to(dtype=torch.long)
            cache_loc = forward_batch.out_cache_loc
            if (
                req_ids.numel() != rows
                or seq_lens.numel() != rows
                or cache_loc.numel() != rows
            ):
                raise ValueError(
                    "Native FIA decode KV rows must match request/sequence/cache metadata."
                )
            token_pos = seq_lens - 1
            valid = (cache_loc >= 0) & (seq_lens > 0)
        else:
            lens = forward_batch.extend_seq_lens
            prefix = forward_batch.extend_prefix_lens
            if lens is None or prefix is None:
                raise ValueError(
                    "Native FIA prefill offload requires extend sequence and prefix lengths."
                )
            batch_size = int(req_ids.numel())
            if batch_size == 0:
                return
            lens = lens.to(dtype=torch.long)
            prefix = prefix.to(dtype=torch.long)
            if lens.numel() != batch_size or prefix.numel() > batch_size:
                raise ValueError(
                    "Native FIA prefill metadata does not match batch size."
                )
            if prefix.numel() < batch_size:
                prefix = torch.cat(
                    [
                        prefix,
                        torch.zeros(
                            batch_size - prefix.numel(),
                            dtype=torch.long,
                            device=src.device,
                        ),
                    ]
                )
            # _pad_inputs_to_size appends token padding after all compact
            # request rows. It does not interleave padding into each request.
            # Prefill graphs are unsupported, so the CPU total is available.
            lens_cpu = forward_batch.extend_seq_lens_cpu
            total = (
                int(sum(lens_cpu[:batch_size]))
                if lens_cpu is not None
                else int(lens.sum().item())
            )
            if not 0 <= total <= rows:
                raise ValueError(
                    "Native FIA prefill extend lengths exceed compact KV rows."
                )
            starts = torch.cumsum(lens, dim=0) - lens
            req_ids = torch.repeat_interleave(req_ids, lens, output_size=total)
            token_pos = (
                torch.repeat_interleave(prefix, lens, output_size=total)
                + src_index[:total]
                - torch.repeat_interleave(starts, lens, output_size=total)
            )
            if total < rows:
                padding = torch.zeros(rows - total, dtype=torch.long, device=src.device)
                req_ids = torch.cat([req_ids, padding])
                token_pos = torch.cat([token_pos, padding])
            valid = src_index < total
            cache_loc = forward_batch.out_cache_loc
            if cache_loc is not None and cache_loc.numel() == rows:
                valid = valid & (cache_loc >= 0)
        valid = (
            valid
            & (req_ids > 0)
            & (req_ids < self.size)
            & (token_pos >= 0)
            & (token_pos < self.max_context_len)
        )
        dst_index = req_ids * self.max_context_len + token_pos
        for host, ptr, start, end in self._host_shards[layer_idx]:
            sparse_kv_ops.unidex_copy_inplace(
                src,
                host,
                src_index,
                (dst_index - start).contiguous(),
                (valid & (dst_index >= start) & (dst_index < end)).contiguous(),
                1,
                1,
                block_dim=48,
                dst_ptr=ptr,
            )

    def prefetch_native_fia(self, layer, forward_batch, topk_indices):
        """Return persistent (NoPE pages, RoPE pages, batch block table).

        Short sequences select every token in order, independent of indexer
        padding. Long sequences use the indexer's K positions. Valid entries
        are packed in order; invalid entries and padding read zero KV. The
        caller supplies FIA lengths min(sequence length, K), with dummy rows
        handled by its graph metadata. Full sequence metadata is never changed.
        """
        batch_size = int(forward_batch.req_pool_indices.numel())
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Native FIA batch {batch_size} exceeds reserved capacity {self.max_batch_size}."
            )
        block_table = self.block_table[:batch_size]
        result = (self.selected_k_nope, self.selected_k_rope, block_table)
        if forward_batch.forward_mode.is_idle() or batch_size == 0:
            return result
        if not forward_batch.forward_mode.is_decode():
            raise ValueError("Native FIA selected-KV prefetch is decode-only.")
        layer_idx = self._layer_index(layer)
        if (
            topk_indices.dim() not in (2, 3, 4)
            or topk_indices.shape[0] != batch_size
            or topk_indices.shape[-1] != self.sparse_context_len
            or any(dim != 1 for dim in topk_indices.shape[1:-1])
        ):
            raise ValueError(
                "Native FIA top-k must have shape [B, K], [B, 1, K], or [B, 1, 1, K]."
            )
        seq_lens = forward_batch.seq_lens.to(dtype=torch.long).view(batch_size, 1)
        req_ids = forward_batch.req_pool_indices.to(dtype=torch.long).view(
            batch_size, 1
        )
        topk = topk_indices.reshape(batch_size, self.sparse_context_len).to(
            dtype=torch.long
        )
        positions = self._positions[: self.sparse_context_len].view(1, -1)
        selected = torch.where(seq_lens <= self.sparse_context_len, positions, topk)
        valid = (
            (req_ids > 0)
            & (req_ids < self.size)
            & (seq_lens > 0)
            & (seq_lens <= self.max_context_len)
            & (selected >= 0)
            & (selected < seq_lens)
        )
        packed = torch.cumsum(valid.to(dtype=torch.long), dim=1) - 1
        dst_index = (
            (
                block_table[:, :1].to(dtype=torch.long) * self.page_size
                + packed.clamp(min=0)
            )
            .reshape(-1)
            .contiguous()
        )
        src_index = (req_ids * self.max_context_len + selected).reshape(-1)
        valid = valid.reshape(-1)
        pages = batch_size * self.blocks_per_request
        # Clear active rows every layer: masked long-sequence holes and graph
        # padding must never expose KV retained from a previous request/layer.
        self.selected_k_nope[:pages].zero_()
        self.selected_k_rope[:pages].zero_()
        for host, ptr, start, end in self._host_shards[layer_idx]:
            sparse_kv_ops.unidex_split_copy_inplace(
                host,
                self.selected_k_nope,
                self.selected_k_rope,
                (src_index - start).contiguous(),
                dst_index,
                (valid & (src_index >= start) & (src_index < end)).contiguous(),
                1,
                2,
                block_dim=48,
                src_ptr=ptr,
            )
        return result

    def get_forward_kv(self, layer, forward_batch, stream=None):
        """Eager prefill gather in compact [request tokens] TND order."""
        self._check_stream(stream)
        if forward_batch.forward_mode.is_decode():
            raise ValueError("Native FIA full-KV gather is eager prefill-only.")
        layer_idx = self._layer_index(layer)
        req_ids = forward_batch.req_pool_indices.to(
            device=self.device, dtype=torch.long
        )
        lens = forward_batch.seq_lens.to(device=self.device, dtype=torch.long)
        if req_ids.numel() != lens.numel():
            raise ValueError(
                "Native FIA full-KV request and sequence lengths must match."
            )
        if forward_batch.forward_mode.is_idle():
            total = 0
        else:
            invalid = (
                (lens < 0)
                | (lens > self.max_context_len)
                | ((lens > 0) & ((req_ids < 0) | (req_ids >= self.size)))
            )
            if bool(invalid.any().item()):
                raise ValueError(
                    "Native FIA prefill has invalid request IDs or sequence lengths (row 0 is padding)."
                )
            total = int(lens.sum().item())
        if total * self.head_dim * 2 > _MAX_COPY_BYTES:
            raise ValueError(
                "Native FIA prefill gather exceeds the copy kernel's 4 GiB output limit."
            )
        # DP can convert an idle rank into a fabricated EXTEND request with
        # positive length and req_id=0. Preserve its TND offsets/length, but
        # provide zero KV without ever reading host padding row 0.
        kv = torch.zeros(
            (total, self.head_num, self.head_dim),
            dtype=self.store_dtype,
            device=self.device,
        )
        if total:
            dst_index = torch.arange(total, dtype=torch.long, device=self.device)
            starts = torch.cumsum(lens, dim=0) - lens
            src_req_ids = torch.repeat_interleave(req_ids, lens, output_size=total)
            src_index = (
                src_req_ids * self.max_context_len
                + dst_index
                - torch.repeat_interleave(starts, lens, output_size=total)
            )
            for host, ptr, start, end in self._host_shards[layer_idx]:
                sparse_kv_ops.unidex_copy_inplace(
                    host,
                    kv,
                    (src_index - start).contiguous(),
                    dst_index,
                    (
                        (src_req_ids > 0) & (src_index >= start) & (src_index < end)
                    ).contiguous(),
                    1,
                    1,
                    block_dim=48,
                    src_ptr=ptr,
                )
        nope, rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return nope.contiguous(), rope.contiguous()
