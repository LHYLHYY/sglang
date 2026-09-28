"""CPU numerical regression tests; no torch/torch_npu installation required.

Execute the production manager class from its AST with NumPy tensor/copy
stand-ins. This checks addressing, packing, and lifecycle contracts, not NPU
kernel execution or graph capture support.
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/native_fia_manager.py"
)


class Tensor(np.ndarray):
    device = "cpu"

    def numel(self):
        return self.size

    def dim(self):
        return self.ndim

    def element_size(self):
        return self.itemsize

    def view(self, *shape):
        return self.reshape(*shape)

    def to(self, *, dtype=None, device=None):
        return self.astype(dtype or self.dtype, copy=False)

    def contiguous(self):
        return tensor(np.ascontiguousarray(self))

    def expand(self, *shape):
        return tensor(np.broadcast_to(self, shape))

    def clamp(self, min=None, max=None):
        return tensor(np.clip(self, min, max))

    def zero_(self):
        self.fill(0)
        return self

    def split(self, sizes, dim):
        return [
            tensor(part) for part in np.split(self, np.cumsum(sizes)[:-1], axis=dim)
        ]


def tensor(data, dtype=None):
    return np.asarray(data, dtype=dtype).view(Tensor)


class FakeOps:
    def __init__(self):
        self.calls = []
        self.allocations = []

    def create_shm_tensor(self, *, shape, dtype, device_id, name):
        host = tensor(np.zeros(shape, dtype=dtype))
        ptr = 100000 + len(self.allocations) * 100000
        self.allocations.append((host, ptr, name))
        return host, ptr, ptr

    @staticmethod
    def flat(t, address_ndims):
        return np.asarray(t).reshape(int(np.prod(t.shape[:address_ndims])), -1)

    def check_raw_pointer(self, view, pointer):
        for host, base, _ in self.allocations:
            if np.shares_memory(view, host):
                offset = (
                    view.__array_interface__["data"][0]
                    - host.__array_interface__["data"][0]
                )
                assert pointer == base + offset
                return
        raise AssertionError("copy does not reference registered host storage")

    def unidex_copy_inplace(
        self,
        src,
        dst,
        src_index,
        dst_index,
        valid,
        src_address_ndims,
        dst_address_ndims,
        **kwargs
    ):
        if "src_ptr" in kwargs:
            self.check_raw_pointer(src, kwargs["src_ptr"])
        if "dst_ptr" in kwargs:
            self.check_raw_pointer(dst, kwargs["dst_ptr"])
        self.calls.append(("copy", np.asarray(valid).copy()))
        src_flat = self.flat(src, src_address_ndims)
        dst_flat = self.flat(dst, dst_address_ndims)
        dst_flat[dst_index[valid]] = src_flat[src_index[valid]]

    def unidex_split_copy_inplace(
        self,
        src,
        nope,
        rope,
        src_index,
        dst_index,
        valid,
        src_address_ndims,
        dst_address_ndims,
        **kwargs
    ):
        self.check_raw_pointer(src, kwargs["src_ptr"])
        self.calls.append(("split", np.asarray(valid).copy()))
        src_flat = self.flat(src, src_address_ndims)
        nope_flat = self.flat(nope, dst_address_ndims)
        rope_flat = self.flat(rope, dst_address_ndims)
        selected = src_flat[src_index[valid]]
        nope_flat[dst_index[valid]] = selected[:, : nope_flat.shape[1]]
        rope_flat[dst_index[valid]] = selected[:, nope_flat.shape[1] :]


class FakePool:
    start_layer = 3
    layer_num = 2
    kv_lora_rank = 16
    qk_rope_head_dim = 16
    store_dtype = np.float16


def load_manager(*, max_copy_bytes=(1 << 32) - 1, missing_kernel=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ops = FakeOps()
    native = {
        name: object()
        for name in (
            "shm_allocator_create_and_register",
            "unidex_copy",
            "unidex_split_copy",
        )
        if name != missing_kernel
    }
    torch = SimpleNamespace(
        long=np.int64,
        int32=np.int32,
        bool=np.bool_,
        float16=np.float16,
        bfloat16="bfloat16",
        ops=SimpleNamespace(npu=SimpleNamespace(**native)),
        npu=SimpleNamespace(current_device=lambda: 0, current_stream=lambda: "stream0"),
        zeros=lambda shape, dtype, device=None: tensor(np.zeros(shape, dtype=dtype)),
        ones=lambda shape, dtype, device=None: tensor(np.ones(shape, dtype=dtype)),
        empty=lambda shape, dtype, device=None: tensor(np.empty(shape, dtype=dtype)),
        arange=lambda end, dtype, device=None: tensor(np.arange(end, dtype=dtype)),
        cat=lambda tensors, dim=0: tensor(np.concatenate(tensors, axis=dim)),
        cumsum=lambda t, dim: tensor(np.cumsum(t, axis=dim)),
        where=lambda condition, x, y: tensor(np.where(condition, x, y)),
        repeat_interleave=lambda t, repeats, output_size: tensor(np.repeat(t, repeats)),
    )
    namespace = {
        "torch": torch,
        "sparse_kv_ops": ops,
        "MLATokenToKVPool": FakePool,
        "logger": logging.getLogger(__name__),
        "_MAX_COPY_BYTES": max_copy_bytes,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["NativeFIAOffloadManager"], ops


def batch(req_ids, seq_lens, mode="decode", **kwargs):
    return SimpleNamespace(
        req_pool_indices=tensor(req_ids, np.int64),
        seq_lens=tensor(seq_lens, np.int64),
        out_cache_loc=tensor(np.arange(len(req_ids)), np.int64),
        forward_mode=SimpleNamespace(
            is_decode=lambda: mode == "decode", is_idle=lambda: mode == "idle"
        ),
        **kwargs,
    )


class TestNativeFIAOffloadManager(unittest.TestCase):
    def make_manager(self, *, batch_size=4, max_copy_bytes=(1 << 32) - 1):
        cls, self.ops = load_manager(max_copy_bytes=max_copy_bytes)
        req_pool = SimpleNamespace(
            req_to_token=np.empty((4, 9)), max_context_len=9, device="cpu"
        )
        return cls(
            req_pool,
            SimpleNamespace(get_kvcache=lambda: FakePool()),
            topk=4,
            max_batch_size=batch_size,
            page_size=4,
        )

    def fill_host(self, manager):
        for layer, host in enumerate(manager.host_kv_buffer):
            values = np.arange(36).reshape(4, 9, 1, 1) + layer * 100
            host[...] = values

    def test_decode_includes_first_token_and_excludes_invalid_requests(self):
        manager = self.make_manager()
        b = batch([1, 0, 4, 3], [1, 1, 2, 10])
        manager.offload_v2(
            tensor(np.ones((4, 1, 16))), tensor(np.full((4, 1, 16), 2)), 3, b
        )
        np.testing.assert_array_equal(manager.host_kv_buffer[0][1, 0, 0, :16], 1)
        np.testing.assert_array_equal(manager.host_kv_buffer[0][1, 0, 0, 16:], 2)
        self.assertEqual(np.count_nonzero(manager.host_kv_buffer[0]), 32)

    def test_short_sequences_ignore_indexer_holes_and_long_sequences_compact(self):
        manager = self.make_manager()
        self.fill_host(manager)
        b = batch([1, 2, 0, 3], [3, 8, 1, 9])
        indices = tensor(
            [[-1, -1, 7, -1], [7, -1, 2, 99], [0, 1, 2, 3], [8, 1, 5, 3]], np.int32
        )
        original = b.seq_lens.copy()
        nope, rope, table = manager.prefetch_native_fia(
            3, b, indices.reshape(4, 1, 1, 4)
        )
        np.testing.assert_array_equal(
            nope[:, :, 0],
            [[9, 10, 11, 0], [25, 20, 0, 0], [0, 0, 0, 0], [35, 28, 32, 30]],
        )
        np.testing.assert_array_equal(rope[:, :, 0], nope[:, :, 0])
        np.testing.assert_array_equal(table, [[0], [1], [2], [3]])
        np.testing.assert_array_equal(b.seq_lens, original)
        self.assertIs(nope, manager.selected_k_nope)
        # The second layer reuses storage and clears the previous layer's tail.
        b.seq_lens[0] = 1
        manager.prefetch_native_fia(4, b, indices)
        np.testing.assert_array_equal(nope[0, :, 0], [109, 0, 0, 0])

    def test_idle_emits_no_copies_and_initial_staging_is_zero(self):
        manager = self.make_manager()
        b = batch([0], [1], mode="idle")
        manager.offload_v2(None, None, 3, b)
        nope, rope, _ = manager.prefetch_native_fia(3, b, None)
        self.assertFalse(self.ops.calls)
        self.assertEqual(np.count_nonzero(nope) + np.count_nonzero(rope), 0)

    def test_host_shards_use_correct_offsets_in_all_copy_directions(self):
        # Each 64-byte row allows 16 rows per mocked shard, so request 1
        # straddles shards and request 3 resides in the final shard.
        manager = self.make_manager(batch_size=2, max_copy_bytes=1024)
        self.assertEqual(len(manager._host_shards[0]), 3)
        b = batch(
            [1, 3],
            [9, 2],
            mode="extend",
            extend_seq_lens=tensor([9, 2]),
            extend_prefix_lens=tensor([0, 0]),
            extend_seq_lens_cpu=[9, 2],
        )
        b.out_cache_loc = tensor(np.arange(11))
        values = np.arange(11, dtype=np.float16).reshape(11, 1, 1) + 1
        manager.offload_v2(
            tensor(np.broadcast_to(values, (11, 1, 16))),
            tensor(np.broadcast_to(values + 20, (11, 1, 16))),
            3,
            b,
        )
        nope, rope = manager.get_forward_kv(3, b)
        np.testing.assert_array_equal(nope[:, 0, 0], np.arange(11) + 1)
        np.testing.assert_array_equal(rope[:, 0, 0], np.arange(11) + 21)
        decode = batch([1, 3], [9, 2])
        nope, rope, _ = manager.prefetch_native_fia(
            3, decode, tensor([[8, 0, 7, 2], [-1, -1, -1, -1]])
        )
        np.testing.assert_array_equal(nope[:, :, 0], [[9, 1, 8, 3], [10, 11, 0, 0]])
        np.testing.assert_array_equal(rope[:, :, 0], [[29, 21, 28, 23], [30, 31, 0, 0]])

    def test_padded_prefill_masks_padding_and_request_zero(self):
        manager = self.make_manager()
        b = batch(
            [1, 0],
            [3, 0],
            mode="extend",
            extend_seq_lens=tensor([2, 0]),
            extend_prefix_lens=tensor([1]),
            extend_seq_lens_cpu=[2, 0],
        )
        b.out_cache_loc = tensor(np.arange(6))
        manager.offload_v2(
            tensor(np.ones((6, 1, 16))), tensor(np.ones((6, 1, 16))), 3, b
        )
        self.assertEqual(np.count_nonzero(manager.host_kv_buffer[0]), 64)
        np.testing.assert_array_equal(manager.host_kv_buffer[0][1, 1:3], 1)

    def test_dp_prefill_padding_follows_all_compact_requests(self):
        for rows in (16, 17):
            with self.subTest(rows=rows):
                manager = self.make_manager()
                b = batch(
                    [1, 2],
                    [3, 7],
                    mode="extend",
                    extend_seq_lens=tensor([3, 5]),
                    extend_prefix_lens=tensor([0, 2]),
                    extend_seq_lens_cpu=[3, 5],
                )
                b.out_cache_loc = tensor(np.arange(rows))
                values = np.arange(rows, dtype=np.float16).reshape(rows, 1, 1) + 1
                key = tensor(np.broadcast_to(values, (rows, 1, 16)))
                manager.offload_v2(key, key, 3, b)
                np.testing.assert_array_equal(
                    manager.host_kv_buffer[0][1, :3, 0, 0], [1, 2, 3]
                )
                np.testing.assert_array_equal(
                    manager.host_kv_buffer[0][2, 2:7, 0, 0], [4, 5, 6, 7, 8]
                )
                self.assertEqual(np.count_nonzero(manager.host_kv_buffer[0]), 8 * 32)

    def test_prefill_gather_preserves_dummy_request_offsets_without_host_io(self):
        manager = self.make_manager()
        self.fill_host(manager)
        # Poison row 0 to make an accidental host padding read observable.
        manager.host_kv_buffer[0][0] = 77
        for req_ids, lengths, expected in (
            ([1, 0, 2], [2, 3, 1], [9, 10, 0, 0, 0, 18]),
            ([0], [4], [0, 0, 0, 0]),
        ):
            with self.subTest(req_ids=req_ids):
                b = batch(req_ids, lengths, mode="extend")
                nope, rope = manager.get_forward_kv(3, b)
                np.testing.assert_array_equal(nope[:, 0, 0], expected)
                np.testing.assert_array_equal(rope[:, 0, 0], expected)

    def test_missing_kernel_fails_before_allocating_host(self):
        cls, ops = load_manager(missing_kernel="unidex_split_copy")
        with self.assertRaisesRegex(RuntimeError, "unidex_split_copy"):
            cls(None, None, topk=4, max_batch_size=2, page_size=4)
        self.assertFalse(ops.allocations)

    def test_decode_paths_do_not_read_device_values_on_cpu(self):
        manager = self.make_manager()
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        prefetch = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "prefetch_native_fia"
        )
        forbidden = {"item", "cpu", "tolist", "synchronize", "nonzero"}
        self.assertFalse(
            [
                node.attr
                for node in ast.walk(prefetch)
                if isinstance(node, ast.Attribute) and node.attr in forbidden
            ]
        )
        self.assertFalse(hasattr(manager, "device_slot_map"))
        self.assertFalse(hasattr(manager, "_prefetch_h2d_miss_stream"))


if __name__ == "__main__":
    unittest.main()
