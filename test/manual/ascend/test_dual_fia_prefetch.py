"""CPU regressions for fused split-FIA partition preparation and cache refill.

Execute production manager methods from their AST with NumPy tensor and kernel
stand-ins. These checks cover values, persistent buffers, and host enqueue order;
they do not establish NPU kernel execution or device-side graph correctness.

Run: python test/manual/ascend/test_dual_fia_prefetch.py -v
"""

import ast
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/manager.py"
)
FIA_MODE = "split_graph_dual_fia"


class Tensor(np.ndarray):
    device = "npu"
    runtime = None

    def dim(self):
        return self.ndim

    def numel(self):
        return self.size

    def element_size(self):
        return self.itemsize

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return self.reshape(*shape)

    def to(self, dtype=None, device=None):
        return tensor(np.asarray(self, dtype=dtype or self.dtype))

    def contiguous(self):
        return tensor(np.ascontiguousarray(self))

    def is_contiguous(self):
        return self.flags.c_contiguous

    def unsqueeze(self, dim):
        return tensor(np.expand_dims(self, dim))

    def squeeze(self, dim=None):
        return tensor(np.squeeze(np.asarray(self), axis=dim))

    def expand(self, *shape):
        shape = tuple(old if new == -1 else new for old, new in zip(self.shape, shape))
        return tensor(np.broadcast_to(self, shape))

    def clamp(self, min=None, max=None):
        return tensor(np.clip(self, min, max))

    def mul(self, other):
        return self * other

    def sum(self, dim=None):
        return tensor(np.asarray(self).sum(axis=dim))

    def zero_(self):
        if self.runtime is not None:
            self.runtime.log.append(("zero", self.runtime.current.name, self.size))
        self.fill(0)
        return self

    def index_fill_(self, dim, indices, value):
        if dim != 0:
            raise AssertionError("Only row index_fill_ is needed in these tests")
        self[indices] = value
        return self

    def scatter_(self, dim, indices, values):
        np.put_along_axis(self, indices, values, axis=dim)
        return self

    def cpu(self):
        raise AssertionError("Prefetch must not synchronize counts to CPU")

    def tolist(self):
        raise AssertionError("Prefetch must not read device values as Python")


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Stream:
    def __init__(self, runtime, name):
        self.runtime = runtime
        self.name = name

    def record_event(self, event):
        self.runtime.log.append(("record", self.name, event.name))

    def wait_event(self, event):
        self.runtime.log.append(("wait", self.name, event.name))


class Runtime:
    def __init__(self):
        self.log = []
        self.main = Stream(self, "main")
        self.hit = Stream(self, "hit")
        self.miss = Stream(self, "miss")
        self.refill = Stream(self, "refill")
        self.current = self.main
        self.event_count = 0
        Tensor.runtime = self

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def event(self):
        self.event_count += 1
        return SimpleNamespace(name=f"event{self.event_count}")


class CopyOps:
    """Model the existing kernels: invalid descriptors skip every output."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []

    @staticmethod
    def flat(value, address_ndims):
        return np.asarray(value).reshape(int(np.prod(value.shape[:address_ndims])), -1)

    def unidex_copy_inplace(
        self, src, dst, src_index, dst_index, valid, src_ndims, dst_ndims, **kwargs
    ):
        self.runtime.log.append(("copy", self.runtime.current.name))
        self.calls.append(("copy", kwargs.copy()))
        src_flat, dst_flat = self.flat(src, src_ndims), self.flat(dst, dst_ndims)
        for si, di, enabled in zip(src_index, dst_index, valid):
            if enabled and 0 <= si < len(src_flat) and 0 <= di < len(dst_flat):
                dst_flat[di] = src_flat[si]
        return dst

    def unidex_split_copy_promote_inplace(
        self,
        src,
        key,
        rope,
        staging,
        src_index,
        dst_index,
        hot_dst_index,
        valid,
        src_ndims,
        dst_ndims,
        hot_ndims,
        **kwargs,
    ):
        self.runtime.log.append(("split_promote", self.runtime.current.name))
        self.calls.append(("split_promote", kwargs.copy()))
        src_flat = self.flat(src, src_ndims)
        key_flat, rope_flat = self.flat(key, dst_ndims), self.flat(rope, dst_ndims)
        hot_flat = self.flat(staging, hot_ndims)
        if not (
            key.is_contiguous() and rope.is_contiguous() and staging.is_contiguous()
        ):
            raise AssertionError("Split-copy destinations must be contiguous")
        for si, di, hi, enabled in zip(src_index, dst_index, hot_dst_index, valid):
            if not (
                enabled
                and 0 <= si < len(src_flat)
                and 0 <= di < len(key_flat)
                and 0 <= hi < len(hot_flat)
            ):
                continue
            row = src_flat[si]
            key_flat[di] = row[: key_flat.shape[1]]
            rope_flat[di] = row[key_flat.shape[1] :]
            hot_flat[hi] = row
        return key, rope, staging


def load_manager(runtime):
    ops = CopyOps(runtime)
    torch = SimpleNamespace(
        Tensor=Tensor,
        int32=np.int32,
        int64=np.int64,
        long=np.int64,
        bool=np.bool_,
        empty=lambda shape, dtype, device=None: tensor(
            np.full(shape, np.nan, dtype=dtype)
        ),
        full=lambda shape, fill, dtype, device=None: tensor(
            np.full(shape, fill, dtype=dtype)
        ),
        zeros=lambda shape, dtype, device=None: tensor(np.zeros(shape, dtype=dtype)),
        arange=lambda end, dtype=None, device=None: tensor(np.arange(end, dtype=dtype)),
        where=lambda condition, x, y: tensor(np.where(condition, x, y)),
        full_like=lambda value, fill, dtype=None: tensor(
            np.full_like(value, fill, dtype=dtype)
        ),
        zeros_like=lambda value: tensor(np.zeros_like(value)),
        cumsum=lambda value, dim: tensor(np.cumsum(value, axis=dim)),
        npu=SimpleNamespace(
            stream=runtime.stream,
            current_stream=lambda: runtime.current,
            Event=runtime.event,
        ),
    )

    def slot_map_lookup(slot_map, req_ids, topk):
        positions = np.full(topk.shape, -1, dtype=np.int32)
        for row, req in enumerate(req_ids):
            for col, token in enumerate(topk[row]):
                if 0 <= req < len(slot_map) and 0 <= token < slot_map.shape[1]:
                    positions[row, col] = slot_map[req, token]
        return tensor(positions >= 0, np.int32), tensor(positions)

    method_names = {
        "_make_attention_partition",
        "_copy_attention_partition",
        "prefetch_partitions_graph_dual",
        "prefetch_partitions",
        "commit_graph_dual_refill",
    }
    function_names = {
        "_normalize_topk_indices_2d",
        "_build_partition_sparse_indices",
        "_record_stream_event",
        "_wait_stream_event",
    }
    class_names = {
        "SparseKVPartition",
        "SparseKVFiaPartition",
        "SparseKVGraphDualPrefetch",
        "SparseKVPrefetchTicket",
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    manager_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SparseKVCacheManager"
    )
    methods = [
        node
        for node in manager_cls.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    if {node.name for node in methods} != method_names:
        raise AssertionError("Missing production FIA prefetch helper")
    manager_cls.body = methods
    selected = [
        node
        for node in tree.body
        if (
            (isinstance(node, ast.ClassDef) and node.name in class_names)
            or (isinstance(node, ast.FunctionDef) and node.name in function_names)
        )
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
            manager_cls,
        ],
        type_ignores=[],
    )
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "torch": torch,
        "sparse_kv_ops": ops,
        "unidex_copy_inplace": ops.unidex_copy_inplace,
        "slot_map_lookup": slot_map_lookup,
        "SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA": FIA_MODE,
        "_profile_push": lambda name: None,
        "_profile_pop": lambda token: None,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["SparseKVCacheManager"], namespace, ops


class TestDualFiaPrefetch(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        cls, self.namespace, self.ops = load_manager(self.runtime)
        self.manager = cls()
        self.manager.attn_impl = FIA_MODE
        self.manager.kv_lora_rank = 512
        self.manager.qk_rope_head_dim = 64
        self.manager.head_num = 1
        self.manager.head_dim = 576
        self.manager.store_dtype = np.float16
        self.manager.device = "npu"
        self.manager.size = 4
        self.manager.max_context_len = 8
        self.manager.sparse_context_len = 4
        self.manager.start_layer = 7
        self.manager.layer_num = 1
        self.manager._slot_map_width = 16
        self.manager._device_cache_slot_ids = tensor(np.arange(4), np.int64)
        self.manager._prefetch_d2d_hit_stream = self.runtime.hit
        self.manager._prefetch_h2d_miss_stream = self.runtime.miss
        self.manager._prefetch_refill_stream = self.runtime.refill
        self.manager.dev_ptr_list = [123456]
        self.manager.host_kv_buffer = [tensor(np.empty((4, 8, 1, 576), np.float16))]
        for req in range(4):
            for token in range(8):
                row = self.manager.host_kv_buffer[0][req, token, 0]
                row[:512] = req * 100 + token
                row[512:] = req * 100 + token + 0.5
        self.manager.device_kv_buffer = [
            tensor(np.full((4, 4, 1, 576), np.nan, np.float16))
        ]
        self.manager.device_slot_map = [tensor(np.full((5, 16), -1, np.int32))]
        # Nontrivial old slots ensure compaction cannot accidentally use Top-K order.
        for req, slots in ((1, {2: 3, 5: 0}), (2, {1: 2})):
            for token, slot in slots.items():
                self.manager.device_slot_map[0][req, token] = slot
                self.manager.device_kv_buffer[0][
                    req, slot
                ] = self.manager.host_kv_buffer[0][req, token]
        shape = (3, 4, 1, 576)
        self.manager._graph_dual_state = SimpleNamespace(
            max_batch_size=3,
            hit_kv=tensor(np.full(shape, np.nan, np.float16)),
            miss_kv=tensor(np.full(shape, np.nan, np.float16)),
            hit_fia_buffer=tensor(np.full(np.prod(shape), np.nan, np.float16)),
            miss_fia_buffer=tensor(np.full(np.prod(shape), np.nan, np.float16)),
            miss_stream=self.runtime.miss,
            layer_events=[
                SimpleNamespace(
                    **{
                        name: SimpleNamespace(name=name)
                        for name in (
                            "inputs_ready",
                            "hit_copy_done",
                            "miss_attention_done",
                            "refill_done",
                        )
                    }
                )
            ],
        )
        self.layer = SimpleNamespace(layer_id=7)

    def run_prefetch(self, graph=True, topk=None, req_ids=None, seq_lens=None):
        if topk is None:
            topk = [[6, 2, -1, 5], [1, 4, 3, 99], [0, 1, 2, 3]]
        req_ids = [1, 2, 0] if req_ids is None else req_ids
        seq_lens = [8, 8, 0] if seq_lens is None else seq_lens
        batch = SimpleNamespace(
            req_pool_indices=tensor(req_ids, np.int64),
            seq_lens=tensor(seq_lens, np.int64),
        )
        fn = (
            self.manager.prefetch_partitions_graph_dual
            if graph
            else self.manager.prefetch_partitions
        )
        with patch.dict(
            self.namespace,
            {
                "_build_partition_sparse_indices": lambda *args: self.fail(
                    "FIA built SFA metadata"
                )
            },
        ):
            return fn(
                self.layer, batch, tensor(topk, np.int32), self.runtime.main, np.float16
            )

    def assert_partition(self, part, selected):
        self.assertIsInstance(part, self.namespace["SparseKVFiaPartition"])
        self.assertFalse(hasattr(part, "sparse_indices"))
        self.assertFalse(hasattr(part, "actual_seq_lengths_kv"))
        self.assertTrue(part.key.is_contiguous())
        self.assertTrue(part.key_rope.is_contiguous())
        for row, tokens in enumerate(selected):
            self.assertEqual(int(part.true_counts[row]), len(tokens))
            for col, (req, token) in enumerate(tokens):
                expected = self.manager.host_kv_buffer[0][req, token]
                np.testing.assert_array_equal(part.kv[row, col], expected)
                np.testing.assert_array_equal(part.key[row, col], expected[:, :512])
                np.testing.assert_array_equal(
                    part.key_rope[row, col], expected[:, 512:]
                )
            np.testing.assert_array_equal(part.key[row, len(tokens) :], 0)
            np.testing.assert_array_equal(part.key_rope[row, len(tokens) :], 0)

    def assert_refill(self):
        cache, slot_map = (
            self.manager.device_kv_buffer[0],
            self.manager.device_slot_map[0],
        )
        for req, selected in ((1, {0: 6, 1: 2, 3: 5}), (2, {0: 1, 1: 4, 2: 3})):
            for slot, token in selected.items():
                np.testing.assert_array_equal(
                    cache[req, slot], self.manager.host_kv_buffer[0][req, token]
                )
                self.assertEqual(slot_map[req, token], slot)
            self.assertEqual(np.count_nonzero(slot_map[req, :8] >= 0), len(selected))

    def test_make_fia_partition_has_views_but_no_metadata_or_clear(self):
        kv = tensor(np.full((2, 4, 1, 576), np.nan, np.float16))
        buffer = tensor(np.full(3 * kv.numel(), np.nan, np.float16))
        with patch.dict(
            self.namespace,
            {
                "_build_partition_sparse_indices": lambda *args: self.fail(
                    "FIA built SFA metadata"
                )
            },
        ):
            part = self.manager._make_attention_partition(
                kv, tensor([2, 0], np.int32), self.runtime.main, buffer
            )
        self.assertEqual(part.buffer.numel(), kv.numel())
        self.assertTrue(np.shares_memory(part.key, buffer))
        self.assertTrue(np.shares_memory(part.key_rope, buffer))
        self.assertFalse(np.shares_memory(part.key, part.key_rope))
        self.assertTrue(np.isnan(part.buffer).all())
        self.assertTrue(np.isnan(kv).all())
        self.assertFalse(self.runtime.log)

    def test_graph_mixed_partitions_fused_copy_and_delayed_refill(self):
        old_cache = self.manager.device_kv_buffer[0].copy()
        ticket = self.run_prefetch()
        self.assert_partition(ticket.hit, [[(1, 2), (1, 5)], [(2, 1)], []])
        self.assert_partition(ticket.miss, [[(1, 6)], [(2, 4), (2, 3)], []])
        np.testing.assert_array_equal(self.manager.device_kv_buffer[0], old_cache)
        self.assertEqual([name for name, _ in self.ops.calls], ["split_promote"] * 2)
        self.assertEqual(self.ops.calls[0][1].get("src_ptr"), 123456)
        self.assertEqual(
            [entry for entry in self.runtime.log if entry[0] == "zero"],
            [
                ("zero", "miss", ticket.miss.buffer.numel()),
                ("zero", "main", ticket.hit.buffer.numel()),
            ],
        )
        self.manager.commit_graph_dual_refill(ticket)
        self.assert_refill()
        log = self.runtime.log
        self.assertLess(
            log.index(("wait", "miss", "inputs_ready")),
            log.index(("split_promote", "miss")),
        )
        self.assertLess(
            log.index(("split_promote", "main")),
            log.index(("record", "main", "hit_copy_done")),
        )
        self.assertLess(
            log.index(("wait", "miss", "hit_copy_done")), log.index(("copy", "miss"))
        )
        self.assertEqual(log[-1], ("record", "miss", "refill_done"))

    def test_reuse_shrinking_and_empty_partitions_clears_stale_nan_tails(self):
        first = self.run_prefetch()
        self.manager.commit_graph_dual_refill(first)
        first.hit.buffer.fill(np.nan)
        first.miss.buffer.fill(np.nan)
        second = self.run_prefetch(topk=[[2, -1, -1, -1], [-1] * 4, [0, 1, 2, 3]])
        self.assert_partition(second.hit, [[(1, 2)], [], []])
        self.assert_partition(second.miss, [[], [], []])
        self.assertTrue(np.shares_memory(first.hit.buffer, second.hit.buffer))
        self.assertTrue(np.shares_memory(first.miss.buffer, second.miss.buffer))

    def test_graph_batch_bucket_views_use_only_active_storage(self):
        state = self.manager._graph_dual_state
        ticket = self.run_prefetch(topk=[[2, 6, -1, -1]], req_ids=[1], seq_lens=[8])
        self.assert_partition(ticket.hit, [[(1, 2)]])
        self.assert_partition(ticket.miss, [[(1, 6)]])
        self.assertEqual(ticket.hit.buffer.numel(), 4 * 576)
        self.assertTrue(np.isnan(state.hit_fia_buffer[4 * 576 :]).all())
        self.assertTrue(np.isnan(state.miss_fia_buffer[4 * 576 :]).all())

    def test_alternating_graph_batch_buckets_reuses_buffer_without_layout_alias(self):
        wide = self.run_prefetch()
        narrow = self.run_prefetch(topk=[[2, 6, -1, -1]], req_ids=[1], seq_lens=[8])
        self.assert_partition(narrow.hit, [[(1, 2)]])
        self.assert_partition(narrow.miss, [[(1, 6)]])
        again = self.run_prefetch()
        self.assert_partition(again.hit, [[(1, 2), (1, 5)], [(2, 1)], []])
        self.assert_partition(again.miss, [[(1, 6)], [(2, 4), (2, 3)], []])
        self.assertTrue(np.shares_memory(wide.hit.buffer, narrow.hit.buffer))
        self.assertTrue(np.shares_memory(wide.hit.key_rope, again.hit.key_rope))

    def test_eager_prefetch_preserves_separate_producers_and_refill_waits(self):
        ticket = self.run_prefetch(graph=False)
        self.assert_partition(ticket.hit, [[(1, 2), (1, 5)], [(2, 1)], []])
        self.assert_partition(ticket.miss, [[(1, 6)], [(2, 4), (2, 3)], []])
        self.assert_refill()
        self.assertEqual(
            [name for name, _ in self.ops.calls],
            ["split_promote", "split_promote", "copy", "copy"],
        )
        log = self.runtime.log
        copy_index = log.index(("copy", "refill"))
        waits = [entry for entry in log[:copy_index] if entry[:2] == ("wait", "refill")]
        self.assertEqual(len(waits), 2)
        records = {
            entry[2]: entry[1] for entry in log[:copy_index] if entry[0] == "record"
        }
        self.assertEqual({records[entry[2]] for entry in waits}, {"hit", "miss"})
        self.assertEqual(
            [entry[1] for entry in log if entry[0] == "zero"], ["hit", "miss"]
        )

    def test_sfa_fallback_still_builds_dummy_metadata_and_uses_original_copy(self):
        self.manager.attn_impl = "split_graph_dual"
        kv = tensor(np.full((2, 4, 1, 576), np.nan, np.float16))
        part = self.manager._make_attention_partition(
            kv, tensor([1, 0], np.int32), self.runtime.main
        )
        self.assertIsInstance(part, self.namespace["SparseKVPartition"])
        np.testing.assert_array_equal(part.kv[:, 0], 0)
        self.assertTrue(np.isnan(part.kv[:, 1:]).all())
        np.testing.assert_array_equal(part.actual_seq_lengths_kv, [1, 1])
        np.testing.assert_array_equal(
            part.sparse_indices.reshape(2, 4), [[0, -1, -1, -1]] * 2
        )
        self.manager._copy_attention_partition(
            self.manager.host_kv_buffer[0],
            part,
            tensor([8, -1], np.int64),
            tensor([0, 4], np.int64),
            tensor([True, False]),
            src_ptr=123456,
        )
        np.testing.assert_array_equal(
            part.kv[0, 0], self.manager.host_kv_buffer[0][1, 0]
        )
        self.assertEqual([name for name, _ in self.ops.calls], ["copy"])
        self.assertEqual(
            len([entry for entry in self.runtime.log if entry[0] == "zero"]), 1
        )


if __name__ == "__main__":
    unittest.main()
