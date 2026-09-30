"""CPU numerical and scheduling checks for split FIA attention.

The production helpers are loaded from their AST with NumPy tensor stand-ins.
These tests exercise masks, partition normalization, and host enqueue order;
they do not establish CANN operator support or device-side graph correctness.

Run: python test/manual/ascend/test_split_fia_attention.py -v
"""

import ast
import logging
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/attention.py"
)


class Tensor(np.ndarray):
    device = "npu"
    runtime = None

    def dim(self):
        return self.ndim

    def numel(self):
        return self.size

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return self.reshape(*shape)

    def contiguous(self):
        return tensor(np.ascontiguousarray(self))

    def is_contiguous(self):
        return self.flags.c_contiguous

    def permute(self, *dims):
        return tensor(np.transpose(self, dims))

    def float(self):
        return tensor(np.asarray(self, dtype=np.float32))

    def to(self, dtype=None, device=None):
        return tensor(np.asarray(self, dtype=dtype or self.dtype))

    def clamp(self, min=None, max=None):
        return tensor(np.clip(self, min, max))

    def clamp_min(self, value):
        return tensor(np.maximum(self, value))

    def split(self, sizes, dim):
        return [
            tensor(part) for part in np.split(self, np.cumsum(sizes)[:-1], axis=dim)
        ]

    def unsqueeze(self, dim):
        return tensor(np.expand_dims(self, dim))

    def record_stream(self, stream):
        if self.runtime is not None:
            self.runtime.log.append(("tensor.record_stream", stream.name))
            self.runtime.recorded_tensors.append((self.data_ptr(), stream.name))

    def cpu(self):
        raise AssertionError("Split attention must not copy counts to CPU")

    def tolist(self):
        raise AssertionError("Split attention must not read device counts as Python")

    def data_ptr(self):
        return self.__array_interface__["data"][0]


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
        self.recorded_tensors = []
        self.main = Stream(self, "main")
        self.hit = Stream(self, "hit")
        self.miss = Stream(self, "miss")
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


class FIA:
    """Reference kernel implementing the public v2 .out tensor contract."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []
        self.workspace_calls = []
        self.workspace = tensor(np.zeros(16, dtype=np.uint8))

    def get_workspace(self, *args, **kwargs):
        self.workspace_calls.append((args, kwargs.copy()))
        self.workspace = tensor(np.zeros(16, dtype=np.uint8))
        return self.workspace

    def out(self, query, key, value, **kwargs):
        self.runtime.log.append(("fia", self.runtime.current.name))
        self.calls.append((query, key, value, kwargs.copy()))
        batch, query_len, heads, dim = query.shape
        block_size = kwargs["block_size"]
        table = np.asarray(kwargs["block_table"])
        k = np.asarray(key)[table].reshape(batch, -1, dim)
        v = np.asarray(value)[table].reshape(batch, -1, dim)
        rope_dim = kwargs["query_rope"].shape[-1]
        kr = np.asarray(kwargs["key_rope"])[table].reshape(batch, -1, rope_dim)
        if key.shape[1] != block_size:
            raise AssertionError("KV page shape does not match block_size")
        if not np.isfinite(k).all() or not np.isfinite(kr).all():
            raise AssertionError("Invalid KV tails reached FIA without clearing")
        scores = np.einsum("bshd,bkd->bhsk", query.astype(np.float64), k)
        scores += np.einsum(
            "bshd,bkd->bhsk", kwargs["query_rope"].astype(np.float64), kr
        )
        scores *= kwargs["softmax_scale"]
        mask = np.asarray(kwargs["atten_mask"])
        scores = np.where(mask, -np.inf, scores)
        for row, length in enumerate(kwargs["actual_seq_kvlen"]):
            scores[row, :, :, length:] = -np.inf
        maximum = scores.max(axis=-1, keepdims=True)
        numerator = np.exp(scores - maximum)
        denominator = numerator.sum(axis=-1, keepdims=True)
        probability = numerator / denominator
        result = np.einsum("bhsk,bkd->bshd", probability, v)
        lse = maximum + np.log(denominator)
        output, stats = kwargs["out"]
        output[...] = result
        stats[...] = lse
        if tuple(output.shape) != (batch, query_len, heads, dim):
            raise AssertionError("FIA out buffer has wrong shape")
        return output, stats


def load_helpers(runtime, fia):
    torch = SimpleNamespace(
        Tensor=Tensor,
        int32=np.int32,
        int64=np.int64,
        float32=np.float32,
        float16=np.float16,
        bfloat16="bfloat16",
        bool=np.bool_,
        arange=lambda end, dtype=None, device=None: tensor(np.arange(end, dtype=dtype)),
        where=lambda condition, x, y: tensor(np.where(condition, x, y)),
        zeros_like=lambda x, **kwargs: tensor(
            np.zeros_like(x, dtype=kwargs.get("dtype"))
        ),
        ones_like=lambda x, **kwargs: tensor(
            np.ones_like(x, dtype=kwargs.get("dtype"))
        ),
        full_like=lambda x, value, **kwargs: tensor(
            np.full_like(x, value, dtype=kwargs.get("dtype"))
        ),
        empty_like=lambda x, **kwargs: tensor(
            np.empty_like(x, dtype=kwargs.get("dtype"))
        ),
        empty=lambda *shape, dtype=None, device=None: tensor(
            np.empty(shape[0] if len(shape) == 1 else shape, dtype=dtype)
        ),
        ones=lambda *shape, dtype=None, device=None: tensor(
            np.ones(shape[0] if len(shape) == 1 else shape, dtype=dtype)
        ),
        maximum=lambda x, y: tensor(np.maximum(x, y)),
        exp=lambda x: tensor(np.exp(x)),
        finfo=np.finfo,
        profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
        npu=SimpleNamespace(
            current_stream=lambda: runtime.current,
            get_device_name=lambda: "Ascend910B",
            stream=runtime.stream,
            Event=runtime.event,
        ),
    )
    names = {
        "_run_decode_fia_partition",
        "_merge_decode_sfa_partitions_python",
        "_run_split_decode_attention",
        "_run_split_decode_attention_graph_dual",
        "_record_stream_event",
        "_wait_stream_event",
        "validate_split_fia_support",
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in nodes}
    if found != names:
        raise AssertionError(f"Missing production helpers: {names - found}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "torch_npu": SimpleNamespace(
            npu_fused_infer_attention_score_v2=SimpleNamespace(out=fia.out),
            _npu_fused_infer_attention_score_v2_get_max_workspace=fia.get_workspace,
        ),
        "_SfaPartitionState": SimpleNamespace,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)

    def merge(hit, miss, merge_impl):
        runtime.log.append(("merge", runtime.current.name))
        return namespace["_merge_decode_sfa_partitions_python"](hit, miss)

    namespace["_merge_decode_sfa_partitions"] = merge
    namespace["_run_decode_sfa_partition"] = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("FIA mode dispatched SFA"))
    return namespace


class TestSplitFIASupport(unittest.TestCase):
    REGISTRY_MODULE = "torch_npu.npu._npugraph_handlers.npugraph_handler"
    REGISTRY_KEY = "npu_fused_infer_attention_score_v2.out"

    def setUp(self):
        runtime = Runtime()
        self.helpers = load_helpers(runtime, FIA(runtime))
        self.validate = self.helpers["validate_split_fia_support"]

    @contextmanager
    def registry_import(self, *, registry=None, error=None):
        module = ModuleType(self.REGISTRY_MODULE)
        module._NPU_GRAPH_OP_HANDLERS = registry or {}

        def import_mock(name, globals=None, locals=None, fromlist=(), level=0):
            # Never fall through to Python's actual importer: these checks must
            # not load a locally installed torch_npu or initialize any device.
            self.assertEqual(name, self.REGISTRY_MODULE)
            self.assertEqual(fromlist, ("_NPU_GRAPH_OP_HANDLERS",))
            self.assertEqual(level, 0)
            if error is not None:
                raise error
            return module

        with patch("builtins.__import__", side_effect=import_mock) as importer:
            yield importer

    def test_missing_v2_out_or_workspace_is_rejected(self):
        runtime_api = self.helpers["torch_npu"]
        for missing in ("v2", "out", "workspace", "workspace_not_callable"):
            with self.subTest(missing=missing):
                mocked_api = SimpleNamespace(**vars(runtime_api))
                if missing == "v2":
                    del mocked_api.npu_fused_infer_attention_score_v2
                elif missing == "out":
                    mocked_api.npu_fused_infer_attention_score_v2 = SimpleNamespace()
                elif missing == "workspace":
                    del mocked_api._npu_fused_infer_attention_score_v2_get_max_workspace
                else:
                    mocked_api._npu_fused_infer_attention_score_v2_get_max_workspace = (
                        None
                    )
                self.helpers["torch_npu"] = mocked_api
                with patch("builtins.__import__") as importer:
                    with self.assertRaisesRegex(RuntimeError, "FIA v2 .out.*workspace"):
                        self.validate(graph_enabled=True)
                    importer.assert_not_called()

    def test_950_is_rejected_before_handler_import(self):
        self.helpers["torch"].npu.get_device_name = lambda: "Ascend950PR"
        with patch("builtins.__import__") as importer:
            with self.assertRaisesRegex(RuntimeError, "950 MLA decode mask"):
                self.validate(graph_enabled=True)
            importer.assert_not_called()

    def test_eager_support_does_not_import_graph_registry(self):
        with patch("builtins.__import__") as importer:
            self.validate(graph_enabled=False)
            importer.assert_not_called()

    def test_graph_support_requires_registered_v2_out(self):
        with self.registry_import(registry={self.REGISTRY_KEY: object()}) as importer:
            self.validate(graph_enabled=True)
            importer.assert_called_once()

    def test_graph_registry_without_v2_out_is_rejected(self):
        with self.registry_import(
            registry={"npu_fused_infer_attention_score.out": object()}
        ) as importer:
            with self.assertRaisesRegex(
                RuntimeError, "FIA v2 .out graph-update handler"
            ):
                self.validate(graph_enabled=True)
            importer.assert_called_once()

    def test_unavailable_registry_preserves_original_import_error(self):
        original = ImportError("No module named torch_npu.npu._npugraph_handlers")
        with self.registry_import(error=original):
            with self.assertRaisesRegex(
                RuntimeError, "auto-dispatch handler registry"
            ) as context:
                self.validate(graph_enabled=True)
        self.assertIs(context.exception.__cause__, original)

    def test_unrelated_registry_initialization_error_is_not_hidden(self):
        original = RuntimeError("unrelated runtime initialization failure")
        with self.registry_import(error=original):
            with self.assertRaises(RuntimeError) as context:
                self.validate(graph_enabled=True)
        self.assertIs(context.exception, original)


class TestSplitFIA(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.fia = FIA(self.runtime)
        self.helpers = load_helpers(self.runtime, self.fia)

    def make_case(self, *, hit_counts, miss_counts, heads=4, capacity=16, seed=314):
        rng = np.random.default_rng(seed)
        batch = len(hit_counts)
        query = tensor(rng.normal(0, 0.1, (batch, 1, heads, 512)), np.float16)
        rope = tensor(rng.normal(0, 0.1, (batch, 1, heads, 64)), np.float16)
        hit = tensor(np.full((batch, capacity, 1, 576), np.nan), np.float16)
        miss = tensor(np.full_like(hit, np.nan))
        for row, (nhit, nmiss) in enumerate(zip(hit_counts, miss_counts)):
            hit[row, :nhit] = rng.normal(0, 0.1, (nhit, 1, 576))
            miss[row, :nmiss] = rng.normal(0, 0.1, (nmiss, 1, 576))
        def make_partition(kv, counts, stream):
            buffer = tensor(np.empty(batch * capacity * 576), np.float16)
            key_size = batch * capacity * 512
            partition = SimpleNamespace(
                kv=kv,
                buffer=buffer,
                key=buffer[:key_size].view(batch, capacity, 1, 512),
                key_rope=buffer[key_size:].view(batch, capacity, 1, 64),
                true_counts=tensor(counts, np.int32),
                stream=stream,
            )
            self.prepare_partition(partition)
            return partition

        return SimpleNamespace(
            query=query,
            query_rope=rope,
            hit=make_partition(hit, hit_counts, self.runtime.hit),
            miss=make_partition(miss, miss_counts, self.runtime.miss),
            capacity=capacity,
            scale=0.37,
        )

    @staticmethod
    def prepare_partition(partition):
        # Model the producer's zero + fused gather/split. The refill snapshot
        # deliberately keeps NaN tails so attention must use prepared storage.
        partition.buffer.fill(0)
        for row, count in enumerate(partition.true_counts):
            partition.key[row, :count] = partition.kv[row, :count, :, :512]
            partition.key_rope[row, :count] = partition.kv[row, :count, :, 512:]

    def run_partition(self, case, partition, **kwargs):
        return self.helpers["_run_decode_fia_partition"](
            partition,
            query=case.query,
            query_rope=case.query_rope,
            nope_head_dim=512,
            rope_head_dim=64,
            scale_value=case.scale,
            page_size=16,
            **kwargs,
        )

    @staticmethod
    def reference(case):
        output = np.zeros(case.query.shape, dtype=np.float64)
        for row, (nhit, nmiss) in enumerate(
            zip(case.hit.true_counts, case.miss.true_counts)
        ):
            selected = np.concatenate(
                (case.hit.kv[row, :nhit, 0], case.miss.kv[row, :nmiss, 0])
            )
            if len(selected) == 0:
                continue
            q = np.concatenate(
                (case.query[row, 0], case.query_rope[row, 0]), axis=-1
            ).astype(np.float64)
            scores = (q @ selected.astype(np.float64).T) * case.scale
            p = np.exp(scores - scores.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            output[row, 0] = p @ selected[:, :512].astype(np.float64)
        return output

    def test_split_matches_union_for_mixed_empty_and_different_heads(self):
        for heads in (1, 4, 16):
            with self.subTest(heads=heads):
                case = self.make_case(
                    hit_counts=(5, 16, 0, 0), miss_counts=(11, 0, 16, 0), heads=heads
                )
                hit = self.run_partition(case, case.hit)
                miss = self.run_partition(case, case.miss)
                merged = self.helpers["_merge_decode_sfa_partitions_python"](hit, miss)
                np.testing.assert_allclose(
                    merged, self.reference(case), atol=8e-5, rtol=2e-3
                )
                self.assertTrue(np.isfinite(merged).all())
                np.testing.assert_array_equal(merged[-1], 0)
                self.assertEqual(hit.softmax_max.shape, (4, 1, 1, heads))
                self.assertEqual(hit.softmax_max.dtype, np.float32)
                np.testing.assert_array_equal(hit.softmax_sum, 1)

    def test_public_v2_contract_masks_and_out_buffers(self):
        case = self.make_case(hit_counts=(3, 0), miss_counts=(13, 16), heads=8)
        state = self.run_partition(case, case.hit, record_stream=False)
        q, key, value, kwargs = self.fia.calls[-1]
        self.assertIs(q, case.query)
        self.assertIs(key, value)
        self.assertEqual(key.data_ptr(), case.hit.key.data_ptr())
        self.assertEqual(kwargs["key_rope"].data_ptr(), case.hit.key_rope.data_ptr())
        self.assertTrue(np.shares_memory(key, case.hit.buffer))
        self.assertTrue(np.shares_memory(kwargs["key_rope"], case.hit.buffer))
        self.assertIs(kwargs["workspace"], self.fia.workspace)
        self.assertEqual(kwargs["actual_seq_kvlen"], [16, 16])
        self.assertTrue(kwargs["return_softmax_lse"])
        self.assertEqual(kwargs["input_layout"], "BSND")
        self.assertEqual(kwargs["num_query_heads"], 8)
        self.assertEqual(kwargs["num_key_value_heads"], 1)
        self.assertEqual(kwargs["softmax_scale"], case.scale)
        self.assertEqual(kwargs["sparse_mode"], 0)
        self.assertEqual(kwargs["atten_mask"].shape, (2, 1, 1, 16))
        self.assertEqual(kwargs["atten_mask"].dtype, np.bool_)
        np.testing.assert_array_equal(kwargs["atten_mask"][0, 0, 0], np.arange(16) >= 3)
        np.testing.assert_array_equal(kwargs["atten_mask"][1, 0, 0], np.arange(16) >= 1)
        np.testing.assert_array_equal(key[0, 3:], 0)
        np.testing.assert_array_equal(key[1], 0)
        self.assertEqual(kwargs["out"][0].dtype, case.query.dtype)
        self.assertEqual(kwargs["out"][1].dtype, np.float32)
        self.assertEqual(kwargs["out"][1].shape, (2, 8, 1, 1))
        np.testing.assert_array_equal(state.output[1], 0)
        np.testing.assert_array_equal(state.softmax_max[1], 0)
        workspace_kwargs = self.fia.workspace_calls[-1][1]
        for name in (
            "return_softmax_lse",
            "actual_seq_kvlen",
            "num_query_heads",
            "sparse_mode",
        ):
            self.assertEqual(workspace_kwargs[name], kwargs[name])
        self.assertFalse(
            any(entry[0] == "tensor.record_stream" for entry in self.runtime.log)
        )

    def test_lse_weights_do_not_equal_count_weights(self):
        case = self.make_case(hit_counts=(2,), miss_counts=(2,), heads=1)
        case.query.fill(0)
        case.query_rope.fill(0)
        case.query[..., 0] = 1
        case.hit.kv[:, :2] = 0
        case.miss.kv[:, :2] = 0
        case.hit.kv[:, :2, :, 0] = 5
        case.miss.kv[:, :2, :, 0] = -5
        self.prepare_partition(case.hit)
        self.prepare_partition(case.miss)
        hit = self.run_partition(case, case.hit)
        miss = self.run_partition(case, case.miss)
        merged = self.helpers["_merge_decode_sfa_partitions_python"](hit, miss)
        np.testing.assert_allclose(merged, self.reference(case), atol=0.003, rtol=0.001)
        self.assertGreater(float(merged[0, 0, 0, 0]), 4)

    def test_two_partitions_keep_independent_workspaces_and_outputs(self):
        case = self.make_case(hit_counts=(5,), miss_counts=(11,))
        hit = self.run_partition(case, case.hit)
        hit_snapshot = hit.output.copy()
        self.run_partition(case, case.miss)
        hit_args = self.fia.calls[0][3]
        miss_args = self.fia.calls[1][3]
        for name in ("workspace", "atten_mask", "key_rope"):
            self.assertFalse(np.shares_memory(hit_args[name], miss_args[name]), name)
        for index in (0, 1):
            self.assertFalse(
                np.shares_memory(hit_args["out"][index], miss_args["out"][index])
            )
        np.testing.assert_array_equal(hit.output, hit_snapshot)

    def test_device_counts_change_mask_without_host_length_changes(self):
        case = self.make_case(hit_counts=(16,), miss_counts=(0,))
        first = self.run_partition(case, case.hit)
        case.hit.true_counts[...] = 3
        case.hit.kv[:, 3:] = np.nan
        self.prepare_partition(case.hit)
        second = self.run_partition(case, case.hit)
        self.assertEqual(
            self.fia.calls[0][3]["actual_seq_kvlen"],
            self.fia.calls[1][3]["actual_seq_kvlen"],
        )
        self.assertFalse(np.allclose(first.output, second.output))
        np.testing.assert_allclose(
            second.output, self.reference(case), atol=8e-5, rtol=2e-3
        )

    def test_multiple_pages_preserve_per_request_selection(self):
        case = self.make_case(hit_counts=(17, 3), miss_counts=(15, 29), capacity=32)
        original_hit = case.hit.kv.copy()
        hit = self.run_partition(case, case.hit)
        miss = self.run_partition(case, case.miss)
        merged = self.helpers["_merge_decode_sfa_partitions_python"](hit, miss)
        np.testing.assert_allclose(merged, self.reference(case), atol=8e-5, rtol=2e-3)
        np.testing.assert_array_equal(
            self.fia.calls[0][3]["block_table"], [[0, 1], [2, 3]]
        )
        # Refill may read its combined snapshot on another stream. Attention
        # consumes the prepared split buffers without modifying that snapshot.
        np.testing.assert_array_equal(case.hit.kv, original_hit)

    def test_prepared_kv_needs_no_sfa_metadata_or_layout_copies(self):
        case = self.make_case(hit_counts=(17, 0), miss_counts=(15, 32), capacity=32)
        expected = self.reference(case)
        # Refill storage is not an attention input. Removing it also catches
        # accidental fallback to splitting/sanitizing the combined KV tensor.
        del case.hit.kv
        del case.miss.kv
        hit = self.run_partition(case, case.hit)
        miss = self.run_partition(case, case.miss)
        output = self.helpers["_merge_decode_sfa_partitions_python"](hit, miss)
        np.testing.assert_allclose(output, expected, atol=8e-5, rtol=2e-3)
        for partition, (_, key, value, kwargs) in zip(
            (case.hit, case.miss), self.fia.calls
        ):
            self.assertEqual(key.data_ptr(), partition.key.data_ptr())
            self.assertEqual(
                kwargs["key_rope"].data_ptr(), partition.key_rope.data_ptr()
            )
            self.assertIs(key, value)
            self.assertTrue(np.shares_memory(key, partition.buffer))
            self.assertTrue(np.shares_memory(kwargs["key_rope"], partition.buffer))

    def test_prepared_buffers_are_recorded_on_consumer_stream(self):
        case = self.make_case(hit_counts=(3,), miss_counts=(13,))
        self.run_partition(case, case.hit)
        for prepared in (
            case.hit.buffer,
            case.hit.key,
            case.hit.key_rope,
            case.hit.true_counts,
            case.query,
            case.query_rope,
        ):
            self.assertIn(
                (prepared.data_ptr(), "hit"), self.runtime.recorded_tensors
            )

    def test_malformed_prepared_shapes_dtypes_and_strides_are_rejected(self):
        def bad_key_shape(partition):
            partition.key = partition.key[..., :256]

        def bad_rope_shape(partition):
            partition.key_rope = partition.key_rope[..., :32]

        def bad_counts_shape(partition):
            partition.true_counts = partition.true_counts.view(1, 1)

        def bad_buffer_shape(partition):
            partition.buffer = partition.buffer.view(16, 576)

        def bad_buffer_size(partition):
            partition.buffer = partition.buffer[:-1]

        def bad_key_stride(partition):
            partition.key = partition.key[..., ::-1]

        def bad_rope_stride(partition):
            partition.key_rope = partition.key_rope[..., ::-1]

        def bad_key_dtype(partition):
            partition.key = partition.key.astype(np.float32)

        def bad_rope_dtype(partition):
            partition.key_rope = partition.key_rope.astype(np.float32)

        def bad_buffer_dtype(partition):
            partition.buffer = partition.buffer.astype(np.float32)

        for mutate, message in (
            (bad_key_shape, "MLA decode"),
            (bad_rope_shape, "MLA decode"),
            (bad_counts_shape, "MLA decode"),
            (bad_buffer_shape, "contiguous NoPE/RoPE"),
            (bad_buffer_size, "contiguous NoPE/RoPE"),
            (bad_key_stride, "contiguous NoPE/RoPE"),
            (bad_rope_stride, "contiguous NoPE/RoPE"),
            (bad_key_dtype, "same dtype"),
            (bad_rope_dtype, "same dtype"),
            (bad_buffer_dtype, "same dtype"),
        ):
            with self.subTest(mutate=mutate.__name__):
                case = self.make_case(hit_counts=(3,), miss_counts=(13,))
                mutate(case.hit)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_partition(case, case.hit)
        self.assertEqual(self.fia.calls, [])

    def test_invalid_layout_and_page_fail_before_calling_fia(self):
        for page_size in (0, 8, 48, 2048):
            with self.subTest(page_size=page_size):
                case = self.make_case(hit_counts=(3,), miss_counts=(13,))
                with self.assertRaisesRegex(ValueError, "page"):
                    self.helpers["_run_decode_fia_partition"](
                        case.hit,
                        query=case.query,
                        query_rope=case.query_rope,
                        nope_head_dim=512,
                        rope_head_dim=64,
                        scale_value=case.scale,
                        page_size=page_size,
                    )
        case = self.make_case(hit_counts=(3,), miss_counts=(13,), heads=3)
        with self.assertRaisesRegex(ValueError, "MLA decode"):
            self.run_partition(case, case.hit)
        self.assertEqual(self.fia.calls, [])

    def test_unsupported_cann_contract_has_actionable_error(self):
        case = self.make_case(hit_counts=(3,), miss_counts=(13,))
        original_error = RuntimeError("unsupported mask with MLA")

        def fail(*args, **kwargs):
            raise original_error

        self.helpers["torch_npu"].npu_fused_infer_attention_score_v2.out = fail
        with self.assertRaisesRegex(RuntimeError, "FIA v2 BSND MLA") as context:
            self.run_partition(case, case.hit)
        self.assertIs(context.exception.__cause__, original_error)

    def test_malformed_out_and_lse_are_rejected(self):
        for output_index, dtype, flatten in (
            (0, np.int16, False),
            (1, np.int32, False),
            (1, None, True),
        ):
            with self.subTest(output_index=output_index, dtype=dtype, flatten=flatten):
                case = self.make_case(hit_counts=(3,), miss_counts=(13,))

                def malformed(*args, **kwargs):
                    self.fia.out(*args, **kwargs)
                    target = kwargs["out"][output_index]
                    if dtype is not None:
                        target.dtype = dtype
                    if flatten:
                        target.shape = (target.size,)

                self.helpers[
                    "torch_npu"
                ].npu_fused_infer_attention_score_v2.out = malformed
                with self.assertRaisesRegex(RuntimeError, "output.*LSE contract"):
                    self.run_partition(case, case.hit)

    def test_eager_streams_join_before_merge(self):
        case = self.make_case(hit_counts=(5,), miss_counts=(11,))
        ticket = SimpleNamespace(
            hit=case.hit, miss=case.miss, refill_done=SimpleNamespace(name="refill")
        )
        output = self.helpers["_run_split_decode_attention"](
            ticket,
            query=case.query,
            query_rope=case.query_rope,
            nope_head_dim=512,
            rope_head_dim=64,
            scale_value=case.scale,
            merge_impl="python",
            stream=self.runtime.main,
            use_fia=True,
            page_size=16,
        )
        np.testing.assert_allclose(output, self.reference(case), atol=8e-5, rtol=2e-3)
        order = [
            entry for entry in self.runtime.log if entry[0] != "tensor.record_stream"
        ]
        self.assertEqual(
            order,
            [
                ("fia", "hit"),
                ("record", "hit", "event1"),
                ("fia", "miss"),
                ("record", "miss", "event2"),
                ("wait", "main", "event1"),
                ("wait", "main", "event2"),
                ("wait", "main", "refill"),
                ("merge", "main"),
            ],
        )

    def test_graph_stream_hit_overlaps_miss_and_merge_precedes_refill_join(self):
        case = self.make_case(hit_counts=(5,), miss_counts=(11,))
        events = SimpleNamespace(
            miss_attention_done=SimpleNamespace(name="miss_attention"),
            refill_done=SimpleNamespace(name="refill"),
        )
        ticket = SimpleNamespace(hit=case.hit, miss=case.miss, events=events)

        def commit_refill(passed):
            self.assertIs(passed, ticket)
            self.runtime.log.append(("refill", self.runtime.current.name))
            self.runtime.miss.record_event(events.refill_done)

        manager = SimpleNamespace(
            merge_impl="python", commit_graph_dual_refill=commit_refill
        )
        output = self.helpers["_run_split_decode_attention_graph_dual"](
            ticket,
            manager,
            query=case.query,
            query_rope=case.query_rope,
            nope_head_dim=512,
            rope_head_dim=64,
            scale_value=case.scale,
            stream=self.runtime.main,
            use_fia=True,
            page_size=16,
        )
        np.testing.assert_allclose(output, self.reference(case), atol=8e-5, rtol=2e-3)
        order = [
            entry for entry in self.runtime.log if entry[0] != "tensor.record_stream"
        ]
        self.assertEqual(
            order,
            [
                ("fia", "main"),
                ("fia", "miss"),
                ("record", "miss", "miss_attention"),
                ("refill", "miss"),
                ("record", "miss", "refill"),
                ("wait", "main", "miss_attention"),
                ("merge", "main"),
                ("wait", "main", "refill"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
