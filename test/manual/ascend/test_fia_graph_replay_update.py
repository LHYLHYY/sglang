"""CPU-only regression checks for the FIA graph-update smoke-test patch.

Run with Python directly or pytest; no torch/torch_npu installation is needed.
The backend class is loaded from its AST to avoid importing the NPU runtime.
These mocks check routing, not device-side capture or event correctness.
"""

import ast
import logging
import os
import sys
import threading
import time
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

NPU_ROOT = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/hardware_backend/npu"
)


def load_functions(path, names, namespace, *, constants=False):
    """Execute production function bodies with explicit CPU test dependencies."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (constants and isinstance(node, (ast.Assign, ast.AnnAssign)))
    ]
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    missing = set(names) - found
    if missing:
        raise AssertionError(f"Missing production functions in {path}: {missing}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_combined_fia(torch_module, torch_npu_module):
    """Load only the production FIA helper, without importing custom KV ops."""
    path = NPU_ROOT / "sparsity_driven_kv_offload/attention.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_combined_decode_fia"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0,
            ),
            helper,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch_module, "torch_npu": torch_npu_module}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_run_combined_decode_fia"]


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


class FakeTensor:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.device = "npu"

    def view(self, *shape):
        return FakeTensor(self.data.reshape(shape))

    reshape = view

    def contiguous(self):
        return FakeTensor(np.ascontiguousarray(self.data))

    def new_zeros(self, shape):
        return FakeTensor(np.zeros(shape, dtype=self.dtype))

    def tolist(self):
        return self.data.tolist()

    def numel(self):
        return self.data.size

    def dim(self):
        return self.data.ndim

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, axis=dim))

    def transpose(self, dim0, dim1):
        return FakeTensor(np.swapaxes(self.data, dim0, dim1))

    def split(self, sizes, dim):
        return [
            FakeTensor(part)
            for part in np.split(self.data, np.cumsum(sizes)[:-1], axis=dim)
        ]

    def to(self, *, device, dtype):
        return FakeTensor(self.data.astype(dtype))

    def __getitem__(self, index):
        return FakeTensor(self.data[index])


def load_sparse_kv_forward(torch_module, torch_npu_module):
    """Load the real forward/config decisions, without the model/NPU imports."""
    namespace = {
        "torch": torch_module,
        "torch_npu": torch_npu_module,
        "os": os,
        "logger": logging.getLogger(__name__),
        "_warned_bool_env_var_keys": set(),
        "is_dsa_enable_prefill_cp": lambda: False,
    }
    sources = (
        (
            NPU_ROOT / "sparsity_driven_kv_offload/config.py",
            {"get_sparse_kv_fia_skip_kv_io"},
            "SPARSE_KV_",
        ),
        (
            NPU_ROOT / "sparsity_driven_kv_offload/attention.py",
            {
                "_get_sparse_kv_manager",
                "_expand_dsa_sparse_indices",
                "_select_split_decode_mode",
                "forward_sparsity_driven_kv_offload",
            },
            "_SPLIT_MODE_",
        ),
        (NPU_ROOT.parents[1] / "utils/common.py", {"get_bool_env_var"}, None),
    )
    for path, functions, prefix in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in functions)
            or (
                prefix is not None
                and isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith(prefix)
            )
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0,
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_backend_class():
    path = NPU_ROOT / "graph_runner/npu_cudagraph_backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUCudaGraphBackend"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0,
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {
        "BaseCudaGraphBackend": object,
        "contextmanager": contextmanager,
        "empty_context": nullcontext,
        "logger": logging.getLogger(__name__),
        "os": os,
        "threading": threading,
        "time": time,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["NPUCudaGraphBackend"], namespace


class TestFIAGraphReplayUpdate(unittest.TestCase):
    def setUp(self):
        cls, self.namespace = load_backend_class()
        self.backend = cls.__new__(cls)
        self.backend._graphs = {}
        self.backend._outputs = {}
        self.backend._fia_update_tasks = {}
        self.backend._pool = None
        self.backend._capture_stream = None
        self.backend._memory_saver_adapter = None
        self.backend._enable_torch_compile = False
        self.backend._device_module = Mock()
        self.backend._device_id = 0
        self.backend._tp_group = Mock()
        self.backend._tp_group.rank_in_group = 3
        self.backend._graph_debug = False
        self.backend._debug_replay_id = 0

    def capture(self, key, op_names=(), has_dispatch_mode=True):
        graph = SimpleNamespace(replay=Mock(), update=Mock())
        if has_dispatch_mode:
            graph.graph_dispatch_mode = SimpleNamespace(
                graph_dispatch_records=[
                    SimpleNamespace(op_cache_entry=SimpleNamespace(__name__=name))
                    for name in op_names
                ]
            )
        self.namespace["torch"] = SimpleNamespace(
            Tensor=type("Tensor", (), {}),
            npu=SimpleNamespace(
                NPUGraph=lambda: graph, graph=lambda *args, **kwargs: nullcontext(),
            ),
        )
        output = object()
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            self.backend.capture_one(key, lambda: output)
        return graph, output

    def test_fia_and_v2_overloads_preserve_captured_lengths(self):
        for base in (
            "npu_fused_infer_attention_score",
            "npu_fused_infer_attention_score_v2",
        ):
            for suffix in ("", ".default", ".out"):
                with self.subTest(op=base + suffix):
                    graph, output = self.capture(1, [base + suffix])
                    self.assertIs(self.backend.replay(1, None), output)
                    graph.update.assert_called_once_with(cpu_update_input=[{}])
                    graph.replay.assert_called_once_with()
                    self.assertEqual(self.backend._fia_update_tasks[1], 1)

    def test_update_runs_on_every_replay(self):
        graph, output = self.capture(1, ["npu_fused_infer_attention_score.out"])
        ready = threading.Event()
        graph.update.side_effect = lambda **kwargs: ready.set()

        def replay():
            self.assertTrue(ready.wait(timeout=2), "FIA update was not submitted")
            ready.clear()

        graph.replay.side_effect = replay
        for _ in range(2):
            self.assertIs(self.backend.replay(1, None), output)
        self.assertEqual(graph.update.call_count, 2)

    def test_sfa_and_empty_graphs_keep_direct_replay(self):
        for names in ([], ["npu_sparse_flash_attention.default"], ["other.out"]):
            with self.subTest(ops=names):
                graph, output = self.capture(1, names)
                self.assertIs(self.backend.replay(1, None), output)
                graph.update.assert_not_called()
                graph.replay.assert_called_once_with()

    def test_missing_dispatch_metadata_keeps_direct_replay(self):
        graph, _ = self.capture(1, has_dispatch_mode=False)
        self.backend.replay(1, None)
        graph.update.assert_not_called()

    def test_batch_buckets_are_independent_and_recapture_resets_detection(self):
        fia_graph, _ = self.capture(1, ["npu_fused_infer_attention_score.out"])
        sfa_graph, _ = self.capture(2, ["npu_sparse_flash_attention.default"])
        self.backend.replay(2, None)
        sfa_graph.update.assert_not_called()
        self.backend.replay(1, None)
        fia_graph.update.assert_called_once_with(cpu_update_input=[{}])
        replacement, _ = self.capture(1)
        self.backend.replay(1, None)
        replacement.update.assert_not_called()

    def test_normal_mla_explicit_length_update_is_unchanged(self):
        graph, output = self.capture(1, ["npu_fused_infer_attention_score.out"])
        self.assertIs(
            self.backend.replay_with_input_update(
                1, seq_lens=[32768], attr_name="actual_seq_lengths_kv"
            ),
            output,
        )
        graph.update.assert_called_once_with(
            cpu_update_input=[{"actual_seq_lengths_kv": [32768]}]
        )
        graph.replay.assert_called_once_with()

    def test_cleanup_clears_detection(self):
        self.capture(1, ["npu_fused_infer_attention_score.out"])
        self.backend.cleanup()
        self.assertEqual(self.backend._fia_update_tasks, {})

    def test_debug_disabled_does_not_log_or_synchronize_replay(self):
        graph, output = self.capture(1, ["npu_fused_infer_attention_score.out"])
        self.backend._device_module.reset_mock()
        with patch.dict(self.namespace, {"logger": Mock()}) as namespace:
            self.assertIs(self.backend.replay(1, None), output)
            namespace["logger"].info.assert_not_called()
        self.backend._device_module.synchronize.assert_not_called()
        self.assertEqual(self.backend._debug_replay_id, 0)
        graph.update.assert_called_once_with(cpu_update_input=[{}])

    def test_model_forward_debug_uses_actual_module_imports(self):
        path = NPU_ROOT.parents[1] / "model_executor/model_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
        )
        log_method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_log_npu_graph_forward"
        )
        # Use imports present in the production module, not an injected `os`.
        # This must fail if the logging helper's module forgets that import.
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.Import)
            and all(alias.name in {"os", "logging"} for alias in node.names)
        ]
        module = ast.Module(body=imports + [log_method], type_ignores=[])
        log = Mock()
        namespace = {"logger": log}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        log_forward = namespace["_log_npu_graph_forward"]

        runner = SimpleNamespace(_npu_graph_debug=False)
        log_forward(runner, "forward.enter", object())
        log.info.assert_not_called()

        runner = SimpleNamespace(
            _npu_graph_debug=True,
            gpu_id=5,
            ps=SimpleNamespace(tp_rank=5),
            forward_pass_id=1,
            is_draft_worker=False,
        )
        for mode, decode_graph in (("EXTEND", False), ("DECODE", True)):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(name=mode), batch_size=1
                )
                log_forward(runner, "forward.route", batch, decode_graph)
                fmt, *args = log.info.call_args.args
                message = fmt % tuple(args)
                self.assertIn(f"pid={os.getpid()} ", message)
                self.assertIn("tp_rank=5 ", message)
                self.assertIn(f"mode={mode} ", message)
                self.assertIn(f"decode_graph={decode_graph}", message)

    def test_debug_records_update_replay_and_join_with_shared_id(self):
        graph, output = self.capture(1, ["npu_fused_infer_attention_score.out"])
        self.backend._graph_debug = True
        self.backend._device_module.reset_mock()
        with self.assertLogs(self.namespace["logger"], level="INFO") as logs:
            self.assertIs(self.backend.replay(1, None), output)
        stages = [message.split("stage=")[1].split()[0] for message in logs.output]
        for stage in (
            "replay.route",
            "update.prepare",
            "update.thread.start",
            "update.thread.enter",
            "update.set_device.begin",
            "update.set_device.returned",
            "update.begin",
            "update.returned",
            "replay.begin",
            "replay.returned",
            "update.join.begin",
            "update.join.returned",
        ):
            self.assertIn(stage, stages)
        for begin, end in (
            ("update.begin", "update.returned"),
            ("replay.begin", "replay.returned"),
            ("update.returned", "update.join.returned"),
            ("replay.returned", "update.join.begin"),
        ):
            self.assertLess(stages.index(begin), stages.index(end))
        for message in logs.output:
            self.assertIn("replay=1 ", message)
            self.assertIn("tp_rank=3 ", message)
            self.assertIn("shape=1 ", message)
        self.assertTrue(any("input_keys=[[]]" in message for message in logs.output))
        self.backend._device_module.synchronize.assert_not_called()
        graph.update.assert_called_once_with(cpu_update_input=[{}])

    def test_update_exception_is_logged_with_traceback(self):
        graph, _ = self.capture(1, ["npu_fused_infer_attention_score.out"])
        graph.update.side_effect = RuntimeError("mock update failure")
        with self.assertLogs(self.namespace["logger"], level="ERROR") as logs:
            with patch.object(threading, "excepthook") as thread_error:
                self.backend.replay(1, None)
        self.assertIn("stage=update.error", logs.output[0])
        self.assertIn("tp_rank=3", logs.output[0])
        self.assertIn("Traceback", logs.output[0])
        self.assertIn("mock update failure", logs.output[0])
        # Logging must not silently swallow the original thread exception.
        thread_error.assert_called_once()

    def test_replay_exception_is_logged_and_reraised(self):
        graph, _ = self.capture(1, ["npu_fused_infer_attention_score.out"])
        updated = threading.Event()
        graph.update.side_effect = lambda **kwargs: updated.set()

        def replay():
            self.assertTrue(updated.wait(timeout=2))
            raise RuntimeError("mock replay failure")

        graph.replay.side_effect = replay
        with self.assertLogs(self.namespace["logger"], level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "mock replay failure"):
                self.backend.replay(1, None)
        self.assertIn("stage=replay.error", logs.output[0])
        self.assertIn("Traceback", logs.output[0])

    def make_runner(
        self, *, manager=None, bs=1, raw_bs=1, is_dsa=True, native=False, mode="DECODE",
    ):
        path = NPU_ROOT / "graph_runner/npu_graph_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        execute = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "execute"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0,
                ),
                execute,
            ],
            type_ignores=[],
        )
        output = SimpleNamespace(
            next_token_logits=[1], full_logits=None, hidden_states=None
        )
        self.backend._outputs[bs] = output
        namespace = {
            "is_deepseek_dsa": lambda config: is_dsa,
            "is_deepseek_v4": lambda config: False,
            "LogitsProcessorOutput": SimpleNamespace,
            "SPARSE_KV_ATTN_IMPL_COMBINED": "combined",
            "SPARSE_KV_ATTN_IMPL_NATIVE_FIA": "native_fia",
            "SPARSE_KV_ATTN_IMPL_SPLIT_EAGER": "split_eager",
            "SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA": "split_graph_dual_fia",
        }
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        runner = SimpleNamespace(
            backend=self.backend,
            buffers=SimpleNamespace(seq_lens=MagicMock()),
            attn_backend=SimpleNamespace(
                sparse_kv_manager=manager, dsa_fia_native=native
            ),
            load_batch=Mock(),
            _make_graph_key=lambda bs: bs,
            _get_update_attr_name=lambda: "actual_seq_lengths_kv",
            _get_update_attr_type=lambda: [],
            bs=bs,
            raw_bs=raw_bs,
            raw_num_token=raw_bs,
            is_dllm=False,
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=object())
            ),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                name=mode,
                is_decode=lambda: mode == "DECODE",
                is_idle=lambda: mode == "IDLE",
                is_target_verify=lambda: False,
            ),
            batch_size=raw_bs,
            seq_lens=Mock(),
            seq_lens_cpu=None,
            needs_forward_metadata_init=lambda: True,
        )
        return namespace["execute"], runner, batch

    def test_runner_traces_metadata_before_fia_replay(self):
        graph, _ = self.capture(1, ["npu_fused_infer_attention_score.out"])
        self.backend._graph_debug = True
        execute, runner, batch = self.make_runner()
        with self.assertLogs(self.namespace["logger"], level="INFO") as logs:
            result = execute(runner, batch)
        stages = [message.split("stage=")[1].split()[0] for message in logs.output]
        self.assertEqual(
            stages[:4],
            [
                "execute.enter",
                "load_batch.begin",
                "load_batch.returned",
                "execute.graph_selected",
            ],
        )
        self.assertEqual(stages[-1], "execute.backend_returned")
        self.assertEqual(result.next_token_logits, [1])
        self.assertEqual(self.backend._debug_replay_id, 1)
        runner.load_batch.assert_called_once_with(batch, None)
        graph.update.assert_called_once_with(cpu_update_input=[{}])

    def test_combined_runner_updates_selected_capacity_not_context_length(self):
        for impl in ("combined", "split_eager"):
            for raw_bs in (1, 2):
                with self.subTest(impl=impl, raw_bs=raw_bs):
                    graph, _ = self.capture(2, ["npu_fused_infer_attention_score.out"])
                    execute, runner, batch = self.make_runner(
                        manager=SimpleNamespace(
                            attn_impl=impl, sparse_context_len=2048
                        ),
                        bs=2,
                        raw_bs=raw_bs,
                    )
                    batch.seq_lens.cpu.side_effect = AssertionError(
                        "Must not use full context lengths for selected KV"
                    )
                    graph_lens = runner.buffers.seq_lens.__getitem__.return_value
                    graph_lens.cpu.return_value.tolist.return_value = [7, 10001]
                    execute(runner, batch)
                    graph_lens.cpu.assert_called_once_with()
                    graph.update.assert_called_once_with(
                        cpu_update_input=[
                            {
                                "actual_seq_lengths_kv": [2048] * raw_bs
                                + [0] * (2 - raw_bs)
                            }
                        ]
                    )
                    graph.replay.assert_called_once_with()

    def test_split_sfa_runner_does_not_get_fia_length_updates(self):
        for impl in (
            "split_graph",
            "split_graph_dual",
            "split_graph_dual_v2",
            "pa_graph",
        ):
            for mode in ("DECODE", "IDLE"):
                with self.subTest(impl=impl, mode=mode):
                    graph, _ = self.capture(1, ["npu_sparse_flash_attention.default"])
                    execute, runner, batch = self.make_runner(
                        manager=SimpleNamespace(
                            attn_impl=impl, sparse_context_len=2048
                        ),
                        mode=mode,
                        raw_bs=1 if mode == "DECODE" else 0,
                    )
                    execute(runner, batch)
                    runner.buffers.seq_lens.__getitem__.assert_not_called()
                    batch.seq_lens.cpu.assert_not_called()
                    graph.update.assert_not_called()
                    graph.replay.assert_called_once_with()

    def test_dual_fia_updates_both_partitions_in_every_layer_and_masks_idle(self):
        # NPUGraph.update broadcasts a one-element input list to all captured
        # dispatch records. Both partitions use the same physical capacity;
        # their device masks retain the different runtime hit/miss lengths.
        for layers in (1, 3):
            for mode, raw_bs in (
                ("DECODE", 1), ("DECODE", 2), ("IDLE", 0), ("IDLE", 2)
            ):
                with self.subTest(layers=layers, mode=mode, raw_bs=raw_bs):
                    graph, _ = self.capture(
                        4, ["npu_fused_infer_attention_score_v2.out"] * (2 * layers)
                    )
                    self.backend._graph_debug = True
                    execute, runner, batch = self.make_runner(
                        manager=SimpleNamespace(
                            attn_impl="split_graph_dual_fia", sparse_context_len=2048
                        ),
                        mode=mode,
                        bs=4,
                        raw_bs=raw_bs,
                    )
                    batch.seq_lens_cpu = FakeTensor(np.array([1, 2, 3, 4]))
                    batch.seq_lens.cpu.side_effect = AssertionError(
                        "Dual FIA must read the nonempty loaded graph buffer"
                    )
                    graph_lens = runner.buffers.seq_lens.__getitem__.return_value
                    expected_lengths = (
                        [2048] * raw_bs + [0] * (4 - raw_bs)
                        if mode == "DECODE"
                        else [0] * 4
                    )
                    for replay_id, length in enumerate((7, 10001), start=1):
                        graph_lens.cpu.return_value.tolist.return_value = [
                            length, 2048, 99, 98
                        ]
                        with self.assertLogs(
                            self.namespace["logger"], level="INFO"
                        ) as logs:
                            execute(runner, batch)
                        self.assertTrue(
                            any("route=sparse_fia_dual" in line for line in logs.output)
                        )
                        runner.buffers.seq_lens.__getitem__.assert_called_with(
                            slice(None, 4)
                        )
                        self.assertEqual(graph_lens.cpu.call_count, replay_id)
                        self.assertEqual(graph.update.call_count, replay_id)
                        self.assertEqual(graph.replay.call_count, replay_id)
                        # FIA v2's update attribute differs from FIA v1.
                        graph.update.assert_called_with(
                            cpu_update_input=[{"actual_seq_kvlen": expected_lengths}]
                        )
                    self.assertEqual(self.backend._fia_update_tasks[4], 2 * layers)
                    batch.seq_lens.cpu.assert_not_called()
                    np.testing.assert_array_equal(batch.seq_lens_cpu.data, [1, 2, 3, 4])

    def test_skip_kv_io_runner_keeps_fia_update_and_replay(self):
        graph, _ = self.capture(2, ["npu_fused_infer_attention_score.out"])
        execute, runner, batch = self.make_runner(
            manager=SimpleNamespace(
                attn_impl="combined", sparse_context_len=2048, fia_skip_kv_io=True
            ),
            bs=2,
            raw_bs=1,
        )
        execute(runner, batch)
        graph_lens = runner.buffers.seq_lens.__getitem__.return_value
        graph_lens.cpu.assert_called_once_with()
        graph.update.assert_called_once_with(
            cpu_update_input=[{"actual_seq_lengths_kv": [2048, 0]}]
        )
        graph.replay.assert_called_once_with()

    def test_normal_mla_runner_still_updates_full_context_length(self):
        graph, _ = self.capture(2, ["npu_fused_infer_attention_score.out"])
        execute, runner, batch = self.make_runner(is_dsa=False, bs=2, raw_bs=1)
        batch.seq_lens.cpu.return_value.tolist.return_value = [32768]
        execute(runner, batch)
        graph.update.assert_called_once_with(
            cpu_update_input=[{"actual_seq_lengths_kv": [32768, 0]}]
        )

    def test_native_dsa_runner_updates_full_context_lengths_and_padding(self):
        for lengths in ([10001], [10001, 32768]):
            with self.subTest(lengths=lengths):
                graph, _ = self.capture(4, ["npu_fused_infer_attention_score.out"])
                execute, runner, batch = self.make_runner(
                    native=True, bs=4, raw_bs=len(lengths)
                )
                batch.seq_lens.cpu.return_value.tolist.return_value = lengths
                execute(runner, batch)
                graph.update.assert_called_once_with(
                    cpu_update_input=[
                        {"actual_seq_lengths_kv": lengths + [0] * (4 - len(lengths))}
                    ]
                )
                graph.replay.assert_called_once_with()

    def test_native_dsa_idle_runner_updates_zero_lengths(self):
        graph, _ = self.capture(2, ["npu_fused_infer_attention_score.out"])
        execute, runner, batch = self.make_runner(
            native=True, mode="IDLE", bs=2, raw_bs=0
        )
        batch.seq_lens.cpu.return_value.tolist.return_value = []
        execute(runner, batch)
        graph.update.assert_called_once_with(
            cpu_update_input=[{"actual_seq_lengths_kv": [0, 0]}]
        )
        graph.replay.assert_called_once_with()

    def test_native_dsa_runner_logs_native_route(self):
        self.capture(2, ["npu_fused_infer_attention_score.out"])
        self.backend._graph_debug = True
        execute, runner, batch = self.make_runner(native=True, bs=2, raw_bs=1)
        batch.seq_lens.cpu.return_value.tolist.return_value = [10001]
        with self.assertLogs(self.namespace["logger"], level="INFO") as logs:
            execute(runner, batch)
        self.assertTrue(any("route=native_mla_fia" in line for line in logs.output))

    def test_fia_offload_reads_loaded_graph_lengths_even_when_cpu_metadata_exists(
        self,
    ):
        for impl in ("native_fia", "combined", "split_eager"):
            with self.subTest(impl=impl):
                graph, _ = self.capture(4, ["npu_fused_infer_attention_score.out"])
                execute, runner, batch = self.make_runner(
                    manager=SimpleNamespace(attn_impl=impl, sparse_context_len=2048),
                    bs=4,
                    raw_bs=3,
                )
                # A cached host copy must not remove the device-to-host ordering
                # point inherited from the working stock MLA replay sequence.
                batch.seq_lens_cpu = FakeTensor(np.array([1, 2, 3, 4]))
                batch.seq_lens.cpu.side_effect = AssertionError(
                    "Read the loaded graph buffer"
                )
                graph_lens = runner.buffers.seq_lens.__getitem__.return_value
                graph_lens.cpu.return_value.tolist.return_value = [7, 2048, 10001, 9999]
                execute(runner, batch)
                runner.buffers.seq_lens.__getitem__.assert_called_once_with(
                    slice(None, 4)
                )
                graph_lens.cpu.assert_called_once_with()
                graph.update.assert_called_once_with(
                    cpu_update_input=[
                        {
                            "actual_seq_lengths_kv": [7, 2048, 2048, 0]
                            if impl == "native_fia"
                            else [2048, 2048, 2048, 0]
                        }
                    ]
                )
                graph.replay.assert_called_once_with()
                np.testing.assert_array_equal(batch.seq_lens_cpu.data, [1, 2, 3, 4])

    def test_native_offload_loaded_graph_lengths_are_clamped_before_replay(self):
        graph, _ = self.capture(4, ["npu_fused_infer_attention_score.out"])
        execute, runner, batch = self.make_runner(
            manager=SimpleNamespace(attn_impl="native_fia", sparse_context_len=128),
            bs=4,
            raw_bs=3,
        )
        graph_lens = runner.buffers.seq_lens.__getitem__.return_value
        graph_lens.cpu.return_value.tolist.return_value = [-1, 1, 10001, 9999]
        execute(runner, batch)
        graph_lens.cpu.assert_called_once_with()
        graph.update.assert_called_once_with(
            cpu_update_input=[{"actual_seq_lengths_kv": [0, 1, 128, 0]}]
        )

    def test_fia_offload_idle_reads_nonempty_graph_buffer_then_updates_zeros(self):
        for impl in ("native_fia", "combined", "split_eager"):
            for raw_bs in (0, 2):
                with self.subTest(impl=impl, raw_bs=raw_bs):
                    graph, _ = self.capture(4, ["npu_fused_infer_attention_score.out"])
                    execute, runner, batch = self.make_runner(
                        manager=SimpleNamespace(
                            attn_impl=impl, sparse_context_len=2048
                        ),
                        mode="IDLE",
                        bs=4,
                        raw_bs=raw_bs,
                    )
                    batch.seq_lens_cpu = Mock(
                        tolist=Mock(
                            side_effect=AssertionError("IDLE must not read stale lengths")
                        )
                    )
                    batch.seq_lens.cpu.return_value.tolist.return_value = []
                    graph_lens = runner.buffers.seq_lens.__getitem__.return_value
                    graph_lens.cpu.return_value.tolist.return_value = [
                        10001, 2048, 99, 98
                    ]
                    execute(runner, batch)
                    runner.buffers.seq_lens.__getitem__.assert_called_once_with(
                        slice(None, 4)
                    )
                    graph_lens.cpu.assert_called_once_with()
                    batch.seq_lens.cpu.assert_not_called()
                    graph.update.assert_called_once_with(
                        cpu_update_input=[{"actual_seq_lengths_kv": [0, 0, 0, 0]}]
                    )
                    graph.replay.assert_called_once_with()

    def test_fia_offload_waits_for_device_snapshot_before_update_and_replay(self):
        for impl, skip_kv_io in (
            ("native_fia", False),
            ("combined", False),
            ("split_eager", False),
            ("split_graph_dual_fia", False),
            ("combined", True),
        ):
            for mode in ("DECODE", "IDLE"):
                with self.subTest(impl=impl, mode=mode, skip_kv_io=skip_kv_io):
                    dual_fia = impl == "split_graph_dual_fia"
                    graph, _ = self.capture(
                        2,
                        ["npu_fused_infer_attention_score_v2.out"] * 2
                        if dual_fia
                        else ["npu_fused_infer_attention_score.out"],
                    )
                    self.backend._graph_debug = True
                    execute, runner, batch = self.make_runner(
                        manager=SimpleNamespace(
                            attn_impl=impl,
                            sparse_context_len=2048,
                            fia_skip_kv_io=skip_kv_io,
                        ),
                        mode=mode,
                        bs=2,
                        raw_bs=1 if mode == "DECODE" else 0,
                    )
                    entered, release = threading.Event(), threading.Event()
                    errors = []

                    def device_snapshot():
                        entered.set()
                        if not release.wait(timeout=2):
                            raise RuntimeError(
                                "Test did not release the device snapshot"
                            )
                        return FakeTensor(np.array([10001, 999]))

                    def run():
                        try:
                            execute(runner, batch)
                        except BaseException as exc:
                            errors.append(exc)

                    graph_lens = runner.buffers.seq_lens.__getitem__.return_value
                    graph_lens.cpu.side_effect = device_snapshot
                    batch.seq_lens.cpu.side_effect = AssertionError(
                        "Read the loaded graph buffer"
                    )
                    batch.seq_lens_cpu = FakeTensor(np.array([10001]))
                    worker = threading.Thread(target=run, daemon=True)
                    with self.assertLogs(self.namespace["logger"], level="INFO") as logs:
                        worker.start()
                        try:
                            self.assertTrue(
                                entered.wait(timeout=2), "Device snapshot was skipped"
                            )
                            graph.update.assert_not_called()
                            graph.replay.assert_not_called()
                        finally:
                            release.set()
                            worker.join(timeout=2)
                        self.assertFalse(
                            worker.is_alive(), "Replay did not finish after snapshot"
                        )
                    self.assertEqual(errors, [])
                    attr_name = (
                        "actual_seq_kvlen" if dual_fia else "actual_seq_lengths_kv"
                    )
                    graph.update.assert_called_once_with(
                        cpu_update_input=[
                            {
                                attr_name: [2048, 0]
                                if mode == "DECODE"
                                else [0, 0]
                            }
                        ]
                    )
                    graph.replay.assert_called_once_with()
                    stages = [
                        message.split("stage=")[1].split()[0] for message in logs.output
                    ]
                    self.assertLess(
                        stages.index("seq_lens.cpu.begin"),
                        stages.index("seq_lens.cpu.returned"),
                    )
                    for stage in ("update.begin", "replay.begin"):
                        self.assertLess(
                            stages.index("seq_lens.cpu.returned"), stages.index(stage)
                        )

    def test_fia_offload_takes_a_fresh_device_snapshot_on_every_replay(self):
        for impl in ("native_fia", "combined", "split_eager"):
            with self.subTest(impl=impl):
                graph, _ = self.capture(2, ["npu_fused_infer_attention_score.out"])
                execute, runner, batch = self.make_runner(
                    manager=SimpleNamespace(attn_impl=impl, sparse_context_len=2048),
                    bs=2,
                    raw_bs=1,
                )
                batch.seq_lens_cpu = FakeTensor(np.array([99]))
                graph_lens = runner.buffers.seq_lens.__getitem__.return_value
                for index, length in enumerate((7, 8, 10001), start=1):
                    graph_lens.cpu.return_value.tolist.return_value = [length, 999]
                    execute(runner, batch)
                    self.assertEqual(graph_lens.cpu.call_count, index)
                    self.assertEqual(graph.update.call_count, index)
                    self.assertEqual(graph.replay.call_count, index)
                    selected = min(length, 2048) if impl == "native_fia" else 2048
                    graph.update.assert_called_with(
                        cpu_update_input=[{"actual_seq_lengths_kv": [selected, 0]}]
                    )

    def test_combined_fia_matches_mla_out_and_page_mapping(self):
        torch_mock = SimpleNamespace(
            int32=np.int32,
            arange=lambda n, **kwargs: FakeTensor(np.arange(n, dtype=kwargs["dtype"])),
            empty_like=lambda x, **kwargs: FakeTensor(np.empty_like(x.data)),
            empty=lambda n, **kwargs: FakeTensor(np.empty(n, dtype=kwargs["dtype"])),
        )
        workspace = object()
        fia = Mock(side_effect=AssertionError("Use the explicit .out overload"))
        npu_mock = SimpleNamespace(
            _npu_fused_infer_attention_score_get_max_workspace=Mock(
                return_value=workspace
            ),
            npu_fused_infer_attention_score=fia,
        )
        call = load_combined_fia(torch_mock, npu_mock)
        query = FakeTensor(np.zeros((2, 1, 16, 512), dtype=np.float32))
        query_rope = FakeTensor(np.zeros((2, 1, 16, 64), dtype=np.float32))
        key = FakeTensor(np.zeros((2, 2048, 1, 512), dtype=np.float32))
        key_rope = FakeTensor(np.zeros((2, 2048, 1, 64), dtype=np.float32))
        output = call(
            query, query_rope, key, key_rope, page_size=128, scale_value=0.125
        )
        fia.assert_not_called()
        args, kwargs = fia.out.call_args
        self.assertIs(args[0], query)
        self.assertIs(args[1], args[2])
        self.assertEqual(args[1].shape, (32, 128, 512))
        self.assertEqual(kwargs["key_rope"].shape, (32, 128, 64))
        self.assertTrue(np.shares_memory(args[1].data, key.data))
        self.assertTrue(np.shares_memory(kwargs["key_rope"].data, key_rope.data))
        np.testing.assert_array_equal(
            kwargs["block_table"].data, np.arange(32, dtype=np.int32).reshape(2, 16)
        )
        self.assertEqual(kwargs["block_size"], 128)
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [2048, 2048])
        self.assertEqual(kwargs["input_layout"], "BSND")
        self.assertEqual(kwargs["sparse_mode"], 0)
        self.assertEqual(kwargs["antiquant_mode"], 0)
        self.assertIsNone(kwargs["antiquant_scale"])
        self.assertIs(kwargs["workspace"], workspace)
        self.assertIs(kwargs["out"][0], output)
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(kwargs["out"][1].shape, (1,))
        (
            ws_args,
            ws_kwargs,
        ) = npu_mock._npu_fused_infer_attention_score_get_max_workspace.call_args
        self.assertEqual(ws_args, args)
        self.assertEqual(
            ws_kwargs,
            {k: v for k, v in kwargs.items() if k not in ("workspace", "out")},
        )
        for page_size in (0, 3, 4096):
            with self.subTest(page_size=page_size):
                with self.assertRaisesRegex(ValueError, "must be divisible"):
                    call(
                        query,
                        query_rope,
                        key,
                        key_rope,
                        page_size=page_size,
                        scale_value=0.125,
                    )


class TestSharedMLAFIA(unittest.TestCase):
    def test_stock_and_offload_share_workspace_out_padding_and_crop(self):
        for heads, padded_heads, graph_padding in (
            (2, 2, False),
            (3, 4, False),
            (3, 4, True),
        ):
            with self.subTest(heads=heads, graph_padding=graph_padding):
                torch_mock = SimpleNamespace(
                    cat=lambda tensors, dim: FakeTensor(
                        np.concatenate([tensor.data for tensor in tensors], axis=dim)
                    ),
                    empty_like=lambda tensor, **kw: FakeTensor(
                        np.empty_like(tensor.data)
                    ),
                    empty=lambda shape, **kw: FakeTensor(
                        np.empty(shape, dtype=kw["dtype"])
                    ),
                )
                workspace = object()
                fia = Mock(side_effect=AssertionError("Use the explicit .out overload"))

                def fill_output(*args, **kwargs):
                    output = kwargs["out"][0].data
                    output[...] = np.arange(output.shape[2])[None, None, :, None]

                fia.out.side_effect = fill_output
                npu_mock = SimpleNamespace(
                    _npu_fused_infer_attention_score_get_max_workspace=Mock(
                        return_value=workspace
                    ),
                    npu_fused_infer_attention_score=fia,
                )
                forward = load_method(
                    NPU_ROOT / "attention/ascend_backend.py",
                    "AscendAttnBackend",
                    "forward_mla_fia",
                    {"torch": torch_mock, "torch_npu": npu_mock},
                )
                tensor = lambda shape: FakeTensor(np.ones(shape, dtype=np.float32))
                metadata = SimpleNamespace()
                if graph_padding:
                    metadata.nope_padding = FakeTensor(np.zeros((2, 1, 1, 512)))
                    metadata.rope_padding = FakeTensor(np.zeros((2, 1, 1, 64)))
                backend = SimpleNamespace(
                    forward_metadata=metadata,
                    q_head_num_padding=padded_heads,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    page_size=128,
                )
                layer = SimpleNamespace(
                    tp_q_head_num=heads, tp_k_head_num=1, scaling=0.125
                )
                key, rope = tensor((4, 128, 512)), tensor((4, 128, 64))
                page_table = FakeTensor(np.arange(4, dtype=np.int32).reshape(2, 2))
                lengths = [7, 256]
                result = forward(
                    backend,
                    tensor((2, heads, 512)),
                    tensor((2, heads, 64)),
                    layer,
                    key,
                    rope,
                    page_table,
                    lengths,
                )
                fia.assert_not_called()
                args, kwargs = fia.out.call_args
                self.assertIs(args[1], key)
                self.assertIs(args[2], key)
                self.assertIs(kwargs["key_rope"], rope)
                self.assertIs(kwargs["block_table"], page_table)
                self.assertIs(kwargs["actual_seq_lengths_kv"], lengths)
                self.assertEqual(args[0].shape, (2, 1, padded_heads, 512))
                self.assertEqual(kwargs["query_rope"].shape, (2, 1, padded_heads, 64))
                self.assertEqual(kwargs["num_heads"], padded_heads)
                self.assertEqual(kwargs["num_key_value_heads"], 1)
                self.assertEqual(kwargs["block_size"], 128)
                self.assertEqual(kwargs["input_layout"], "BSND")
                self.assertEqual(kwargs["scale"], 0.125)
                self.assertEqual(kwargs["sparse_mode"], 0)
                self.assertEqual(kwargs["antiquant_mode"], 0)
                self.assertIsNone(kwargs["antiquant_scale"])
                self.assertIs(kwargs["workspace"], workspace)
                self.assertEqual(kwargs["out"][1].shape, (1,))
                (
                    ws_args,
                    ws_kwargs,
                ) = (
                    npu_mock._npu_fused_infer_attention_score_get_max_workspace.call_args
                )
                self.assertEqual(ws_args, args)
                self.assertEqual(
                    ws_kwargs,
                    {
                        key: value
                        for key, value in kwargs.items()
                        if key not in ("workspace", "out")
                    },
                )
                if padded_heads > heads:
                    self.assertTrue(np.all(args[0].data[:, :, heads:, :] == 0))
                    self.assertTrue(
                        np.all(kwargs["query_rope"].data[:, :, heads:, :] == 0)
                    )
                self.assertEqual(result.shape, (2, heads * 512))
                np.testing.assert_array_equal(
                    result.data.reshape(2, heads, 512),
                    np.broadcast_to(np.arange(heads)[None, :, None], (2, heads, 512)),
                )


class TestDSAFIANativeConfig(unittest.TestCase):
    def setUp(self):
        self.namespace = {
            "os": os,
            "logger": logging.getLogger(__name__),
            "_warned_bool_env_var_keys": set(),
            "is_npu": Mock(return_value=True),
            "is_deepseek_dsa": lambda config: config.index_topk is not None,
        }
        load_functions(
            NPU_ROOT.parents[1] / "utils/common.py",
            {"get_bool_env_var"},
            self.namespace,
        )
        load_functions(
            NPU_ROOT / "sparsity_driven_kv_offload/config.py",
            {
                "is_dsa_fia_native_requested",
                "is_dsa_fia_native_enabled",
                "get_dsa_fia_native_cell_size",
                "is_sparsity_driven_kv_offload_requested",
            },
            self.namespace,
            constants=True,
        )
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def config_args(self):
        return dict(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV32ForCausalLM"], index_topk=2048,
                ),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
            ),
            server_args=SimpleNamespace(
                attention_backend="ascend",
                kv_cache_dtype="auto",
                enable_prefill_cp=False,
                enable_dsa_prefill_context_parallel=False,
                attn_cp_size=1,
                dcp_size=1,
                speculative_algorithm=None,
                enable_torch_compile=False,
            ),
            use_mla_backend=True,
        )

    def test_native_is_opt_in_and_never_overrides_offload_on_non_npu(self):
        requested = self.namespace["is_dsa_fia_native_requested"]
        offload = self.namespace["is_sparsity_driven_kv_offload_requested"]
        self.assertFalse(requested())
        self.assertFalse(offload())
        os.environ["SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD"] = "1"
        self.assertTrue(offload())
        for value in ("1", "true"):
            with self.subTest(value=value):
                os.environ["SGLANG_NPU_DSA_FIA_NATIVE"] = value
                self.assertTrue(requested())
                self.assertFalse(offload())
        self.namespace["is_npu"].return_value = False
        self.assertFalse(requested())
        self.assertTrue(offload())
        self.namespace["is_npu"].return_value = True
        os.environ["SGLANG_NPU_DSA_FIA_NATIVE"] = "0"
        self.assertFalse(requested())
        self.assertTrue(offload())

    def test_disabled_native_does_not_change_pool_sizing(self):
        args = self.config_args()
        self.assertFalse(self.namespace["is_dsa_fia_native_enabled"](**args))
        self.assertIsNone(
            self.namespace["get_dsa_fia_native_cell_size"](
                **args, num_layers=61, element_size=2
            )
        )

    def test_native_full_cache_sizing_includes_unused_index_pool(self):
        os.environ["SGLANG_NPU_DSA_FIA_NATIVE"] = "1"
        args = self.config_args()
        for architecture in ("DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"):
            with self.subTest(architecture=architecture):
                args["model_config"].hf_config.architectures = [architecture]
                self.assertTrue(self.namespace["is_dsa_fia_native_enabled"](**args))
                self.assertEqual(
                    self.namespace["get_dsa_fia_native_cell_size"](
                        **args, num_layers=61, element_size=2
                    ),
                    (512 + 64 + 128) * 61 * 2,
                )

    def test_native_rejects_non_dsa_wrong_model_and_backend(self):
        os.environ["SGLANG_NPU_DSA_FIA_NATIVE"] = "1"
        changes = (
            (
                "non_dsa",
                lambda a: setattr(a["model_config"].hf_config, "index_topk", None),
            ),
            (
                "other_architecture",
                lambda a: setattr(
                    a["model_config"].hf_config,
                    "architectures",
                    ["GlmMoeDsaForCausalLM"],
                ),
            ),
            (
                "other_backend",
                lambda a: setattr(a["server_args"], "attention_backend", "dsa"),
            ),
            ("non_mla", lambda a: a.update(use_mla_backend=False)),
        )
        for name, change in changes:
            with self.subTest(case=name):
                args = self.config_args()
                change(args)
                with self.assertRaises(ValueError):
                    self.namespace["is_dsa_fia_native_enabled"](**args)

    def test_native_accepts_unquantized_kv_and_accounts_for_dtype_size(self):
        os.environ["SGLANG_NPU_DSA_FIA_NATIVE"] = "1"
        for dtype in ("auto", "bf16", "bfloat16"):
            for num_layers, element_size in ((61, 2), (3, 4)):
                with self.subTest(dtype=dtype, layers=num_layers, bytes=element_size):
                    args = self.config_args()
                    args["server_args"].kv_cache_dtype = dtype
                    self.assertEqual(
                        self.namespace["get_dsa_fia_native_cell_size"](
                            **args, num_layers=num_layers, element_size=element_size
                        ),
                        704 * num_layers * element_size,
                    )

    def test_native_rejects_incompatible_runtime_features(self):
        os.environ["SGLANG_NPU_DSA_FIA_NATIVE"] = "1"
        for field, value in (
            ("enable_prefill_cp", True),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("speculative_algorithm", "EAGLE"),
            ("enable_torch_compile", True),
            ("kv_cache_dtype", "fp8_e4m3"),
        ):
            with self.subTest(field=field):
                args = self.config_args()
                setattr(args["server_args"], field, value)
                with self.assertRaises(ValueError):
                    self.namespace["is_dsa_fia_native_enabled"](**args)
        for env_name in ("SGLANG_NPU_USE_MLAPO", "SGLANG_USE_FIA_NZ"):
            with self.subTest(env=env_name), patch.dict(os.environ, {env_name: "1"}):
                with self.assertRaises(ValueError):
                    self.namespace["is_dsa_fia_native_enabled"](**self.config_args())


class TestNativeFIAOffloadConfig(unittest.TestCase):
    def setUp(self):
        TestDSAFIANativeConfig.setUp(self)
        self.graph = SimpleNamespace(
            decode=SimpleNamespace(backend="full", max_bs=64, bs=[1, 8, 80]),
            prefill=SimpleNamespace(backend="disabled"),
        )
        self.namespace["get_exec"] = Mock(side_effect=ValueError("not initialized"))
        load_functions(
            NPU_ROOT / "sparsity_driven_kv_offload/config.py",
            {
                "get_sparse_kv_attn_impl",
                "_get_native_fia_offload_graph_config",
                "get_native_fia_offload_max_batch_size",
                "get_native_fia_offload_buffer_size",
                "is_sparsity_driven_kv_offload_enabled",
                "get_sparsity_driven_kv_offload_cell_size",
            },
            self.namespace,
        )
        os.environ.update(
            SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD="1",
            SGLANG_NPU_DSA_FIA_NATIVE="0",
            SGLANG_NPU_SPARSE_KV_ATTN_IMPL="native_fia",
        )

    def config_args(self):
        args = TestDSAFIANativeConfig.config_args(self)
        vars(args["server_args"]).update(
            max_running_requests=32,
            cuda_graph_config=self.graph,
            tp_size=16,
            enable_pdmux=False,
            enable_two_batch_overlap=False,
            disable_radix_cache=True,
            disaggregation_mode="null",
        )
        return args

    def test_native_offload_keeps_only_index_pool_per_token_and_one_staging_buffer(
        self,
    ):
        args = self.config_args()
        args["model_config"].hf_config.index_topk = 2050
        self.assertTrue(self.namespace["is_sparsity_driven_kv_offload_enabled"](**args))
        self.assertEqual(
            self.namespace["get_sparsity_driven_kv_offload_cell_size"](
                **args, num_layers=61, element_size=2
            ),
            128 * 61 * 2,
        )
        # Cover the largest graph bucket, round K to complete pages, and do not
        # multiply staging by layer count or by the full context length.
        padded_topk, pages, batch_size = 2176, 17, 80
        self.assertEqual(
            self.namespace["get_native_fia_offload_buffer_size"](
                model_config=args["model_config"],
                server_args=args["server_args"],
                page_size=128,
                element_size=2,
            ),
            batch_size * padded_topk * 576 * 2
            + batch_size * pages * 4
            + padded_topk * 8,
        )

    def test_resolved_graph_capacity_overrides_server_args_and_covers_tp_padding(self):
        args = self.config_args()["server_args"]
        get_capacity = self.namespace["get_native_fia_offload_max_batch_size"]
        self.assertEqual(get_capacity(args), 80)
        published = SimpleNamespace(
            decode=SimpleNamespace(backend="full", max_bs=97, bs=[128]),
            prefill=SimpleNamespace(backend="disabled"),
        )
        self.namespace["get_exec"].side_effect = None
        self.namespace["get_exec"].return_value = SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=published)
        )
        self.assertEqual(get_capacity(args), 128)
        published.decode.backend = "disabled"
        args.max_running_requests = 33
        self.assertEqual(get_capacity(args), 48)

    def test_native_only_and_legacy_modes_do_not_reserve_offload_staging(self):
        args = self.config_args()
        reserve = self.namespace["get_native_fia_offload_buffer_size"]
        for environment in (
            {"SGLANG_NPU_DSA_FIA_NATIVE": "1"},
            {"SGLANG_NPU_SPARSE_KV_ATTN_IMPL": "combined"},
            {"SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD": "0"},
        ):
            with self.subTest(environment=environment), patch.dict(
                os.environ, environment
            ):
                self.assertEqual(
                    reserve(
                        model_config=args["model_config"],
                        server_args=args["server_args"],
                        page_size=128,
                        element_size=2,
                    ),
                    0,
                )

    def test_unsupported_ownership_graph_and_execution_modes_fail_at_startup(self):
        enabled = self.namespace["is_sparsity_driven_kv_offload_enabled"]
        for field, value in (
            ("disable_radix_cache", False),
            ("enable_pdmux", True),
            ("enable_two_batch_overlap", True),
            ("disaggregation_mode", "decode"),
            ("max_running_requests", None),
            ("max_running_requests", 0),
            ("enable_prefill_cp", True),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("speculative_algorithm", "EAGLE"),
            ("enable_torch_compile", True),
            ("kv_cache_dtype", "fp8_e4m3"),
        ):
            with self.subTest(field=field, value=value):
                args = self.config_args()
                setattr(args["server_args"], field, value)
                with self.assertRaises(ValueError):
                    enabled(**args)
        for value in (True, 0, -1, 2.5):
            with self.subTest(index_topk=value):
                args = self.config_args()
                args["model_config"].hf_config.index_topk = value
                with self.assertRaisesRegex(ValueError, "positive integer index_topk"):
                    enabled(**args)
        self.graph.prefill.backend = "full"
        with self.assertRaisesRegex(ValueError, "prefill graph"):
            enabled(**self.config_args())

    def test_native_offload_requires_real_io_and_nd_fia(self):
        for env_name in (
            "SGLANG_NPU_USE_MLAPO",
            "SGLANG_USE_FIA_NZ",
            "SGLANG_NPU_SPARSE_KV_FIA_SKIP_KV_IO",
        ):
            with self.subTest(env=env_name), patch.dict(os.environ, {env_name: "1"}):
                with self.assertRaises(ValueError):
                    self.namespace["is_sparsity_driven_kv_offload_enabled"](
                        **self.config_args()
                    )

    def test_pool_capacity_subtracts_fixed_staging_before_token_page_alignment(self):
        calculate = load_method(
            NPU_ROOT.parents[1] / "model_executor/pool_configurator.py",
            "DefaultPoolConfigurator",
            "calculate_pool_sizes",
            {"MemoryPoolConfig": SimpleNamespace},
        )
        pool = SimpleNamespace(
            _native_fia_offload_buffer_bytes=123456, _cell_size=128 * 61 * 2,
        )
        available = pool._native_fia_offload_buffer_bytes + pool._cell_size * 1001
        result = calculate(pool, available, 128)
        self.assertEqual(result.max_total_num_tokens, 896)
        for available in (123456, 123455):
            with self.subTest(available=available), self.assertRaisesRegex(
                ValueError, "staging"
            ):
                calculate(pool, available, 128)
        pool._native_fia_offload_buffer_bytes = 0
        self.assertEqual(
            calculate(pool, pool._cell_size * 1001, 128).max_total_num_tokens, 896
        )


class TestDualFIAOffloadConfig(unittest.TestCase):
    def setUp(self):
        TestNativeFIAOffloadConfig.setUp(self)
        os.environ["SGLANG_NPU_SPARSE_KV_ATTN_IMPL"] = "split_graph_dual_fia"
        load_functions(
            NPU_ROOT / "sparsity_driven_kv_offload/config.py",
            {"is_sparse_kv_decode_graph_enabled"},
            self.namespace,
        )

    def config_args(self):
        args = TestNativeFIAOffloadConfig.config_args(self)
        args["server_args"].page_size = 128
        return args

    def test_dual_fia_opt_in_keeps_hot_cache_pool_accounting(self):
        self.assertEqual(
            self.namespace["get_sparse_kv_attn_impl"](), "split_graph_dual_fia"
        )
        for graph_backend in ("disabled", "full"):
            for page_size in (16, 128, 1024):
                for dtype in ("auto", "bf16", "bfloat16"):
                    with self.subTest(graph=graph_backend, page=page_size, dtype=dtype):
                        self.graph.decode.backend = graph_backend
                        args = self.config_args()
                        args["server_args"].page_size = page_size
                        args["server_args"].kv_cache_dtype = dtype
                        self.assertTrue(
                            self.namespace["is_sparsity_driven_kv_offload_enabled"](
                                **args
                            )
                        )
                        self.assertEqual(
                            self.namespace["get_sparsity_driven_kv_offload_cell_size"](
                                **args, num_layers=61, element_size=2
                            ),
                            128 * 61 * 2,
                        )
                        self.assertEqual(
                            self.namespace["get_native_fia_offload_buffer_size"](
                                model_config=args["model_config"],
                                server_args=args["server_args"],
                                page_size=page_size,
                                element_size=2,
                            ),
                            0,
                        )

    def test_dual_fia_rejects_unsupported_partition_dimensions_and_pages(self):
        enabled = self.namespace["is_sparsity_driven_kv_offload_enabled"]
        for field, value in (("kv_lora_rank", 256), ("qk_rope_head_dim", 32)):
            with self.subTest(field=field):
                args = self.config_args()
                setattr(args["model_config"], field, value)
                with self.assertRaisesRegex(ValueError, "512.*64"):
                    enabled(**args)
        for topk in (None, True, 0, -1, 2.5, 1024, 2050):
            with self.subTest(topk=topk):
                args = self.config_args()
                args["model_config"].hf_config.index_topk = topk
                with self.assertRaises(ValueError):
                    enabled(**args)
        for page_size in (None, True, 0, -16, 8, 15, 48, 2048, 128.0):
            with self.subTest(page_size=page_size):
                args = self.config_args()
                args["server_args"].page_size = page_size
                with self.assertRaisesRegex(ValueError, "page size"):
                    enabled(**args)

    def test_dual_fia_rejects_unsupported_runtime_and_ownership_modes(self):
        enabled = self.namespace["is_sparsity_driven_kv_offload_enabled"]
        for field, value in (
            ("max_running_requests", None),
            ("max_running_requests", 0),
            ("max_running_requests", -1),
            ("attention_backend", "dsa"),
            ("kv_cache_dtype", "fp8_e4m3"),
            ("enable_prefill_cp", True),
            ("enable_dsa_prefill_context_parallel", True),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("speculative_algorithm", "EAGLE"),
            ("enable_torch_compile", True),
            ("enable_pdmux", True),
            ("enable_two_batch_overlap", True),
            ("disable_radix_cache", False),
            ("disaggregation_mode", "decode"),
        ):
            with self.subTest(field=field, value=value):
                args = self.config_args()
                setattr(args["server_args"], field, value)
                with self.assertRaises(ValueError):
                    enabled(**args)
        for env_name in (
            "SGLANG_NPU_USE_MLAPO",
            "SGLANG_USE_FIA_NZ",
            "SGLANG_NPU_SPARSE_KV_FIA_SKIP_KV_IO",
        ):
            with self.subTest(environment=env_name), patch.dict(
                os.environ, {env_name: "1"}
            ):
                with self.assertRaises(ValueError):
                    enabled(**self.config_args())
        self.graph.prefill.backend = "full"
        with self.assertRaisesRegex(ValueError, "prefill graph"):
            enabled(**self.config_args())

    def test_decode_graph_detection_uses_resolved_execution_config(self):
        detect = self.namespace["is_sparse_kv_decode_graph_enabled"]
        args = self.config_args()["server_args"]
        self.assertTrue(detect(args))
        self.graph.decode.backend = "disabled"
        self.assertFalse(detect(args))
        published = SimpleNamespace(
            decode=SimpleNamespace(backend="full"),
            prefill=SimpleNamespace(backend="disabled"),
        )
        self.namespace["get_exec"].side_effect = None
        self.namespace["get_exec"].return_value = SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=published)
        )
        self.assertTrue(detect(args))


class TestDSAFIANativeAttentionRouting(unittest.TestCase):
    def test_native_prefill_and_decode_reuse_ordinary_attention_routes(self):
        methods = SimpleNamespace(MHA_NPU="mha", MLA_NPU="mla", DSA_NPU="dsa")
        namespace = load_functions(
            NPU_ROOT.parents[1] / "models/deepseek_common/attention_backend_handler.py",
            {"handle_attention_ascend"},
            {"AttnForwardMethod": methods},
        )
        handler = namespace["handle_attention_ascend"]
        for mode in ("EXTEND", "DECODE", "IDLE", "TARGET_VERIFY", "DRAFT_EXTEND_V2"):
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_extend=lambda: mode
                    in ("EXTEND", "TARGET_VERIFY", "DRAFT_EXTEND_V2"),
                    is_target_verify=lambda: mode == "TARGET_VERIFY",
                    is_draft_extend_v2=lambda: mode == "DRAFT_EXTEND_V2",
                )
            )
            for use_dsa, native in ((True, True), (True, False), (False, False)):
                with self.subTest(mode=mode, use_dsa=use_dsa, native=native):
                    attn = SimpleNamespace(use_dsa=use_dsa, dsa_fia_native=native)
                    expected = (
                        "dsa"
                        if use_dsa and not native
                        else ("mha" if mode == "EXTEND" else "mla")
                    )
                    self.assertEqual(handler(attn, batch), expected)
            # Existing models do not have to define the diagnostic attribute.
            self.assertEqual(handler(SimpleNamespace(use_dsa=True), batch), "dsa")

    def prepare_case(self):
        tensor = lambda shape: FakeTensor(np.ones(shape, dtype=np.float32))
        context = SimpleNamespace(fetch_qkv_latent=lambda: tensor((2, 8)))
        model = SimpleNamespace(
            use_dsa=True,
            dsa_fia_native=True,
            q_lora_rank=2,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            qk_nope_head_dim=2,
            qk_head_dim=4,
            num_local_heads=2,
            q_a_layernorm=Mock(side_effect=lambda value: value, variance_epsilon=1e-6),
            kv_a_layernorm=lambda value: value,
            q_b_proj=lambda value: (tensor((2, 8)),),
            w_kc=tensor((2, 2, 4)),
            rotary_emb=lambda positions, q, k: (q, k),
            use_deepseek_yarn_rope=False,
            layer_id=3,
            indexer=Mock(return_value="selected_indices"),
        )
        namespace = {
            "get_attn_tp_context": lambda: context,
            "_use_ag_after_qlora": False,
            "is_mla_preprocess_enabled": lambda: False,
            "dsa_use_prefill_cp": lambda batch: False,
            "fused_split_qk_norm": lambda *args, **kwargs: (
                tensor((2, 2)),
                tensor((2, 1, 4)),
                tensor((2, 1, 2)),
            ),
            "torch": SimpleNamespace(
                bmm=lambda a, b: FakeTensor(np.matmul(a.data, b.data))
            ),
        }
        load_functions(
            NPU_ROOT / "modules/deepseek_v2_attention_mla_npu.py",
            {"forward_mha_prepare_npu", "forward_mla_prepare_npu"},
            namespace,
        )
        return model, namespace

    def test_native_mla_prepare_skips_indexer_and_returns_no_topk(self):
        for native in (False, True):
            with self.subTest(native=native):
                model, namespace = self.prepare_case()
                model.dsa_fia_native = native
                result = namespace["forward_mla_prepare_npu"](
                    model, object(), object(), object(), object(), object()
                )
                if native:
                    model.indexer.assert_not_called()
                    self.assertIsNone(result[-1])
                else:
                    model.indexer.assert_called_once()
                    self.assertEqual(result[-1], "selected_indices")

    def test_native_mha_prepare_skips_indexer_before_writing_paged_kv(self):
        class ReachedKVNormalization(Exception):
            pass

        for native in (False, True):
            with self.subTest(native=native):
                model, namespace = self.prepare_case()
                model.dsa_fia_native = native
                model.kv_a_layernorm = Mock(side_effect=ReachedKVNormalization)
                with self.assertRaises(ReachedKVNormalization):
                    namespace["forward_mha_prepare_npu"](
                        model, object(), object(), object(), object(), object()
                    )
                if native:
                    model.indexer.assert_not_called()
                else:
                    model.indexer.assert_called_once()
                    self.assertFalse(model.indexer.call_args.kwargs["return_indices"])


class TestFIASkipKVIO(unittest.TestCase):
    def setUp(self):
        self.torch_mock = SimpleNamespace(
            int32=np.int32,
            zeros=lambda shape, **kw: FakeTensor(np.zeros(shape, dtype=kw["dtype"])),
            cumsum=lambda tensor, dim: FakeTensor(np.cumsum(tensor.data, axis=dim)),
            npu=SimpleNamespace(current_stream=Mock(return_value="main_stream")),
        )
        self.npu_mock = SimpleNamespace(npu_sparse_flash_attention=Mock())
        self.namespace = load_sparse_kv_forward(self.torch_mock, self.npu_mock)
        self.call_fia = Mock(side_effect=lambda query, *args, **kwargs: query)
        self.namespace["_run_combined_decode_fia"] = self.call_fia
        self.forward = self.namespace["forward_sparsity_driven_kv_offload"]

    def make_case(self, *, skip=False, prefill=False, graph_mode=True):
        def tensor(shape):
            return FakeTensor(np.ones(shape, dtype=np.float32))

        batch_size, heads, nope_dim, rope_dim = 2, 2, 512, 64
        q = tensor((batch_size, heads, nope_dim))
        k = tensor((batch_size, 1, nope_dim))
        q_rope = tensor((batch_size, heads, rope_dim))
        k_rope = tensor((batch_size, 1, rope_dim))
        self.manager = SimpleNamespace(
            fia_skip_kv_io=skip,
            attn_impl="combined",
            sparse_context_len=128,
            offload_v2=Mock(),
            prefetch=Mock(),
            get_forward_kv=Mock(return_value=(k, k_rope)),
        )
        # A recognizable value verifies that the normal path still consumes
        # prefetch output, whereas the diagnostic consumes the initialized zeros.
        self.manager.prefetch.side_effect = lambda layer, batch, topk, selected, stream: selected.data.fill(
            7
        )
        backend = SimpleNamespace(
            sparse_kv_manager=self.manager,
            device="npu",
            kv_lora_rank=nope_dim,
            qk_rope_head_dim=rope_dim,
            page_size=128,
            graph_mode=graph_mode,
            forward_metadata=SimpleNamespace(
                actual_seq_lengths_q=tensor((batch_size,)),
                actual_seq_lengths_kv=tensor((batch_size,)),
            ),
        )
        self.npu_mock.npu_sparse_flash_attention.return_value = (q, None, None)
        batch = SimpleNamespace(
            batch_size=batch_size,
            seq_lens=tensor((batch_size,)),
            forward_mode=SimpleNamespace(
                is_decode=lambda: not prefill,
                is_extend_without_speculative=lambda: prefill,
            ),
        )
        layer = SimpleNamespace(tp_k_head_num=1, tp_q_head_num=heads, scaling=0.125)
        return dict(
            backend=backend,
            q=q,
            k=k,
            v=k,
            layer=layer,
            forward_batch=batch,
            q_rope=q_rope,
            k_rope=k_rope,
            topk_indices=tensor((batch_size, 128)),
        )

    def test_diagnostic_defaults_off_and_rejects_other_attention_modes(self):
        get_flag = self.namespace["get_sparse_kv_fia_skip_kv_io"]
        env_name = self.namespace["SPARSE_KV_FIA_SKIP_KV_IO_ENV_VAR"]
        with patch.dict(os.environ, {}, clear=True):
            for impl in self.namespace["SPARSE_KV_ATTN_IMPL_CHOICES"]:
                self.assertFalse(get_flag(impl))
            for value in ("1", "true"):
                os.environ[env_name] = value
                self.assertTrue(get_flag("combined"))
                for impl in self.namespace["SPARSE_KV_ATTN_IMPL_CHOICES"]:
                    if impl != "combined":
                        with self.assertRaisesRegex(ValueError, "requires"):
                            get_flag(impl)
            os.environ[env_name] = "0"
            self.assertFalse(get_flag("combined"))

    def test_decode_skips_both_io_calls_but_keeps_zero_kv_fia(self):
        for graph_mode in (False, True):
            with self.subTest(graph_mode=graph_mode):
                case = self.make_case(skip=True, graph_mode=graph_mode)
                self.call_fia.reset_mock()
                output = self.forward(**case)
                self.manager.offload_v2.assert_not_called()
                self.manager.prefetch.assert_not_called()
                self.call_fia.assert_called_once()
                args, kwargs = self.call_fia.call_args
                self.assertEqual(args[0].shape, (2, 1, 2, 512))
                self.assertEqual(args[2].shape, (2, 128, 1, 512))
                self.assertEqual(args[3].shape, (2, 128, 1, 64))
                self.assertTrue(np.all(args[2].data == 0))
                self.assertTrue(np.all(args[3].data == 0))
                self.assertEqual(kwargs, {"page_size": 128, "scale_value": 0.125})
                self.assertEqual(output.shape, (2, 1024))
        self.npu_mock.npu_sparse_flash_attention.assert_not_called()

    def test_switch_off_preserves_decode_io_and_uses_prefetched_kv(self):
        self.forward(**self.make_case())
        self.manager.offload_v2.assert_called_once()
        self.manager.prefetch.assert_called_once()
        self.call_fia.assert_called_once()
        self.assertTrue(np.all(self.call_fia.call_args.args[2].data == 7))
        self.assertTrue(np.all(self.call_fia.call_args.args[3].data == 7))

    def test_switch_does_not_bypass_prefill_offload_or_attention(self):
        self.forward(**self.make_case(skip=True, prefill=True))
        self.manager.offload_v2.assert_called_once()
        self.manager.get_forward_kv.assert_called_once()
        self.manager.prefetch.assert_not_called()
        self.call_fia.assert_not_called()
        self.npu_mock.npu_sparse_flash_attention.assert_called_once()

    def test_save_kv_cache_false_still_prefetches_when_diagnostic_is_off(self):
        self.forward(**self.make_case(), save_kv_cache=False)
        self.manager.offload_v2.assert_not_called()
        self.manager.prefetch.assert_called_once()
        self.call_fia.assert_called_once()


class TestDualFIAOffloadAttentionRouting(unittest.TestCase):
    def setUp(self):
        TestFIASkipKVIO.setUp(self)

    def make_case(self, *, impl="split_graph_dual_fia", graph_mode=True, prefill=False):
        case = TestFIASkipKVIO.make_case(self, graph_mode=graph_mode, prefill=prefill)
        self.manager.attn_impl = impl
        self.manager.merge_impl = "python"
        self.manager._split_graph_dual_logged = True
        self.manager.prefetch_partitions_graph_dual = Mock(return_value=object())
        self.manager.prefetch_partitions = Mock(return_value=object())
        self.graph_attention = Mock(return_value=case["q"].view(2, 1, 2, 512))
        self.eager_attention = Mock(return_value=case["q"].view(2, 1, 2, 512))
        self.namespace["_run_split_decode_attention_graph_dual"] = self.graph_attention
        self.namespace["_run_split_decode_attention"] = self.eager_attention
        return case

    def test_selector_adds_dual_fia_without_rerouting_existing_modes(self):
        select = self.namespace["_select_split_decode_mode"]
        expected = {
            "combined": (None, None),
            "native_fia": (None, None),
            "split_eager": (None, "parallel"),
            "split_graph": ("single_stream", "parallel"),
            "split_graph_dual": ("dual_stream", "parallel"),
            "split_graph_dual_fia": ("dual_stream", "parallel"),
            "split_graph_dual_v2": ("dual_stream_v2", "parallel"),
            "pa_graph": ("pa_hot_cache", "pa_hot_cache"),
        }
        self.assertEqual(
            set(expected), set(self.namespace["SPARSE_KV_ATTN_IMPL_CHOICES"])
        )
        for impl, (graph, eager) in expected.items():
            with self.subTest(impl=impl):
                self.assertEqual(select(impl, True), graph)
                self.assertEqual(select(impl, False), eager)

    def test_dual_fia_keeps_prefetch_tickets_and_both_stream_execution_routes(self):
        for impl in ("split_graph_dual_fia", "split_graph_dual"):
            for graph_mode in (False, True):
                with self.subTest(impl=impl, graph_mode=graph_mode):
                    case = self.make_case(impl=impl, graph_mode=graph_mode)
                    self.call_fia.reset_mock()
                    self.npu_mock.npu_sparse_flash_attention.reset_mock()
                    ordered = Mock()
                    ordered.attach_mock(self.manager.offload_v2, "offload")
                    prefetch = (
                        self.manager.prefetch_partitions_graph_dual
                        if graph_mode
                        else self.manager.prefetch_partitions
                    )
                    other_prefetch = (
                        self.manager.prefetch_partitions
                        if graph_mode
                        else self.manager.prefetch_partitions_graph_dual
                    )
                    attention = (
                        self.graph_attention if graph_mode else self.eager_attention
                    )
                    other_attention = (
                        self.eager_attention if graph_mode else self.graph_attention
                    )
                    ordered.attach_mock(prefetch, "prefetch")
                    ordered.attach_mock(attention, "attention")
                    output = self.forward(**case)
                    self.assertEqual(output.shape, (2, 1024))
                    self.assertEqual(
                        [call[0] for call in ordered.mock_calls],
                        ["offload", "prefetch", "attention"],
                    )
                    prefetch.assert_called_once_with(
                        case["layer"], case["forward_batch"], case["topk_indices"],
                        "main_stream", dtype=case["k"].dtype,
                    )
                    self.assertIs(attention.call_args.args[0], prefetch.return_value)
                    if graph_mode:
                        self.assertIs(attention.call_args.args[1], self.manager)
                    kwargs = attention.call_args.kwargs
                    self.assertEqual(kwargs["use_fia"], impl == "split_graph_dual_fia")
                    self.assertEqual(kwargs["page_size"], case["backend"].page_size)
                    self.assertEqual(kwargs["stream"], "main_stream")
                    self.assertEqual(kwargs["query"].shape, (2, 1, 2, 512))
                    self.assertEqual(kwargs["query_rope"].shape, (2, 1, 2, 64))
                    other_prefetch.assert_not_called()
                    other_attention.assert_not_called()
                    self.manager.prefetch.assert_not_called()
                    self.call_fia.assert_not_called()
                    self.npu_mock.npu_sparse_flash_attention.assert_not_called()

    def test_dual_fia_prefill_keeps_original_offload_and_sfa(self):
        case = self.make_case(prefill=True, graph_mode=False)
        self.forward(**case)
        self.manager.offload_v2.assert_called_once()
        self.manager.get_forward_kv.assert_called_once()
        self.manager.prefetch_partitions_graph_dual.assert_not_called()
        self.manager.prefetch_partitions.assert_not_called()
        self.graph_attention.assert_not_called()
        self.eager_attention.assert_not_called()
        self.npu_mock.npu_sparse_flash_attention.assert_called_once()


class TestNativeFIAOffloadAttention(unittest.TestCase):
    def setUp(self):
        TestFIASkipKVIO.setUp(self)

    def make_case(self, *, prefill=False, graph_mode=True, cpu_tensor=False):
        case = TestFIASkipKVIO.make_case(self, prefill=prefill, graph_mode=graph_mode)
        self.manager.attn_impl = "native_fia"
        self.manager.sparse_context_len = 2048
        tensor = lambda shape: FakeTensor(np.ones(shape, dtype=np.float32))
        self.pages = tensor((32, 128, 512))
        self.rope_pages = tensor((32, 128, 64))
        self.selected_table = FakeTensor(np.arange(32, dtype=np.int32).reshape(2, 16))
        self.manager.prefetch_native_fia = Mock(
            return_value=(self.pages, self.rope_pages, self.selected_table)
        )
        metadata = case["backend"].forward_metadata
        metadata.seq_lens_cpu_int = (
            FakeTensor(np.array([7, 10001])) if cpu_tensor else None
        )
        metadata.seq_lens_cpu_list = [7, 10001]
        metadata.block_tables = object()
        case["backend"].forward_mla_fia = Mock(return_value="native_attention_output")
        return case

    def test_decode_orders_real_io_then_shared_fia_and_preserves_full_metadata(self):
        for graph_mode in (False, True):
            for cpu_tensor in (False, True):
                with self.subTest(graph_mode=graph_mode, cpu_tensor=cpu_tensor):
                    case = self.make_case(graph_mode=graph_mode, cpu_tensor=cpu_tensor)
                    metadata = case["backend"].forward_metadata
                    metadata_before = vars(metadata).copy()
                    ordered = Mock()
                    ordered.attach_mock(self.manager.offload_v2, "offload")
                    ordered.attach_mock(self.manager.prefetch_native_fia, "gather")
                    ordered.attach_mock(case["backend"].forward_mla_fia, "fia")
                    output = self.forward(**case)
                    self.assertEqual(output, "native_attention_output")
                    self.assertEqual(
                        [call[0] for call in ordered.mock_calls],
                        ["offload", "gather", "fia"],
                    )
                    self.manager.offload_v2.assert_called_once()
                    self.assertEqual(
                        self.manager.offload_v2.call_args.args[-1], "main_stream"
                    )
                    self.manager.prefetch_native_fia.assert_called_once_with(
                        case["layer"], case["forward_batch"], case["topk_indices"]
                    )
                    args = case["backend"].forward_mla_fia.call_args.args
                    self.assertIs(args[0], case["q"])
                    self.assertIs(args[1], case["q_rope"])
                    self.assertIs(args[2], case["layer"])
                    self.assertIs(args[3], self.pages)
                    self.assertIs(args[4], self.rope_pages)
                    self.assertIs(args[5], self.selected_table)
                    self.assertEqual(args[6], [7, 2048])
                    for name, value in metadata_before.items():
                        self.assertIs(getattr(metadata, name), value)
                    self.assertEqual(metadata.seq_lens_cpu_list, [7, 10001])
                    if cpu_tensor:
                        np.testing.assert_array_equal(
                            metadata.seq_lens_cpu_int.data, [7, 10001]
                        )
                    self.manager.prefetch.assert_not_called()
                    self.manager.get_forward_kv.assert_not_called()
                    self.call_fia.assert_not_called()
                    self.npu_mock.npu_sparse_flash_attention.assert_not_called()

    def test_decode_without_new_kv_still_gathers_selected_host_pages(self):
        case = self.make_case()
        self.forward(**case, save_kv_cache=False)
        self.manager.offload_v2.assert_not_called()
        self.manager.prefetch_native_fia.assert_called_once()
        case["backend"].forward_mla_fia.assert_called_once()

    def test_eager_dp_padding_extends_local_cpu_lengths_with_zero(self):
        for length in (7, 10001):
            with self.subTest(length=length):
                case = self.make_case(graph_mode=False, cpu_tensor=True)
                metadata = case["backend"].forward_metadata
                metadata.seq_lens_cpu_int = FakeTensor(np.array([length]))
                metadata.seq_lens_cpu_list = [length]
                original_cpu_lengths = metadata.seq_lens_cpu_int
                original_block_tables = metadata.block_tables
                # Eager attention DP pads Q and device metadata to another
                # rank's batch size after creating the local CPU metadata.
                batch = case["forward_batch"]
                batch.seq_lens = FakeTensor(np.array([length, 0]))
                batch.req_pool_indices = FakeTensor(np.array([1, 0]))
                self.assertEqual(case["q"].shape[0], 2)
                self.assertEqual(self.selected_table.shape[0], 2)
                self.forward(**case)
                fia_args = case["backend"].forward_mla_fia.call_args.args
                self.assertEqual(fia_args[-1], [min(length, 2048), 0])
                self.assertIs(metadata.seq_lens_cpu_int, original_cpu_lengths)
                self.assertEqual(metadata.seq_lens_cpu_int.tolist(), [length])
                self.assertEqual(metadata.seq_lens_cpu_list, [length])
                self.assertIs(metadata.block_tables, original_block_tables)

    def test_metadata_longer_than_selected_batch_is_rejected_before_fia(self):
        case = self.make_case(graph_mode=False)
        case["backend"].forward_metadata.seq_lens_cpu_list = [7, 10001, 99]
        with self.assertRaises(ValueError):
            self.forward(**case)
        case["backend"].forward_mla_fia.assert_not_called()

    def test_prefill_keeps_real_offload_and_original_dsa_sfa(self):
        case = self.make_case(prefill=True, graph_mode=False)
        self.forward(**case)
        self.manager.offload_v2.assert_called_once()
        self.manager.get_forward_kv.assert_called_once()
        self.manager.prefetch_native_fia.assert_not_called()
        case["backend"].forward_mla_fia.assert_not_called()
        self.npu_mock.npu_sparse_flash_attention.assert_called_once()


if __name__ == "__main__":
    unittest.main()
