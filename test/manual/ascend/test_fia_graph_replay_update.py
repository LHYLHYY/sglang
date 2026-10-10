"""CPU-only regression checks for FIA graph updates and DSA offload routing.

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
            set(),
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

    def test_fia_offload_reads_loaded_graph_lengths_even_when_cpu_metadata_exists(self):
        graph, _ = self.capture(4, ["npu_fused_infer_attention_score_v2.out"] * 2)
        execute, runner, batch = self.make_runner(
            manager=SimpleNamespace(
                attn_impl="split_graph_dual_fia", sparse_context_len=2048
            ),
            bs=4,
            raw_bs=3,
        )
        # A cached host copy must not remove the device-to-host ordering point.
        batch.seq_lens_cpu = FakeTensor(np.array([1, 2, 3, 4]))
        batch.seq_lens.cpu.side_effect = AssertionError("Read the loaded graph buffer")
        graph_lens = runner.buffers.seq_lens.__getitem__.return_value
        graph_lens.cpu.return_value.tolist.return_value = [7, 2048, 10001, 9999]
        execute(runner, batch)
        runner.buffers.seq_lens.__getitem__.assert_called_once_with(slice(None, 4))
        graph_lens.cpu.assert_called_once_with()
        graph.update.assert_called_once_with(
            cpu_update_input=[{"actual_seq_kvlen": [2048, 2048, 2048, 0]}]
        )
        graph.replay.assert_called_once_with()
        np.testing.assert_array_equal(batch.seq_lens_cpu.data, [1, 2, 3, 4])


    def test_fia_offload_idle_reads_nonempty_graph_buffer_then_updates_zeros(self):
        for raw_bs in (0, 2):
            with self.subTest(raw_bs=raw_bs):
                graph, _ = self.capture(
                    4, ["npu_fused_infer_attention_score_v2.out"] * 2
                )
                execute, runner, batch = self.make_runner(
                    manager=SimpleNamespace(
                        attn_impl="split_graph_dual_fia", sparse_context_len=2048
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
                graph_lens.cpu.return_value.tolist.return_value = [10001, 2048, 99, 98]
                execute(runner, batch)
                runner.buffers.seq_lens.__getitem__.assert_called_once_with(
                    slice(None, 4)
                )
                graph_lens.cpu.assert_called_once_with()
                batch.seq_lens.cpu.assert_not_called()
                graph.update.assert_called_once_with(
                    cpu_update_input=[{"actual_seq_kvlen": [0, 0, 0, 0]}]
                )
                graph.replay.assert_called_once_with()

    def test_fia_offload_waits_for_device_snapshot_before_update_and_replay(self):
        for mode in ("DECODE", "IDLE"):
            with self.subTest(mode=mode):
                graph, _ = self.capture(
                    2,
                    ["npu_fused_infer_attention_score_v2.out"] * 2,
                )
                self.backend._graph_debug = True
                execute, runner, batch = self.make_runner(
                    manager=SimpleNamespace(
                        attn_impl="split_graph_dual_fia",
                        sparse_context_len=2048,
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
                graph.update.assert_called_once_with(
                    cpu_update_input=[
                        {
                            "actual_seq_kvlen": [2048, 0]
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
        graph, _ = self.capture(2, ["npu_fused_infer_attention_score_v2.out"] * 2)
        execute, runner, batch = self.make_runner(
            manager=SimpleNamespace(
                attn_impl="split_graph_dual_fia", sparse_context_len=2048
            ),
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
            graph.update.assert_called_with(
                cpu_update_input=[{"actual_seq_kvlen": [2048, 0]}]
            )


class TestSharedMLAFIA(unittest.TestCase):
    def test_mla_fia_workspace_out_padding_and_crop(self):
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


class TestDualFIAOffloadConfig(unittest.TestCase):
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
                "_get_sparse_kv_graph_config",
                "is_sparse_kv_decode_graph_enabled",
                "is_sparsity_driven_kv_offload_enabled",
                "get_sparsity_driven_kv_offload_cell_size",
            },
            self.namespace,
        )
        os.environ.update(
            SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD="1",
            SGLANG_NPU_DSA_FIA_NATIVE="0",
        )

    def config_args(self):
        args = TestDSAFIANativeConfig.config_args(self)
        vars(args["server_args"]).update(
            max_running_requests=32,
            cuda_graph_config=self.graph,
            tp_size=16,
            page_size=128,
            enable_pdmux=False,
            enable_two_batch_overlap=False,
            disable_radix_cache=True,
            disaggregation_mode="null",
        )
        return args

    def test_dual_fia_default_keeps_hot_cache_pool_accounting(self):
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

    def test_retired_modes_are_rejected_before_offload_is_enabled(self):
        for value in (
            "combined", "native_fia", "split_eager", "split_graph",
            "split_graph_dual", "split_graph_dual_v2", "pa_graph",
        ):
            with self.subTest(value=value), patch.dict(
                os.environ, {"SGLANG_NPU_SPARSE_KV_ATTN_IMPL": value}
            ):
                with self.assertRaisesRegex(ValueError, "SGLANG_NPU_SPARSE_KV_ATTN_IMPL"):
                    self.namespace["get_sparse_kv_attn_impl"]()
                with self.assertRaisesRegex(ValueError, "SGLANG_NPU_SPARSE_KV_ATTN_IMPL"):
                    self.namespace["is_sparsity_driven_kv_offload_enabled"](
                        **self.config_args()
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


class TestDualFIAOffloadAttentionRouting(unittest.TestCase):
    def setUp(self):
        self.torch_mock = SimpleNamespace(
            int32=np.int32,
            cumsum=lambda tensor, dim: FakeTensor(np.cumsum(tensor.data, axis=dim)),
            npu=SimpleNamespace(current_stream=Mock(return_value="main_stream")),
        )
        self.npu_mock = SimpleNamespace(npu_sparse_flash_attention=Mock())
        self.namespace = load_sparse_kv_forward(self.torch_mock, self.npu_mock)
        self.forward = self.namespace["forward_sparsity_driven_kv_offload"]

    def make_case(self, *, prefill=False, graph_mode=True):
        def tensor(shape):
            return FakeTensor(np.ones(shape, dtype=np.float32))

        batch_size, heads, nope_dim, rope_dim = 2, 2, 512, 64
        q = tensor((batch_size, heads, nope_dim))
        k = tensor((batch_size, 1, nope_dim))
        q_rope = tensor((batch_size, heads, rope_dim))
        k_rope = tensor((batch_size, 1, rope_dim))
        self.manager = SimpleNamespace(
            attn_impl="split_graph_dual_fia",
            merge_impl="python",
            _split_graph_dual_logged=True,
            offload_v2=Mock(),
            prefetch_partitions_graph_dual=Mock(return_value=object()),
            prefetch_partitions=Mock(return_value=object()),
            get_forward_kv=Mock(return_value=(k, k_rope)),
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
        self.graph_attention = Mock(return_value=q.view(2, 1, 2, 512))
        self.eager_attention = Mock(return_value=q.view(2, 1, 2, 512))
        self.namespace["_run_split_decode_attention_graph_dual"] = self.graph_attention
        self.namespace["_run_split_decode_attention"] = self.eager_attention
        return dict(
            backend=backend,
            q=q,
            k=k,
            v=k,
            layer=layer,
            forward_batch=batch,
            q_rope=q_rope,
            k_rope=k_rope,
            topk_indices=tensor((batch_size, 2048)),
        )

    def test_selector_keeps_dual_fia_graph_and_eager_and_rejects_retired_modes(self):
        select = self.namespace["_select_split_decode_mode"]
        self.assertEqual(
            self.namespace["SPARSE_KV_ATTN_IMPL_CHOICES"], ("split_graph_dual_fia",)
        )
        self.assertEqual(select("split_graph_dual_fia", True), "dual_stream")
        self.assertEqual(select("split_graph_dual_fia", False), "parallel")
        for impl in (
            "combined", "native_fia", "split_eager", "split_graph",
            "split_graph_dual", "split_graph_dual_v2", "pa_graph",
        ):
            for graph_mode in (False, True):
                with self.subTest(impl=impl, graph_mode=graph_mode):
                    with self.assertRaisesRegex(ValueError, "Unsupported sparse KV"):
                        select(impl, graph_mode)

    def test_dual_fia_keeps_prefetch_tickets_and_both_stream_execution_routes(self):
        for graph_mode in (False, True):
            with self.subTest(graph_mode=graph_mode):
                case = self.make_case(graph_mode=graph_mode)
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
                self.assertEqual(kwargs["page_size"], case["backend"].page_size)
                self.assertEqual(kwargs["stream"], "main_stream")
                self.assertEqual(kwargs["query"].shape, (2, 1, 2, 512))
                self.assertEqual(kwargs["query_rope"].shape, (2, 1, 2, 64))
                other_prefetch.assert_not_called()
                other_attention.assert_not_called()
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


    def test_save_kv_cache_false_still_prefetches_and_runs_fia(self):
        for graph_mode in (False, True):
            with self.subTest(graph_mode=graph_mode):
                case = self.make_case(graph_mode=graph_mode)
                self.forward(**case, save_kv_cache=False)
                self.manager.offload_v2.assert_not_called()
                prefetch = (
                    self.manager.prefetch_partitions_graph_dual
                    if graph_mode else self.manager.prefetch_partitions
                )
                attention = self.graph_attention if graph_mode else self.eager_attention
                prefetch.assert_called_once()
                attention.assert_called_once()
                self.assertIs(attention.call_args.args[0], prefetch.return_value)


if __name__ == "__main__":
    unittest.main()
