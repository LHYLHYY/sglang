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
from unittest.mock import Mock, patch

NPU_ROOT = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/hardware_backend/npu"
)


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

    def test_runner_traces_metadata_before_fia_replay(self):
        graph, _ = self.capture(1, ["npu_fused_infer_attention_score.out"])
        self.backend._graph_debug = True
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
        self.backend._outputs[1] = output
        namespace = {
            "is_deepseek_dsa": lambda config: True,
            "LogitsProcessorOutput": SimpleNamespace,
        }
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        runner = SimpleNamespace(
            backend=self.backend,
            load_batch=Mock(),
            _make_graph_key=lambda bs: bs,
            bs=1,
            raw_num_token=1,
            is_dllm=False,
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=object())
            ),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="DECODE"),
            batch_size=1,
            needs_forward_metadata_init=lambda: True,
        )
        with self.assertLogs(self.namespace["logger"], level="INFO") as logs:
            result = namespace["execute"](runner, batch)
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

    def test_attention_accepts_tuple_list_and_tensor_results(self):
        path = NPU_ROOT / "sparsity_driven_kv_offload/attention.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.IfExp)
            and isinstance(node.value.orelse, ast.Name)
            and node.value.orelse.id == "ret"
            and any(
                isinstance(target, ast.Name) and target.id == "attn_out"
                for target in node.targets
            )
        )
        module = ast.Module(body=[assignment], type_ignores=[])
        code = compile(ast.fix_missing_locations(module), str(path), "exec")
        output = object()
        for result in ((output, None), [output, None], output):
            namespace = {"ret": result}
            exec(code, namespace)
            self.assertIs(namespace["attn_out"], output)


if __name__ == "__main__":
    unittest.main()
