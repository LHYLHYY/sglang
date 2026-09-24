"""CPU-only regression checks for the FIA graph-update smoke-test patch.

Run with Python directly or pytest; no torch/torch_npu installation is needed.
The backend class is loaded from its AST to avoid importing the NPU runtime.
These mocks check routing, not device-side capture or event correctness.
"""

import ast
import logging
import sys
import threading
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
        "threading": threading,
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
