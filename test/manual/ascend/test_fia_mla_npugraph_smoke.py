"""NPU-only smoke test of the combined KV FIA call aligned with ordinary MLA.

Run on Ascend (an outer timeout also covers a hang inside a host runtime API):
    timeout 180s python test/manual/ascend/test_fia_mla_npugraph_smoke.py -v

Optional: FIA_TEST_HEADS=16 FIA_TEST_PAGE_SIZE=128 SGLANG_NPU_GRAPH_DEBUG=1
No model weights or custom sparse-KV kernels are needed. This test checks the
dense selected-KV FIA/update contract, not DSA accuracy or real H2D/refill.
"""

import importlib.util
import logging
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestFIAMlaNPUGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            raise unittest.SkipTest("Requires torch and torch_npu") from exc
        if not torch.npu.is_available():
            raise unittest.SkipTest("Requires an Ascend NPU")
        cls.torch = torch
        cls.torch_npu = torch_npu
        # Reuse the AST loaders so this standalone test does not import sglang's
        # model stack or require the custom sgl-kernel-npu KV-offload extension.
        path = Path(__file__).with_name("test_fia_graph_replay_update.py")
        spec = importlib.util.spec_from_file_location("fia_smoke_helpers", path)
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        cls.helpers = helpers

    def wait_for_device(self, event, stage):
        deadline = time.monotonic() + 20
        while not event.query():
            if time.monotonic() >= deadline:
                self.fail(
                    f"{stage}: host returned but device event did not finish in 20s"
                )
            time.sleep(0.05)
        print(f"{stage}: device completed", flush=True)

    def test_mla_out_workspace_explicit_lengths_and_repeated_replay(self):
        torch = self.torch
        device = torch.device("npu", torch.npu.current_device())
        heads = int(os.getenv("FIA_TEST_HEADS", "16"))
        page_size = int(os.getenv("FIA_TEST_PAGE_SIZE", "128"))
        batch, capacity, dim, rope_dim = 2, 2048, 512, 64
        scale = (dim + rope_dim) ** -0.5
        print(
            f"torch={torch.__version__} torch_npu={self.torch_npu.__version__} "
            f"device={torch.npu.get_device_name(device)} batch={batch} "
            f"heads={heads} capacity={capacity} page_size={page_size}",
            flush=True,
        )
        call_fia = self.helpers.load_combined_fia(torch, self.torch_npu)
        backend_cls, namespace = self.helpers.load_backend_class()
        namespace["torch"] = torch
        backend = backend_cls.__new__(backend_cls)
        backend._graphs = {}
        backend._outputs = {}
        backend._fia_update_tasks = {}
        backend._pool = None
        backend._capture_stream = torch.npu.Stream()
        backend._memory_saver_adapter = None
        backend._enable_torch_compile = False
        backend._device_module = torch.npu
        backend._device_id = device.index
        backend._tp_group = SimpleNamespace(rank_in_group=0, barrier=lambda: None)
        backend._graph_debug = os.getenv("SGLANG_NPU_GRAPH_DEBUG", "0").lower() in (
            "1",
            "true",
        )
        backend._debug_replay_id = 0

        torch.manual_seed(2026)
        with torch.no_grad():
            inputs = [
                torch.randn(shape, dtype=torch.bfloat16, device=device)
                for shape in (
                    (batch, 1, heads, dim),
                    (batch, 1, heads, rope_dim),
                    (batch, capacity, 1, dim),
                    (batch, capacity, 1, rope_dim),
                )
            ]
            input_ptrs = [tensor.data_ptr() for tensor in inputs]

            def forward():
                return call_fia(*inputs, page_size=page_size, scale_value=scale)

            print("warmup/capture.begin", flush=True)
            backend.capture_one(batch, forward)
            self.assertEqual(backend._fia_update_tasks[batch], 1)
            output_ptr = backend._outputs[batch].data_ptr()
            print("capture.returned: 1 FIA update task", flush=True)

            # Full batch -> padded batch -> full batch. Keep all Tensor addresses
            # fixed and update the CPU lengths just as the production runner does.
            for index, raw_bs in enumerate((2, 1, 2), start=1):
                for tensor in inputs:
                    tensor.copy_(torch.randn_like(tensor))
                reference = forward()
                ready = torch.npu.Event()
                ready.record(torch.npu.current_stream())
                self.wait_for_device(ready, f"case {index} eager reference")
                lengths = [capacity] * raw_bs + [0] * (batch - raw_bs)
                print(f"case {index}: replay.begin lengths={lengths}", flush=True)
                output = backend.replay_with_input_update(
                    batch,
                    seq_lens=lengths,
                    attr_name="actual_seq_lengths_kv",
                    attr_type=[],
                )
                print(f"case {index}: update/replay/join host returned", flush=True)
                done = torch.npu.Event()
                done.record(torch.npu.current_stream())
                self.wait_for_device(done, f"case {index} replay")
                self.assertEqual(output.data_ptr(), output_ptr)
                self.assertEqual([tensor.data_ptr() for tensor in inputs], input_ptrs)
                # Padded rows are not consumed by the model; compare active rows.
                actual = output[:raw_bs].float().cpu()
                expected = reference[:raw_bs].float().cpu()
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
                print(
                    f"case {index}: max_abs_error="
                    f"{(actual - expected).abs().max().item():.6e}",
                    flush=True,
                )
        print(
            "PASSED: MLA-style FIA .out + workspace + explicit-length NPUGraph replay",
            flush=True,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
