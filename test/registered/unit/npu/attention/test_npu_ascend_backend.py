"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_backend.
"""

import os
import sys
import unittest
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.sparsity_driven_kv_offload",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMaskBuilder,
    AscendAttnMultiStepDraftBackend,
    ForwardMetadata,
    _expand_dsa_sparse_indices,
    _reshape_kv_for_fia_nz,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
    attention as sparse_kv_attention,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.attention import (
    _merge_decode_sfa_partitions,
    _select_split_decode_mode,
    _SfaPartitionState,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    SPARSE_KV_ATTN_IMPL_COMBINED,
    SPARSE_KV_ATTN_IMPL_ENV_VAR,
    SPARSE_KV_ATTN_IMPL_SPLIT_EAGER,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2,
    SPARSE_KV_MERGE_IMPL_AUTO,
    SPARSE_KV_MERGE_IMPL_ENV_VAR,
    SPARSE_KV_MERGE_IMPL_FUSED,
    SPARSE_KV_MERGE_IMPL_PYTHON,
    get_sparse_kv_attn_impl,
    get_sparse_kv_merge_impl,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    _assign_miss_cache_slots,
)


class TestSparseKVAttentionImplementation(unittest.TestCase):
    def test_default_is_combined(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SPARSE_KV_ATTN_IMPL_ENV_VAR, None)
            self.assertEqual(get_sparse_kv_attn_impl(), SPARSE_KV_ATTN_IMPL_COMBINED)

    def test_split_implementations_are_normalized(self):
        cases = (
            ("  SPLIT_EAGER  ", SPARSE_KV_ATTN_IMPL_SPLIT_EAGER),
            ("  SPLIT_GRAPH  ", SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH),
            ("  SPLIT_GRAPH_DUAL  ", SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL),
            (
                "  SPLIT_GRAPH_DUAL_V2  ",
                SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2,
            ),
        )
        for value, expected in cases:
            with self.subTest(value=value), patch.dict(
                os.environ, {SPARSE_KV_ATTN_IMPL_ENV_VAR: value}
            ):
                self.assertEqual(get_sparse_kv_attn_impl(), expected)

    def test_invalid_implementation_is_rejected(self):
        with patch.dict(os.environ, {SPARSE_KV_ATTN_IMPL_ENV_VAR: "invalid"}):
            with self.assertRaisesRegex(ValueError, SPARSE_KV_ATTN_IMPL_ENV_VAR):
                get_sparse_kv_attn_impl()

    def test_split_decode_mode_matrix(self):
        cases = (
            (SPARSE_KV_ATTN_IMPL_COMBINED, False, None),
            (SPARSE_KV_ATTN_IMPL_COMBINED, True, None),
            (SPARSE_KV_ATTN_IMPL_SPLIT_EAGER, False, "parallel"),
            (SPARSE_KV_ATTN_IMPL_SPLIT_EAGER, True, None),
            (SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH, False, "parallel"),
            (SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH, True, "single_stream"),
            (SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL, False, "parallel"),
            (SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL, True, "dual_stream"),
            (SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2, False, "parallel"),
            (SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2, True, "dual_stream_v2"),
        )
        for attn_impl, graph_mode, expected in cases:
            with self.subTest(attn_impl=attn_impl, graph_mode=graph_mode):
                self.assertEqual(
                    _select_split_decode_mode(attn_impl, graph_mode), expected
                )


class TestSparseKVCacheSlotPlanner(unittest.TestCase):
    def test_misses_use_only_the_complement_of_hit_slots(self):
        hit_valid = torch.tensor(
            [
                [True, False, True, False, True, False, False, False],
                [False, False, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
            ]
        )
        miss_valid = torch.tensor(
            [
                [False, True, False, True, False, True, False, False],
                [True, True, True, True, False, False, False, False],
                [False, False, False, False, False, False, False, False],
            ]
        )
        device_token_pos = torch.tensor(
            [
                [6, -1, 1, -1, 4, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [7, 5, 3, 1, 6, 4, 2, 0],
            ],
            dtype=torch.int32,
        )
        miss_ranks = torch.cumsum(miss_valid.to(torch.int32), dim=1).to(
            torch.int64
        ) - 1

        assigned = _assign_miss_cache_slots(
            hit_valid, device_token_pos, miss_ranks, topk_len=8
        )

        self.assertEqual(assigned[0, miss_valid[0]].tolist(), [0, 2, 3])
        self.assertEqual(assigned[1, miss_valid[1]].tolist(), [0, 1, 2, 3])
        hit_slots = set(device_token_pos[0, hit_valid[0]].tolist())
        miss_slots = set(assigned[0, miss_valid[0]].tolist())
        self.assertTrue(hit_slots.isdisjoint(miss_slots))
        self.assertEqual(len(miss_slots), int(miss_valid[0].sum()))


class TestSparseKVAttentionMergeImplementation(unittest.TestCase):
    def test_default_is_auto(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SPARSE_KV_MERGE_IMPL_ENV_VAR, None)
            self.assertEqual(get_sparse_kv_merge_impl(), SPARSE_KV_MERGE_IMPL_AUTO)

    def test_merge_implementations_are_normalized(self):
        cases = (
            ("  AUTO  ", SPARSE_KV_MERGE_IMPL_AUTO),
            ("  PYTHON  ", SPARSE_KV_MERGE_IMPL_PYTHON),
            ("  FUSED  ", SPARSE_KV_MERGE_IMPL_FUSED),
        )
        for value, expected in cases:
            with self.subTest(value=value), patch.dict(
                os.environ, {SPARSE_KV_MERGE_IMPL_ENV_VAR: value}
            ):
                self.assertEqual(get_sparse_kv_merge_impl(), expected)

    def test_invalid_merge_implementation_is_rejected(self):
        with patch.dict(os.environ, {SPARSE_KV_MERGE_IMPL_ENV_VAR: "invalid"}):
            with self.assertRaisesRegex(ValueError, SPARSE_KV_MERGE_IMPL_ENV_VAR):
                get_sparse_kv_merge_impl()

    @staticmethod
    def _make_states():
        hit = _SfaPartitionState(
            output=torch.randn(2, 1, 3, 4, dtype=torch.bfloat16),
            softmax_max=torch.randn(2, 1, 1, 3, dtype=torch.float32),
            softmax_sum=torch.rand(2, 1, 1, 3, dtype=torch.float32) + 1.0,
            true_counts=torch.tensor([4, 0], dtype=torch.int32),
        )
        miss = _SfaPartitionState(
            output=torch.randn(2, 1, 3, 4, dtype=torch.bfloat16),
            softmax_max=torch.randn(2, 1, 1, 3, dtype=torch.float32),
            softmax_sum=torch.rand(2, 1, 1, 3, dtype=torch.float32) + 1.0,
            true_counts=torch.tensor([2, 5], dtype=torch.int32),
        )
        return hit, miss

    def test_fused_merge_dispatches_to_kernel_wrapper(self):
        hit, miss = self._make_states()
        expected = _merge_decode_sfa_partitions(
            hit, miss, SPARSE_KV_MERGE_IMPL_PYTHON
        )

        fused = MagicMock(side_effect=lambda *args: args[-1].copy_(expected))
        with patch.object(
            sparse_kv_attention, "_fused_sfa_state_merge_inplace", fused
        ), patch.object(
            sparse_kv_attention,
            "_is_fused_sfa_state_merge_available",
            return_value=True,
        ):
            actual = _merge_decode_sfa_partitions(
                hit, miss, SPARSE_KV_MERGE_IMPL_FUSED
            )

        torch.testing.assert_close(actual, expected)
        fused.assert_called_once()
        fused_args = fused.call_args.args
        expected_inputs = (
            hit.output,
            hit.softmax_max,
            hit.softmax_sum,
            miss.output,
            miss.softmax_max,
            miss.softmax_sum,
            hit.true_counts,
            miss.true_counts,
        )
        self.assertEqual(len(fused_args), 9)
        for actual_arg, expected_arg in zip(fused_args[:8], expected_inputs):
            self.assertIs(actual_arg, expected_arg)
        self.assertEqual(fused_args[-1].shape, hit.output.shape)
        self.assertEqual(fused_args[-1].dtype, hit.output.dtype)
        self.assertEqual(fused_args[-1].device, hit.output.device)

    def test_auto_uses_fused_merge_when_native_schema_is_available(self):
        hit, miss = self._make_states()
        expected = _merge_decode_sfa_partitions(
            hit, miss, SPARSE_KV_MERGE_IMPL_PYTHON
        )
        fused = MagicMock(side_effect=lambda *args: args[-1].copy_(expected))
        with patch.object(
            sparse_kv_attention, "_fused_sfa_state_merge_inplace", fused
        ), patch.object(
            sparse_kv_attention,
            "_is_fused_sfa_state_merge_available",
            return_value=True,
        ):
            actual = _merge_decode_sfa_partitions(
                hit, miss, SPARSE_KV_MERGE_IMPL_AUTO
            )

        torch.testing.assert_close(actual, expected)
        fused.assert_called_once()

    def test_auto_falls_back_once_when_native_schema_is_missing(self):
        hit, miss = self._make_states()
        expected = _merge_decode_sfa_partitions(
            hit, miss, SPARSE_KV_MERGE_IMPL_PYTHON
        )
        fused = MagicMock()
        with patch.object(
            sparse_kv_attention, "_fused_sfa_state_merge_inplace", fused
        ), patch.object(
            sparse_kv_attention,
            "_is_fused_sfa_state_merge_available",
            return_value=False,
        ), patch.object(
            sparse_kv_attention, "_FUSED_MERGE_FALLBACK_LOGGED", False
        ), patch.object(sparse_kv_attention.logger, "warning") as warning:
            actual_first = _merge_decode_sfa_partitions(
                hit, miss, SPARSE_KV_MERGE_IMPL_AUTO
            )
            actual_second = _merge_decode_sfa_partitions(
                hit, miss, SPARSE_KV_MERGE_IMPL_AUTO
            )

        torch.testing.assert_close(actual_first, expected)
        torch.testing.assert_close(actual_second, expected)
        fused.assert_not_called()
        warning.assert_called_once()

    def test_strict_fused_merge_rejects_missing_kernel(self):
        hit, miss = self._make_states()
        with patch.object(
            sparse_kv_attention, "_fused_sfa_state_merge_inplace", None
        ), patch.object(
            sparse_kv_attention,
            "_is_fused_sfa_state_merge_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "sgl-kernel-npu"):
                _merge_decode_sfa_partitions(
                    hit, miss, SPARSE_KV_MERGE_IMPL_FUSED
                )


class TestSparseKVDualGraphIntegration(unittest.TestCase):
    def test_dual_v2_overlaps_miss_copy_but_serializes_sfa(self):
        call_order = []
        main_stream = object()
        miss_stream = object()
        events = SimpleNamespace(
            hit_copy_done=object(),
            hit_attention_done=object(),
            miss_attention_done=object(),
        )
        hit_partition = SimpleNamespace(name="hit")
        miss_partition = SimpleNamespace(name="miss", stream=miss_stream)
        ticket = SimpleNamespace(
            hit=hit_partition,
            miss=miss_partition,
            events=events,
        )

        def make_state(name):
            return SimpleNamespace(
                output=MagicMock(name=f"{name}_output"),
                softmax_max=MagicMock(name=f"{name}_max"),
                softmax_sum=MagicMock(name=f"{name}_sum"),
                true_counts=MagicMock(name=f"{name}_counts"),
            )

        states = {"hit": make_state("hit"), "miss": make_state("miss")}
        merged = object()
        manager = SimpleNamespace(
            merge_impl="fused",
            publish_graph_dual_v2_slot_map=lambda _ticket, _stream: call_order.append(
                ("publish", _stream)
            ),
        )

        def run_partition(partition, **_kwargs):
            call_order.append(("sfa", partition.name))
            return states[partition.name]

        with (
            patch.object(
                sparse_kv_attention.torch,
                "npu",
                SimpleNamespace(stream=lambda _stream: nullcontext()),
                create=True,
            ),
            patch.object(
                sparse_kv_attention.torch.profiler,
                "record_function",
                side_effect=lambda _name: nullcontext(),
            ),
            patch.object(
                sparse_kv_attention,
                "_wait_stream_event",
                side_effect=lambda stream, event: call_order.append(
                    ("wait", stream, event)
                ),
            ),
            patch.object(
                sparse_kv_attention,
                "_record_stream_event",
                side_effect=lambda stream, event: call_order.append(
                    ("record", stream, event)
                ),
            ),
            patch.object(
                sparse_kv_attention,
                "_run_decode_sfa_indexed_partition",
                side_effect=run_partition,
            ),
            patch.object(
                sparse_kv_attention,
                "_merge_decode_sfa_partitions",
                side_effect=lambda *_args: call_order.append(("merge",)) or merged,
            ),
        ):
            result = sparse_kv_attention._run_split_decode_attention_graph_dual_v2(
                ticket,
                manager,
                query=MagicMock(),
                query_rope=MagicMock(),
                scale_value=1.0,
                stream=main_stream,
            )

        self.assertIs(result, merged)
        self.assertEqual(
            call_order,
            [
                ("wait", main_stream, events.hit_copy_done),
                ("sfa", "hit"),
                ("record", main_stream, events.hit_attention_done),
                ("wait", miss_stream, events.hit_attention_done),
                ("sfa", "miss"),
                ("record", miss_stream, events.miss_attention_done),
                ("wait", main_stream, events.miss_attention_done),
                ("merge",),
                ("publish", main_stream),
            ],
        )

    def test_pdmux_rejected_before_sparse_manager_construction(self):
        backend_module = sys.modules[AscendAttnBackend.__module__]
        manager_constructor = MagicMock(name="SparseKVCacheManager")
        manager_module_name = (
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager"
        )
        fake_manager_module = SimpleNamespace(
            SparseKVCacheManager=manager_constructor,
            register_sparse_kv_manager=MagicMock(),
        )
        model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            attention_arch=object(),
            use_alibi=False,
            hf_config=SimpleNamespace(architectures=[]),
            context_len=4096,
        )
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=1,
            model_config=model_config,
            req_to_token_pool=SimpleNamespace(req_to_token=MagicMock()),
            token_to_kv_pool=MagicMock(),
            server_args=SimpleNamespace(enable_pdmux=True),
            use_mla_backend=True,
        )

        for attn_impl in (
            SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
            SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2,
        ):
            with (
                self.subTest(attn_impl=attn_impl),
                patch.dict(sys.modules, {manager_module_name: fake_manager_module}),
                patch.object(backend_module.torch, "tensor", return_value=MagicMock()),
                patch.object(
                    backend_module,
                    "AscendTorchNativeAttnBackend",
                    return_value=MagicMock(),
                ),
                patch.object(
                    backend_module,
                    "is_sparsity_driven_kv_offload_enabled",
                    return_value=True,
                ),
                patch.object(
                    backend_module,
                    "get_sparse_kv_attn_impl",
                    return_value=attn_impl,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "do not support PDMux"):
                    AscendAttnBackend(model_runner)

            manager_constructor.assert_not_called()

    def test_dual_graph_metadata_registers_main_and_miss_streams(self):
        backend_module = sys.modules[AscendAttnBackend.__module__]
        register_stream = MagicMock(name="register_npu_host_callback_stream")
        host_callback_module_name = (
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload."
            "host_callback"
        )
        fake_host_callback_module = SimpleNamespace(
            register_npu_host_callback_stream=register_stream
        )
        main_stream = object()
        miss_stream = object()
        device = torch.device("cpu")

        backend = object.__new__(AscendAttnBackend)
        backend.enable_sparsity_driven_kv_offload = True
        backend.device = device
        backend.sparse_kv_manager = SimpleNamespace(
            attn_impl=SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
            _graph_dual_state=SimpleNamespace(miss_stream=miss_stream),
        )
        backend.graph_metadata = {"block_tables": torch.empty((2, 4))}
        backend.is_hybrid_swa = False
        backend.use_sliding_window_kv_pool = False
        backend.speculative_num_draft_tokens = None
        backend.q_head_num_padding = None
        forward_mode = MagicMock()
        forward_mode.is_target_verify.return_value = False
        forward_mode.is_draft_extend_v2.return_value = False
        forward_mode.is_dllm_extend.return_value = False
        fake_npu = SimpleNamespace(current_stream=MagicMock(return_value=main_stream))

        with (
            patch.dict(
                sys.modules,
                {host_callback_module_name: fake_host_callback_module},
            ),
            patch.object(backend_module.torch, "npu", fake_npu, create=True),
        ):
            backend._init_cuda_graph_metadata(
                bs=2,
                forward_mode=forward_mode,
                seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            )

        self.assertEqual(
            register_stream.call_args_list,
            [call(main_stream, device), call(miss_stream, device)],
        )

    def test_dual_v2_graph_metadata_registers_main_and_miss_streams(self):
        backend_module = sys.modules[AscendAttnBackend.__module__]
        register_stream = MagicMock(name="register_npu_host_callback_stream")
        host_callback_module_name = (
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload."
            "host_callback"
        )
        main_stream = object()
        miss_stream = object()
        backend = object.__new__(AscendAttnBackend)
        backend.enable_sparsity_driven_kv_offload = True
        backend.device = torch.device("cpu")
        backend.sparse_kv_manager = SimpleNamespace(
            attn_impl=SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2,
            _graph_dual_v2_state=SimpleNamespace(miss_stream=miss_stream),
        )
        backend.graph_metadata = {"block_tables": torch.empty((2, 4))}
        backend.is_hybrid_swa = False
        backend.use_sliding_window_kv_pool = False
        backend.speculative_num_draft_tokens = None
        backend.q_head_num_padding = None
        forward_mode = MagicMock()
        forward_mode.is_target_verify.return_value = False
        forward_mode.is_draft_extend_v2.return_value = False
        forward_mode.is_dllm_extend.return_value = False

        with (
            patch.dict(
                sys.modules,
                {
                    host_callback_module_name: SimpleNamespace(
                        register_npu_host_callback_stream=register_stream
                    )
                },
            ),
            patch.object(
                backend_module.torch,
                "npu",
                SimpleNamespace(current_stream=MagicMock(return_value=main_stream)),
                create=True,
            ),
        ):
            backend._init_cuda_graph_metadata(
                bs=2,
                forward_mode=forward_mode,
                seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            )

        self.assertEqual(
            register_stream.call_args_list,
            [call(main_stream, backend.device), call(miss_stream, backend.device)],
        )


class TestExpandDsaSparseIndices(unittest.TestCase):
    def test_2d_input_adds_unsqueeze(self):
        """A [T, K] tensor becomes [T, 1, K]."""
        topk = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 1, 3))
        self.assertTrue(torch.equal(result.squeeze(1), topk))

    def test_3d_input_passthrough(self):
        topk = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertTrue(torch.equal(result, topk))

    def test_2d_shape_correctness(self):
        topk = torch.zeros(5, 8)
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.dim(), 3)
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1], 1)
        self.assertEqual(result.shape[2], 8)

    def test_2d_single_row(self):
        topk = torch.tensor([[1, 2, 3, 4]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (1, 1, 4))


class TestReshapeKvForFiaNz(unittest.TestCase):
    def test_output_shape(self):
        """Output shape is (-1, 1, num_heads*head_dim//16, page_size, 16)."""
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (1, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_element_preservation(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 2 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.numel(), tensor.numel())
        self.assertTrue(torch.equal(result.flatten(), tensor))

    def test_different_parameters(self):
        num_heads = 4
        head_dim = 128
        page_size = 32
        total = 3 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.randn(total)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (3, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_view_relationship(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.data_ptr(), tensor.data_ptr())


class TestForwardMetadata(unittest.TestCase):
    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(ForwardMetadata))

    def test_all_fields_default_none(self):
        metadata = ForwardMetadata()
        for f in fields(ForwardMetadata):
            self.assertIsNone(
                getattr(metadata, f.name),
                f"Field '{f.name}' should default to None",
            )

    def test_create_with_values(self):
        block_tables = torch.tensor([[1, 2], [3, 4]])
        seq_lens = torch.tensor([10, 20])
        metadata = ForwardMetadata(
            block_tables=block_tables,
            seq_lens=seq_lens,
            seq_lens_cpu_list=[10, 20],
        )
        self.assertTrue(torch.equal(metadata.block_tables, block_tables))
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertEqual(metadata.seq_lens_cpu_list, [10, 20])

    def test_partial_assignment(self):
        metadata = ForwardMetadata(swa_mask=torch.ones(3, 3))
        self.assertIsNotNone(metadata.swa_mask)
        self.assertIsNone(metadata.block_tables)
        self.assertIsNone(metadata.seq_lens)

    def test_field_names(self):
        names = {f.name for f in fields(ForwardMetadata)}
        expected = {
            "block_tables",
            "block_tables_swa",
            "swa_out_cache_loc",
            "extend_seq_lens_cpu_int",
            "seq_lens_cpu_int",
            "seq_lens_cpu_list",
            "seq_lens_list_cumsum",
            "seq_lens",
            "actual_seq_lengths_q",
            "actual_seq_lengths_q_pa",
            "actual_seq_lengths_kv",
            "swa_mask",
            "prefix_lens",
            "flatten_prefix_block_tables",
        }
        self.assertEqual(names, expected)


class TestGenerateMaskFlag(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_bool(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertEqual(mask.dtype, torch.bool)

    def test_upper_triangular_pattern(self):
        """generate_mask_flag returns ~tril, i.e. True above the diagonal."""
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[0, 3].item())
        self.assertTrue(mask[2, 3].item())
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[1, 0].item())
        self.assertFalse(mask[3, 3].item())

    def test_diagonal_is_false(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(5)
        for i in range(5):
            self.assertFalse(mask[i, i].item())

    def test_1x1(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(1)
        self.assertEqual(mask.shape, (1, 1))
        self.assertFalse(mask[0, 0].item())

    def test_symmetric_upper(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_mask_flag(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertTrue(mask[i, j].item())
                self.assertFalse(mask[j, i].item())


class TestGenerateAttnMask(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(8, "norm", torch.float16)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)

    def test_default_dtype_float16(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm")
        self.assertEqual(mask.dtype, torch.float16)

    def test_mix_mode_float16(self):
        """mix + float16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertEqual(mask[1, 0].item(), 0.0)

    def test_mix_mode_bfloat16(self):
        """mix + bfloat16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[1, 1].item(), 0.0)

    def test_norm_mode_float16(self):
        """norm + float16 -> upper triangle is -inf (overflow), lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_bfloat16(self):
        """norm + bfloat16 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_float32(self):
        """norm + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_mix_mode_float32(self):
        """mix + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_lower_triangle_all_zero(self):
        """The lower triangle (including diagonal) is always zero."""
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0.0)

    def test_upper_triangle_all_masked(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1.0)

    def test_diagonal_is_zero(self):
        """Diagonal is always zero regardless of mode/dtype."""
        for mode in ("mix", "norm"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                mask = AscendAttnMaskBuilder.generate_attn_mask(4, mode, dtype)
                for i in range(4):
                    self.assertEqual(mask[i, i].item(), 0.0)


class TestGetAttentionMaskId(unittest.TestCase):
    def test_flat_tensor(self):
        """Produces a flat tensor of arange ranges concatenated."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([7, 8, 9, 15, 16, 17, 18, 19])
        self.assertEqual(result.dim(), 1)
        self.assertTrue(torch.equal(result, expected))

    def test_single_sequence(self):
        seq_lens = torch.tensor([5])
        extend_lens = torch.tensor([2])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([3, 4])
        self.assertTrue(torch.equal(result, expected))

    def test_multiple_sequences(self):
        seq_lens = torch.tensor([3, 6, 10])
        extend_lens = torch.tensor([1, 2, 4])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([2, 4, 5, 6, 7, 8, 9])
        self.assertTrue(torch.equal(result, expected))

    def test_total_length(self):
        """Result length equals sum of extend_lens per row."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected_len = 3 + 5
        self.assertEqual(result.numel(), expected_len)

    def test_values_correct(self):
        seq_lens = torch.tensor([4, 8])
        extend_lens = torch.tensor([2, 3])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        self.assertEqual(result[0].item(), 2)
        self.assertEqual(result[1].item(), 3)
        self.assertEqual(result[2].item(), 5)
        self.assertEqual(result[4].item(), 7)


class TestUpdateAttnCache(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_seqlen_greater_than_cached(self):
        """When seqlen > cached, the mask is regenerated and cache updated."""
        old_mask = torch.zeros(4, 4)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.shape, (8, 8))
        self.assertEqual(result_mask.dtype, torch.float16)

    def test_seqlen_less_equal_cached(self):
        """When seqlen <= cached, the existing mask is kept."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_dtype_change(self):
        """When dtype differs, the mask is converted but not regenerated."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.dtype, torch.float32)
        self.assertIsNot(result_mask, old_mask)

    def test_no_change(self):
        """When seqlen <= cached and dtype matches, nothing changes."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_seqlen_greater_and_dtype_change(self):
        """Both regeneration and dtype conversion happen."""
        old_mask = torch.zeros(4, 4, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=16,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 16)
        self.assertEqual(result_mask.shape, (16, 16))
        self.assertEqual(result_mask.dtype, torch.float32)

    def test_seqlen_equal_cached_no_regen(self):
        """seqlen == cached should not trigger regeneration."""
        old_mask = torch.ones(8, 8, dtype=torch.float32)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="mix",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)


class TestGetSplitfuseAttnMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        mask = self.builder.get_splitfuse_attn_mask(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_int8(self):
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask.dtype, torch.int8)

    def test_upper_triangular(self):
        """Upper triangle (excluding diagonal) is 1, rest is 0."""
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask[0, 1].item(), 1)
        self.assertEqual(mask[0, 3].item(), 1)
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertEqual(mask[1, 0].item(), 0)
        self.assertEqual(mask[1, 1].item(), 0)

    def test_lower_triangle_zero(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0)

    def test_upper_triangle_one(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1)


class TestGetSwaMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        """Output shape is (batch, 1, s2)."""
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_1d_input_unsqueezed(self):
        """1-D input of shape (B,) is handled; output still (B, 1, s2)."""
        seq_lens = torch.tensor([3, 6, 9])
        mask = self.builder.get_swa_mask(seq_lens, s2=12, left_context=512)
        self.assertEqual(mask.shape, (3, 1, 12))

    def test_2d_input(self):
        """2-D input of shape (B, 1) is also accepted."""
        seq_lens = torch.tensor([[5], [10]])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_mask_values_large_left_context(self):
        """With left_context >= max seq_len, only indices >= seq_lens are True."""
        seq_lens = torch.tensor([3, 5])
        mask = self.builder.get_swa_mask(seq_lens, s2=8, left_context=512)
        row0 = mask[0, 0]
        self.assertFalse(row0[0].item())
        self.assertFalse(row0[2].item())
        self.assertTrue(row0[3].item())
        self.assertTrue(row0[7].item())
        row1 = mask[1, 0]
        self.assertFalse(row1[4].item())
        self.assertTrue(row1[5].item())
        self.assertTrue(row1[7].item())

    def test_mask_values_small_left_context(self):
        """With a small left_context, earlier positions are also masked."""
        seq_lens = torch.tensor([10, 20])
        mask = self.builder.get_swa_mask(seq_lens, s2=30, left_context=5)
        row0 = mask[0, 0]
        self.assertTrue(row0[0].item())
        self.assertTrue(row0[4].item())
        self.assertFalse(row0[5].item())
        self.assertFalse(row0[9].item())
        self.assertTrue(row0[10].item())
        self.assertTrue(row0[29].item())
        row1 = mask[1, 0]
        self.assertTrue(row1[0].item())
        self.assertTrue(row1[14].item())
        self.assertFalse(row1[15].item())
        self.assertFalse(row1[19].item())
        self.assertTrue(row1[20].item())

    def test_clamp_to_zero(self):
        """When seq_len < left_context, start is clamped to 0."""
        seq_lens = torch.tensor([2])
        mask = self.builder.get_swa_mask(seq_lens, s2=5, left_context=10)
        row = mask[0, 0]
        self.assertFalse(row[0].item())
        self.assertFalse(row[1].item())
        self.assertTrue(row[2].item())
        self.assertTrue(row[4].item())

    def test_default_left_context(self):
        """Default left_context is 512."""
        seq_lens = torch.tensor([10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15)
        self.assertEqual(mask.shape, (1, 1, 15))
        row = mask[0, 0]
        self.assertFalse(row[9].item())
        self.assertTrue(row[10].item())

    def test_dtype_bool(self):
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.dtype, torch.bool)


class TestCanUseTnd(unittest.TestCase):
    def test_128_128(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=128)
            )
        )

    def test_192_192(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=192)
            )
        )

    def test_256_256(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=256, v_head_dim=256)
            )
        )

    def test_192_128(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=128)
            )
        )

    def test_64_64(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=64, v_head_dim=64)
            )
        )

    def test_128_256(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=256)
            )
        )

    def test_256_128(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=256, v_head_dim=128)
            )
        )

    def test_128_192(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=192)
            )
        )

    def test_192_256(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=256)
            )
        )

    def test_96_96(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=96, v_head_dim=96)
            )
        )


class TestGenerateAlibiBias(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(AscendAttnBackend)

    def test_shape(self):
        """Output shape is (num_heads, 1, seq_len)."""
        slopes = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = self.backend._generate_alibi_bias(
            seq_len=8,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (4, 1, 8))

    def test_values(self):
        """Each element is slopes[h] * position."""
        slopes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        seq_len = 5
        result = self.backend._generate_alibi_bias(
            seq_len=seq_len,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(4):
            for p in range(seq_len):
                expected = slopes[h].item() * p
                self.assertAlmostEqual(result[h, 0, p].item(), expected, places=5)

    def test_dtype(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_single_head(self):
        slopes = torch.tensor([1.5])
        result = self.backend._generate_alibi_bias(
            seq_len=3,
            slopes=slopes,
            num_heads=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(result[0, 0, 1].item(), 1.5)
        self.assertAlmostEqual(result[0, 0, 2].item(), 3.0)

    def test_zero_position_is_zero(self):
        """Position 0 always yields 0 regardless of slopes."""
        slopes = torch.tensor([1.0, 2.0, 3.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(3):
            self.assertAlmostEqual(result[h, 0, 0].item(), 0.0)

    def test_default_dtype_bfloat16(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.dtype, torch.bfloat16)


class TestGetCudaGraphSeqLenFillValue(unittest.TestCase):
    def test_returns_zero(self):
        backend = object.__new__(AscendAttnBackend)
        self.assertEqual(backend.get_cuda_graph_seq_len_fill_value(), 0)


class TestGetVerifyBuffers(unittest.TestCase):
    def test_no_verify_mask(self):
        backend = object.__new__(AscendAttnBackend)
        self.assertIsNone(backend.verify_mask)

    def test_update_is_noop(self):
        backend = object.__new__(AscendAttnBackend)
        backend.update_verify_buffers_to_fill_after_draft(None, None)
        backend.update_verify_buffers_to_fill_after_draft(MagicMock(), 4)
        backend.update_verify_buffers_to_fill_after_draft(None, 16)


class TestCommonTemplate(unittest.TestCase):
    @staticmethod
    def _make_draft_backend(speculative_num_steps):
        backend = object.__new__(AscendAttnMultiStepDraftBackend)
        backend.speculative_num_steps = speculative_num_steps
        return backend

    def test_calls_fn_for_each_step(self):
        """call_fn is invoked for steps 0..speculative_num_steps-2."""
        backend = self._make_draft_backend(speculative_num_steps=4)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 3)
        for i in range(3):
            call_fn.assert_any_call(i, forward_batch)

    def test_zero_steps(self):
        """speculative_num_steps=1 -> no calls (range(0))."""
        backend = self._make_draft_backend(speculative_num_steps=1)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        call_fn.assert_not_called()

    def test_two_steps(self):
        """speculative_num_steps=2 -> exactly one call with index 0."""
        backend = self._make_draft_backend(speculative_num_steps=2)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        call_fn.assert_called_once_with(0, forward_batch)

    def test_call_indices(self):
        backend = self._make_draft_backend(speculative_num_steps=5)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        indices = []
        backend.common_template(forward_batch, lambda i, fb: indices.append(i))
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_assert_spec_info_not_none(self):
        """Raises AssertionError when forward_batch.spec_info is None."""
        backend = self._make_draft_backend(speculative_num_steps=4)
        forward_batch = MagicMock()
        forward_batch.spec_info = None
        with self.assertRaises(AssertionError):
            backend.common_template(forward_batch, MagicMock())

    def test_passes_same_forward_batch(self):
        backend = self._make_draft_backend(speculative_num_steps=3)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        for call in call_fn.call_args_list:
            self.assertIs(call.args[1], forward_batch)


if __name__ == "__main__":
    unittest.main()
