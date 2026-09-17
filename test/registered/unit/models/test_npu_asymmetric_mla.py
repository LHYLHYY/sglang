"""CPU contracts for GLM-5.2 asymmetric MLA with attention TP2 and DP8.

Run directly with ``python test/registered/unit/models/test_npu_asymmetric_mla.py``.
Only the standard library is required. Production definitions are AST-loaded
without importing the serving runtime, and fake layers/collectives execute their
actual control flow. Kernel numerics and graph replay require a 16-NPU test.
"""

import ast
import copy
import math
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"

# CI registration is a runtime no-op. Load its stdlib-only module directly so
# standalone execution does not import sglang (and its optional dependencies).
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def load_definitions(relative_path, names, namespace, class_name=None):
    """Execute unchanged production definitions without module-level imports."""
    path = SRT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_nodes = tree.body
    if class_name is not None:
        source_nodes = next(
            node.body
            for node in source_nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    definitions = [
        copy.deepcopy(node)
        for node in source_nodes
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in definitions} != set(names):
        raise AssertionError(f"Missing production definition in {path}: {names}")
    if class_name is not None:
        definitions = [
            ast.ClassDef(
                name=class_name,
                bases=[],
                keywords=[],
                body=definitions,
                decorator_list=[],
            )
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def env_namespace(enabled=True, **overrides):
    values = {
        "SGLANG_NPU_USE_ASYM_MLA": enabled,
        "SGLANG_USE_AG_AFTER_QLORA": False,
        "SGLANG_NPU_USE_MLAPO": False,
    }
    values.update(overrides)
    return SimpleNamespace(
        **{
            name: SimpleNamespace(get=lambda value=value: value)
            for name, value in values.items()
        }
    )


class FakeLayer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.quant_method = SimpleNamespace()
        self.weight = SimpleNamespace(dtype=None, shape=(0, 0))
        self.reduce_results = kwargs.get("reduce_results", False)


class FakeTensor:
    """Shape-only tensor for testing control flow, never kernel numerics."""

    def __init__(self, shape, dtype="int32", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "npu"
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        if self._contiguous:
            return self
        return FakeTensor(self.shape, self.dtype)

    def to(self, dtype):
        return self if self.dtype == dtype else FakeTensor(self.shape, dtype)

    def squeeze(self, dim):
        assert self.shape[dim] == 1
        return FakeTensor(self.shape[:dim] + self.shape[dim + 1 :], self.dtype)

    def unsqueeze(self, dim):
        return FakeTensor(self.shape[:dim] + (1,) + self.shape[dim:], self.dtype)

    def split(self, sizes, dim=-1):
        dim %= self.dim()
        assert sum(sizes) == self.shape[dim]
        return tuple(
            FakeTensor(self.shape[:dim] + (size,) + self.shape[dim + 1 :], self.dtype)
            for size in sizes
        )

    def view(self, *shape):
        shape = list(shape)
        if -1 in shape:
            index = shape.index(-1)
            shape[index] = math.prod(self.shape) // -math.prod(shape)
        assert math.prod(shape) == math.prod(self.shape)
        return FakeTensor(shape, self.dtype)

    def transpose(self, first, second):
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        return FakeTensor(shape, self.dtype)

    def unflatten(self, dim, sizes):
        sizes = list(sizes)
        if -1 in sizes:
            sizes[sizes.index(-1)] = self.shape[dim] // -math.prod(sizes)
        return FakeTensor(
            self.shape[:dim] + tuple(sizes) + self.shape[dim + 1 :], self.dtype
        )


def make_attention(local_rank, enabled=True, use_dsa=True, tp_size=2, layer_id=0):
    server_args = SimpleNamespace(kv_cache_dtype="auto", device="npu")
    parallel = SimpleNamespace(
        attn_tp_rank=local_rank,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        attn_dcp_size=1,
        dcp_enabled=False,
        attn_tp_group=SimpleNamespace(ranks=[0, 1], world_size=tp_size),
    )
    namespace = {
        "_is_npu": True,
        "_is_cuda": False,
        "logger": Mock(),
        "envs": env_namespace(enabled),
        "get_parallel": lambda: parallel,
        "get_server_args": lambda: server_args,
        "is_deepseek_dsa": lambda config: use_dsa,
        "get_dsa_index_topk": lambda config: 2048,
        "get_dsa_index_n_heads": lambda config: 32,
        "get_dsa_index_head_dim": lambda config: 128,
        "add_prefix": lambda name, prefix: f"{prefix}.{name}",
        "fused_a_gemm_weight_eligible": lambda layer: False,
        "ReplicatedLinear": FakeLayer,
        "ColumnParallelLinear": FakeLayer,
        "RowParallelLinear": FakeLayer,
        "RMSNorm": FakeLayer,
        "Indexer": FakeLayer,
        "RadixAttention": FakeLayer,
        "torch": SimpleNamespace(bfloat16=object()),
    }
    load_definitions("configs/model_config.py", ["dsa_layer_skips_topk"], namespace)
    load_definitions(
        "models/deepseek_v2.py",
        ["__init__", "forward_prepare", "forward_core"],
        namespace,
        class_name="DeepseekV2AttentionMLA",
    )
    attention_class = namespace["DeepseekV2AttentionMLA"]
    attention_class._get_q_b_proj_quant_config = lambda self, config: config
    for name in (
        "init_mha_forward",
        "init_mla_forward",
        "init_mla_fused_rope_rocm_forward",
        "init_mla_fused_rope_cpu_forward",
    ):
        setattr(attention_class, name, lambda self: None)
    attention = attention_class(
        config=SimpleNamespace(
            rms_norm_eps=1e-6, index_topk_freq=4, index_skip_topk_offset=3
        ),
        hidden_size=6144,
        num_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        q_lora_rank=2048,
        kv_lora_rank=512,
        layer_id=layer_id,
        reduce_results=False,
        skip_rope=True,
        quant_config=SimpleNamespace(get_name=lambda: "modelslim"),
    )
    return attention, namespace


class TestAsymmetricMLATopology(unittest.TestCase):
    def test_eight_disjoint_attention_pairs(self):
        namespace = load_definitions(
            "layers/dp_attention.py", ["compute_dp_attention_world_info"], {}
        )
        pairs = [[] for _ in range(8)]
        for global_rank in range(16):
            local_rank, local_size, dp_rank, dp_size = namespace[
                "compute_dp_attention_world_info"
            ](True, global_rank, 16, 8, 1)
            self.assertEqual((local_size, dp_size), (2, 8))
            self.assertEqual(local_rank, global_rank % 2)
            pairs[dp_rank].append(global_rank)
        self.assertEqual(pairs, [[2 * rank, 2 * rank + 1] for rank in range(8)])

    def test_roles_and_glm_complete_compute_projections(self):
        controller, _ = make_attention(0)
        compute, _ = make_attention(1)
        self.assertTrue(controller.is_asym_controller)
        self.assertFalse(controller.is_asym_compute)
        self.assertTrue(hasattr(controller, "indexer"))
        for name in ("q_b_proj", "kv_b_proj", "o_proj"):
            self.assertFalse(hasattr(controller, name))
        self.assertTrue(compute.is_asym_compute)
        self.assertFalse(compute.is_asym_controller)
        self.assertFalse(hasattr(compute, "indexer"))
        self.assertEqual(compute.num_local_heads, 64)
        self.assertEqual(compute.asym_index_topk, 2048)
        for name in ("q_b_proj", "kv_b_proj", "o_proj"):
            projection = getattr(compute, name)
            self.assertEqual(projection.kwargs["tp_rank"], 0)
            self.assertEqual(projection.kwargs["tp_size"], 1)
            self.assertIs(projection.kwargs["quant_config"], compute.quant_config)
        self.assertEqual(compute.q_b_proj.args[:2], (2048, 64 * 256))
        self.assertEqual(compute.kv_b_proj.args[:2], (512, 64 * (192 + 256)))
        self.assertEqual(compute.o_proj.args[:2], (64 * 256, 6144))
        self.assertEqual(compute.attn_mqa.args[0], 64)
        self.assertEqual(compute.attn_mha.args[0], 64)
        self.assertFalse(compute.o_proj.kwargs["reduce_results"])

    def test_disabled_feature_preserves_symmetric_sharding(self):
        for rank in range(2):
            attention, _ = make_attention(rank, enabled=False)
            self.assertFalse(attention.is_asym_dsa_npu)
            self.assertEqual(attention.num_local_heads, 32)
            self.assertTrue(hasattr(attention, "indexer"))
            for name in ("q_b_proj", "kv_b_proj", "o_proj"):
                self.assertEqual(getattr(attention, name).kwargs["tp_rank"], rank)
                self.assertEqual(getattr(attention, name).kwargs["tp_size"], 2)

    def test_invalid_attention_group_size_is_rejected(self):
        for tp_size in (1, 4, 16):
            with self.subTest(tp_size=tp_size), self.assertRaises(ValueError):
                make_attention(0, tp_size=tp_size)

    def test_glm_topk_schedule_matches_on_both_roles(self):
        for layer_id in range(12):
            expected_skip = layer_id not in (0, 1, 2, 6, 10)
            for rank in range(2):
                attention, _ = make_attention(rank, layer_id=layer_id)
                self.assertEqual(attention.skip_topk, expected_skip)
                self.assertEqual(
                    attention.next_skip_topk,
                    layer_id + 1 not in (0, 1, 2, 6, 10, 14),
                )

    def test_backend_padding_matches_all_glm_attention_heads(self):
        runner = SimpleNamespace(
            device="npu",
            page_size=128,
            model_config=SimpleNamespace(
                dtype="bfloat16",
                attention_arch="mla",
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                qk_nope_head_dim=192,
                context_len=32768,
                num_attention_heads=64,
                hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
            ),
            req_to_token_pool=SimpleNamespace(req_to_token=object()),
            token_to_kv_pool=object(),
            server_args=SimpleNamespace(speculative_num_draft_tokens=None),
            ps=SimpleNamespace(attn_cp_size=1),
            is_hybrid_swa=False,
        )
        mask_builder = SimpleNamespace(
            mask=None,
            fia_mask=None,
            mtp_mask=None,
            mixed_chunk_attn_mask=None,
            ringmla_mask=None,
        )
        for enabled, expected_heads in ((True, 64), (False, 32)):
            namespace = load_definitions(
                "hardware_backend/npu/attention/ascend_backend.py",
                ["__init__"],
                {
                    "torch": SimpleNamespace(tensor=Mock()),
                    "AttentionArch": SimpleNamespace(MLA="mla"),
                    "AscendTorchNativeAttnBackend": Mock(),
                    "AscendAttnMaskBuilder": Mock(return_value=mask_builder),
                    "get_bool_env_var": lambda *args: False,
                    "get_parallel": lambda: SimpleNamespace(attn_tp_size=2),
                    "get_flags": lambda: SimpleNamespace(
                        capture=SimpleNamespace(enable_torch_compile=False)
                    ),
                    "envs": env_namespace(enabled),
                    "is_deepseek_dsa": lambda config: True,
                    "SWAKVPool": type("SWAKVPool", (), {}),
                    "DllmConfig": SimpleNamespace(from_server_args=lambda args: None),
                },
                class_name="AscendAttnBackend",
            )
            backend = namespace["AscendAttnBackend"](runner)
            self.assertEqual(backend.tp_q_head_num, expected_heads)
            self.assertEqual(backend.q_head_num_padding, expected_heads)


class TestAsymmetricMLADispatch(unittest.TestCase):
    def test_prefill_decode_and_speculative_dispatch(self):
        methods = SimpleNamespace(
            DSA_NPU_ASYM=object(), DSA_NPU=object(), MHA_NPU=object(), MLA_NPU=object()
        )
        namespace = load_definitions(
            "models/deepseek_common/attention_backend_handler.py",
            ["handle_attention_ascend"],
            {"AttnForwardMethod": methods},
        )
        dispatch = namespace["handle_attention_ascend"]
        for phase in ("prefill", "decode", "verify", "draft_extend"):
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_extend=lambda: phase != "decode",
                    is_target_verify=lambda: phase == "verify",
                    is_draft_extend_v2=lambda: phase == "draft_extend",
                )
            )
            for rank in range(2):
                attention, _ = make_attention(rank)
                self.assertIs(dispatch(attention, batch), methods.DSA_NPU_ASYM)
            attention, _ = make_attention(0, enabled=False)
            self.assertIs(dispatch(attention, batch), methods.DSA_NPU)

    def test_idle_controller_does_not_access_missing_projection(self):
        controller, namespace = make_attention(0)
        namespace["get_attn_tp_context"] = lambda: SimpleNamespace(input_scattered=False)
        hidden_states = SimpleNamespace(shape=(0, 6144))
        batch = object()
        state = controller.forward_prepare(None, hidden_states, batch, None)
        self.assertIs(controller.forward_core(state), hidden_states)


class TestAsymmetricMLAConfiguration(unittest.TestCase):
    def setUp(self):
        self.model_module = SimpleNamespace(is_deepseek_dsa=lambda config: config.is_dsa)
        module_patch = patch.dict(
            sys.modules, {"sglang.srt.configs.model_config": self.model_module}
        )
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def make_args(self, enabled=True, env_overrides=None, **overrides):
        env_overrides = env_overrides or {}
        namespace = load_definitions(
            "server_args.py",
            ["_handle_npu_asymmetric_mla"],
            {
                "envs": env_namespace(enabled, **env_overrides),
                "get_bool_env_var": lambda name: env_overrides.get(name, False),
            },
            class_name="ServerArgs",
        )
        args = namespace["ServerArgs"]()
        defaults = dict(
            device="npu",
            tp_size=16,
            dp_size=8,
            enable_dp_attention=True,
            attn_cp_size=1,
            dcp_size=1,
            pp_size=1,
            enable_prefill_cp=False,
            cp_strategy=None,
            enable_prefill_context_parallel=False,
            enable_dsa_prefill_context_parallel=False,
            enable_dsa_cache_layer_split=False,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
            disaggregation_mode="null",
            disaggregation_transfer_backend="ascend",
            attention_backend="ascend",
            prefill_attention_backend="ascend",
            decode_attention_backend="ascend",
        )
        defaults.update(overrides)
        args.__dict__.update(defaults)
        args._resolved = lambda: SimpleNamespace(**defaults)
        args._resolved_attention_backends = lambda: (
            defaults["prefill_attention_backend"],
            defaults["decode_attention_backend"],
        )
        args.get_model_config = lambda: SimpleNamespace(
            hf_config=SimpleNamespace(is_dsa=True)
        )
        return args

    def test_supported_configuration(self):
        self.make_args()._handle_npu_asymmetric_mla()

    def test_ascend_pd_decode_is_supported(self):
        self.make_args(disaggregation_mode="decode")._handle_npu_asymmetric_mla()

    def test_pd_prefill_requires_symmetric_attention(self):
        # The deployment uses P TP16/DP4 and D TP16/DP8. Diagnose an inherited
        # asymmetric flag on P before reporting its different attention TP.
        with self.assertRaisesRegex(ValueError, "SGLANG_NPU_USE_ASYM_MLA=0"):
            self.make_args(
                disaggregation_mode="prefill", dp_size=4
            )._handle_npu_asymmetric_mla()
        self.make_args(
            enabled=False, disaggregation_mode="prefill", dp_size=4
        )._handle_npu_asymmetric_mla()

    def test_pd_decode_requires_ascend_transfer(self):
        for backend in ("mooncake", "nixl", "fake"):
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ValueError, "--disaggregation-transfer-backend ascend"
            ):
                self.make_args(
                    disaggregation_mode="decode",
                    disaggregation_transfer_backend=backend,
                )._handle_npu_asymmetric_mla()

    def test_incompatible_configurations_fail_early(self):
        cases = (
            {"device": "cuda"},
            {"tp_size": 2},
            {"dp_size": 1},
            {"dp_size": 16},
            {"enable_dp_attention": False},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
            {"pp_size": 2},
            {"enable_prefill_cp": True},
            {"cp_strategy": "round-robin"},
            {"enable_prefill_context_parallel": True},
            {"enable_dsa_prefill_context_parallel": True},
            {"enable_dsa_cache_layer_split": True},
            {"enable_two_batch_overlap": True},
            {"disaggregation_mode": "prefill"},
            {"decode_attention_backend": "triton"},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.make_args(**case)._handle_npu_asymmetric_mla()

    def test_incompatible_runtime_optimizations_fail_early(self):
        for name in (
            "SGLANG_USE_AG_AFTER_QLORA",
            "SGLANG_NPU_USE_MLAPO",
            "SGLANG_USE_FIA_NZ",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.make_args(env_overrides={name: True})._handle_npu_asymmetric_mla()

    def test_resolved_configuration_takes_precedence(self):
        args = self.make_args()
        args.tp_size = 2
        args.device = "cuda"
        args._handle_npu_asymmetric_mla()

    def test_non_dsa_model_fails_early(self):
        args = self.make_args()
        args.get_model_config = lambda: SimpleNamespace(
            hf_config=SimpleNamespace(is_dsa=False)
        )
        with self.assertRaisesRegex(ValueError, "DSA model"):
            args._handle_npu_asymmetric_mla()

    def test_disabled_feature_does_not_constrain_other_deployments(self):
        self.make_args(
            enabled=False, device="cuda", tp_size=4, dp_size=1
        )._handle_npu_asymmetric_mla()


class TestAsymmetricMLATopK(unittest.TestCase):
    def setUp(self):
        self.collective = Mock(side_effect=lambda indices: indices)
        self.namespace = load_definitions(
            "hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py",
            [
                "_normalize_asym_topk_indices",
                "forward_dsa_asym_prepare_npu",
                "forward_dsa_asym_core_npu",
            ],
            {
                "torch": SimpleNamespace(
                    int32="int32",
                    zeros=Mock(side_effect=lambda shape, **kw: FakeTensor(shape)),
                    bmm=lambda left, right: FakeTensor(
                        (left.shape[0], left.shape[1], right.shape[2]), "bfloat16"
                    ),
                ),
                "_all_reduce_asym_topk_indices": self.collective,
            },
        )

    def make_role(self, controller, skip_topk, is_nextn=False):
        model = SimpleNamespace(
            is_asym_controller=controller,
            skip_topk=skip_topk,
            is_nextn=is_nextn,
            next_skip_topk=True,
            layer_id=3,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            qk_head_dim=256,
            num_local_heads=64,
            asym_index_topk=2048,
            hidden_size=6144,
            alt_stream=None,
            w_kc=FakeTensor((64, 192, 512), "bfloat16"),
            fused_qkv_a_proj_with_mqa=Mock(
                return_value=(FakeTensor((7, 2048 + 512 + 64), "bfloat16"), None)
            ),
            q_a_layernorm=Mock(side_effect=lambda value: value),
            kv_a_layernorm=Mock(side_effect=lambda value: value),
            q_b_proj=Mock(return_value=(FakeTensor((7, 64 * 256), "bfloat16"), None)),
            indexer=Mock(return_value=FakeTensor((7, 2048))),
        )
        model.rotary_emb = Mock(side_effect=lambda positions, query, key: (query, key))
        model.rotary_emb.is_neox_style = True
        return model

    def prepare(self, model, previous=None):
        return self.namespace["forward_dsa_asym_prepare_npu"](
            model,
            object(),
            FakeTensor((7, 6144), "bfloat16"),
            object(),
            None,
            None,
            previous,
        )

    def test_normalizes_legacy_layout_dtype_and_contiguity(self):
        normalize = self.namespace["_normalize_asym_topk_indices"]
        current = FakeTensor((7, 2048))
        self.assertIs(normalize(current), current)
        legacy = normalize(FakeTensor((7, 1, 2048), "int64", contiguous=False))
        self.assertEqual(legacy.shape, (7, 2048))
        self.assertEqual(legacy.dtype, "int32")
        self.assertTrue(legacy.is_contiguous())
        for shape in ((2048,), (7, 2, 2048), (7, 1, 1, 2048)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                normalize(FakeTensor(shape))

    def test_collective_uses_only_its_attention_pair(self):
        for dp_rank in range(8):
            pair = SimpleNamespace(
                world_size=2,
                ranks=[2 * dp_rank, 2 * dp_rank + 1],
                device_group=object(),
            )
            all_reduce = Mock()
            namespace = load_definitions(
                "hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py",
                ["_all_reduce_asym_topk_indices"],
                {
                    "get_parallel": lambda: SimpleNamespace(attn_tp_group=pair),
                    "torch": SimpleNamespace(int32="int32"),
                    "dist": SimpleNamespace(
                        all_reduce=all_reduce, ReduceOp=SimpleNamespace(SUM="sum")
                    ),
                },
            )
            indices = FakeTensor((7, 2048))
            reduce_topk = namespace["_all_reduce_asym_topk_indices"]
            self.assertIs(reduce_topk(indices), indices)
            all_reduce.assert_called_once_with(
                indices, op="sum", group=pair.device_group
            )
            for invalid in (
                FakeTensor((7, 2048), "int64"),
                FakeTensor((7, 2048), contiguous=False),
            ):
                with self.assertRaises(ValueError):
                    reduce_topk(invalid)
            pair.world_size = 16
            with self.assertRaises(ValueError):
                reduce_topk(indices)

    def test_shared_topk_skips_collective_on_both_roles(self):
        previous = FakeTensor((7, 2048))
        for controller in (True, False):
            model = self.make_role(controller, skip_topk=True)
            state = self.prepare(model, previous)
            self.assertIs(state[4], previous)
            model.indexer.assert_not_called()
            if controller:
                model.fused_qkv_a_proj_with_mqa.assert_not_called()
            else:
                model.q_b_proj.assert_called_once()
        # A SUM on already-replicated TopK would double every token index.
        self.collective.assert_not_called()

    def test_shared_layer_without_previous_topk_fails(self):
        for controller in (True, False):
            model = self.make_role(controller, skip_topk=True)
            with self.assertRaisesRegex(ValueError, "previous TopK"):
                self.prepare(model)
            model.fused_qkv_a_proj_with_mqa.assert_not_called()
            model.indexer.assert_not_called()
        self.collective.assert_not_called()

    def test_fresh_topk_controller_supplies_indices_compute_supplies_zeros(self):
        controller = self.make_role(True, skip_topk=False)
        compute = self.make_role(False, skip_topk=False)
        self.prepare(controller)
        controller.indexer.assert_called_once()
        self.collective.assert_called_once_with(controller.indexer.return_value)
        self.collective.reset_mock()
        self.prepare(compute)
        compute.indexer.assert_not_called()
        zeros = self.namespace["torch"].zeros
        zeros.assert_called_once_with((7, 2048), dtype="int32", device="npu")
        self.collective.assert_called_once()
        self.assertEqual(self.collective.call_args.args[0].shape, (7, 2048))

    def test_first_mtp_iteration_recomputes_topk_but_later_reuses(self):
        controller = self.make_role(True, skip_topk=True, is_nextn=True)
        initial_state = self.prepare(controller)
        controller.indexer.assert_called_once()
        self.collective.assert_called_once()
        previous = initial_state[4]
        controller.indexer.reset_mock()
        self.collective.reset_mock()
        self.assertIs(self.prepare(controller, previous)[4], previous)
        controller.indexer.assert_not_called()
        self.collective.assert_not_called()

    def test_glm_interleaved_rope_uses_dynamic_linear_and_full_head_bmm(self):
        fused_norm = Mock(
            return_value=(
                FakeTensor((7, 2048), "bfloat16"),
                FakeTensor((7, 1, 512), "bfloat16"),
                FakeTensor((7, 1, 64), "bfloat16"),
            )
        )
        self.namespace["fused_split_qk_norm"] = fused_norm
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                compute = self.make_role(False, skip_topk=reuse)
                compute.rotary_emb.is_neox_style = False
                previous = FakeTensor((7, 2048)) if reuse else None
                self.collective.reset_mock()
                state = self.prepare(compute, previous)
                self.assertEqual(state[0].shape, (7, 64, 64))
                self.assertEqual(state[2].shape, (7, 64, 512))
                self.assertEqual(state[3].shape, (7, 1, 512))
                compute.q_b_proj.assert_called_once_with(fused_norm.return_value[0])
                compute.indexer.assert_not_called()
                if reuse:
                    self.assertIs(state[4], previous)
                    self.collective.assert_not_called()
                else:
                    self.collective.assert_called_once()

    def test_controller_core_returns_zero_and_preserves_indexshare(self):
        zero_output = object()
        hidden_states = SimpleNamespace(
            shape=(7, 6144), new_zeros=Mock(return_value=zero_output)
        )
        topk = FakeTensor((7, 2048))
        for next_skip_topk in (True, False):
            model = SimpleNamespace(
                is_asym_controller=True, hidden_size=6144, next_skip_topk=next_skip_topk
            )
            output, carry = self.namespace["forward_dsa_asym_core_npu"](
                model, None, None, None, None, topk, hidden_states, None, None, None
            )
            self.assertIs(output, zero_output)
            self.assertIs(carry, topk if next_skip_topk else None)
            hidden_states.new_zeros.assert_called_with((7, 6144))

    def test_compute_core_delegates_to_standard_dynamic_quantization_path(self):
        core = Mock(return_value=(object(), object()))
        self.namespace["forward_dsa_core_npu"] = core
        model = SimpleNamespace(is_asym_controller=False)
        inner = tuple(object() for _ in range(9))
        result = self.namespace["forward_dsa_asym_core_npu"](model, *inner)
        self.assertIs(result, core.return_value)
        core.assert_called_once_with(model, *inner[:5], *inner[6:])


class TestAsymmetricMLAWeights(unittest.TestCase):
    def make_loader(self, attention, enabled=True):
        namespace = load_definitions(
            "models/deepseek_common/deepseek_weight_loader.py",
            ["_is_missing_asym_dsa_attention_weight", "post_load_weights"],
            {
                "envs": env_namespace(enabled),
                "torch": SimpleNamespace(
                    float8_e4m3fn="fp8", float8_e4m3fnuz="fp8uz", int8="int8"
                ),
                "_use_aiter_gfx95": False,
                "_is_npu": True,
                "_is_musa": False,
                "bind_or_assign": lambda old, new: new,
            },
            class_name="DeepseekV2WeightLoaderMixin",
        )
        loader = namespace["DeepseekV2WeightLoaderMixin"]()
        loader.get_submodule = Mock(return_value=attention)
        loader.config = SimpleNamespace(num_hidden_layers=78)
        loader.model = SimpleNamespace(
            start_layer=0,
            end_layer=1,
            layers=[SimpleNamespace(self_attn=attention)],
            decoder=SimpleNamespace(self_attn=attention),
        )
        loader.quant_config = SimpleNamespace(get_name=lambda: "modelslim")
        return loader

    def test_missing_role_projection_weights_and_scales_are_filtered(self):
        for rank in (0, 1):
            attention, _ = make_attention(rank)
            loader = self.make_loader(attention)
            for prefix in ("model.layers.0", "model.decoder"):
                for projection in ("q_b_proj", "kv_b_proj", "o_proj", "indexer"):
                    for suffix in ("weight", "weight_scale", "weight_offset"):
                        parameter = (
                            "wq_b." + suffix if projection == "indexer" else suffix
                        )
                        name = f"{prefix}.self_attn.{projection}.{parameter}"
                        skipped = loader._is_missing_asym_dsa_attention_weight(name, {})
                        self.assertEqual(skipped, not hasattr(attention, projection))
                        self.assertFalse(
                            loader._is_missing_asym_dsa_attention_weight(
                                name, {name: object()}
                            )
                        )
            for name in (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale",
                "model.layers.0.self_attn.q_a_layernorm.weight",
            ):
                self.assertFalse(loader._is_missing_asym_dsa_attention_weight(name, {}))

    def test_disabled_feature_does_not_filter_absent_weights(self):
        controller, _ = make_attention(0)
        loader = self.make_loader(controller, enabled=False)
        self.assertFalse(
            loader._is_missing_asym_dsa_attention_weight(
                "model.layers.0.self_attn.q_b_proj.weight_scale", {}
            )
        )

    def test_controller_skips_kv_b_postprocessing_in_target_and_mtp(self):
        controller, _ = make_attention(0)
        loader = self.make_loader(controller)
        loader.post_load_weights()
        loader.post_load_weights(is_nextn=True)
        loader.post_load_weights(
            weight_names=["model.layers.0.self_attn.kv_b_proj.weight"]
        )
        self.assertIsNone(controller.w_kc)
        self.assertIsNone(controller.w_vc)

    def test_compute_float_kv_b_is_split_into_all_glm_heads(self):
        compute, _ = make_attention(1)
        loader = self.make_loader(compute)
        compute.kv_b_proj.weight = FakeTensor((64 * (192 + 256), 512), "bfloat16")
        loader.post_load_weights()
        self.assertEqual(compute.w_kc.shape, (64, 192, 512))
        self.assertEqual(compute.w_vc.shape, (64, 512, 256))
        compute.kv_b_proj.weight = FakeTensor((32 * (192 + 256), 512), "bfloat16")
        with self.assertRaisesRegex(ValueError, "all attention heads"):
            loader.post_load_weights()


if __name__ == "__main__":
    unittest.main()
