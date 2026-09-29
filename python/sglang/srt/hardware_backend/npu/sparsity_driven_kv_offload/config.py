"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

_ENABLE_ENV_VAR = "SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD"

SPARSE_KV_ATTN_IMPL_ENV_VAR = "SGLANG_NPU_SPARSE_KV_ATTN_IMPL"
SPARSE_KV_ATTN_IMPL_COMBINED = "combined"
SPARSE_KV_ATTN_IMPL_NATIVE_FIA = "native_fia"
SPARSE_KV_ATTN_IMPL_SPLIT_EAGER = "split_eager"
SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH = "split_graph"
SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL = "split_graph_dual"
SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA = "split_graph_dual_fia"
SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2 = "split_graph_dual_v2"
SPARSE_KV_ATTN_IMPL_PA_GRAPH = "pa_graph"
SPARSE_KV_ATTN_IMPL_CHOICES = (
    SPARSE_KV_ATTN_IMPL_COMBINED,
    SPARSE_KV_ATTN_IMPL_NATIVE_FIA,
    SPARSE_KV_ATTN_IMPL_SPLIT_EAGER,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA,
    SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_V2,
    SPARSE_KV_ATTN_IMPL_PA_GRAPH,
)

# Diagnostic experiment A: keep FIA but remove real decode KV traffic.
SPARSE_KV_FIA_SKIP_KV_IO_ENV_VAR = "SGLANG_NPU_SPARSE_KV_FIA_SKIP_KV_IO"

# Full-model diagnostic: reuse stock Ascend attention without changing index_topk.
DSA_FIA_NATIVE_ENV_VAR = "SGLANG_NPU_DSA_FIA_NATIVE"

SPARSE_KV_MERGE_IMPL_ENV_VAR = "SGLANG_NPU_SPARSE_KV_MERGE_IMPL"
SPARSE_KV_MERGE_IMPL_AUTO = "auto"
SPARSE_KV_MERGE_IMPL_PYTHON = "python"
SPARSE_KV_MERGE_IMPL_FUSED = "fused"
SPARSE_KV_MERGE_IMPL_CHOICES = (
    SPARSE_KV_MERGE_IMPL_AUTO,
    SPARSE_KV_MERGE_IMPL_PYTHON,
    SPARSE_KV_MERGE_IMPL_FUSED,
)


def is_dsa_fia_native_requested() -> bool:
    return is_npu() and get_bool_env_var(DSA_FIA_NATIVE_ENV_VAR)


def is_dsa_fia_native_enabled(
    *, model_config: ModelConfig, server_args: ServerArgs, use_mla_backend: bool,
) -> bool:
    """Validate the startup-only, full-model native FIA diagnostic.

    Keep the DSA config and checkpoint, but use dense attention and native HBM
    KV throughout prefill/decode. This overrides offload, including allocation
    hooks and host callbacks; it is not an accuracy-preserving DSA backend.
    """
    if not is_dsa_fia_native_requested():
        return False
    if not (
        server_args.attention_backend == "ascend"
        and use_mla_backend
        and is_deepseek_dsa(model_config.hf_config)
        and model_config.hf_config.architectures[0]
        in ("DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM")
    ):
        raise ValueError(
            f"{DSA_FIA_NATIVE_ENV_VAR}=1 requires a DeepSeek V3/V3.2 DSA "
            "model with --attention-backend ascend and MLA enabled. "
            "Keep index_topk in the model config."
        )
    if server_args.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise ValueError(
            f"{DSA_FIA_NATIVE_ENV_VAR}=1 requires unquantized native MLA KV; "
            "use --kv-cache-dtype auto or bfloat16."
        )
    if (
        server_args.enable_prefill_cp
        or (server_args.attn_cp_size or 1) > 1
        or server_args.dcp_size > 1
        or server_args.speculative_algorithm is not None
        or server_args.enable_torch_compile
        or get_bool_env_var("SGLANG_NPU_USE_MLAPO")
        or get_bool_env_var("SGLANG_USE_FIA_NZ")
    ):
        raise ValueError(
            f"{DSA_FIA_NATIVE_ENV_VAR}=1 uses the stock non-speculative ND "
            "MLA/FIA path. Disable context parallelism, speculative decoding, "
            "torch.compile, MLAPO and FIA_NZ for this diagnostic."
        )
    return True


def get_dsa_fia_native_cell_size(
    *,
    model_config: ModelConfig,
    server_args: ServerArgs,
    use_mla_backend: bool,
    num_layers: int,
    element_size: int,
) -> Optional[int]:
    if not is_dsa_fia_native_enabled(
        model_config=model_config,
        server_args=server_args,
        use_mla_backend=use_mla_backend,
    ):
        return None
    # NPUMLATokenToKVPool retains its index buffer, even though this diagnostic
    # never runs the indexer. Unlike the GPU DSA pool, all three NPU buffers use
    # the KV dtype; account for them before deciding the available token count.
    return (
        (
            model_config.kv_lora_rank
            + model_config.qk_rope_head_dim
            + model_config.index_head_dim
        )
        * num_layers
        * element_size
    )


def is_sparsity_driven_kv_offload_requested() -> bool:
    return get_bool_env_var(_ENABLE_ENV_VAR) and not is_dsa_fia_native_requested()


def _get_native_fia_offload_graph_config(server_args: ServerArgs):
    """Use the published phase settings once runtime overrides are resolved."""
    try:
        config = get_exec().graph.cuda_graph_config
    except ValueError:
        config = None
    return config if config is not None else server_args.cuda_graph_config


def is_sparse_kv_decode_graph_enabled(server_args: ServerArgs) -> bool:
    """Read the resolved decode setting before validating FIA graph support."""
    return _get_native_fia_offload_graph_config(server_args).decode.backend != "disabled"


def get_native_fia_offload_max_batch_size(server_args: ServerArgs) -> int:
    """Conservative per-rank bound shared by staging allocation and budgeting."""
    if (
        server_args.max_running_requests is None
        or server_args.max_running_requests <= 0
    ):
        raise ValueError("native_fia requires a positive --max-running-requests.")
    max_batch_size = server_args.max_running_requests
    decode = _get_native_fia_offload_graph_config(server_args).decode
    if decode.backend != "disabled":
        if decode.max_bs is not None:
            max_batch_size = max(max_batch_size, decode.max_bs)
        if decode.bs:
            max_batch_size = max(max_batch_size, max(decode.bs))
    # A DP rank can replay an otherwise empty batch padded to another rank's
    # active bucket. Keep the global request limit, rather than dividing by DP,
    # and conservatively cover eager / graph attention-TP alignment as well.
    alignment = max(1, server_args.tp_size)
    return (max_batch_size + alignment - 1) // alignment * alignment


def get_native_fia_offload_buffer_size(
    *,
    model_config: ModelConfig,
    server_args: ServerArgs,
    page_size: int,
    element_size: int,
) -> int:
    """Persistent selected-KV staging bytes, excluding the per-token index pool."""
    if (
        not is_sparsity_driven_kv_offload_requested()
        or get_sparse_kv_attn_impl() != SPARSE_KV_ATTN_IMPL_NATIVE_FIA
    ):
        return 0
    is_sparsity_driven_kv_offload_enabled(
        model_config=model_config, server_args=server_args, use_mla_backend=True,
    )
    if page_size <= 0 or element_size <= 0:
        raise ValueError("native_fia staging requires positive page and element sizes.")
    topk = model_config.hf_config.index_topk
    padded_topk = (topk + page_size - 1) // page_size * page_size
    max_batch_size = get_native_fia_offload_max_batch_size(server_args)
    kv_bytes = (
        max_batch_size
        * padded_topk
        * (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
        * element_size
    )
    block_table_bytes = max_batch_size * (padded_topk // page_size) * 4
    token_position_bytes = padded_topk * 8
    return kv_bytes + block_table_bytes + token_position_bytes


def get_sparse_kv_attn_impl() -> str:
    value = (
        os.getenv(SPARSE_KV_ATTN_IMPL_ENV_VAR, SPARSE_KV_ATTN_IMPL_COMBINED)
        .strip()
        .lower()
    )
    if value not in SPARSE_KV_ATTN_IMPL_CHOICES:
        allowed_values = ", ".join(SPARSE_KV_ATTN_IMPL_CHOICES)
        raise ValueError(
            f"{SPARSE_KV_ATTN_IMPL_ENV_VAR} must be one of: {allowed_values}; "
            f"got {value!r}."
        )
    return value


def get_sparse_kv_fia_skip_kv_io(attn_impl: str) -> bool:
    """Read the decode-only FIA diagnostic switch once at manager startup.

    This invalidates generated outputs. Restart the service to change the
    switch, since it changes which operations are captured in the graph.
    """
    enabled = get_bool_env_var(SPARSE_KV_FIA_SKIP_KV_IO_ENV_VAR)
    if enabled and attn_impl != SPARSE_KV_ATTN_IMPL_COMBINED:
        raise ValueError(
            f"{SPARSE_KV_FIA_SKIP_KV_IO_ENV_VAR}=1 requires "
            f"{SPARSE_KV_ATTN_IMPL_ENV_VAR}={SPARSE_KV_ATTN_IMPL_COMBINED}; "
            f"got {attn_impl!r}. This diagnostic is only supported for combined."
        )
    return enabled


def get_sparse_kv_merge_impl() -> str:
    """Select the split-attention state merge implementation.

    ``auto`` uses the fused sgl-kernel-npu operator when it is installed and
    otherwise falls back to the graph-safe PyTorch implementation. ``fused``
    is a strict opt-in that fails on the first merge call when the matching
    kernel wheel is unavailable.
    """

    value = (
        os.getenv(SPARSE_KV_MERGE_IMPL_ENV_VAR, SPARSE_KV_MERGE_IMPL_AUTO)
        .strip()
        .lower()
    )
    if value not in SPARSE_KV_MERGE_IMPL_CHOICES:
        allowed_values = ", ".join(SPARSE_KV_MERGE_IMPL_CHOICES)
        raise ValueError(
            f"{SPARSE_KV_MERGE_IMPL_ENV_VAR} must be one of: {allowed_values}; "
            f"got {value!r}."
        )
    return value


def is_sparsity_driven_kv_offload_enabled(
    *, model_config: ModelConfig, server_args: ServerArgs, use_mla_backend: bool,
) -> bool:
    if not is_sparsity_driven_kv_offload_requested():
        return False

    if not (
        is_npu()
        and server_args.attention_backend == "ascend"
        and use_mla_backend
        and is_deepseek_dsa(model_config.hf_config)
    ):
        raise ValueError(
            f"{_ENABLE_ENV_VAR} requires an NPU DeepSeek DSA model using "
            "the Ascend MLA attention backend."
        )
    if server_args.max_running_requests is None:
        raise ValueError(
            f"{_ENABLE_ENV_VAR} requires an explicit "
            "--max-running-requests to bound the per-process host KV allocation."
        )
    attn_impl = get_sparse_kv_attn_impl()
    if attn_impl in (
        SPARSE_KV_ATTN_IMPL_NATIVE_FIA,
        SPARSE_KV_ATTN_IMPL_SPLIT_GRAPH_DUAL_FIA,
    ):
        if model_config.hf_config.architectures[0] not in (
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
        ):
            raise ValueError(
                f"{attn_impl} offload requires a DeepSeek V3/V3.2 DSA model."
            )
        topk = model_config.hf_config.index_topk
        if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
            raise ValueError(
                f"{attn_impl} offload requires a positive integer index_topk."
            )
        if server_args.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
            raise ValueError(
                f"{attn_impl} offload requires unquantized KV; "
                "use --kv-cache-dtype auto or bfloat16."
            )
        if (
            server_args.enable_prefill_cp
            or (server_args.attn_cp_size or 1) > 1
            or server_args.dcp_size > 1
            or server_args.speculative_algorithm is not None
            or server_args.enable_torch_compile
            or get_bool_env_var("SGLANG_NPU_USE_MLAPO")
            or get_bool_env_var("SGLANG_USE_FIA_NZ")
        ):
            raise ValueError(
                f"{attn_impl} offload requires the non-speculative ND MLA/FIA path. "
                "Disable context parallelism, speculative decoding, torch.compile, "
                "MLAPO and FIA_NZ."
            )
        if server_args.enable_pdmux or server_args.enable_two_batch_overlap:
            raise ValueError(
                f"{attn_impl} offload shares selected-KV staging across layers; "
                "disable PDMux and two-batch overlap."
            )
        if (
            _get_native_fia_offload_graph_config(server_args).prefill.backend
            != "disabled"
        ):
            raise ValueError(
                f"{attn_impl} offload requires the prefill graph backend disabled."
            )
        if not server_args.disable_radix_cache:
            raise ValueError(
                f"{attn_impl} offload requires --disable-radix-cache: "
                "host KV is owned by individual requests and cannot share prefixes."
            )
        if server_args.disaggregation_mode != "null":
            raise ValueError(f"{attn_impl} offload does not support PD disaggregation.")
        if get_bool_env_var(SPARSE_KV_FIA_SKIP_KV_IO_ENV_VAR):
            raise ValueError(
                f"{attn_impl} offload requires real KV IO; disable FIA_SKIP_KV_IO."
            )
        if attn_impl == SPARSE_KV_ATTN_IMPL_NATIVE_FIA:
            get_native_fia_offload_max_batch_size(server_args)
        else:
            if server_args.max_running_requests <= 0:
                raise ValueError(
                    "split_graph_dual_fia requires a positive --max-running-requests."
                )
            if model_config.kv_lora_rank != 512 or model_config.qk_rope_head_dim != 64:
                raise ValueError(
                    "split_graph_dual_fia requires MLA KV dimensions 512 + 64."
                )
            # The existing hot-cache manager and its compact partition buffers
            # use this fixed capacity. Do not accept a model whose indexer would
            # generate a different number of selected rows.
            if topk != 2048:
                raise ValueError("split_graph_dual_fia requires index_topk=2048.")
            page_size = server_args.page_size
            if (
                not isinstance(page_size, int)
                or isinstance(page_size, bool)
                or page_size <= 0
                or page_size > 1024
                or page_size % 16
                or topk % page_size
            ):
                raise ValueError(
                    "split_graph_dual_fia requires a page size divisible by 16, "
                    "at most 1024, and dividing the selected-KV capacity 2048."
                )
            if server_args.enable_dsa_prefill_context_parallel:
                raise ValueError(
                    "split_graph_dual_fia does not support DSA prefill context parallelism."
                )
    return True


def get_sparsity_driven_kv_offload_cell_size(
    *,
    model_config: ModelConfig,
    server_args: ServerArgs,
    use_mla_backend: bool,
    num_layers: int,
    element_size: int,
) -> Optional[int]:
    if not is_sparsity_driven_kv_offload_enabled(
        model_config=model_config,
        server_args=server_args,
        use_mla_backend=use_mla_backend,
    ):
        return None

    index_head_dim = model_config.index_head_dim
    if index_head_dim is None:
        raise ValueError("Sparsity-driven KV offload requires an index KV cache.")
    return index_head_dim * num_layers * element_size
