"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.npu.NPUGraph per shape; attention metadata captured
    inside the graph. replay_with_input_update substitutes fresh
    seq_lens without re-recording."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._fia_update_tasks: Dict[Any, int] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._device_id = self._device_module.current_device()
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._graph_debug = get_bool_env_var("SGLANG_NPU_GRAPH_DEBUG")
        self._debug_replay_id = 0
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )
        self.debug_log("backend.init", torch_compile=self._enable_torch_compile)

    def next_debug_replay_id(self):
        if not self._graph_debug:
            return None
        self._debug_replay_id += 1
        return self._debug_replay_id

    def debug_log(self, stage, shape_key=None, debug_id=None, **details):
        """Host-side breadcrumbs only: never read tensor contents or synchronize.

        Enable SGLANG_NPU_GRAPH_DEBUG before starting the server. A `returned`
        marker means the host API returned, NOT that NPU work has completed.
        """
        if self._graph_debug:
            logger.info(
                "[NPU_GRAPH_DEBUG] pid=%d device=%s tp_rank=%s thread=%s "
                "replay=%s shape=%s stage=%s host_time=%.6f %s",
                os.getpid(),
                self._device_id,
                self._tp_group.rank_in_group,
                threading.current_thread().name,
                debug_id,
                shape_key,
                stage,
                time.monotonic(),
                " ".join(f"{key}={value}" for key, value in details.items()),
            )

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = torch.npu.graph

        with (
            skip_guard_context,
            graph_ctx(
                graph,
                pool=self._pool,
                stream=self._capture_stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out
        # Auto-dispatch can insert ExternalEvent waits for FIA (including its
        # default overload). Even fixed-length calls need update to signal them.
        dispatch_mode = getattr(graph, "graph_dispatch_mode", None)
        records = getattr(dispatch_mode, "graph_dispatch_records", ())
        fia_ops = {
            "npu_fused_infer_attention_score",
            "npu_fused_infer_attention_score.default",
            "npu_fused_infer_attention_score.out",
            "npu_fused_infer_attention_score_v2",
            "npu_fused_infer_attention_score_v2.default",
            "npu_fused_infer_attention_score_v2.out",
        }
        self._fia_update_tasks[shape_key] = sum(
            getattr(getattr(record, "op_cache_entry", None), "__name__", "") in fia_ops
            for record in records
        )
        if self._fia_update_tasks[shape_key]:
            logger.info(
                "NPU graph %s captured %d FIA update task(s); replay requires "
                "input updates.",
                shape_key,
                self._fia_update_tasks[shape_key],
            )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        debug_id = kwargs.get("debug_id")
        if debug_id is None:
            debug_id = self.next_debug_replay_id()
        self.debug_log(
            "replay.route",
            shape_key,
            debug_id,
            fia_tasks=self._fia_update_tasks[shape_key],
            route="captured_lengths" if self._fia_update_tasks[shape_key] else "direct",
        )
        if self._fia_update_tasks[shape_key]:
            # Fallback for callers without explicit CPU metadata. The decode
            # runner supplies selected-KV lengths for the combined FIA smoke
            # test via replay_with_input_update, just like ordinary MLA.
            return self.replay_with_input_update(
                shape_key, seq_lens=None, cpu_update_input=[{}], debug_id=debug_id
            )
        self.debug_log("replay.begin", shape_key, debug_id)
        self._graphs[shape_key].replay()
        self.debug_log("replay.returned", shape_key, debug_id)
        return self._outputs[shape_key]

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
        debug_id: Optional[int] = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph in a background
        thread, then replay. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph = self._graphs[shape_key]
        if debug_id is None:
            debug_id = self.next_debug_replay_id()
        if self._graph_debug:
            self.debug_log(
                "update.prepare",
                shape_key,
                debug_id,
                fia_tasks=self._fia_update_tasks.get(shape_key, 0),
                # Keys only: logging seq_lens tensors could synchronize the NPU.
                input_keys=[list(item) for item in cpu_update_input],
            )

        def _update():
            self.debug_log("update.thread.enter", shape_key, debug_id)
            try:
                self.debug_log("update.set_device.begin", shape_key, debug_id)
                self._device_module.set_device(self._device_id)
                self.debug_log("update.set_device.returned", shape_key, debug_id)
                self.debug_log("update.begin", shape_key, debug_id)
                graph.update(cpu_update_input=cpu_update_input)
                self.debug_log("update.returned", shape_key, debug_id)
            except Exception:
                # Keep the original failure behavior, but identify the rank and
                # graph even if the main thread is stuck in replay or later work.
                logger.exception(
                    "[NPU_GRAPH_DEBUG] pid=%d device=%s tp_rank=%s "
                    "replay=%s shape=%s stage=update.error",
                    os.getpid(),
                    self._device_id,
                    self._tp_group.rank_in_group,
                    debug_id,
                    shape_key,
                )
                raise

        thread = threading.Thread(target=_update)
        self.debug_log("update.thread.start", shape_key, debug_id)
        thread.start()
        self.debug_log("replay.begin", shape_key, debug_id)
        try:
            graph.replay()
        except Exception:
            logger.exception(
                "[NPU_GRAPH_DEBUG] pid=%d device=%s replay=%s shape=%s "
                "stage=replay.error",
                os.getpid(),
                self._device_id,
                debug_id,
                shape_key,
            )
            raise
        self.debug_log("replay.returned", shape_key, debug_id)
        self.debug_log("update.join.begin", shape_key, debug_id)
        thread.join()
        self.debug_log("update.join.returned", shape_key, debug_id)
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._fia_update_tasks.clear()
        self._pool = None
