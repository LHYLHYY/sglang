# Run native MLA FIA with a full DSA checkpoint

## Dual-stream FIA with hot-cache overlap

Select `split_graph_dual_fia` to use FIA for both hit and miss attention while
retaining the original compact dual-stream prefetch and hot-cache refill:

```bash
export SGLANG_NPU_DSA_FIA_NATIVE=0
export SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD=1
export SGLANG_NPU_SPARSE_KV_ATTN_IMPL=split_graph_dual_fia
export ASCEND_USE_FIA=1
export SGLANG_NPU_GRAPH_DEBUG=1
unset SGLANG_NPU_SPARSE_KV_FIA_SKIP_KV_IO
unset SGLANG_NPU_USE_MLAPO
unset SGLANG_USE_FIA_NZ
```

Keep the complete model and benchmark from the successful `combined` run. Use
`--attention-backend ascend --kv-cache-dtype auto`,
`--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled`,
`--disable-radix-cache` and the original explicit `--max-running-requests`.
An explicit `--cuda-graph-config` overrides the convenience graph flags.
The current hot-cache manager requires `index_topk=2048`; MLA dimensions must
be 512 + 64. Page size must divide 2048, be a multiple of 16 and be <=1024
(for example 128). CP, speculative decoding, torch.compile, MLAPO, FIA_NZ,
PDMux, two-batch overlap and PD disaggregation are rejected in this mode.

This mode requires a matching **torch_npu/CANN with FIA v2 MLA decode mask and
LSE support on Atlas A2/A3**. The legacy FIA API used by `combined` does not
support LSE for MLA D=512, so a successful combined test alone does not establish
this capability. See the [legacy FIA restrictions](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md)
and [FIA v2 contract](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md).
Startup checks the v2 `.out`, maximum-workspace API and graph handler. Full-model
warmup then exercises the actual masked MLA/LSE combination; older CANN versions
can still reject it. Ascend 950 MLA decode does not support the mask used here
and is rejected. The mode never silently falls back to SFA or combined FIA.
The installed `sgl_kernel_npu` must also provide both the Python wrapper
`unidex_split_copy_promote_inplace` and its registered NPU kernel
`unidex_split_copy_promote`; startup reports a missing dependency explicitly.

During graph decode, the main stream runs the HBM hit copy and hit FIA while
the worker stream runs the host miss copy, miss FIA and refill. Merge waits
for miss attention; the layer joins refill before reusing the shared workspace.
The hit/miss KV buffers are independent. This reuses the compact
`split_graph_dual` implementation, rather than the shared noncompact workspace
of `split_graph_dual_v2`. Both graph and eager decode use the existing split-copy
promotion kernel to read each valid source KV row once and write contiguous
NoPE/RoPE plus a combined compact snapshot for refill. Here the promotion output
is the snapshot, not the live hot cache: refill still waits for hit gathering
before replacing cache slots.

Per-layer hit/miss counts stay on the device. Each partition passes fixed
selected-buffer capacity as `actual_seq_kvlen` and uses a `[B,1,1,K]` device mask
to exclude its unused rows. FIA partitions do not build SFA `sparse_indices` or
`actual_seq_lengths_kv`. Each producer stream clears one flat FIA buffer before
the fused gather; its NoPE/RoPE views then contain zero tails even when counts
shrink. The copy kernel skips invalid descriptors, so this one clear is needed
to prevent masked V from containing stale NaNs. The former combined-KV dummy
clear, whole-KV `where` and two contiguous layout copies are removed. FIA only
views the prepared storage as pages; concurrent refill reads the separate
combined snapshot. Graph capture preallocates and shares these buffers across
layers, joining refill before reuse.
Empty partitions get one zero dummy token and are neutralized before merge.
FIA v2 returns FP32 LSE; passing `max=LSE, sum=1` to the existing stable merge
weights both outputs by their softmax mass. Eager decode uses the original
parallel prefetch with these same FIA partitions; prefill remains SFA.

Every graph replay retains the blocking nonempty graph-buffer read before
update/replay. The update uses **`actual_seq_kvlen`**, broadcast to every layer's
two FIA tasks; padding and IDLE lengths are zero. Expected replay logs include
`seq_lens.cpu.returned`, then `seq_lens.selected_kv route=sparse_fia_dual` and
`update.prepare input_keys=[['actual_seq_kvlen']]`. Capture should report two
FIA update tasks per attention layer in the captured graph.

The CPU checks cover partition masking/merge and stream dependency scheduling.
Use the unchanged full-model benchmark below to validate device completion,
accuracy and performance. The additional mask and partition work means an actual
speedup must be measured; CPU tests cannot establish device overlap or throughput.

## Original DSA offload with hot-cache prefetch

The replay ordering fix also applies to the original `combined` route and to
`split_eager` when it falls back to combined FIA under decode graphs. These
modes keep `SparseKVCacheManager`, host KV storage, hot-cache prefetch, request
reset hooks and callback streams. They do not select `NativeFIAOffloadManager`.

To return from the working `native_fia` integration to the original path, set
these variables in the **server** shell and restart the same full-model server:

```bash
export SGLANG_NPU_DSA_FIA_NATIVE=0
export SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD=1
export SGLANG_NPU_SPARSE_KV_ATTN_IMPL=combined
export ASCEND_USE_FIA=1
export SGLANG_NPU_GRAPH_DEBUG=1
unset SGLANG_NPU_SPARSE_KV_FIA_SKIP_KV_IO
```

Keep the working model, TP/DP, memory settings and launch arguments, including
`--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled`.
Run the unchanged full-model benchmark below. Replay logs should contain
`seq_lens.cpu.begin`, `seq_lens.cpu.returned`, then `seq_lens.selected_kv`
with `route=sparse_fia_mla` and `skip_kv_io=False` before update/replay starts.

Every DECODE and IDLE replay now reads the nonempty loaded graph length buffer
before starting FIA task updates, even when lengths are available on the CPU.
The original combined FIA continues to use selected-buffer capacity for real
requests; graph padding and IDLE ranks explicitly use zero. The device readback
provides ordering and must not substitute full-context lengths for selected-KV
lengths. This preserves the existing combined FIA diagnostic's treatment of
selected-buffer padding and does not establish numerical accuracy.

`split_eager` uses this same fix for its graph-mode FIA fallback; its eager
split-SFA path is unchanged. `split_graph`, `split_graph_dual`,
`split_graph_dual_v2` and `pa_graph` retain their SFA replay paths without this
FIA CPU update. Full-model completion has been reported for both `native_fia`
and `combined` after the ordering fix.

## DSA with real KV offload

`SGLANG_NPU_SPARSE_KV_ATTN_IMPL=native_fia` restores the real DSA indexer
and host KV storage while sharing the same `AscendAttnBackend.forward_mla_fia`
implementation as the working native FIA diagnostic. Keep the complete
DeepSeek V3/V3.2 model, its original `index_topk`, weights, TP/DP and benchmark.

In the **server** shell, set:

```bash
export SGLANG_NPU_DSA_FIA_NATIVE=0
export SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD=1
export SGLANG_NPU_SPARSE_KV_ATTN_IMPL=native_fia
export ASCEND_USE_FIA=1
export SGLANG_NPU_GRAPH_DEBUG=1
unset SGLANG_NPU_SPARSE_KV_FIA_SKIP_KV_IO
unset SGLANG_NPU_USE_MLAPO
unset SGLANG_USE_FIA_NZ
```

Use these arguments with the original full-model launch command:

```bash
--attention-backend ascend \
--kv-cache-dtype auto \
--cuda-graph-backend-decode full \
--cuda-graph-backend-prefill disabled \
--disable-radix-cache \
--max-running-requests 32
```

The explicit request limit bounds host KV allocation; keep a different limit
if required by the original deployment. If `--cuda-graph-config` is supplied,
its JSON overrides the graph convenience flags: set its prefill backend to
`disabled` too. Context parallelism, speculative decoding, torch.compile,
MLAPO, FIA_NZ, PDMux, two-batch overlap and PD disaggregation are rejected.
The native-only switch must be **0**: setting it to 1 disables offload.

Install matching `sgl_kernel_npu` Python wrappers and native operators with
`unidex_split_copy_inplace` support. The selected-KV gather uses this operator
to copy registered host KV directly into the separate nope/rope pages; an old
kernel wheel or wrapper without it is rejected at startup.

The new path performs the following work:

- Prefill runs the original DSA indexer and eager SFA while writing full KV to
  host memory. The index KV pool remains in HBM.
- Decode writes the new token's KV, gathers all DSA-selected tokens from host
  into persistent, separate nope/rope pages, and calls stock MLA FIA on the
  same current NPU stream. One staging allocation is shared across layers;
  it does not use hot-cache hit/miss logic, request reset hooks or host callbacks.
- FIA receives `min(sequence_length, index_topk)` as its CPU KV length. Graph
  padding and IDLE ranks receive zero. The indexer retains full-context lengths
  and page tables. Staging capacity is page-aligned and reserved before sizing
  the index pool, including the largest configured decode graph bucket.

This first integration transfers all selected KV on each decode step. It
restores functional offload but does not include the old hot-cache performance
optimization. Prefix sharing is disabled because host KV belongs to individual
requests. Restart the server when changing modes.

Expected startup log: `NPU DSA native FIA offload: indexer enabled, full KV on
host`. Decode replay logs contain `stage=seq_lens.selected_kv` and
`route=native_fia_offload`. Run the unchanged full-model benchmark shown below
and verify completion beyond warmup. This integration has CPU regression
coverage; NPU graph capture, multi-rank completion and accuracy still require
validation on the Ascend server.

## Native FIA baseline without offload

`SGLANG_NPU_DSA_FIA_NATIVE=1` is a startup-only diagnostic for DeepSeek V3/V3.2
DSA models. It keeps the original model configuration (including `index_topk`),
model size, checkpoint and indexer weights. It executes the stock Ascend MHA
prefill and MLA decode paths instead of sparse attention. Generated output is
not an accuracy check for DSA.

### What changes

| Component | DSA offload, combined FIA | Native FIA diagnostic |
| --- | --- | --- |
| Prefill | DSA preparation, indexer, host KV, SFA | Stock `forward_mha_prepare_npu` / `forward_mha_core_npu` |
| Decode preparation | DSA indexer and selected KV gathering | Stock `forward_mla_prepare_npu` / `forward_mla_core_npu`, no indexer |
| KV | Host backing plus selected/hot-cache buffers | Persistent native paged HBM nope/rope buffers |
| Decode attention | `_run_combined_decode_fia` | `forward_decode_graph` through shared `forward_mla_fia` |
| Page table | Contiguous selected-KV page IDs | Stock request page table |
| Graph CPU lengths | Selected capacity, normally 2048 | Actual full context lengths; zero for graph padding |
| Offload setup | Manager, host registration, request reset hook, callback streams | None, even if the old offload environment switch remains set |

The FIA workspace query, `.out` call, BSND layout, head padding and output
slicing are the existing stock implementation, not a copy of that code. The
diagnostic retains the unused index KV buffer and accounts for all three native
buffers using the NPU KV dtype when calculating capacity.

### Full-model baseline server

In the **server** shell, before launching the original full model:

```bash
export SGLANG_NPU_DSA_FIA_NATIVE=1
export ASCEND_USE_FIA=1
export SGLANG_NPU_GRAPH_DEBUG=1
unset SGLANG_NPU_SPARSE_KV_FIA_SKIP_KV_IO
unset SGLANG_NPU_USE_MLAPO
unset SGLANG_USE_FIA_NZ
```

Keep the original model path, quantization, TP/DP and other normal launch
arguments. Use `--attention-backend ascend --cuda-graph-backend-decode full`
and unquantized KV (`--kv-cache-dtype auto` or `bfloat16`). Remove any
`index_topk: null` model override: the diagnostic requires the original DSA
configuration. Context parallelism, speculative decoding, torch.compile,
MLAPO and FIA_NZ are rejected by this first diagnostic implementation.

Full HBM KV uses more device memory than offload. Let the server recalculate
capacity; do not assume an offload-sized `--max-total-tokens` still fits.
The unchanged `--max-running-requests` controls the full-model request pool.

## Full-model benchmark for either mode

After restarting the server, run the same benchmark:

```bash
python -m sglang.bench_serving \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --backend sglang --host 127.0.0.1 --port 6688 \
  --max-concurrency 32 --dataset-name random \
  --random-input-len 10000 --random-output-len 32 \
  --num-prompts 16 --random-range-ratio 1 --request-rate 7
```

The baseline startup log is `NPU DSA native FIA diagnostic: stock MHA prefill +
MLA decode`, with replay logs `stage=seq_lens.native_mla` and
`route=native_mla_fia`. Confirm that graph capture reports FIA update tasks and
that the benchmark progresses beyond `Starting warmup with 1 sequences...`
to its final results. HTTP 200 and host `update.returned` messages alone do not
establish device completion.

Disable the diagnostic by unsetting `SGLANG_NPU_DSA_FIA_NATIVE` and restarting
the server. The existing DSA/offload switches then take effect again.

## If eager offload succeeds but decode graph replay stalls

Keep the full model, offload mode, TP/DP and benchmark unchanged and restart
with `--cuda-graph-backend-decode full` and `SGLANG_NPU_GRAPH_DEBUG=1`.
Keep prefill graphs disabled. Check any `--cuda-graph-config` JSON too: its
decode setting overrides the convenience flag.

The `native_fia`, `combined`, `split_graph_dual_fia` and graph-mode `split_eager`
replay paths read
`self.buffers.seq_lens[:self.bs].cpu().tolist()`
from the loaded graph buffer before starting FIA's background `graph.update`
and `graph.replay`. This device-to-host snapshot restores the ordering point
present in the working native MLA baseline that cached CPU lengths or fixed
selected-KV capacity omit.
The graph bucket is nonempty even on IDLE ranks; reading an empty raw batch
would not provide that wait. IDLE ranks still pass zero lengths to FIA. The
change is a targeted alignment with the baseline; it does not establish the
cause of the device stall.

For each rank, inspect the debug stages in this order:

1. `seq_lens.cpu.begin`, then `seq_lens.cpu.returned`.
2. `seq_lens.selected_kv` with `route=native_fia_offload` for `native_fia`, or
   `route=sparse_fia_mla` for `combined` / graph-mode `split_eager`, or
   `route=sparse_fia_dual` for `split_graph_dual_fia`.
3. `update.begin` / `replay.begin`, followed by their return and join stages.

If the benchmark still stalls, retain each rank's final stages and the earliest
Ascend runtime error. A stall before `seq_lens.cpu.returned` locates the host
wait before graph update; a later stall needs investigation in the subsequent
update, replay or device work. Neither the CPU mocks nor host return messages
prove successful NPU execution. Confirm the benchmark finishes and produces
its final measurements.

## Local regression checks

```bash
python test/manual/ascend/test_fia_graph_replay_update.py -v
python test/manual/ascend/test_native_fia_offload_manager.py -v
python test/manual/ascend/test_split_fia_attention.py -v
python test/manual/ascend/test_dual_fia_prefetch.py -v
```

These CPU mocks check configuration, staging memory accounting, shared FIA
arguments and head padding, real offload routing, and selected/full-context
graph updates including idle/padded batches. They also verify that native and
original combined FIA update/replay wait for each device-length snapshot to
return, while SFA graph routes remain unchanged. They do not validate NPU
kernels, multi-rank completion or numerical accuracy. Use the full-model run
above for device validation.
