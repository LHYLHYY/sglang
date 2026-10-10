# Dual-stream FIA for DSA KV offload

## Dual-stream FIA with hot-cache overlap

`split_graph_dual_fia` is the only supported offload attention implementation
and the default when `SGLANG_NPU_SPARSE_KV_ATTN_IMPL` is unset. It uses FIA for
both hit and miss attention, with dual-stream prefetch and hot-cache refill.
The explicit setting below is optional. Restart the service after changing
startup configuration.

The retired values `combined`, `native_fia`, `split_eager`, `split_graph`,
`split_graph_dual`, `split_graph_dual_v2` and `pa_graph` are rejected at startup.
Remove an old setting or replace it with `split_graph_dual_fia`.
The separate `SGLANG_NPU_DSA_FIA_NATIVE` dense diagnostic and
`SGLANG_NPU_GRAPH_DEBUG` logging switch remain available.

In the server shell:

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

Keep the complete model, TP/DP configuration and benchmark. Use
`--attention-backend ascend --kv-cache-dtype auto`,
`--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled`,
`--disable-radix-cache` and the original explicit `--max-running-requests`.
An explicit `--cuda-graph-config` overrides the convenience graph flags.
The current hot-cache manager requires `index_topk=2048`; MLA dimensions must
be 512 + 64. Page size must divide 2048, be a multiple of 16 and be <=1024
(for example 128). CP, speculative decoding, torch.compile, MLAPO, FIA_NZ,
PDMux, two-batch overlap and PD disaggregation are rejected in this mode.

This mode requires a matching **torch_npu/CANN with FIA v2 MLA decode mask and
LSE support on Atlas A2/A3**. See the
[FIA v2 contract](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md).
Startup checks the v2 `.out`, maximum-workspace API and graph handler. Full-model
warmup then exercises the actual masked MLA/LSE combination; older CANN versions
can still reject it. Ascend 950 MLA decode does not support the mask used here
and is rejected. Decode always uses FIA v2, including the eager path.
The installed `sgl_kernel_npu` must also provide both the Python wrapper
`unidex_split_copy_promote_inplace` and its registered NPU kernel
`unidex_split_copy_promote`; startup reports a missing dependency explicitly.

During graph decode, the main stream runs the HBM hit copy and hit FIA while
the worker stream runs the host miss copy, miss FIA and refill. Merge waits
for miss attention; the layer joins refill before reusing the shared workspace.
The hit/miss KV buffers are independent. Graph FIA preserves each selected KV
at its original Top-K position `j`, using the same preallocated `b*K+j`
destination indices for both partitions. There are no hit/miss `cumsum`, rank
clamps, or batch-offset calculations on this graph path. Eager FIA prefetch
still compacts its partitions and retains its corresponding prefix masks.
Both graph and eager decode use the existing split-copy promotion kernel to
read each valid source KV row once and write contiguous NoPE/RoPE plus a
combined snapshot for refill. Here the promotion output is the snapshot, not
the live hot cache: refill still waits for hit gathering before replacing
cache slots. Graph FIA refills snapshot position `b*K+j` into hot-cache
position `req_id*K+j`, retaining the existing `token -> j` slot-map mapping.

Per-layer hit/miss counts stay on the device. Each partition passes fixed
selected-buffer capacity as `actual_seq_kvlen` and uses a `[B,1,1,K]` device mask
to exclude its unused rows. Graph masks use the actual hit/miss validity at each
Top-K position, including holes; only eager masks use a valid prefix defined by
the count. Attention does not mutate the validity masks used by refill.
FIA partitions do not build SFA `sparse_indices` or
`actual_seq_lengths_kv`. Each producer stream clears one flat FIA buffer before
the fused gather; its NoPE/RoPE views then contain zeros in unused positions,
including when the selected positions change without changing their count.
The copy kernel skips invalid descriptors, so this one clear is needed
to prevent masked V from containing stale NaNs. The former combined-KV dummy
clear, whole-KV `where` and two contiguous layout copies are removed. FIA only
views the prepared storage as pages; concurrent refill reads the separate
combined snapshot. Graph capture preallocates and shares these buffers across
layers, joining refill before reuse.
Empty partitions alone unmask the zero dummy at position zero and are
neutralized before merge; nonempty partitions retain their exact validity mask.
FIA v2 returns FP32 LSE; passing `max=LSE, sum=1` to the existing stable merge
weights both outputs by their softmax mass. Eager decode uses the original
parallel prefetch with compact FIA partitions; prefill remains SFA.

Every graph replay retains the blocking nonempty graph-buffer read before
update/replay. The update uses **`actual_seq_kvlen`**, broadcast to every layer's
two FIA tasks; padding and IDLE lengths are zero. Expected replay logs include
`seq_lens.cpu.returned`, then `seq_lens.selected_kv route=sparse_fia_dual` and
`update.prepare input_keys=[['actual_seq_kvlen']]`. Capture should report two
FIA update tasks per attention layer in the captured graph.

The CPU checks cover original-position gathering/refill, mixed and empty
partitions, mask changes at fixed counts, buffer reuse, the absence of graph
FIA prefix scans, and stream dependency scheduling.
Use the unchanged full-model benchmark below to validate device completion,
accuracy and performance. The additional mask and partition work means an actual
speedup must be measured; CPU tests cannot establish device overlap or throughput.

## Native FIA baseline without offload

`SGLANG_NPU_DSA_FIA_NATIVE=1` is a startup-only diagnostic for DeepSeek V3/V3.2
DSA models. It keeps the original model configuration (including `index_topk`),
model size, checkpoint and indexer weights. It executes the stock Ascend MHA
prefill and MLA decode paths instead of sparse attention. Generated output is
not an accuracy check for DSA.

### What changes

| Component | DSA offload, dual-stream FIA | Native FIA diagnostic |
| --- | --- | --- |
| Prefill | DSA preparation, indexer, host KV, SFA | Stock `forward_mha_prepare_npu` / `forward_mha_core_npu` |
| Decode preparation | DSA indexer and selected KV gathering | Stock `forward_mla_prepare_npu` / `forward_mla_core_npu`, no indexer |
| KV | Host backing plus selected/hot-cache buffers | Persistent native paged HBM nope/rope buffers |
| Decode attention | Two `_run_decode_fia_partition` calls and LSE merge | `forward_decode_graph` through shared `forward_mla_fia` |
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

## Full-model benchmark

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

Keep the full model, TP/DP and benchmark unchanged. Enable
`SGLANG_NPU_GRAPH_DEBUG=1` and keep the prefill graph backend disabled.
An explicit `--cuda-graph-config` overrides the graph convenience flags.

Before each FIA decode replay, the runner reads
`self.buffers.seq_lens[:self.bs].cpu().tolist()` from the loaded, nonempty graph
bucket. This blocking device-to-host read supplies the ordering point required
before background `graph.update` and `graph.replay`; do not remove it because
FIA uses fixed capacity rather than the returned values. An IDLE rank still
reads its nonempty bucket and passes zero lengths to FIA. Both FIA tasks in
every layer receive the `actual_seq_kvlen` update, with zeros for padding.

Check each rank's debug stages in this order:

1. `seq_lens.cpu.begin`, then `seq_lens.cpu.returned`.
2. `seq_lens.selected_kv` with `route=sparse_fia_dual`.
3. `update.begin` / `replay.begin`, followed by their return and join stages.

Retain each rank's final stages and the earliest Ascend runtime error if a
stall recurs. Host return messages alone do not establish device completion;
verify that the full-model benchmark completes.

## Local regression checks

```bash
python test/manual/ascend/test_fia_graph_replay_update.py -v
python test/manual/ascend/test_split_fia_attention.py -v
python test/manual/ascend/test_dual_fia_prefetch.py -v
```

These CPU checks cover the default and retired-mode rejection, graph/eager
routing, prefill SFA, FIA masks and LSE merge, original-position gathering and
refill, buffer reuse, stream scheduling, and graph update ordering for active,
IDLE and padded batches. The independent native MLA diagnostic also retains
its configuration and routing checks. CPU mocks cannot validate NPU kernel
support, actual device overlap, multi-rank completion or throughput; use the
full-model benchmark for those checks.
