# Run native MLA FIA with a full DSA checkpoint

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

## Local regression checks

```bash
python test/manual/ascend/test_fia_graph_replay_update.py -v
python test/manual/ascend/test_native_fia_offload_manager.py -v
```

These CPU mocks check configuration, staging memory accounting, shared FIA
arguments and head padding, real offload routing, and selected/full-context
graph updates including idle/padded batches. They do not validate NPU kernels,
multi-rank completion or numerical accuracy. Use the full-model run above for
device validation.
