# Run the stock Ascend FIA path with a full DSA checkpoint

`SGLANG_NPU_DSA_FIA_NATIVE=1` is a startup-only diagnostic for DeepSeek V3/V3.2
DSA models. It keeps the original model configuration (including `index_topk`),
model size, checkpoint and indexer weights. It executes the stock Ascend MHA
prefill and MLA decode paths instead of sparse attention. Generated output is
not an accuracy check for DSA.

## What changes

| Component | DSA offload, combined FIA | Native FIA diagnostic |
| --- | --- | --- |
| Prefill | DSA preparation, indexer, host KV, SFA | Stock `forward_mha_prepare_npu` / `forward_mha_core_npu` |
| Decode preparation | DSA indexer and selected KV gathering | Stock `forward_mla_prepare_npu` / `forward_mla_core_npu`, no indexer |
| KV | Host backing plus selected/hot-cache buffers | Persistent native paged HBM nope/rope buffers |
| Decode attention | `_run_combined_decode_fia` | Existing `AscendAttnBackend.forward_decode_graph` |
| Page table | Contiguous selected-KV page IDs | Stock request page table |
| Graph CPU lengths | Selected capacity, normally 2048 | Actual full context lengths; zero for graph padding |
| Offload setup | Manager, host registration, request reset hook, callback streams | None, even if the old offload environment switch remains set |

The FIA workspace query, `.out` call, BSND layout, head padding and output
slicing are the existing stock implementation, not a copy of that code. The
diagnostic retains the unused index KV buffer and accounts for all three native
buffers using the NPU KV dtype when calculating capacity.

## Full-model server and benchmark

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

After restarting the server, run the same benchmark:

```bash
python -m sglang.bench_serving \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --backend sglang --host 127.0.0.1 --port 6688 \
  --max-concurrency 32 --dataset-name random \
  --random-input-len 10000 --random-output-len 32 \
  --num-prompts 16 --random-range-ratio 1 --request-rate 7
```

Expected startup log: `NPU DSA native FIA diagnostic: stock MHA prefill + MLA
decode`. Decode replay logs include `stage=seq_lens.native_mla` and
`route=native_mla_fia`. Confirm that graph capture reports FIA update tasks and
that the benchmark progresses beyond `Starting warmup with 1 sequences...`
to its final results. HTTP 200 and host `update.returned` messages alone do not
establish device completion.

Disable the diagnostic by unsetting `SGLANG_NPU_DSA_FIA_NATIVE` and restarting
the server. The existing DSA/offload switches then take effect again.

## Local regression checks

```bash
python test/manual/ascend/test_fia_graph_replay_update.py -v
```

These CPU mocks check configuration and routing, including full-context graph
updates for DSA and idle/padded batches. They do not validate NPU kernels,
multi-rank completion or numerical accuracy. Use the full-model run above for
device validation.
