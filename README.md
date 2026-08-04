# KernelCubed

Reproducible attention-backend benchmarks for the unmodified Qwen3-0.6B
architecture.

The first benchmark compares:

- PyTorch scaled dot-product attention (SDPA)
- FlashAttention (flash-attn)
- FlashInfer

It uses the model's real attention geometry from config.json (16 query heads,
8 key/value heads, head dimension 128 by default) and checks every accelerated
backend against SDPA before reporting timings.

## Scope

This is a per-layer attention-kernel benchmark. Q, K, and V are synthetic but
have the exact shapes and dtype used after Qwen's projection and RoPE steps.
That isolates attention implementation performance without changing weights or
model semantics. It is not an end-to-end tokens/second benchmark.

Prefill measures causal self-attention over a complete prompt. Decode measures
one new query token attending to an existing KV cache.

## Quick start

The repository defaults to the adjacent local checkpoint at
../ai/qwen3-0.6b/model when it exists:

```bash
cd /home/base/KernelCubed
/home/base/ai/qwen3-0.6b/.venv/bin/python -m kernelcubed.benchmark
```

Write machine-readable results while changing the tested lengths:

```bash
/home/base/ai/qwen3-0.6b/.venv/bin/python -m kernelcubed.benchmark --prefill-lengths 128,512,2048 --decode-lengths 128,2048,8192,32768 --output results/qwen3-0.6b.json
```

Useful options:

```text
--phase {both,prefill,decode}
--dtype {auto,float16,bfloat16,float32}
--backends sdpa,flash-attn,flashinfer
--warmup 5
--repeats 20
--seed 0
```

Run python -m kernelcubed.benchmark --help for the complete interface.

## Dependencies

PyTorch is required. FlashAttention and FlashInfer are optional imports so the
harness remains usable while a backend is being installed or ported. Missing or
unsupported backends appear as SKIP rows with the reason; they are never
silently replaced by another implementation.

```bash
python -m pip install torch
python -m pip install flashinfer-python
python -m pip install flash-attn --no-build-isolation
```

FlashInfer can use a large precompiled cache instead of JIT compilation. The
core and cache versions must match. For this repository's CUDA 13 environment:

```bash
python -m pip install "flashinfer-python==0.6.14"
python -m pip install "flashinfer-jit-cache==0.6.14+cu130" --index-url https://flashinfer.ai/whl/cu130
```

FlashAttention installation is sensitive to the Python, PyTorch, CUDA toolkit,
compiler, and GPU combination. Use a build compatible with the target machine.
The current PyTorch 2.11/CUDA 13 environment has no matching official
FlashAttention wheel, so that backend reports SKIP until a compatible build is
installed.

## Reading results

- latency_ms is the median of synchronized CUDA measurements.
- p20_ms and p80_ms expose run-to-run spread.
- speedup_vs_sdpa uses the matching SDPA case as the baseline.
- max_abs_error, mean_abs_error, and cosine_similarity compare output tensors
  with SDPA in float32.
- peak_extra_mib is peak allocator growth above the already allocated QKV
  tensors. It is not total process VRAM.

CUDA kernels compile and autotune during warmup. The harness intentionally does
not include that one-time startup work in steady-state latency.

## Recorded baseline

results/rtx4070-laptop-cu130.json records 20 warmups and 100 measured runs
per case on the RTX 4070 Laptop GPU. It includes exact package versions,
environment metadata, numerical comparisons, and the explicit FlashAttention
skip reason. Treat it as a machine baseline rather than a universal backend
ranking.

## Custom CUDA decode-attention prototype

`kernelcubed-cuda` is an inference-only C++/CUDA backend specialized for the
unmodified Qwen3-0.6B decode geometry:

- one query token, 16 query heads, 8 KV heads, and head dimension 128
- contiguous NHD KV-cache layout
- BF16 or FP16 inputs with FP32 dot products, softmax state, and accumulation
- warp-per-query-head execution through 256 cached tokens
- paired-head split-KV execution for longer contexts; each warp maintains two
  independent softmax states while sharing every K/V cache load
- 128-token split partitions, capped at 64, to retain GPU occupancy

- a reusable 0.51 MiB FP32 split workspace per CUDA stream, allocated once
  outside the measured decode path
The shared model environment contains CUDA 13.0 headers and a CUDA 13.3
compiler. Install the pinned 13.0 compiler into an isolated, ignored project
directory so the extension does not modify that environment:

```bash
/home/base/ai/qwen3-0.6b/.venv/bin/python -m pip install \
  --target .toolchains/cuda130 --no-deps \
  -r requirements-cuda-build.txt
```

The first use JIT-compiles the extension for Ada compute capability 8.9. Run
the focused comparison with:

```bash
/home/base/ai/qwen3-0.6b/.venv/bin/python -m kernelcubed.benchmark \
  --phase decode \
  --backends sdpa,flashinfer,kernelcubed-cuda \
  --decode-lengths 128,256,512,2048,8192,32768 \
  --warmup 20 --repeats 100
```

`results/decode-prototype-rtx4070-cu130.json` records the tuned prototype.
Against SDPA, its measured speedups were 1.97x at 128 tokens, 1.39x at 256,
1.19x at 512, 1.00x at 2,048, 1.19x at 8,192, and 0.99x at 32,768. It was
competitive with but did not consistently beat FlashInfer. These are isolated
attention latencies, not end-to-end model tokens per second.

The paired-GQA checkpoint is recorded in
`results/decode-gqa2-rtx4070-cu130.json`. In that 100-sample trial, the custom
kernel measured 1.60x, 1.26x, 1.15x, 1.49x, 1.20x, and 1.19x versus SDPA at
128, 256, 512, 2,048, 8,192, and 32,768 tokens respectively. At 32,768 tokens,
paired loading measured 0.674 ms versus 0.701 ms for the initial custom
prototype and 0.765 ms for FlashInfer. Laptop GPU scheduling and clocking still
produce run-to-run variance, so compare saved raw percentile data as well.


`results/decode-workspace-rtx4070-cu130.json` records the allocator-free
checkpoint. After warmup, the custom backend reported about 0.0039 MiB of
incremental allocation per invocation at every tested length, compared with
0.0674 MiB at 2,048 tokens for the allocating split path. The persistent
workspace itself occupies about 0.51 MiB per CUDA stream and can be released
with `clear_decode_workspaces()`.
## SWE-bench Mini

The SWE workflow uses a deterministic 12-task subset of the official
[SWE-bench Lite oracle-retrieval dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_oracle).
It favors short prompts, one-file changes, small gold patches, and at most three
fail-to-pass tests. Gold patch statistics are used only to select the easy
subset; solution patches are never written into model prompts.
The downloader strips the Oracle dataset's bundled example output so small
models cannot inflate results by copying its demonstration patch.

Download the suite:

```bash
/home/base/ai/qwen3-0.6b/.venv/bin/python -m kernelcubed.swebench download
```

Run a quick three-task generation benchmark:

```bash
/home/base/ai/qwen3-0.6b/.venv/bin/python -m kernelcubed.swebench run \
  --limit 3 \
  --max-tokens 256 \
  --output-dir results/swebench-mini/qwen3-0.6b-smoke
```

Remove --limit and use the default 512 output tokens for the complete mini
suite. Each run produces:

- predictions.jsonl in the format expected by the official harness
- generations.jsonl with raw output and per-request token/timing information
- metrics.json with aggregate prompt tokens, output tokens, tokens per second,
  structurally complete diff rate, truncation count, backend, model, GPU, and
  run settings

For an attention-kernel comparison, keep the task manifest, seed, model,
max-model-len, max-tokens, batch size, and thinking setting unchanged. Change
only the runtime/kernel selection, using --attention-backend when it maps to a
vLLM backend.

Official resolved-rate scoring applies each generated patch and runs repository
tests in Docker. Install the
[official SWE-bench harness](https://www.swebench.com/SWE-bench/guides/quickstart/)
and run:

```bash
python -m pip install swebench
python -m kernelcubed.swebench evaluate \
  --predictions results/swebench-mini/qwen3-0.6b-baseline/predictions.jsonl
```

The Docker daemon must be running. Image downloads can consume substantial disk
space, so the wrapper defaults to one worker and environment-level caching.
If Docker reports a socket permission error, add the current user to the
`docker` group and start a new login session before running the evaluator.

## Tests

The unit tests run on CPU and do not require optional GPU backends:

```bash
python -m unittest discover -s tests -v
```
