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

## Tests

The unit tests run on CPU and do not require optional GPU backends:

```bash
python -m unittest discover -s tests -v
```
