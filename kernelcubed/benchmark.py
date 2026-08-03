"""Compare attention backends using the unmodified Qwen attention geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Iterable, Protocol

import torch
import torch.nn.functional as F


DEFAULT_MODEL_PATH = Path(
    os.environ.get("QWEN_MODEL_PATH", "../ai/qwen3-0.6b/model")
)


def configure_cuda_home() -> Path | None:
    """Use a pip-bundled CUDA toolkit when CUDA_HOME is not configured."""
    configured = os.environ.get("CUDA_HOME")
    if configured:
        return Path(configured)

    candidates: list[Path] = []
    for entry in sys.path:
        nvidia_root = Path(entry) / "nvidia"
        if nvidia_root.is_dir():
            candidates.extend(nvidia_root.glob("cu*/bin/nvcc"))

    if not candidates:
        return None

    cuda_home = sorted(candidates, reverse=True)[0].parent.parent
    os.environ["CUDA_HOME"] = str(cuda_home)
    executable_dir = Path(sys.executable).parent
    os.environ["PATH"] = (
        f"{executable_dir}{os.pathsep}{cuda_home / 'bin'}"
        f"{os.pathsep}{os.environ.get('PATH', '')}"
    )
    return cuda_home


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int

    @classmethod
    def from_config(cls, path: Path) -> "ModelSpec":
        config_path = path if path.name == "config.json" else path / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        hidden_size = int(config["hidden_size"])
        query_heads = int(config["num_attention_heads"])
        return cls(
            model_type=str(config.get("model_type", "unknown")),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=query_heads,
            num_key_value_heads=int(config.get("num_key_value_heads", query_heads)),
            head_dim=int(config.get("head_dim", hidden_size // query_heads)),
            max_position_embeddings=int(config["max_position_embeddings"]),
        )

    def validate(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )


@dataclass(frozen=True)
class Case:
    phase: str
    sequence_length: int

    @property
    def query_length(self) -> int:
        return self.sequence_length if self.phase == "prefill" else 1


@dataclass
class BenchmarkResult:
    backend: str
    phase: str
    sequence_length: int
    status: str
    latency_ms: float | None = None
    p20_ms: float | None = None
    p80_ms: float | None = None
    query_tokens_per_second: float | None = None
    speedup_vs_sdpa: float | None = None
    peak_extra_mib: float | None = None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    cosine_similarity: float | None = None
    reason: str | None = None


class Backend(Protocol):
    name: str

    def availability(self) -> tuple[bool, str | None]: ...

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phase: str,
    ) -> torch.Tensor: ...


class SdpaBackend:
    name = "sdpa"

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        # PyTorch layout: batch, heads, sequence, head_dim.
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = k.transpose(0, 1).unsqueeze(0)
        v_t = v.transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            dropout_p=0.0,
            is_causal=phase == "prefill",
            enable_gqa=q.shape[1] != k.shape[1],
        )
        return output.squeeze(0).transpose(0, 1)


class FlashAttentionBackend:
    name = "flash-attn"

    def __init__(self) -> None:
        self._function: Callable[..., torch.Tensor] | None = None
        self._error: str | None = None
        try:
            from flash_attn import flash_attn_func

            self._function = flash_attn_func
        except Exception as exc:  # Optional binary dependency.
            self._error = f"{type(exc).__name__}: {exc}"

    def availability(self) -> tuple[bool, str | None]:
        if self._function is None:
            return False, self._error or "flash-attn is unavailable"
        return True, None

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        assert self._function is not None
        return self._function(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            dropout_p=0.0,
            causal=phase == "prefill",
        ).squeeze(0)


class FlashInferBackend:
    name = "flashinfer"

    def __init__(self) -> None:
        self._module = None
        self._error: str | None = None
        try:
            import flashinfer

            self._module = flashinfer
        except Exception as exc:  # Optional binary/JIT dependency.
            self._error = f"{type(exc).__name__}: {exc}"

    def availability(self) -> tuple[bool, str | None]:
        if self._module is None:
            return False, self._error or "flashinfer is unavailable"
        return True, None

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        assert self._module is not None
        if phase == "prefill":
            return self._module.single_prefill_with_kv_cache(
                q,
                k,
                v,
                causal=True,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
            )
        return self._module.single_decode_with_kv_cache(
            q.squeeze(0),
            k,
            v,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
        ).unsqueeze(0)


BACKEND_FACTORIES: dict[str, Callable[[], Backend]] = {
    "sdpa": SdpaBackend,
    "flash-attn": FlashAttentionBackend,
    "flashinfer": FlashInferBackend,
}


def parse_lengths(value: str) -> list[int]:
    try:
        lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "lengths must be comma-separated integers"
        ) from exc
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must contain positive integers")
    return lengths


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def make_qkv(
    spec: ModelSpec,
    case: Case,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        case.query_length,
        spec.num_attention_heads,
        spec.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        case.sequence_length,
        spec.num_key_value_heads,
        spec.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        case.sequence_length,
        spec.num_key_value_heads,
        spec.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return q.contiguous(), k.contiguous(), v.contiguous()


def error_metrics(
    output: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float, float]:
    output_f = output.float()
    reference_f = reference.float()
    difference = (output_f - reference_f).abs()
    cosine = F.cosine_similarity(
        output_f.reshape(1, -1), reference_f.reshape(1, -1)
    ).item()
    return difference.max().item(), difference.mean().item(), cosine


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, list[float], float | None]:
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = function()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = function()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        extra_memory = max(
            0, torch.cuda.max_memory_allocated(device) - baseline_memory
        ) / (1024**2)
    else:
        timings = []
        for _ in range(repeats):
            start_time = time.perf_counter()
            output = function()
            timings.append((time.perf_counter() - start_time) * 1000.0)
        extra_memory = None

    assert output is not None
    return output, timings, extra_memory


def benchmark_case(
    spec: ModelSpec,
    case: Case,
    backends: Iterable[Backend],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    seed: int,
) -> list[BenchmarkResult]:
    q, k, v = make_qkv(spec, case, device, dtype, seed)
    reference = SdpaBackend().run(q, k, v, case.phase)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    results: list[BenchmarkResult] = []
    for backend in backends:
        available, reason = backend.availability()
        if not available:
            results.append(
                BenchmarkResult(
                    backend=backend.name,
                    phase=case.phase,
                    sequence_length=case.sequence_length,
                    status="SKIP",
                    reason=reason,
                )
            )
            continue

        try:
            output, timings, extra_memory = measure(
                lambda: backend.run(q, k, v, case.phase),
                device,
                warmup,
                repeats,
            )
            max_error, mean_error, cosine = error_metrics(output, reference)
            median_ms = statistics.median(timings)
            results.append(
                BenchmarkResult(
                    backend=backend.name,
                    phase=case.phase,
                    sequence_length=case.sequence_length,
                    status="OK",
                    latency_ms=median_ms,
                    p20_ms=percentile(timings, 0.2),
                    p80_ms=percentile(timings, 0.8),
                    query_tokens_per_second=case.query_length * 1000.0 / median_ms,
                    peak_extra_mib=extra_memory,
                    max_abs_error=max_error,
                    mean_abs_error=mean_error,
                    cosine_similarity=cosine,
                )
            )
        except Exception as exc:  # Keep one backend failure from aborting the run.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results.append(
                BenchmarkResult(
                    backend=backend.name,
                    phase=case.phase,
                    sequence_length=case.sequence_length,
                    status="ERROR",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )

    sdpa_latency = next(
        (
            result.latency_ms
            for result in results
            if result.backend == "sdpa" and result.status == "OK"
        ),
        None,
    )
    if sdpa_latency:
        for result in results:
            if result.status == "OK" and result.latency_ms:
                result.speedup_vs_sdpa = sdpa_latency / result.latency_ms
    return results


def format_number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'phase':<8} {'length':>8} {'backend':<12} {'status':<7} "
        f"{'median ms':>10} {'speedup':>9} {'max error':>11} {'cosine':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.phase:<8} {result.sequence_length:>8} "
            f"{result.backend:<12} {result.status:<7} "
            f"{format_number(result.latency_ms):>10} "
            f"{format_number(result.speedup_vs_sdpa, 2):>9} "
            f"{format_number(result.max_abs_error, 6):>11} "
            f"{format_number(result.cosine_similarity, 6):>10}"
        )
        if result.reason:
            concise_reason = " ".join(result.reason.splitlines())
            if len(concise_reason) > 300:
                concise_reason = concise_reason[:297] + "..."
            print(f"  {result.backend}: {concise_reason}")


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def environment_metadata(
    device: torch.device, dtype: torch.dtype
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "packages": {
            "flash-attn": package_version("flash-attn"),
            "flashinfer-python": package_version("flashinfer-python"),
            "flashinfer-jit-cache": package_version("flashinfer-jit-cache"),
        },
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
                "cuda_runtime": torch.version.cuda,
                "cuda_home": os.environ.get("CUDA_HOME"),
            }
        )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--phase", choices=("both", "prefill", "decode"), default="both"
    )
    parser.add_argument(
        "--prefill-lengths",
        type=parse_lengths,
        default=parse_lengths("128,512,2048"),
    )
    parser.add_argument(
        "--decode-lengths",
        type=parse_lengths,
        default=parse_lengths("128,2048,8192,32768"),
    )
    parser.add_argument(
        "--backends",
        type=lambda value: [item.strip() for item in value.split(",")],
        default=list(BACKEND_FACTORIES),
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit(
            "--warmup must be non-negative and --repeats must be positive"
        )
    unknown = sorted(set(args.backends) - BACKEND_FACTORIES.keys())
    if unknown:
        raise SystemExit(f"unknown backend(s): {', '.join(unknown)}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        configure_cuda_home()
    dtype = resolve_dtype(args.dtype, device)
    spec = ModelSpec.from_config(args.model_path.expanduser().resolve())
    spec.validate()
    backends = [BACKEND_FACTORIES[name]() for name in args.backends]

    cases: list[Case] = []
    if args.phase in ("both", "prefill"):
        cases.extend(Case("prefill", length) for length in args.prefill_lengths)
    if args.phase in ("both", "decode"):
        cases.extend(Case("decode", length) for length in args.decode_lengths)

    print(
        f"Model: {spec.model_type}; layers={spec.num_hidden_layers}; "
        f"heads={spec.num_attention_heads}/{spec.num_key_value_heads}; "
        f"head_dim={spec.head_dim}"
    )
    metadata = environment_metadata(device, dtype)
    print(
        f"Device: {metadata.get('gpu', metadata['device'])}; "
        f"dtype={metadata['dtype']}; torch={metadata['torch']}"
    )

    results: list[BenchmarkResult] = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            results.extend(
                benchmark_case(
                    spec,
                    case,
                    backends,
                    device,
                    dtype,
                    args.warmup,
                    args.repeats,
                    args.seed + index,
                )
            )

    print()
    print_results(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": str(args.model_path.expanduser().resolve()),
            "model": asdict(spec),
            "environment": metadata,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "results": [asdict(result) for result in results],
        }
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.output}")

    return 1 if any(result.status == "ERROR" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
