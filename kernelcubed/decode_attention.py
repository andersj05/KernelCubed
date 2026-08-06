"""JIT loader and Python contract for the Qwen decode-attention prototype."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import ModuleType

import torch

from .benchmark import configure_cuda_home


SOURCE_DIRECTORY = Path(__file__).parent / "csrc"
LOCAL_CUDA_HOME = Path(__file__).parents[1] / ".toolchains/cuda130/nvidia/cu13"
MAX_SPLIT_PARTITIONS = 64
SPLIT_STATE_SIZE = 130
_WORKSPACE_LOCK = Lock()
_WORKSPACES: dict[tuple[int, int, int], torch.Tensor] = {}


def validate_decode_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> None:
    if query.ndim != 3 or query.shape[0] != 1 or query.shape[2] != 128:
        raise ValueError("query must have shape [1, query_heads, 128]")
    if key_cache.ndim != 3 or key_cache.shape[2] != 128:
        raise ValueError("key_cache must have shape [tokens, kv_heads, 128]")
    if value_cache.shape != key_cache.shape:
        raise ValueError("value_cache must have the same shape as key_cache")
    if key_cache.shape[0] < 1:
        raise ValueError("KV cache must contain at least one token")
    if query.shape[1] % key_cache.shape[1]:
        raise ValueError("query_heads must be divisible by kv_heads")
    if query.shape[1] != 2 * key_cache.shape[1]:
        raise ValueError("this Qwen prototype requires two query heads per KV head")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("only float16 and bfloat16 are supported")
    if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
        raise TypeError("query, key_cache, and value_cache dtypes must match")
    tensors = (query, key_cache, value_cache)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("query, key_cache, and value_cache must be CUDA tensors")
    if query.device != key_cache.device or query.device != value_cache.device:
        raise ValueError("query, key_cache, and value_cache must share one GPU")
    if not query.is_contiguous():
        raise ValueError("query must be contiguous")
    if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("the final K/V cache dimension must be contiguous")


def get_decode_workspace(query: torch.Tensor) -> torch.Tensor:
    """Return the reusable split-KV state for the query's CUDA stream."""
    if not query.is_cuda:
        raise ValueError("workspace requires a CUDA query")
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(query.device).cuda_stream)
    key = (device_index, stream_id, query.shape[1])
    with _WORKSPACE_LOCK:
        workspace = _WORKSPACES.get(key)
        if workspace is None:
            workspace = torch.empty(
                MAX_SPLIT_PARTITIONS,
                query.shape[1],
                SPLIT_STATE_SIZE,
                device=query.device,
                dtype=torch.float32,
            )
            _WORKSPACES[key] = workspace
    return workspace


def clear_decode_workspaces() -> None:
    """Release cached workspaces, primarily for tests and memory management."""
    with _WORKSPACE_LOCK:
        _WORKSPACES.clear()


@lru_cache(maxsize=1)
def load_decode_extension(verbose: bool = False) -> ModuleType:
    if (LOCAL_CUDA_HOME / "bin/nvcc").is_file():
        os.environ["CUDA_HOME"] = str(LOCAL_CUDA_HOME)

        runtime_link = LOCAL_CUDA_HOME / "lib/libcudart.so"
        versioned_runtime = LOCAL_CUDA_HOME / "lib/libcudart.so.13"
        if versioned_runtime.is_file() and not runtime_link.exists():
            runtime_link.symlink_to(versioned_runtime.name)

        executable_directory = Path(sys.executable).parent
        os.environ["PATH"] = os.pathsep.join(
            (
                str(executable_directory),
                str(LOCAL_CUDA_HOME / "bin"),
                os.environ.get("PATH", ""),
            )
        )
    cuda_home = configure_cuda_home()
    if cuda_home is None:
        raise RuntimeError(
            "a CUDA toolkit with nvcc is required to build the extension"
        )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    from torch.utils.cpp_extension import load

    return load(
        name="kernelcubed_qwen_decode_v1",
        sources=[
            str(SOURCE_DIRECTORY / "decode_attention.cpp"),
            str(SOURCE_DIRECTORY / "decode_attention.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=verbose,
    )


def decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    validate_decode_inputs(query, key_cache, value_cache)
    extension = load_decode_extension()
    resolved_scale = scale if scale is not None else 128.0**-0.5
    workspace = get_decode_workspace(query)
    return extension.decode_attention(
        query, key_cache, value_cache, workspace, resolved_scale
    )
