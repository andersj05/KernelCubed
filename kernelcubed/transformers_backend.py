"""Transformers attention adapter for KernelCubed Qwen decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .decode_attention import decode_attention


BACKEND_NAME = "kernelcubed-cuda"
_custom_decode_calls = 0
_sdpa_fallback_calls = 0


@dataclass(frozen=True)
class AttentionStats:
    custom_decode_calls: int
    sdpa_fallback_calls: int


def reset_attention_stats() -> None:
    global _custom_decode_calls, _sdpa_fallback_calls
    _custom_decode_calls = 0
    _sdpa_fallback_calls = 0


def get_attention_stats() -> AttentionStats:
    return AttentionStats(_custom_decode_calls, _sdpa_fallback_calls)


def _can_use_custom_decode(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> bool:
    return (
        attention_mask is None
        and not kwargs.get("output_attentions", False)
        and getattr(module, "sliding_window", None) is None
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[0] == key.shape[0] == value.shape[0] == 1
        and query.shape[2] == 1
        and key.shape == value.shape
        and query.shape[1] == 2 * key.shape[1]
        and query.shape[-1] == key.shape[-1] == 128
        and query.dtype == key.dtype == value.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.is_cuda
        and key.is_cuda
        and value.is_cuda
    )


def kernelcubed_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Use KernelCubed for eligible decode and SDPA for prefill/fallback."""
    global _custom_decode_calls, _sdpa_fallback_calls

    if not _can_use_custom_decode(
        module, query, key, value, attention_mask, kwargs
    ):
        from transformers.integrations.sdpa_attention import (
            sdpa_attention_forward,
        )

        _sdpa_fallback_calls += 1
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    query_nhd = query[0].transpose(0, 1).contiguous()
    key_nhd = key[0].transpose(0, 1)
    value_nhd = value[0].transpose(0, 1)
    output = decode_attention(
        query_nhd,
        key_nhd,
        value_nhd,
        scale=scaling,
    )
    _custom_decode_calls += 1
    return output.unsqueeze(0), None


def register_transformers_backend() -> str:
    """Register the backend globally with the installed Transformers runtime."""
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(BACKEND_NAME, kernelcubed_attention_forward)
    return BACKEND_NAME
