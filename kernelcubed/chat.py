"""Persistent terminal chat runner for the local Qwen3-0.6B model."""

from __future__ import annotations

from collections.abc import Mapping
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .transformers_backend import (
    BACKEND_NAME,
    AttentionStats,
    get_attention_stats,
    register_transformers_backend,
    reset_attention_stats,
)


DEFAULT_MODEL_PATH = Path("/home/base/ai/qwen3-0.6b/model")


@dataclass(frozen=True)
class GenerationReport:
    text: str
    output_tokens: int
    elapsed_seconds: float
    attention: AttentionStats

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.output_tokens / self.elapsed_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with local Qwen using SDPA prefill and custom CUDA decode."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--attention-backend",
        choices=[BACKEND_NAME, "sdpa"],
        default=BACKEND_NAME,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful, concise assistant.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3's explicit thinking mode in the chat template.",
    )
    parser.add_argument(
        "--prompt",
        help="Run one prompt and exit instead of entering the chat loop.",
    )
    return parser


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def sampling_options(
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    if temperature <= 0.0:
        return {"do_sample": False}
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }


def initial_messages(system_prompt: str) -> list[dict[str, str]]:
    if not system_prompt:
        return []
    return [{"role": "system", "content": system_prompt}]


def format_report(report: GenerationReport) -> str:
    stats = report.attention
    return (
        f"{report.output_tokens} tokens in {report.elapsed_seconds:.2f}s "
        f"({report.tokens_per_second:.1f} tok/s); "
        f"custom decode calls={stats.custom_decode_calls}; "
        f"SDPA fallback calls={stats.sdpa_fallback_calls}"
    )


def normalize_prompt_ids(encoded: Any) -> list[int]:
    """Normalize Transformers 4.x/5.x chat-template return types."""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("terminal chat supports one conversation at a time")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(
        isinstance(token, int) for token in encoded
    ):
        raise TypeError("chat template did not return integer token IDs")
    return encoded


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.attention_backend == BACKEND_NAME:
        register_transformers_backend()
        from .decode_attention import load_decode_extension

        load_decode_extension()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attention_backend,
        local_files_only=True,
    )
    model.to(torch.device(args.device))
    model.eval()
    return model, tokenizer


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> GenerationReport:
    encoded_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    prompt_ids = normalize_prompt_ids(encoded_prompt)
    prompt_length = len(prompt_ids)
    if prompt_length + args.max_new_tokens > args.max_context:
        raise ValueError(
            f"prompt ({prompt_length}) + requested output "
            f"({args.max_new_tokens}) exceeds --max-context "
            f"({args.max_context}); use /reset or increase the limit"
        )

    device = torch.device(args.device)
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    options = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        **sampling_options(args.temperature, args.top_p),
    }

    reset_attention_stats()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **options,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    output_ids = generated[0, prompt_length:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return GenerationReport(
        text=text,
        output_tokens=int(output_ids.numel()),
        elapsed_seconds=elapsed,
        attention=get_attention_stats(),
    )


def run_turn(
    user_text: str,
    messages: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> GenerationReport:
    messages.append({"role": "user", "content": user_text})
    try:
        report = generate_reply(model, tokenizer, messages, args)
    except Exception:
        messages.pop()
        raise
    messages.append({"role": "assistant", "content": report.text})
    return report


def print_reply(report: GenerationReport) -> None:
    print(f"assistant> {report.text}")
    print(f"[{format_report(report)}]")


def interactive_chat(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> int:
    messages = initial_messages(args.system_prompt)
    print(
        "Ready. Commands: /reset clears history, /stats shows the last "
        "attention counters, /exit quits."
    )
    last_report: GenerationReport | None = None
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        if user_text == "/reset":
            messages = initial_messages(args.system_prompt)
            last_report = None
            print("History cleared.")
            continue
        if user_text == "/stats":
            if last_report is None:
                print("No generation has run yet.")
            else:
                print(format_report(last_report))
            continue

        try:
            last_report = run_turn(
                user_text,
                messages,
                model,
                tokenizer,
                args,
            )
        except ValueError as exc:
            print(f"error: {exc}")
            continue
        print_reply(last_report)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_context < 1 or args.max_new_tokens < 1:
        raise SystemExit("context and output limits must be positive")
    if not args.model.expanduser().is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")

    torch.manual_seed(args.seed)
    print(
        f"Loading {args.model} on {args.device} with "
        f"{args.attention_backend}..."
    )
    model, tokenizer = load_model_and_tokenizer(args)
    print("Model loaded.")

    if args.prompt is not None:
        messages = initial_messages(args.system_prompt)
        report = run_turn(
            args.prompt,
            messages,
            model,
            tokenizer,
            args,
        )
        print_reply(report)
        return 0
    return interactive_chat(model, tokenizer, args)


if __name__ == "__main__":
    raise SystemExit(main())
