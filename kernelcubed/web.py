"""Small local web UI for the persistent KernelCubed chat runtime."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse
import webbrowser

import torch

from .chat import (
    build_parser as build_chat_parser,
    initial_messages,
    load_model_and_tokenizer,
    normalize_prompt_ids,
    sampling_options,
)
from .transformers_backend import get_attention_stats, reset_attention_stats


WEB_ROOT = Path(__file__).with_name("web_assets")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = build_chat_parser()
    parser.description = "Run the local KernelCubed chat web UI."
    parser.set_defaults(prompt=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the UI in the default browser.",
    )
    return parser


class ChatRuntime:
    """Own one model instance and one browser conversation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.messages = initial_messages(args.system_prompt)
        self.state = "loading"
        self.error: str | None = None
        self.device_name = args.device
        self._generation_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def load(self) -> None:
        try:
            torch.manual_seed(self.args.seed)
            model, tokenizer = load_model_and_tokenizer(self.args)
            device = torch.device(self.args.device)
            device_name = (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device)
            )
        except Exception as exc:
            with self._state_lock:
                self.error = str(exc)
                self.state = "error"
            return
        with self._state_lock:
            self.model = model
            self.tokenizer = tokenizer
            self.device_name = device_name
            self.state = "ready"

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "state": self.state,
                "error": self.error,
                "model": str(self.args.model),
                "backend": self.args.attention_backend,
                "device": self.device_name,
                "dtype": self.args.dtype,
                "max_context": self.args.max_context,
                "max_new_tokens": self.args.max_new_tokens,
                "thinking": self.args.thinking,
            }

    def reset(self) -> bool:
        if not self._generation_lock.acquire(blocking=False):
            return False
        try:
            self.messages = initial_messages(self.args.system_prompt)
            return True
        finally:
            self._generation_lock.release()

    def stream_reply(self, user_text: str) -> Iterator[dict[str, Any]]:
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("A response is already being generated.")
        try:
            yield from self._stream_reply_locked(user_text)
        finally:
            self._generation_lock.release()

    def _stream_reply_locked(self, user_text: str) -> Iterator[dict[str, Any]]:
        if self.state != "ready" or self.model is None or self.tokenizer is None:
            raise RuntimeError("The model is not ready yet.")

        from transformers import TextIteratorStreamer

        runtime = self

        class CountingStreamer(TextIteratorStreamer):
            output_tokens = 0

            def put(self, value: Any) -> None:
                is_prompt = self.skip_prompt and self.next_tokens_are_prompt
                if not is_prompt:
                    self.output_tokens += int(value.numel())
                super().put(value)

        self.messages.append({"role": "user", "content": user_text})
        try:
            encoded = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.args.thinking,
            )
            prompt_ids = normalize_prompt_ids(encoded)
            prompt_length = len(prompt_ids)
            if prompt_length + self.args.max_new_tokens > self.args.max_context:
                raise ValueError(
                    f"This conversation uses {prompt_length} tokens and the "
                    f"response limit is {self.args.max_new_tokens}. Reset the "
                    f"chat or increase --max-context ({self.args.max_context})."
                )
        except Exception:
            self.messages.pop()
            raise

        device = torch.device(self.args.device)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        streamer = CountingStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        options = {
            "max_new_tokens": self.args.max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            **sampling_options(self.args.temperature, self.args.top_p),
        }
        result: dict[str, Any] = {}

        reset_attention_stats()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()

        def generate() -> None:
            try:
                with torch.inference_mode():
                    result["generated"] = runtime.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **options,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except Exception as exc:
                result["error"] = exc
                streamer.end()

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        chunks: list[str] = []
        yield {"type": "start", "prompt_tokens": prompt_length}
        for chunk in streamer:
            chunks.append(chunk)
            elapsed = max(time.perf_counter() - started, 0.000001)
            yield {
                "type": "delta",
                "text": chunk,
                "output_tokens": streamer.output_tokens,
                "elapsed_seconds": elapsed,
                "tokens_per_second": streamer.output_tokens / elapsed,
            }
        worker.join()
        elapsed = max(time.perf_counter() - started, 0.000001)

        if "error" in result:
            self.messages.pop()
            raise result["error"]

        generated = result["generated"]
        output_tokens = int(generated[0, prompt_length:].numel())
        text = "".join(chunks).strip()
        stats = get_attention_stats()
        self.messages.append({"role": "assistant", "content": text})
        yield {
            "type": "done",
            "text": text,
            "prompt_tokens": prompt_length,
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": output_tokens / elapsed,
            "custom_decode_calls": stats.custom_decode_calls,
            "sdpa_fallback_calls": stats.sdpa_fallback_calls,
        }


def encode_event(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")


def read_message(handler: BaseHTTPRequestHandler) -> str:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Invalid request length.") from exc
    if length < 1 or length > 64 * 1024:
        raise ValueError("Message must be between 1 byte and 64 KiB.")
    try:
        payload = json.loads(handler.rfile.read(length))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Request body must be valid JSON.") from exc
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty.")
    return message.strip()


def make_handler(runtime: ChatRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "KernelCubedUI/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/status":
                self.send_json(HTTPStatus.OK, runtime.status())
                return
            asset = ASSETS.get(path)
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            filename, content_type = asset
            body = (WEB_ROOT / filename).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/reset":
                if runtime.reset():
                    self.send_json(HTTPStatus.OK, {"ok": True})
                else:
                    self.send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "Wait for the current response to finish."},
                    )
                return
            if path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                message = read_message(self)
            except ValueError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "application/x-ndjson; charset=utf-8"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            connected = True
            try:
                for event in runtime.stream_reply(message):
                    if connected:
                        try:
                            self.wfile.write(encode_event(event))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            connected = False
            except Exception as exc:
                if connected:
                    try:
                        self.wfile.write(
                            encode_event({"type": "error", "error": str(exc)})
                        )
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass

    return Handler


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    if args.max_context < 1 or args.max_new_tokens < 1:
        raise SystemExit("context and output limits must be positive")
    if not args.model.expanduser().is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")

    runtime = ChatRuntime(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    url = f"http://{args.host}:{server.server_port}"
    threading.Thread(target=runtime.load, daemon=True).start()
    print(f"KernelCubed UI: {url}")
    print("Loading the model in the background. Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping KernelCubed UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
