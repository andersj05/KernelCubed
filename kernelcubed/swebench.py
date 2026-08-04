"""Download, run, and evaluate a small reproducible SWE-bench suite."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ORACLE_DATASET = "princeton-nlp/SWE-bench_Lite_oracle"
EVALUATION_DATASET = "SWE-bench/SWE-bench_Lite"
DATASET_SPLIT = "test"
DATASET_CONFIG = "default"
DATASET_SERVER = "https://datasets-server.huggingface.co"
HUGGING_FACE_API = "https://huggingface.co/api/datasets"
DEFAULT_TASKS = Path("benchmarks/swebench-mini/tasks.jsonl")
DEFAULT_MANIFEST = Path("benchmarks/swebench-mini/manifest.json")
DEFAULT_MODEL = Path("../ai/qwen3-0.6b/model")


@dataclass(frozen=True)
class PatchStats:
    files: int
    changed_lines: int
    added_lines: int
    removed_lines: int


@dataclass(frozen=True)
class SelectionConfig:
    count: int = 12
    max_prompt_chars: int = 24_000
    max_changed_lines: int = 40
    max_files: int = 1
    max_fail_to_pass: int = 3
    max_per_repo: int = 2


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "KernelCubed/0.1"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last_error


def dataset_revision(dataset: str = ORACLE_DATASET) -> str:
    metadata = request_json(f"{HUGGING_FACE_API}/{dataset}")
    revision = metadata.get("sha")
    if not revision:
        raise RuntimeError(f"Hugging Face did not return a revision for {dataset}")
    return str(revision)


def dataset_row_count(
    dataset: str = ORACLE_DATASET,
    split: str = DATASET_SPLIT,
) -> int:
    payload = request_json(f"{DATASET_SERVER}/size", {"dataset": dataset})
    for split_info in payload["size"]["splits"]:
        if (
            split_info["config"] == DATASET_CONFIG
            and split_info["split"] == split
        ):
            return int(split_info["num_rows"])
    raise RuntimeError(f"split {split!r} was not found for {dataset}")


def fetch_rows(
    dataset: str = ORACLE_DATASET,
    split: str = DATASET_SPLIT,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    total = dataset_row_count(dataset, split)
    rows: list[dict[str, Any]] = []
    for offset in range(0, total, page_size):
        payload = request_json(
            f"{DATASET_SERVER}/rows",
            {
                "dataset": dataset,
                "config": DATASET_CONFIG,
                "split": split,
                "offset": offset,
                "length": min(page_size, total - offset),
            },
        )
        rows.extend(item["row"] for item in payload["rows"])
    if len(rows) != total:
        raise RuntimeError(f"expected {total} rows but downloaded {len(rows)}")
    return rows


def patch_stats(patch: str) -> PatchStats:
    files = {
        match.group(1)
        for match in re.finditer(r"^diff --git a/(.+?) b/.+$", patch, re.MULTILINE)
    }
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return PatchStats(
        files=len(files),
        changed_lines=added + removed,
        added_lines=added,
        removed_lines=removed,
    )


def json_list_count(value: Any) -> int:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return 0
    return len(value) if isinstance(value, list) else 0


def sanitize_oracle_prompt(text: str, problem_statement: str) -> str:
    """Keep the issue and retrieved source while removing the bundled demo patch."""
    code_start = text.find("<code>")
    code_end = text.rfind("</code>")
    if code_start < 0 or code_end < code_start:
        raise ValueError("oracle prompt does not contain a complete <code> block")
    code = text[code_start : code_end + len("</code>")]
    return f"<issue>\n{problem_statement.strip()}\n</issue>\n{code}"


def task_candidate(row: dict[str, Any]) -> dict[str, Any]:
    stats = patch_stats(str(row.get("patch", "")))
    problem_statement = str(row["problem_statement"])
    prompt = sanitize_oracle_prompt(str(row["text"]), problem_statement)
    return {
        "instance_id": str(row["instance_id"]),
        "repo": str(row["repo"]),
        "base_commit": str(row["base_commit"]),
        "version": str(row["version"]),
        "problem_statement": problem_statement,
        "oracle_prompt": prompt,
        "prompt_chars": len(prompt),
        "estimated_prompt_tokens": (len(prompt) + 3) // 4,
        "selection_stats": {
            **asdict(stats),
            "fail_to_pass": json_list_count(row.get("FAIL_TO_PASS")),
        },
    }


def select_tasks(
    rows: Iterable[dict[str, Any]],
    config: SelectionConfig,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = task_candidate(row)
        stats = candidate["selection_stats"]
        if not candidate["problem_statement"].strip():
            continue
        if stats["files"] < 1 or stats["files"] > config.max_files:
            continue
        if stats["changed_lines"] < 1:
            continue
        if stats["changed_lines"] > config.max_changed_lines:
            continue
        if stats["fail_to_pass"] > config.max_fail_to_pass:
            continue
        if candidate["prompt_chars"] > config.max_prompt_chars:
            continue
        candidates.append(candidate)

    candidates.sort(
        key=lambda task: (
            task["selection_stats"]["files"],
            task["selection_stats"]["changed_lines"],
            task["selection_stats"]["fail_to_pass"],
            task["prompt_chars"],
            task["instance_id"],
        )
    )
    selected: list[dict[str, Any]] = []
    repo_counts: Counter[str] = Counter()
    for candidate in candidates:
        if repo_counts[candidate["repo"]] >= config.max_per_repo:
            continue
        selected.append(candidate)
        repo_counts[candidate["repo"]] += 1
        if len(selected) == config.count:
            break
    if len(selected) < config.count:
        raise RuntimeError(
            f"only {len(selected)} tasks match the selection criteria; "
            f"{config.count} were requested"
        )
    return selected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def download_command(args: argparse.Namespace) -> int:
    if args.tasks.exists() and not args.force:
        raise SystemExit(
            "task output already exists; pass --force to replace the suite"
        )
    config = SelectionConfig(
        count=args.count,
        max_prompt_chars=args.max_prompt_chars,
        max_changed_lines=args.max_changed_lines,
        max_files=args.max_files,
        max_fail_to_pass=args.max_fail_to_pass,
        max_per_repo=args.max_per_repo,
    )
    print(f"Downloading {ORACLE_DATASET} ({DATASET_SPLIT})...")
    revision = dataset_revision()
    rows = fetch_rows()
    tasks = select_tasks(rows, config)
    write_jsonl(args.tasks, tasks)

    manifest_tasks = [
        {
            key: value
            for key, value in task.items()
            if key not in {"oracle_prompt", "problem_statement"}
        }
        for task in tasks
    ]
    manifest = {
        "format_version": 1,
        "retrieval": "official oracle retrieval; pre-change files; demo removed",
        "oracle_dataset": ORACLE_DATASET,
        "evaluation_dataset": EVALUATION_DATASET,
        "dataset_revision": revision,
        "split": DATASET_SPLIT,
        "selection": asdict(config),
        "tasks": manifest_tasks,
    }
    write_json(args.manifest, manifest)
    print(
        f"Wrote {len(tasks)} tasks to {args.tasks} "
        f"(revision {revision[:12]})."
    )
    for task in tasks:
        stats = task["selection_stats"]
        print(
            f"  {task['instance_id']}: {task['estimated_prompt_tokens']} est. tokens, "
            f"{stats['changed_lines']} changed lines"
        )
    return 0


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    instructions = """
Produce the smallest correct fix for this issue.
Return only a unified git diff that can be applied with git apply.
Do not include explanations, markdown fences, tests, or the known solution.
If no correct patch can be produced, return an empty response.
""".strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a software engineer editing the supplied repository snapshot. "
                "Follow the output contract exactly."
            ),
        },
        {
            "role": "user",
            "content": f"{task['oracle_prompt'].rstrip()}\n\n<instructions>\n"
            f"{instructions}\n</instructions>",
        },
    ]


def extract_patch(text: str) -> str:
    cleaned = re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    diff_start = cleaned.find("diff --git ")
    if diff_start >= 0:
        cleaned = cleaned[diff_start:]
    else:
        header = re.search(r"(?m)^---\s+(?:a/)?\S+\n\+\+\+\s+(?:b/)?\S+", cleaned)
        if header:
            cleaned = cleaned[header.start():]
    return cleaned.strip() + ("\n" if cleaned.strip() else "")


def is_unified_diff(patch: str) -> bool:
    if not patch:
        return False
    if "diff --git " in patch:
        return True
    return bool(re.search(r"(?m)^---\s+\S+\n\+\+\+\s+\S+", patch))


def is_complete_unified_diff(patch: str) -> bool:
    """Return whether every textual hunk contains its declared line counts."""
    if not is_unified_diff(patch):
        return False
    lines = patch.splitlines()
    hunk_pattern = re.compile(
        r"^@@ -(?:\d+)(?:,(\d+))? \+(?:\d+)(?:,(\d+))? @@"
    )
    found_hunk = False
    index = 0
    while index < len(lines):
        match = hunk_pattern.match(lines[index])
        if not match:
            index += 1
            continue
        found_hunk = True
        expected_old = int(match.group(1) or 1)
        expected_new = int(match.group(2) or 1)
        actual_old = 0
        actual_new = 0
        index += 1
        while index < len(lines):
            line = lines[index]
            if hunk_pattern.match(line) or line.startswith("diff --git "):
                break
            if line.startswith("\\ No newline at end of file"):
                index += 1
                continue
            if line.startswith("+"):
                actual_new += 1
            elif line.startswith("-"):
                actual_old += 1
            elif line.startswith(" ") or not line:
                actual_old += 1
                actual_new += 1
            else:
                return False
            index += 1
        if actual_old != expected_old or actual_new != expected_new:
            return False
    return found_hunk


def metric_delta(metrics: Any, start: str, end: str) -> float | None:
    if metrics is None:
        return None
    start_value = getattr(metrics, start, None)
    end_value = getattr(metrics, end, None)
    if start_value is None or end_value is None:
        return None
    delta = float(end_value) - float(start_value)
    return delta if delta >= 0 else None


def mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def ensure_output_dir(path: Path, force: bool) -> None:
    expected = [
        path / "predictions.jsonl",
        path / "generations.jsonl",
        path / "metrics.json",
    ]
    if any(item.exists() for item in expected) and not force:
        raise SystemExit(
            f"{path} already contains a run; pass --force to replace it"
        )
    path.mkdir(parents=True, exist_ok=True)


def run_command(args: argparse.Namespace) -> int:
    tasks = read_jsonl(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("the task file contains no tasks")
    ensure_output_dir(args.output_dir, args.force)

    if args.attention_backend != "auto":
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    from vllm import LLM, SamplingParams

    model_path = args.model.expanduser().resolve()
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    model_load_seconds = time.perf_counter() - load_started
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    prepared: list[tuple[dict[str, Any], list[dict[str, str]], int]] = []
    generations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for task in tasks:
        messages = build_messages(task)
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompt_tokens = len(prompt_ids)
        if prompt_tokens + args.max_tokens > args.max_model_len:
            reason = (
                f"prompt ({prompt_tokens}) + max output ({args.max_tokens}) "
                f"exceeds max model length ({args.max_model_len})"
            )
            generations.append(
                {
                    "instance_id": task["instance_id"],
                    "status": "SKIP",
                    "reason": reason,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": 0,
                    "valid_patch": False,
                    "raw_output": "",
                    "model_patch": "",
                }
            )
            predictions.append(
                {
                    "instance_id": task["instance_id"],
                    "model_name_or_path": args.model_name,
                    "model_patch": "",
                }
            )
            continue
        prepared.append((task, messages, prompt_tokens))

    generation_started = time.perf_counter()
    for offset in range(0, len(prepared), args.batch_size):
        chunk = prepared[offset : offset + args.batch_size]
        conversations = [item[1] for item in chunk]
        outputs = llm.chat(
            conversations,
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
        )
        for (task, _messages, counted_prompt_tokens), request_output in zip(
            chunk, outputs, strict=True
        ):
            completion = request_output.outputs[0]
            raw_output = completion.text
            extracted = extract_patch(raw_output)
            truncated = completion.finish_reason == "length"
            valid_patch = not truncated and is_complete_unified_diff(extracted)
            model_patch = extracted if valid_patch else ""
            prompt_tokens = len(request_output.prompt_token_ids or [])
            if not prompt_tokens:
                prompt_tokens = counted_prompt_tokens
            output_tokens = len(completion.token_ids)
            ttft = metric_delta(
                request_output.metrics, "arrival_time", "first_token_time"
            )
            decode_seconds = metric_delta(
                request_output.metrics, "first_token_time", "finished_time"
            )
            generations.append(
                {
                    "instance_id": task["instance_id"],
                    "status": "OK",
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "time_to_first_token_seconds": ttft,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": (
                        output_tokens / decode_seconds
                        if decode_seconds and output_tokens
                        else None
                    ),
                    "finish_reason": completion.finish_reason,
                    "truncated": truncated,
                    "valid_patch": valid_patch,
                    "raw_output": raw_output,
                    "model_patch": model_patch,
                }
            )
            predictions.append(
                {
                    "instance_id": task["instance_id"],
                    "model_name_or_path": args.model_name,
                    "model_patch": model_patch,
                }
            )
    generation_wall_seconds = time.perf_counter() - generation_started

    ordered_generations = {
        item["instance_id"]: item for item in generations
    }
    generations = [
        ordered_generations[task["instance_id"]] for task in tasks
    ]
    ordered_predictions = {
        item["instance_id"]: item for item in predictions
    }
    predictions = [
        ordered_predictions[task["instance_id"]] for task in tasks
    ]
    completed = [item for item in generations if item["status"] == "OK"]
    prompt_tokens = sum(item["prompt_tokens"] for item in completed)
    output_tokens = sum(item["output_tokens"] for item in completed)
    valid_patches = sum(bool(item["valid_patch"]) for item in completed)
    truncated = sum(bool(item.get("truncated")) for item in completed)
    summary = {
        "format_version": 1,
        "model": str(model_path),
        "model_name": args.model_name,
        "tasks": str(args.tasks),
        "attention_backend": args.attention_backend,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "model_load_seconds": model_load_seconds,
        "generation_wall_seconds": generation_wall_seconds,
        "task_count": len(tasks),
        "completed_count": len(completed),
        "skipped_count": len(tasks) - len(completed),
        "valid_patch_count": valid_patches,
        "valid_patch_rate": valid_patches / len(tasks),
        "truncated_count": truncated,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            output_tokens / generation_wall_seconds
            if generation_wall_seconds
            else None
        ),
        "total_tokens_per_second": (
            (prompt_tokens + output_tokens) / generation_wall_seconds
            if generation_wall_seconds
            else None
        ),
        "mean_time_to_first_token_seconds": mean_or_none(
            item.get("time_to_first_token_seconds") for item in completed
        ),
        "mean_decode_tokens_per_second": mean_or_none(
            item.get("decode_tokens_per_second") for item in completed
        ),
        "official_resolved_rate": None,
        "official_evaluation_note": (
            "Run the evaluate subcommand with Docker to populate resolved rate."
        ),
    }
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_jsonl(args.output_dir / "generations.jsonl", generations)
    write_json(args.output_dir / "metrics.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"Wrote predictions and metrics to {args.output_dir}")
    return 0


def docker_available() -> tuple[bool, str | None]:
    if shutil.which("docker") is None:
        return False, "docker CLI is not installed"
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        reason = result.stderr.strip() or result.stdout.strip()
        return False, reason or "Docker daemon is unavailable"
    return True, None


def evaluate_command(args: argparse.Namespace) -> int:
    available, reason = docker_available()
    if not available:
        print(f"Docker is required for official SWE-bench scoring: {reason}", file=sys.stderr)
        return 2
    if importlib.util.find_spec("swebench") is None:
        print(
            "Install the official harness first: python -m pip install swebench",
            file=sys.stderr,
        )
        return 2

    instance_ids = [
        task["instance_id"] for task in read_jsonl(args.tasks)
    ]
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        EVALUATION_DATASET,
        "--split",
        DATASET_SPLIT,
        "--predictions_path",
        str(args.predictions),
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
        "--cache_level",
        args.cache_level,
        "--instance_ids",
        *instance_ids,
    ]
    print("Running:", " ".join(command))
    return subprocess.run(command, check=False).returncode


def report_list_size(report: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = report.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, int):
            return value
    return None


def summarize_command(args: argparse.Namespace) -> int:
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    summary = dict(metrics)
    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        resolved = report_list_size(
            report, "resolved_ids", "resolved_instances", "resolved"
        )
        submitted = report_list_size(
            report, "submitted_ids", "submitted_instances", "submitted"
        )
        summary["official_resolved"] = resolved
        summary["official_submitted"] = submitted
        summary["official_resolved_rate"] = (
            resolved / submitted
            if resolved is not None and submitted
            else None
        )
        summary["official_report"] = str(args.report)
    print(json.dumps(summary, indent=2))
    if args.output:
        write_json(args.output, summary)
    return 0


def add_download_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("download", help="download and curate the mini suite")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--max-prompt-chars", type=int, default=24_000)
    parser.add_argument("--max-changed-lines", type=int, default=40)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-fail-to-pass", type=int, default=3)
    parser.add_argument("--max-per-repo", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(function=download_command)


def add_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("run", help="generate patches and throughput metrics")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default="qwen3-0.6b-kernelcubed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/swebench-mini/qwen3-0.6b-baseline"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(function=run_command)


def add_evaluate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="run official Docker-based SWE-bench tests"
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-id", default="kernelcubed-qwen3-0.6b")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--cache-level",
        choices=("none", "base", "env", "instance"),
        default="env",
    )
    parser.set_defaults(function=evaluate_command)


def add_summarize_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "summarize", help="combine generation and official test metrics"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(function=summarize_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_download_parser(subparsers)
    add_run_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_summarize_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    sys.exit(main())
