#!/usr/bin/env python3
"""Certify KV/cache stability under repeated OpenAI tool-loop pressure."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, NamedTuple
import urllib.error
import urllib.request


TOOL_NAME = "lookup_probe_fact"
FIXTURE_FACTS = {
    "primary": "KV-STABILITY-PRIMARY-7429",
    "secondary": "KV-STABILITY-SECONDARY-319",
}
KV_PIN = "KV-PIN-8842"
DEFAULT_BASE_URL = "http://127.0.0.1:9337/v1"
DEFAULT_MODELS = "auto"
DEFAULT_ATTEMPTS = 3
DEFAULT_PRESSURE_TURNS = 6
DEFAULT_TIMEOUT = 180.0
DEFAULT_MIN_CACHED_TOKENS = 2048
DEFAULT_SUFFIX_PREFILL_LIMIT = 256
DEFAULT_OUTPUT_DIR = "target/kv-tool-loop-stability/latest"
NATIVE_LOG_CHECKPOINT_TAIL_BYTES = 4096
FAILURE_PATTERNS = (
    "failed to find a memory slot",
    "RuntimeError: llama_decode failed",
    "llama_decode failed",
)


class ToolCall(NamedTuple):
    call_id: str
    key: str


class CacheMetrics(NamedTuple):
    prompt_tokens: int
    cached_tokens: int


class LogFinding(NamedTuple):
    path: str
    line_number: int
    pattern: str
    text: str


class NativeLogCheckpoint(NamedTuple):
    path: Path
    offset: int
    identity: tuple[int, int] | None
    tail: bytes


class ProbeResult(NamedTuple):
    model: str
    attempt: int
    phase: str
    ok: bool
    detail: str
    elapsed_ms: int
    status_code: int | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None


def normalize_v1_base(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("base URL is empty")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise ValueError("at least one model is required")
    return models


def parse_native_logs(values: Iterable[str] | None) -> list[Path]:
    logs: list[Path] = []
    env_value = os.environ.get("MESH_KV_TOOL_LOOP_NATIVE_LOGS")
    if env_value:
        logs.extend(
            Path(part.strip()).expanduser()
            for part in env_value.split(",")
            if part.strip()
        )
    for value in values or []:
        logs.append(Path(value).expanduser())
    return dedupe_paths(logs)


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.fspath(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def build_plan(
    base_url: str,
    models: Iterable[str],
    attempts: int,
    pressure_turns: int,
    timeout: float,
    output_dir: Path,
    min_cached_tokens: int,
    suffix_prefill_limit: int,
    native_logs: Iterable[Path],
) -> dict[str, Any]:
    model_list = list(models)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if pressure_turns < 0:
        raise ValueError("pressure_turns cannot be negative")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if min_cached_tokens < 0:
        raise ValueError("min_cached_tokens cannot be negative")
    if suffix_prefill_limit < 0:
        raise ValueError("suffix_prefill_limit cannot be negative")
    checks = [
        {
            "phase": "tool_loop",
            "models": model_list,
            "attempts": attempts,
            "pressure_turns": pressure_turns,
            "timeout_seconds": timeout,
            "description": "Growing tool-result conversation with a stable long prefix.",
        },
        {
            "phase": "same_prefix_cache",
            "models": model_list,
            "min_cached_tokens": min_cached_tokens,
            "suffix_prefill_limit": suffix_prefill_limit,
            "timeout_seconds": timeout,
            "description": "Same long prefix with a different tiny tail should reuse cached tokens.",
        },
        {
            "phase": "exact_prefix_cache",
            "models": model_list,
            "min_cached_tokens": min_cached_tokens,
            "suffix_prefill_limit": suffix_prefill_limit,
            "timeout_seconds": timeout,
            "description": "Identical body warm request should still reuse cached tokens.",
        },
    ]
    log_paths = [str(path) for path in native_logs]
    if log_paths:
        checks.append(
            {
                "phase": "native_log_scan",
                "paths": log_paths,
                "scan_mode": "appended_since_run_start",
                "fatal_patterns": list(FAILURE_PATTERNS)
                + ["proactive_eviction status=error"],
            }
        )
    return {
        "name": "kv-tool-loop-stability",
        "base_url": normalize_v1_base(base_url),
        "models": model_list,
        "attempts": attempts,
        "pressure_turns": pressure_turns,
        "timeout_seconds": timeout,
        "output_dir": str(output_dir),
        "min_cached_tokens": min_cached_tokens,
        "suffix_prefill_limit": suffix_prefill_limit,
        "native_log_scan_mode": "appended_since_run_start" if log_paths else None,
        "native_logs": log_paths,
        "checks": checks,
        "evidence_files": [
            "manifest.json",
            "results.jsonl",
            "summary.json",
            "summary.md",
            "transcripts/*.jsonl",
        ],
    }


def render_plan(plan: dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, sort_keys=True)


def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Return one deterministic KV/tool-loop probe fact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "enum": sorted(FIXTURE_FACTS),
                        }
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_tool_call_request(
    model: str,
    attempt: int,
    key: str = "primary",
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    request_messages = list(messages) if messages is not None else initial_messages(attempt, key)
    return {
        "model": model,
        "messages": request_messages,
        "tools": tool_schema(),
        "tool_choice": {
            "type": "function",
            "function": {"name": TOOL_NAME},
        },
        "parallel_tool_calls": False,
        "stream": False,
        "temperature": 0,
        "max_tokens": 128,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def initial_messages(attempt: int, key: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": stable_system_prefix(),
        },
        {
            "role": "user",
            "content": (
                f"Attempt {attempt}: call {TOOL_NAME} with key={key}. "
                f"Keep the pinned context value {KV_PIN} in memory for the final answer. "
                "Do not answer directly before the tool call."
            ),
        },
    ]


def build_tool_result_request(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 128,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_cache_request(model: str, tail: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": stable_system_prefix(),
            },
            {
                "role": "user",
                "content": (
                    f"{tail}\nReturn exactly this pinned value in one sentence: {KV_PIN}."
                ),
            },
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 32,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def stable_system_prefix() -> str:
    lines = [
        "You are a deterministic KV/cache stability certification endpoint.",
        f"Pinned recall token: {KV_PIN}.",
        "When a tool is requested, call exactly the requested tool and never invent facts.",
    ]
    for index in range(192):
        lines.append(
            "stable-prefix-block-"
            f"{index:03d}: preserve tool schema, conversation state, and cached prompt geometry."
        )
    return "\n".join(lines)


def extract_tool_call(response: dict[str, Any]) -> ToolCall:
    finish_reason = _first_choice(response).get("finish_reason")
    if finish_reason != "tool_calls":
        raise ValueError(f"tool-call turn finish_reason was not tool_calls: {finish_reason!r}")
    message = _first_message(response)
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("response did not contain tool_calls")
    call = calls[0]
    if not isinstance(call, dict):
        raise ValueError("tool call is not an object")
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("tool call is missing a non-empty id")
    if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
        raise ValueError(f"unexpected tool name: {function!r}")
    arguments = _decode_tool_arguments(function.get("arguments"))
    key = arguments.get("key")
    if key not in FIXTURE_FACTS:
        raise ValueError(f"unsupported fixture key: {key!r}")
    return ToolCall(call_id=call_id, key=key)


def assistant_tool_message(response: dict[str, Any]) -> dict[str, Any]:
    message = dict(_first_message(response))
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
    }


def tool_result_message(call: ToolCall) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.call_id,
        "name": TOOL_NAME,
        "content": json.dumps(
            {"key": call.key, "value": FIXTURE_FACTS[call.key]},
            separators=(",", ":"),
        ),
    }


def extract_cache_metrics(response: dict[str, Any]) -> CacheMetrics:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return CacheMetrics(prompt_tokens=0, cached_tokens=0)
    prompt_tokens = _as_non_negative_int(usage.get("prompt_tokens"))
    details = usage.get("prompt_tokens_details")
    cached_tokens = 0
    if isinstance(details, dict):
        cached_tokens = _as_non_negative_int(details.get("cached_tokens"))
    return CacheMetrics(prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)


def evaluate_cache_threshold(
    metrics: CacheMetrics,
    min_cached_tokens: int,
    suffix_prefill_limit: int,
) -> tuple[bool, str]:
    if metrics.cached_tokens < min_cached_tokens:
        return (
            False,
            (
                f"cached_tokens={metrics.cached_tokens} below required minimum "
                f"{min_cached_tokens}; prompt_tokens={metrics.prompt_tokens}"
            ),
        )
    if metrics.prompt_tokens > 0:
        suffix_prefill_tokens = max(metrics.prompt_tokens - metrics.cached_tokens, 0)
        if suffix_prefill_tokens > suffix_prefill_limit:
            return (
                False,
                (
                    f"suffix_prefill_tokens={suffix_prefill_tokens} above limit "
                    f"{suffix_prefill_limit}; prompt_tokens={metrics.prompt_tokens} "
                    f"cached_tokens={metrics.cached_tokens}"
                ),
            )
    return (
        True,
        (
            f"prompt_tokens={metrics.prompt_tokens} cached_tokens={metrics.cached_tokens} "
            f"min_cached_tokens={min_cached_tokens}"
        ),
    )


def scan_failure_logs(paths: Iterable[Path]) -> list[LogFinding]:
    findings: list[LogFinding] = []
    for path in paths:
        if not path.exists():
            findings.append(
                LogFinding(
                    path=str(path),
                    line_number=0,
                    pattern="missing log",
                    text="native log path did not exist",
                )
            )
            continue
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.decode("utf-8", errors="replace").strip()
                pattern = matched_failure_pattern(line)
                if pattern:
                    findings.append(
                        LogFinding(
                            path=str(path),
                            line_number=line_number,
                            pattern=pattern,
                            text=line[:500],
                        )
                    )
    return findings


def capture_native_log_checkpoints(paths: Iterable[Path]) -> list[NativeLogCheckpoint]:
    checkpoints: list[NativeLogCheckpoint] = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            checkpoints.append(NativeLogCheckpoint(path=path, offset=0, identity=None, tail=b""))
            continue
        checkpoints.append(
            NativeLogCheckpoint(
                path=path,
                offset=stat.st_size,
                identity=file_identity(stat),
                tail=read_checkpoint_tail(path, stat.st_size),
            )
        )
    return checkpoints


def scan_failure_logs_since(
    checkpoints: Iterable[NativeLogCheckpoint],
) -> list[LogFinding]:
    findings: list[LogFinding] = []
    for checkpoint in checkpoints:
        findings.extend(scan_one_log_since(checkpoint))
    return findings


def scan_one_log_since(checkpoint: NativeLogCheckpoint) -> list[LogFinding]:
    path = checkpoint.path
    try:
        stat = path.stat()
    except FileNotFoundError:
        return [
            LogFinding(
                path=str(path),
                line_number=0,
                pattern="missing log",
                text="native log path did not exist",
            )
        ]
    offset = checkpoint.offset
    if (
        checkpoint.identity != file_identity(stat)
        or stat.st_size < checkpoint.offset
        or not checkpoint_tail_matches(checkpoint)
    ):
        offset = 0
    with path.open("rb") as handle:
        start_line_number = line_number_start_for_offset(handle, offset)
        handle.seek(offset)
        return scan_failure_lines(path, handle, start_line_number)


def scan_failure_lines(
    path: Path,
    handle: Iterable[bytes],
    start_line_number: int = 1,
) -> list[LogFinding]:
    findings: list[LogFinding] = []
    for line_number, raw_line in enumerate(handle, start=start_line_number):
        line = raw_line.decode("utf-8", errors="replace").strip()
        pattern = matched_failure_pattern(line)
        if pattern:
            findings.append(
                LogFinding(
                    path=str(path),
                    line_number=line_number,
                    pattern=pattern,
                    text=line[:500],
                )
            )
    return findings


def line_number_start_for_offset(handle: Any, offset: int) -> int:
    if offset <= 0:
        return 1
    handle.seek(0)
    remaining = offset
    newline_count = 0
    while remaining > 0:
        chunk = handle.read(min(remaining, 1024 * 1024))
        if not chunk:
            break
        remaining -= len(chunk)
        newline_count += chunk.count(b"\n")
    return newline_count + 1


def file_identity(stat: os.stat_result) -> tuple[int, int]:
    return (int(stat.st_dev), int(stat.st_ino))


def read_checkpoint_tail(path: Path, offset: int) -> bytes:
    if offset <= 0:
        return b""
    start = max(offset - NATIVE_LOG_CHECKPOINT_TAIL_BYTES, 0)
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(offset - start)


def checkpoint_tail_matches(checkpoint: NativeLogCheckpoint) -> bool:
    if checkpoint.offset <= 0:
        return True
    try:
        return read_checkpoint_tail(checkpoint.path, checkpoint.offset) == checkpoint.tail
    except OSError:
        return False


def matched_failure_pattern(line: str) -> str | None:
    for pattern in FAILURE_PATTERNS:
        if pattern in line:
            return pattern
    if "proactive_eviction" in line and "status=error" in line:
        return "proactive_eviction status=error"
    return None


def run_tool_loop_probe(
    base_url: str,
    model: str,
    attempt: int,
    timeout: float,
    pressure_turns: int,
    transcript_dir: Path,
) -> ProbeResult:
    started = time.monotonic()
    transcript_path = transcript_dir / safe_name(f"{model}-attempt-{attempt}.jsonl")
    messages = initial_messages(attempt, "primary")
    try:
        response, status_code = post_json(
            base_url,
            "/chat/completions",
            build_tool_call_request(model, attempt, messages=messages),
            timeout,
        )
        first_call = extract_tool_call(response)
        record_transcript(transcript_path, "first_tool_call", status_code, first_call.call_id)
        messages.extend([assistant_tool_message(response), tool_result_message(first_call)])
        run_final_after_tool(base_url, model, messages, timeout, [FIXTURE_FACTS[first_call.key]])
        run_pressure_turns(base_url, model, messages, timeout, pressure_turns, transcript_path)
        run_second_tool_loop(base_url, model, messages, timeout, transcript_path)
        run_final_recall(base_url, model, messages, timeout, transcript_path)
        return _result(
            model,
            attempt,
            "tool_loop",
            True,
            f"completed {pressure_turns} pressure turns and two tool calls",
            started,
            status_code,
        )
    except Exception as exc:
        record_transcript(transcript_path, "failure", None, detail=str(exc))
        return _result(model, attempt, "tool_loop", False, str(exc), started)


def run_final_after_tool(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    expected_values: Iterable[str],
) -> None:
    messages.append(
        {
            "role": "user",
            "content": f"Answer with the tool fact and include {KV_PIN}.",
        }
    )
    response, _status_code = post_json(
        base_url,
        "/chat/completions",
        build_tool_result_request(model, messages),
        timeout,
    )
    validate_message(response, [*expected_values, KV_PIN])
    messages.append(_assistant_text_message(response))


def run_pressure_turns(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    pressure_turns: int,
    transcript_path: Path,
) -> None:
    for turn in range(1, pressure_turns + 1):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Pressure turn {turn}: keep the pinned value stable. "
                    f"Return {KV_PIN} and a short confirmation."
                ),
            }
        )
        response, status_code = post_json(
            base_url,
            "/chat/completions",
            build_tool_result_request(model, messages, max_tokens=64),
            timeout,
        )
        validate_message(response, [KV_PIN])
        record_transcript(transcript_path, f"pressure_turn_{turn}", status_code)
        messages.append(_assistant_text_message(response))


def run_second_tool_loop(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    transcript_path: Path,
) -> None:
    messages.append(
        {
            "role": "user",
            "content": (
                f"Now call {TOOL_NAME} with key=secondary. "
                "Do not answer directly before the tool call."
            ),
        }
    )
    response, status_code = post_json(
        base_url,
        "/chat/completions",
        build_tool_call_request(model, attempt=1, key="secondary", messages=messages),
        timeout,
    )
    call = extract_tool_call(response)
    record_transcript(transcript_path, "second_tool_call", status_code, call.call_id)
    messages.extend([assistant_tool_message(response), tool_result_message(call)])


def run_final_recall(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    transcript_path: Path,
) -> None:
    expected = [KV_PIN, FIXTURE_FACTS["primary"], FIXTURE_FACTS["secondary"]]
    messages.append(
        {
            "role": "user",
            "content": "Final recall: include both tool facts and the pinned KV value.",
        }
    )
    response, status_code = post_json(
        base_url,
        "/chat/completions",
        build_tool_result_request(model, messages, max_tokens=128),
        timeout,
    )
    validate_message(response, expected)
    record_transcript(transcript_path, "final_recall", status_code)
    messages.append(_assistant_text_message(response))


def run_cache_probe(
    base_url: str,
    model: str,
    phase: str,
    timeout: float,
    min_cached_tokens: int,
    suffix_prefill_limit: int,
) -> ProbeResult:
    started = time.monotonic()
    try:
        if phase == "exact_prefix_cache":
            warm = build_cache_request(model, "Exact-prefix warmup tail.")
            measured = dict(warm)
        else:
            warm = build_cache_request(model, "Same-prefix warmup tail alpha.")
            measured = build_cache_request(model, "Same-prefix measured tail beta.")
        post_json(base_url, "/chat/completions", warm, timeout)
        response, status_code = post_json(base_url, "/chat/completions", measured, timeout)
        validate_message(response, [KV_PIN])
        metrics = extract_cache_metrics(response)
        ok, detail = evaluate_cache_threshold(
            metrics,
            min_cached_tokens=min_cached_tokens,
            suffix_prefill_limit=suffix_prefill_limit,
        )
        return _result(
            model,
            0,
            phase,
            ok,
            detail,
            started,
            status_code,
            metrics.prompt_tokens,
            metrics.cached_tokens,
        )
    except Exception as exc:
        return _result(model, 0, phase, False, str(exc), started)


def run_native_log_scan(checkpoints: Iterable[NativeLogCheckpoint]) -> ProbeResult:
    started = time.monotonic()
    findings = scan_failure_logs_since(checkpoints)
    if findings:
        detail = "; ".join(
            f"{finding.path}:{finding.line_number} {finding.pattern}: {finding.text[:160]}"
            for finding in findings[:5]
        )
        if len(findings) > 5:
            detail += f"; +{len(findings) - 5} more"
        return _result("native-log", 0, "native_log_scan", False, detail, started)
    return _result("native-log", 0, "native_log_scan", True, "no fatal KV log patterns", started)


def run_certification(
    base_url: str,
    models: Iterable[str],
    attempts: int,
    timeout: float,
    pressure_turns: int,
    min_cached_tokens: int,
    suffix_prefill_limit: int,
    native_logs: Iterable[Path],
    output_dir: Path,
) -> list[ProbeResult]:
    transcript_dir = output_dir / "transcripts"
    prepare_transcript_dir(transcript_dir)
    log_checkpoints = capture_native_log_checkpoints(native_logs)
    results: list[ProbeResult] = []
    for model in models:
        for attempt in range(1, attempts + 1):
            results.append(
                run_tool_loop_probe(
                    base_url,
                    model,
                    attempt,
                    timeout,
                    pressure_turns,
                    transcript_dir,
                )
            )
        results.append(
            run_cache_probe(
                base_url,
                model,
                "same_prefix_cache",
                timeout,
                min_cached_tokens,
                suffix_prefill_limit,
            )
        )
        results.append(
            run_cache_probe(
                base_url,
                model,
                "exact_prefix_cache",
                timeout,
                min_cached_tokens,
                suffix_prefill_limit,
            )
        )
    log_paths = list(native_logs)
    if log_paths:
        results.append(run_native_log_scan(log_checkpoints))
    return results


def prepare_transcript_dir(transcript_dir: Path) -> None:
    if transcript_dir.is_symlink() or transcript_dir.is_file():
        transcript_dir.unlink()
    elif transcript_dir.exists():
        shutil.rmtree(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)


def write_evidence(output_dir: Path, plan: dict[str, Any], results: Iterable[ProbeResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    manifest = dict(plan)
    manifest["created_at"] = utc_now()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row._asdict(), sort_keys=True) + "\n")
    summary = summarize_results(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_summary_markdown(summary, rows),
        encoding="utf-8",
    )


def summarize_results(results: Iterable[ProbeResult]) -> dict[str, Any]:
    rows = list(results)
    passed = sum(1 for row in rows if row.ok)
    failed = len(rows) - passed
    phases: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = phases.setdefault(row.phase, {"passed": 0, "failed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed" if row.ok else "failed"] += 1
    return {
        "ok": failed == 0,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "phases": phases,
    }


def render_summary_markdown(summary: dict[str, Any], results: Iterable[ProbeResult]) -> str:
    rows = list(results)
    status = "PASS" if summary["ok"] else "FAIL"
    lines = [
        "# KV Tool-Loop Stability Summary",
        "",
        f"Status: **{status}**",
        "",
        "| Phase | Passed | Failed | Total |",
        "|---|---:|---:|---:|",
    ]
    for phase, counts in sorted(summary["phases"].items()):
        lines.append(
            f"| {phase} | {counts['passed']} | {counts['failed']} | {counts['total']} |"
        )
    lines.extend(
        [
            "",
            "| Result | Model | Attempt | Phase | Detail |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in rows:
        result = "PASS" if row.ok else "FAIL"
        detail = row.detail.replace("|", "\\|")
        lines.append(f"| {result} | {row.model} | {row.attempt} | {row.phase} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], int]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{normalize_v1_base(base_url)}{path}",
        data=data,
        headers={
            "content-type": "application/json",
            "authorization": "Bearer mesh-llm-kv-tool-loop-probe",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status_code = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        snippet = body.decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"response was not JSON: {snippet}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("response JSON was not an object")
    return decoded, status_code


def validate_message(response: dict[str, Any], expected_values: Iterable[str]) -> None:
    message = _first_message(response)
    if message.get("tool_calls"):
        raise ValueError("expected final text, got another tool call")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response content was empty")
    missing = [value for value in expected_values if value not in content]
    if missing:
        raise ValueError(f"response missing expected values: {', '.join(missing)}")


def record_transcript(
    path: Path,
    phase: str,
    status_code: int | None,
    tool_call_id: str | None = None,
    detail: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "status_code": status_code,
        "tool_call_id": tool_call_id,
        "detail": detail,
        "recorded_at": utc_now(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def print_summary(results: Iterable[ProbeResult], output_dir: Path) -> None:
    rows = list(results)
    passed = sum(1 for row in rows if row.ok)
    print(f"kv/tool-loop stability: {passed}/{len(rows)} phases passed")
    print(f"evidence: {output_dir}")
    for row in rows:
        status = "PASS" if row.ok else "FAIL"
        print(f"{status} model={row.model} attempt={row.attempt} phase={row.phase}: {row.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Certify live mesh-llm KV/cache stability under OpenAI tool-loop pressure.",
    )
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument(
        "--models",
        default=os.environ.get("MESH_KV_TOOL_LOOP_MODELS", DEFAULT_MODELS),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=int(os.environ.get("MESH_KV_TOOL_LOOP_ATTEMPTS", str(DEFAULT_ATTEMPTS))),
    )
    parser.add_argument(
        "--pressure-turns",
        type=int,
        default=int(os.environ.get("MESH_KV_TOOL_LOOP_PRESSURE_TURNS", str(DEFAULT_PRESSURE_TURNS))),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("MESH_KV_TOOL_LOOP_TIMEOUT", str(DEFAULT_TIMEOUT))),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MESH_KV_TOOL_LOOP_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--min-cached-tokens",
        type=int,
        default=int(
            os.environ.get("MESH_KV_TOOL_LOOP_MIN_CACHED_TOKENS", str(DEFAULT_MIN_CACHED_TOKENS))
        ),
    )
    parser.add_argument(
        "--suffix-prefill-limit",
        type=int,
        default=int(
            os.environ.get(
                "MESH_KV_TOOL_LOOP_SUFFIX_PREFILL_LIMIT",
                str(DEFAULT_SUFFIX_PREFILL_LIMIT),
            )
        ),
    )
    parser.add_argument("--native-log", action="append", default=[])
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args()


def default_base_url() -> str:
    env_base = os.environ.get("MESH_KV_TOOL_LOOP_BASE_URL")
    if env_base:
        return env_base
    client_base = os.environ.get("MESH_CLIENT_API_BASE")
    if client_base:
        return f"{client_base.rstrip('/')}/v1"
    return DEFAULT_BASE_URL


def main() -> int:
    args = parse_args()
    base_url = normalize_v1_base(args.base_url)
    models = parse_models(args.models)
    validate_runtime_options(args)
    output_dir = Path(args.output_dir)
    native_logs = parse_native_logs(args.native_log)
    plan = build_plan(
        base_url=base_url,
        models=models,
        attempts=args.attempts,
        pressure_turns=args.pressure_turns,
        timeout=args.timeout,
        output_dir=output_dir,
        min_cached_tokens=args.min_cached_tokens,
        suffix_prefill_limit=args.suffix_prefill_limit,
        native_logs=native_logs,
    )
    if args.print_plan:
        print(render_plan(plan))
        return 0
    results = run_certification(
        base_url=base_url,
        models=models,
        attempts=args.attempts,
        timeout=args.timeout,
        pressure_turns=args.pressure_turns,
        min_cached_tokens=args.min_cached_tokens,
        suffix_prefill_limit=args.suffix_prefill_limit,
        native_logs=native_logs,
        output_dir=output_dir,
    )
    write_evidence(output_dir, plan, results)
    print_summary(results, output_dir)
    return 0 if all(row.ok for row in results) else 1


def validate_runtime_options(args: argparse.Namespace) -> None:
    if args.attempts < 1:
        raise ValueError("attempts must be at least 1")
    if args.pressure_turns < 0:
        raise ValueError("pressure-turns cannot be negative")
    if args.min_cached_tokens < 0:
        raise ValueError("min-cached-tokens cannot be negative")
    if args.suffix_prefill_limit < 0:
        raise ValueError("suffix-prefill-limit cannot be negative")


def _first_message(response: dict[str, Any]) -> dict[str, Any]:
    choice = _first_choice(response)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("first choice did not contain a message object")
    return message


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response had no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("first choice is not an object")
    return choice


def _decode_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments were not valid JSON") from exc
    elif isinstance(arguments, dict):
        decoded = arguments
    else:
        raise ValueError("tool arguments were neither JSON string nor object")
    if not isinstance(decoded, dict):
        raise ValueError("tool arguments were not a JSON object")
    return decoded


def _assistant_text_message(response: dict[str, Any]) -> dict[str, Any]:
    message = _first_message(response)
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("assistant message content was not text")
    return {"role": "assistant", "content": content}


def _as_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _result(
    model: str,
    attempt: int,
    phase: str,
    ok: bool,
    detail: str,
    started: float,
    status_code: int | None = None,
    prompt_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> ProbeResult:
    return ProbeResult(
        model=model,
        attempt=attempt,
        phase=phase,
        ok=ok,
        detail=detail,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        status_code=status_code,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )


def safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return f"{safe}.jsonl"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
