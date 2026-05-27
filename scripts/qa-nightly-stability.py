#!/usr/bin/env python3
"""Run repeatable mesh-llm stability checks and write evidence artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, NamedTuple
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://127.0.0.1:9337/v1"
DEFAULT_MODELS = "auto,mesh"
DEFAULT_ATTEMPTS = 3
DEFAULT_TIMEOUT = 120.0
DEFAULT_OUTPUT_DIR = "target/nightly-stability/latest"
VALID_AGENT_SMOKES = ("opencode", "pi", "goose")


class CommandSpec(NamedTuple):
    name: str
    command: list[str]
    env: dict[str, str]
    log: str
    prerequisite: str | None = None


class CommandResult(NamedTuple):
    name: str
    status: str
    exit_code: int
    elapsed_ms: int
    log: str


class ProbeResult(NamedTuple):
    model: str | None
    attempt: int | None
    phase: str
    ok: bool
    detail: str
    elapsed_ms: int
    status_code: int | None = None
    ttft_ms: int | None = None
    actual_model: str | None = None
    tok_per_sec: float | None = None


class AttestationResult(NamedTuple):
    status: str
    ok: bool
    binary: str | None
    expected_status: str | None
    version: int | None = None
    signer_key_id: str | None = None
    artifact_digest: str | None = None
    error: str | None = None
    elapsed_ms: int = 0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def strip_think_tags(text: str) -> str:
    """Remove <think> and </think> tag markers from the text."""
    return re.sub(r"</?think>\s*", "", text).strip()


def normalize_v1_base(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("base URL is empty")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_models(value: str) -> list[str]:
    models = parse_csv(value)
    if not models:
        raise ValueError("at least one model is required")
    return models


def parse_agent_smokes(value: str) -> list[str]:
    requested = parse_csv(value)
    unknown = sorted(set(requested) - set(VALID_AGENT_SMOKES))
    if unknown:
        joined = ", ".join(unknown)
        allowed = ", ".join(VALID_AGENT_SMOKES)
        raise ValueError(f"unknown agent smoke: {joined}; expected one of {allowed}")
    return requested


def build_plan(
    base_url: str,
    models: list[str],
    attempts: int,
    output_dir: Path,
    agent_smokes: list[str],
    skip_streaming: bool,
    timeout: float,
    mesh_binary: str | None,
    release_attestation_expected_status: str | None,
) -> dict[str, Any]:
    specs = build_command_specs(
        base_url=base_url,
        models=models,
        attempts=attempts,
        output_dir=output_dir,
        agent_smokes=agent_smokes,
        skip_streaming=skip_streaming,
        timeout=timeout,
    )
    steps: list[dict[str, Any]] = [
        {
            "name": "openai-surface-probe",
            "models": models,
            "attempts": attempts,
            "phases": ["models", "chat", *([] if skip_streaming else ["stream_chat"])],
            "output": str(output_dir / "results.jsonl"),
        }
    ]
    steps.extend(
        {
            "name": spec.name,
            "command": spec.command,
            "env": spec.env,
            "log": spec.log,
            "prerequisite": spec.prerequisite,
        }
        for spec in specs
    )
    if mesh_binary:
        steps.append(
            {
                "name": "release-attestation-inspect",
                "binary": mesh_binary,
                "expected_status": release_attestation_expected_status,
                "output": str(output_dir / "release-attestation.json"),
            }
        )
    return {
        "name": "nightly-stability",
        "endpoint": normalize_v1_base(base_url),
        "models": models,
        "attempts": attempts,
        "output_dir": str(output_dir),
        "evidence": [
            "manifest.json",
            "commands.jsonl",
            "results.jsonl",
            "release-attestation.json",
            "summary.json",
            "summary.md",
            "logs/",
        ],
        "steps": steps,
    }


def build_command_specs(
    base_url: str,
    models: list[str],
    attempts: int,
    output_dir: Path,
    agent_smokes: list[str],
    skip_streaming: bool,
    timeout: float,
) -> list[CommandSpec]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    base = normalize_v1_base(base_url)
    logs_dir = Path("logs")
    specs = [
        CommandSpec(
            name="tool-call-reliability",
            command=_tool_call_command(base, models, attempts, output_dir, skip_streaming, timeout),
            env={},
            log=str(logs_dir / "tool-call-reliability.log"),
        )
    ]
    for smoke in agent_smokes:
        specs.append(_agent_smoke_spec(smoke, base, output_dir))
    return specs


def _tool_call_command(
    base_url: str,
    models: list[str],
    attempts: int,
    output_dir: Path,
    skip_streaming: bool,
    timeout: float,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root() / "scripts" / "qa-agent-tool-call-reliability.py"),
        "--base-url",
        base_url,
        "--models",
        ",".join(models),
        "--attempts",
        str(attempts),
        "--timeout",
        _format_float(timeout),
        "--output",
        str(output_dir / "tool-call-results.jsonl"),
    ]
    if skip_streaming:
        command.append("--skip-streaming")
    return command


def _agent_smoke_spec(smoke: str, base_url: str, output_dir: Path) -> CommandSpec:
    logs_dir = Path("logs")
    work_dir = output_dir / "agent-smokes" / smoke
    env = {
        "MESH_AGENT_BASE_URL": base_url,
        "MESH_OPENCODE_BASE_URL": base_url,
    }
    if smoke == "opencode":
        script = "ci-opencode-smoke.sh"
        prerequisite = "opencode"
        env.update(
            {
                "OPENCODE_SMOKE_WORK_DIR": str(work_dir),
                "OPENCODE_SMOKE_OUTPUT": str(work_dir / "opencode-output.jsonl"),
                "OPENCODE_SMOKE_TURN1_OUTPUT": str(work_dir / "opencode-turn1.jsonl"),
                "OPENCODE_SMOKE_TURN2_OUTPUT": str(work_dir / "opencode-turn2.jsonl"),
                "OPENCODE_SMOKE_ERROR_LOG": str(work_dir / "opencode-stderr.log"),
                "OPENCODE_SMOKE_SURFACE_LOG": str(work_dir / "openai-surface.jsonl"),
                "OPENCODE_SMOKE_SURFACE_PROXY_LOG": str(work_dir / "openai-surface-proxy.log"),
            }
        )
    elif smoke == "pi":
        script = "ci-pi-smoke.sh"
        prerequisite = "pi"
        env.update(
            {
                "PI_SMOKE_WORK_DIR": str(work_dir),
                "PI_SMOKE_OUTPUT": str(work_dir / "pi-output.jsonl"),
                "PI_SMOKE_ERROR_LOG": str(work_dir / "pi-stderr.log"),
            }
        )
    elif smoke == "goose":
        script = "ci-goose-smoke.sh"
        prerequisite = "goose"
        env.update(
            {
                "GOOSE_SMOKE_WORK_DIR": str(work_dir),
                "GOOSE_SMOKE_OUTPUT": str(work_dir / "goose-output.jsonl"),
                "GOOSE_SMOKE_ERROR_LOG": str(work_dir / "goose-stderr.log"),
            }
        )
    else:
        raise ValueError(f"unknown agent smoke: {smoke}")
    return CommandSpec(
        name=f"{smoke}-agent-smoke",
        command=[str(repo_root() / "scripts" / script)],
        env=env,
        log=str(logs_dir / f"{smoke}-agent-smoke.log"),
        prerequisite=prerequisite,
    )


def run_commands(specs: Iterable[CommandSpec], output_dir: Path) -> list[CommandResult]:
    results: list[CommandResult] = []
    for spec in specs:
        result = run_command(spec, output_dir)
        results.append(result)
        print(
            f"{result.status} {result.name}: exit={result.exit_code} "
            f"elapsed_ms={result.elapsed_ms} log={result.log}",
            flush=True,
        )
    return results


def run_command(spec: CommandSpec, output_dir: Path) -> CommandResult:
    log_path = output_dir / spec.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.prerequisite and shutil.which(spec.prerequisite) is None:
        log_path.write_text(
            f"Prerequisite command not found on PATH: {spec.prerequisite}\n",
            encoding="utf-8",
        )
        return CommandResult(
            name=spec.name,
            status="PREREQ",
            exit_code=0,
            elapsed_ms=0,
            log=spec.log,
        )
    env = os.environ.copy()
    env.update(spec.env)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shell_join(spec.command)}\n\n")
        process = subprocess.run(
            spec.command,
            cwd=repo_root(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_ms = int((time.monotonic() - start) * 1000)
    status = "PASS" if process.returncode == 0 else "FAIL"
    return CommandResult(
        name=spec.name,
        status=status,
        exit_code=process.returncode,
        elapsed_ms=elapsed_ms,
        log=spec.log,
    )


def _print_probe_result(result: ProbeResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    tok_str = f" tok/s={result.tok_per_sec:.1f}" if result.tok_per_sec is not None else ""
    actual = f" actual={result.actual_model}" if result.actual_model else ""
    print(
        f"{status} {result.phase} model={result.model or '-'}"
        f"{actual} attempt={result.attempt or '-'}"
        f" elapsed_ms={result.elapsed_ms}{tok_str}: {result.detail}",
        flush=True,
    )


def run_surface_probes(
    base_url: str,
    models: list[str],
    attempts: int,
    timeout: float,
    include_streaming: bool,
) -> list[ProbeResult]:
    base = normalize_v1_base(base_url)
    results: list[ProbeResult] = []

    result = run_models_probe(base, timeout)
    results.append(result)
    _print_probe_result(result)

    for model in models:
        for attempt in range(1, attempts + 1):
            result = run_chat_probe(base, model, attempt, timeout)
            results.append(result)
            _print_probe_result(result)
            if include_streaming:
                result = run_stream_chat_probe(base, model, attempt, timeout)
                results.append(result)
                _print_probe_result(result)

    return results


def run_models_probe(base_url: str, timeout: float) -> ProbeResult:
    started = time.monotonic()
    try:
        response, status_code = get_json(base_url, "/models", timeout)
        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("models response did not contain any models")
        return _probe_result(None, None, "models", True, f"{len(data)} models", started, status_code)
    except Exception as exc:
        return _probe_result(None, None, "models", False, str(exc), started)


def run_chat_probe(base_url: str, model: str, attempt: int, timeout: float) -> ProbeResult:
    started = time.monotonic()
    try:
        response, status_code = post_json(
            base_url,
            "/chat/completions",
            build_chat_request(model, attempt, stream=False),
            timeout,
        )
        actual_model = response.get("model")
        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") or 0
        elapsed_s = max(time.monotonic() - started, 0.001)
        tok_per_sec = completion_tokens / elapsed_s if completion_tokens else None
        validate_sentinel(first_message_content(response), "STABILITY_OK")
        return _probe_result(
            model, attempt, "chat", True, "matched STABILITY_OK", started,
            status_code, actual_model=actual_model, tok_per_sec=tok_per_sec,
        )
    except Exception as exc:
        return _probe_result(
            model, attempt, "chat", False, str(exc), started,
            actual_model=actual_model,  # type: ignore[union-attr]
            tok_per_sec=tok_per_sec,  # type: ignore[union-attr]
        )


def run_stream_chat_probe(base_url: str, model: str, attempt: int, timeout: float) -> ProbeResult:
    started = time.monotonic()
    try:
        chunks, status_code, ttft_ms = post_json_stream(
            base_url,
            "/chat/completions",
            build_chat_request(model, attempt, stream=True),
            timeout,
        )
        # Extract model name and usage from stream chunks (before sentinel
        # validation so these are available even on failure).
        actual_model = None
        completion_tokens = 0
        for chunk in chunks:
            if not actual_model and chunk.get("model"):
                actual_model = chunk.get("model")
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens") or 0
        elapsed_s = max(time.monotonic() - started, 0.001)
        tok_per_sec = completion_tokens / elapsed_s if completion_tokens else None
        validate_sentinel(stream_content(chunks), "STREAM_OK")
        return _probe_result(
            model,
            attempt,
            "stream_chat",
            True,
            "matched STREAM_OK",
            started,
            status_code,
            ttft_ms,
            actual_model=actual_model,
            tok_per_sec=tok_per_sec,
        )
    except Exception as exc:
        return _probe_result(
            model,
            attempt,
            "stream_chat",
            False,
            str(exc),
            started,
            actual_model=actual_model,  # type: ignore[union-attr]
            tok_per_sec=tok_per_sec,  # type: ignore[union-attr]
        )


def build_chat_request(model: str, attempt: int, stream: bool) -> dict[str, Any]:
    sentinel = "STREAM_OK" if stream else "STABILITY_OK"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a deterministic mesh-llm stability probe.",
            },
            {
                "role": "user",
                "content": f"Attempt {attempt}: reply with exactly {sentinel} and no extra text.",
            },
        ],
        "stream": stream,
        "max_tokens": 32,
        "temperature": 0,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def get_json(base_url: str, path: str, timeout: float) -> tuple[dict[str, Any], int]:
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status_code = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    return decode_json_object(body), status_code


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], int]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"content-type": "application/json", "authorization": "Bearer mesh-llm-stability"},
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
    return decode_json_object(body), status_code


def post_json_stream(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[list[dict[str, Any]], int, int | None]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"content-type": "application/json", "authorization": "Bearer mesh-llm-stability"},
        method="POST",
    )
    started = time.monotonic()
    first_event_ms: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            chunks: list[dict[str, Any]] = []
            for chunk in parse_sse_lines(line.decode("utf-8", errors="replace") for line in response):
                if first_event_ms is None:
                    first_event_ms = int((time.monotonic() - started) * 1000)
                chunks.append(chunk)
            status_code = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    if not chunks:
        raise RuntimeError("stream returned no events")
    return chunks, status_code, first_event_ms


def decode_json_object(body: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        snippet = body.decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"response was not JSON: {snippet}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("response JSON was not an object")
    return decoded


def parse_sse_lines(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"stream event was not JSON: {data[:200]}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("stream event JSON was not an object")
        yield decoded


def first_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response had no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("first choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("first choice did not contain a message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("message content was not a string")
    return content


def stream_content(chunks: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    saw_choice = False
    for chunk in chunks:
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            saw_choice = True
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    if not saw_choice:
        raise ValueError("stream returned no choices")
    return "".join(parts)


def validate_sentinel(content: str, sentinel: str) -> None:
    cleaned = strip_think_tags(content)
    if cleaned != sentinel and sentinel not in cleaned:
        raise ValueError(f"expected exactly {sentinel}, got {content!r}")


def _probe_result(
    model: str | None,
    attempt: int | None,
    phase: str,
    ok: bool,
    detail: str,
    started: float,
    status_code: int | None = None,
    ttft_ms: int | None = None,
    actual_model: str | None = None,
    tok_per_sec: float | None = None,
) -> ProbeResult:
    return ProbeResult(
        model=model,
        attempt=attempt,
        phase=phase,
        ok=ok,
        detail=detail,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        status_code=status_code,
        ttft_ms=ttft_ms,
        actual_model=actual_model,
        tok_per_sec=tok_per_sec,
    )


def write_evidence(
    output_dir: Path,
    plan: dict[str, Any],
    results: list[CommandResult],
    probe_results: list[ProbeResult] | None = None,
    attestation_result: AttestationResult | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(plan)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "manifest.json", manifest)
    _write_jsonl(output_dir / "commands.jsonl", [result._asdict() for result in results])
    _write_jsonl(output_dir / "results.jsonl", [result._asdict() for result in (probe_results or [])])
    _write_json(
        output_dir / "release-attestation.json",
        (attestation_result or default_attestation_result())._asdict(),
    )
    summary = summarize_evidence(results, probe_results or [], attestation_result)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_summary_markdown(summary, results, probe_results or [], attestation_result),
        encoding="utf-8",
    )


def summarize_results(results: Iterable[CommandResult]) -> dict[str, Any]:
    rows = list(results)
    passed = sum(1 for row in rows if row.status == "PASS")
    failed = sum(1 for row in rows if row.status == "FAIL")
    prereq = sum(1 for row in rows if row.status == "PREREQ")
    elapsed_ms = sum(row.elapsed_ms for row in rows)
    return {
        "ok": failed == 0,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "prereq": prereq,
        "elapsed_ms": elapsed_ms,
    }


def summarize_probe_results(results: Iterable[ProbeResult]) -> dict[str, Any]:
    rows = list(results)
    passed = sum(1 for row in rows if row.ok)
    failed = sum(1 for row in rows if not row.ok)
    elapsed_ms = sum(row.elapsed_ms for row in rows)
    return {
        "ok": failed == 0,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "prereq": 0,
        "elapsed_ms": elapsed_ms,
    }


def default_attestation_result() -> AttestationResult:
    return AttestationResult(
        status="not_configured",
        ok=True,
        binary=None,
        expected_status=None,
    )


def summarize_attestation_result(result: AttestationResult | None) -> dict[str, Any]:
    attestation = result or default_attestation_result()
    return {
        "ok": attestation.ok,
        "total": 1,
        "passed": 1 if attestation.ok and attestation.status != "not_configured" else 0,
        "failed": 0 if attestation.ok else 1,
        "prereq": 1 if attestation.status == "not_configured" else 0,
        "elapsed_ms": attestation.elapsed_ms,
        "status": attestation.status,
    }


def summarize_evidence(
    command_results: Iterable[CommandResult],
    probe_results: Iterable[ProbeResult],
    attestation_result: AttestationResult | None = None,
) -> dict[str, Any]:
    commands = summarize_results(command_results)
    probes = summarize_probe_results(probe_results)
    attestation = summarize_attestation_result(attestation_result)
    failed = commands["failed"] + probes["failed"] + attestation["failed"]
    total = commands["total"] + probes["total"] + attestation["total"]
    return {
        "ok": failed == 0,
        "total": total,
        "passed": commands["passed"] + probes["passed"] + attestation["passed"],
        "failed": failed,
        "prereq": commands["prereq"] + probes["prereq"] + attestation["prereq"],
        "elapsed_ms": commands["elapsed_ms"] + probes["elapsed_ms"] + attestation["elapsed_ms"],
        "commands": commands,
        "probes": probes,
        "attestation": attestation,
        "release_attestation": (attestation_result or default_attestation_result())._asdict(),
    }


def render_summary_markdown(
    summary: dict[str, Any],
    results: Iterable[CommandResult],
    probe_results: Iterable[ProbeResult],
    attestation_result: AttestationResult | None,
) -> str:
    commands = summary.get("commands", {})
    probes = summary.get("probes", {})
    attestation = attestation_result or default_attestation_result()
    lines = [
        "# Nightly Stability Summary",
        "",
        f"- Overall: {'PASS' if summary['ok'] else 'FAIL'}",
        f"- Passed: {summary['passed']}/{summary['total']}",
        f"- Prereq: {summary['prereq']}",
        f"- Elapsed: {summary['elapsed_ms']} ms",
        "",
        "## Timing Snapshot",
        "",
        "| Area | Passed | Failed | Prereq | Elapsed ms |",
        "|---|---:|---:|---:|---:|",
        _summary_timing_row("OpenAI surface probes", probes),
        _summary_timing_row("Command probes", commands),
        _summary_timing_row("Release attestation", summary.get("attestation", {})),
        "",
        "## Release Attestation",
        "",
        f"- Status: `{attestation.status}`",
        f"- Expected: `{attestation.expected_status or ''}`",
        f"- Binary: `{attestation.binary or ''}`",
        f"- Version: `{attestation.version or ''}`",
        f"- Signer: `{attestation.signer_key_id or ''}`",
        f"- Artifact digest: `{attestation.artifact_digest or ''}`",
        f"- Error: `{attestation.error or ''}`",
        "",
        "## OpenAI Surface Probes",
        "",
        "| Phase | Requested Model | Attempt | Status | HTTP | TTFT ms | Elapsed ms | Actual Model | tok/s | Detail |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for result in probe_results:
        tok_str = f"{result.tok_per_sec:.1f}" if result.tok_per_sec is not None else ""
        lines.append(
            f"| `{result.phase}` | `{result.model or ''}` | {result.attempt or ''} | "
            f"{'PASS' if result.ok else 'FAIL'} | {result.status_code or ''} | "
            f"{result.ttft_ms or ''} | {result.elapsed_ms} | "
            f"`{result.actual_model or ''}` | {tok_str} | {result.detail} |"
        )
    lines.extend([
        "",
        "## Command Probes",
        "",
        "| Step | Status | Exit | Elapsed ms | Log |",
        "|---|---:|---:|---:|---|",
    ])
    for result in results:
        lines.append(
            f"| `{result.name}` | {result.status} | {result.exit_code} | "
            f"{result.elapsed_ms} | `{result.log}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _summary_timing_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {summary.get('passed', 0)} | {summary.get('failed', 0)} | "
        f"{summary.get('prereq', 0)} | {summary.get('elapsed_ms', 0)} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeatable mesh-llm stability checks and write manifest.json, "
            "commands.jsonl, results.jsonl, summary.json, and summary.md evidence."
        )
    )
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--models", default=os.environ.get("MESH_STABILITY_MODELS", DEFAULT_MODELS))
    parser.add_argument(
        "--attempts",
        type=int,
        default=int(os.environ.get("MESH_STABILITY_ATTEMPTS", str(DEFAULT_ATTEMPTS))),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("MESH_STABILITY_TIMEOUT", str(DEFAULT_TIMEOUT))),
    )
    parser.add_argument(
        "--agent-smokes",
        default=os.environ.get("MESH_STABILITY_AGENT_SMOKES", ""),
        help="Comma-separated optional agent CLI smokes: opencode,pi,goose.",
    )
    parser.add_argument("--output-dir", default=os.environ.get("MESH_STABILITY_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mesh-binary", default=os.environ.get("MESH_STABILITY_MESH_BINARY"))
    parser.add_argument(
        "--release-attestation-public-key-file",
        default=os.environ.get("MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE"),
    )
    parser.add_argument(
        "--release-attestation-expected-status",
        default=os.environ.get("MESH_STABILITY_RELEASE_ATTESTATION_EXPECTED_STATUS", "valid"),
    )
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args()


def inspect_release_attestation(
    binary: str | None,
    public_key_file: str | None,
    expected_status: str | None,
) -> AttestationResult:
    if not binary:
        return default_attestation_result()

    command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "xtask",
        "--",
        "release-attestation",
        "inspect",
        "--binary",
        binary,
        "--json",
    ]
    if public_key_file:
        command.extend(["--public-key-file", public_key_file])

    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if process.returncode != 0:
        return AttestationResult(
            status="command_failed",
            ok=False,
            binary=binary,
            expected_status=expected_status,
            error=process.stderr.strip() or process.stdout.strip() or "xtask inspect failed",
            elapsed_ms=elapsed_ms,
        )

    payload = json.loads(process.stdout)
    status = str(payload.get("status", "invalid"))
    return AttestationResult(
        status=status,
        ok=status == expected_status,
        binary=binary,
        expected_status=expected_status,
        version=payload.get("version"),
        signer_key_id=payload.get("signer_key_id"),
        artifact_digest=payload.get("artifact_digest"),
        error=payload.get("error"),
        elapsed_ms=elapsed_ms,
    )


def default_base_url() -> str:
    for name in ("MESH_STABILITY_BASE_URL", "MESH_AGENT_BASE_URL", "MESH_OPENCODE_BASE_URL"):
        value = os.environ.get(name)
        if value:
            return value
    client_base = os.environ.get("MESH_CLIENT_API_BASE")
    if client_base:
        return f"{client_base.rstrip('/')}/v1"
    return DEFAULT_BASE_URL


def main() -> int:
    try:
        args = parse_args()
        models = parse_models(args.models)
        agent_smokes = parse_agent_smokes(args.agent_smokes)
        output_dir = Path(args.output_dir)
        plan = build_plan(
            base_url=args.base_url,
            models=models,
            attempts=args.attempts,
            output_dir=output_dir,
            agent_smokes=agent_smokes,
            skip_streaming=args.skip_streaming,
            timeout=args.timeout,
            mesh_binary=args.mesh_binary,
            release_attestation_expected_status=args.release_attestation_expected_status,
        )
        if args.print_plan:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        attestation_result = inspect_release_attestation(
            binary=args.mesh_binary,
            public_key_file=args.release_attestation_public_key_file,
            expected_status=args.release_attestation_expected_status,
        )
        specs = build_command_specs(
            base_url=args.base_url,
            models=models,
            attempts=args.attempts,
            output_dir=output_dir,
            agent_smokes=agent_smokes,
            skip_streaming=args.skip_streaming,
            timeout=args.timeout,
        )
        probe_results = run_surface_probes(
            base_url=args.base_url,
            models=models,
            attempts=args.attempts,
            timeout=args.timeout,
            include_streaming=not args.skip_streaming,
        )
        results = run_commands(specs, output_dir)
        write_evidence(output_dir, plan, results, probe_results, attestation_result)
        summary = summarize_evidence(results, probe_results, attestation_result)
        print_human_summary(output_dir, summary, results, probe_results, attestation_result)
        return 0 if summary["ok"] else 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def print_human_summary(
    output_dir: Path,
    summary: dict[str, Any],
    results: Iterable[CommandResult],
    probe_results: Iterable[ProbeResult],
    attestation_result: AttestationResult | None,
) -> None:
    print(f"nightly stability: {summary['passed']}/{summary['total']} steps passed", flush=True)
    print(f"results: {output_dir}", flush=True)
    attestation = attestation_result or default_attestation_result()
    print(
        "release attestation: "
        f"status={attestation.status} expected={attestation.expected_status or '-'} "
        f"binary={attestation.binary or '-'}",
        flush=True,
    )
    for result in probe_results:
        status = "PASS" if result.ok else "FAIL"
        tok_str = f" tok/s={result.tok_per_sec:.1f}" if result.tok_per_sec is not None else ""
        actual = f" actual={result.actual_model}" if result.actual_model else ""
        print(
            f"{status} {result.phase} model={result.model or '-'}"
            f"{actual} attempt={result.attempt or '-'}"
            f" elapsed_ms={result.elapsed_ms}{tok_str}: {result.detail}",
            flush=True,
        )
    for result in results:
        print(
            f"{result.status} {result.name}: exit={result.exit_code} "
            f"elapsed_ms={result.elapsed_ms} log={result.log}",
            flush=True,
        )


def shell_join(command: Iterable[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "/._-:=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
