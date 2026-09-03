#!/usr/bin/env python3
"""Run ONE side of the event-system A/B benchmark matrix (spec §17.5,
`.omo/plans/event-system.md` task 19).

One invocation drives a fixed, deterministic set of paired trials against
ONE binary in ONE trial mode (`production` or `event-disabled`) and writes a
manifest the paired comparator (`compare-event-benchmark-matrix.py`)
consumes. Certification runs this script three times with the SAME `--seed`:
production on the current binary, `event-disabled` on the current binary,
and production on the verified baseline release binary -- the same script
serves the baseline binary because a manifest records binary identity
(path/sha256/`--version`) rather than assuming which binary produced it.

`--mode event-disabled` forwards the hidden, TEST-ONLY selector
`MESH_LLM_EVENT_SYSTEM_TRIAL_MODE=event-disabled` to the spawned process,
which is accepted ONLY alongside `MESH_LLM_BENCHMARK_TUNE_TRIAL=1` (see
`crates/mesh-llm-config/src/env_overrides.rs`). This script always sets both
gate and selector consistently -- it never asks the binary to run this trial
mode without the trial gate.

Trial unit (mirrors `benchmark_trial_unit_definition()` in
`crates/mesh-llm-commands/src/gpus/tune/output_types.rs` VERBATIM -- see
`TRIAL_UNIT_DEFINITION` below and its cross-check test): one trial is one
fresh process launch, one readiness wait, one warmup request excluded from
metrics, one measured streaming request, and one shutdown. A pair is two
trials, one per side, with the same prompt and seed, side order randomized
per pair. This script produces one SIDE of each pair; running it again with
the identical `--seed`/`--pairs-primary`/`--pairs-scenario`/`--scenario`
arguments for the other side yields an index-aligned trial plan the
comparator pairs by (scenario, pair_index).

This module deliberately never runs the real benchmark end-to-end as part of
its own test suite (see `scripts/tests/test_run_event_benchmark_matrix.py`):
every unit test exercises a pure function or injects a fake trial executor.
Real end-to-end certification runs are Task 21's job.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = 1
METRICS_SCHEMA = "streaming_v1"

# The CLI's `--mode` values ARE the values forwarded verbatim as
# `MESH_LLM_EVENT_SYSTEM_TRIAL_MODE`; kept as an explicit alias map (rather
# than passing `args.mode` straight through) so a future CLI spelling change
# cannot silently become a wire-value change without a deliberate edit here.
MODE_TO_TRIAL_ENV_VALUE: dict[str, str] = {
    "production": "production",
    "event-disabled": "event-disabled",
}
VALID_MODES = tuple(MODE_TO_TRIAL_ENV_VALUE)

TRIAL_ENV_NAME = "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"
TRIAL_GATE_ENV_NAME = "MESH_LLM_BENCHMARK_TUNE_TRIAL"

# Persist RAW values only for this explicit, non-sensitive allowlist,
# normalized as booleans/enums/numbers -- matches the plan's benchmark and
# certification protocol verbatim. Every other `MESH_LLM_*` name is
# redacted to name + `<redacted:present>`, never a raw value.
ENV_ALLOWLIST = (
    "MESH_LLM_LIFECYCLE_LOG_PARSER",
    TRIAL_GATE_ENV_NAME,
    TRIAL_ENV_NAME,
)

# Names matching this pattern are ALWAYS redacted, even if a future
# allowlist entry collided with one of them -- defense in depth, not just
# reliance on the allowlist staying hand-curated correctly.
SENSITIVE_NAME_PATTERN = re.compile(
    r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH|URL|PATH)", re.IGNORECASE
)
REDACTED_PRESENT = "<redacted:present>"

# Frozen `decode_only_tok_s` epsilon -- MUST match
# `streaming::DECODE_ONLY_TOK_S_EPSILON_SECS` in
# `crates/mesh-llm-commands/src/gpus/tune/benchmark/streaming.rs`. Duplicated
# (not imported -- this is a standalone Python tool) rather than redefined
# differently; the cross-check test in the paired test file pins this.
DECODE_ONLY_TOK_S_EPSILON_SECS = 1e-6

# Verbatim mirror of `benchmark_trial_unit_definition()` in
# `crates/mesh-llm-commands/src/gpus/tune/output_types.rs`. This is a REUSE
# of the frozen wording, not a redefinition -- the paired test file parses
# the Rust source and asserts these strings match after whitespace
# normalization, so the two can never silently drift apart.
TRIAL_UNIT_DEFINITION: dict[str, str] = {
    "trial": (
        "One trial is one fresh process launch, one readiness wait, "
        "one warmup request excluded from metrics, one measured "
        "streaming request, and one shutdown."
    ),
    "pair": (
        "A pair is two trials, one per side, with the same prompt "
        "and seed, side order randomized per pair."
    ),
}

PRIMARY_SCENARIO = "__primary__"

CERTIFICATION_HOSTS = {
    ("Darwin", "arm64"): "macos-arm64-metal",
    ("Linux", "x86_64"): "linux-x86_64-cuda",
}

U64_MAX = (1 << 64) - 1

DEFAULT_MAX_TOKENS = 64
DEFAULT_READINESS_TIMEOUT_SECS = 120.0
DEFAULT_REQUEST_TIMEOUT_SECS = 120.0
DEFAULT_READINESS_POLL_INTERVAL_SECS = 0.5
DEFAULT_SHUTDOWN_TIMEOUT_SECS = 15.0


def resolve_trial_env_value(mode: str) -> str:
    """Maps a `--mode` CLI value to the `MESH_LLM_EVENT_SYSTEM_TRIAL_MODE`
    wire value. Raises `ValueError` for any value outside `VALID_MODES` --
    argparse's `choices=` already prevents this in practice, but the mapping
    stays a hard error rather than a silent passthrough for direct callers."""
    try:
        return MODE_TO_TRIAL_ENV_VALUE[mode]
    except KeyError as exc:
        raise ValueError(f"unknown --mode {mode!r}; expected one of {VALID_MODES}") from exc


def validate_seed(seed: int) -> None:
    if seed < 0 or seed > U64_MAX:
        raise ValueError(f"--seed must fit in a u64 (0..={U64_MAX}), got {seed}")


def normalize_env_value(name: str, raw: str) -> Any:
    """Normalizes an allowlisted raw env-var string into a bool/enum/number
    for the manifest. Unknown (forward-compatible) allowlisted names
    normalize to the raw string unchanged."""
    if name == TRIAL_GATE_ENV_NAME:
        return raw == "1"
    return raw


def capture_environment_snapshot(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Builds the manifest's persisted environment snapshot: only
    `MESH_LLM_*` names are recorded at all. An allowlisted, non-sensitive
    name is stored as its normalized value; every other name is stored as
    the redacted-presence marker only -- no raw value ever leaves this
    function for a non-allowlisted name, so a comparator reading two
    manifests in memory can compare presence/equality without ever holding
    a secret in the persisted artifact."""
    snapshot: dict[str, dict[str, Any]] = {}
    for name, raw in environ.items():
        if not name.startswith("MESH_LLM_"):
            continue
        if name in ENV_ALLOWLIST and not SENSITIVE_NAME_PATTERN.search(name):
            snapshot[name] = {"value": normalize_env_value(name, raw), "redacted": False}
        else:
            snapshot[name] = {"value": REDACTED_PRESENT, "redacted": True}
    return snapshot


def compute_file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_binary_identity(
    binary: Path,
    *,
    run_version: Callable[[Path], str | None] | None = None,
) -> dict[str, Any]:
    """Records enough identity for THIS SAME script to serve the baseline
    binary too: absolute path, sha256 of the file bytes (`None` if the path
    does not exist, e.g. a placeholder in a unit test), and `<binary>
    --version` stdout (best-effort; `None` when the binary cannot be
    executed). `run_version` is injectable so tests never spawn a real
    process."""
    resolved = binary.expanduser()
    sha256 = compute_file_sha256(resolved)
    if run_version is not None:
        version_output = run_version(resolved)
    else:
        version_output = _run_real_version_probe(resolved)
    return {"path": str(resolved), "sha256": sha256, "version": version_output}


def _run_real_version_probe(binary: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return output or None


def capture_host_classification() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    certification_host = CERTIFICATION_HOSTS.get((system, machine))
    return {
        "system": system,
        "machine": machine,
        "certification_host": certification_host,
        "p99_gate": "enforced" if certification_host else "informational",
    }


def capture_thermal_state(
    *, run_pmset: Callable[[], subprocess.CompletedProcess[str] | None] | None = None,
    thermal_root: Path = Path("/sys/class/thermal"),
) -> dict[str, Any]:
    """Best-effort thermal/power/clock-state capture "where available" (the
    plan's exact phrase). Always returns a well-formed record -- never
    raises -- so a host without any readable thermal source still produces
    an explicit `{"available": False, ...}` record rather than a missing
    field."""
    system = platform.system()
    if system == "Darwin":
        result = run_pmset() if run_pmset is not None else _run_pmset_therm()
        if result is not None and result.returncode == 0 and result.stdout.strip():
            return {"available": True, "source": "pmset -g therm", "raw": result.stdout.strip()}
        return {"available": False, "source": "pmset -g therm"}
    if system == "Linux":
        zones: dict[str, int] = {}
        if thermal_root.is_dir():
            for zone_dir in sorted(thermal_root.glob("thermal_zone*")):
                temp_file = zone_dir / "temp"
                if not temp_file.is_file():
                    continue
                with contextlib.suppress(OSError, ValueError):
                    zones[zone_dir.name] = int(temp_file.read_text().strip())
        if zones:
            return {"available": True, "source": "sysfs", "zones_millidegrees_c": zones}
        return {"available": False, "source": "sysfs"}
    return {"available": False, "source": None}


def _run_pmset_therm() -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class TrialPlanEntry:
    scenario: str
    pair_index: int
    prompt_seed: int


def build_trial_plan(
    seed: int,
    pairs_primary: int,
    pairs_scenario: int,
    scenarios: Sequence[str],
) -> list[TrialPlanEntry]:
    """Deterministic trial plan from `seed`: `pairs_primary` entries in the
    synthetic `__primary__` group, then `pairs_scenario` entries per named
    `--scenario`, in the order scenarios were given. Two invocations of this
    function with the SAME `seed`/counts/scenarios -- as required when
    running production, event-disabled, and baseline through this script --
    produce an IDENTICAL plan, so the comparator can pair trial i of one
    manifest with trial i of another by (scenario, pair_index): "same
    prompt and seed" pairing without the two sides needing to run in the
    same process."""
    if pairs_primary < 1:
        raise ValueError("--pairs-primary must be at least 1")
    if pairs_scenario < 1:
        raise ValueError("--pairs-scenario must be at least 1")
    if not scenarios:
        raise ValueError("at least one --scenario is required")
    rng = random.Random(seed)
    plan: list[TrialPlanEntry] = []
    for index in range(pairs_primary):
        plan.append(TrialPlanEntry(PRIMARY_SCENARIO, index, rng.getrandbits(64)))
    for scenario in scenarios:
        for index in range(pairs_scenario):
            plan.append(TrialPlanEntry(scenario, index, rng.getrandbits(64)))
    return plan


def compute_decode_only_tok_s(
    completion_tokens: int | None,
    total_elapsed_ms: float | None,
    ttft_ms: float | None,
) -> float | None:
    """`completion_tokens / max(total_elapsed - ttft, epsilon)`, mirroring
    `streaming::decode_only_tok_s` in
    `crates/mesh-llm-commands/src/gpus/tune/benchmark/streaming.rs`
    field-for-field: null (never zero) whenever `ttft_ms` is null or the
    decode interval is zero/negative."""
    if completion_tokens is None or total_elapsed_ms is None or ttft_ms is None:
        return None
    interval_secs = (total_elapsed_ms - ttft_ms) / 1000.0
    if interval_secs <= 0.0:
        return None
    return completion_tokens / max(interval_secs, DECODE_ONLY_TOK_S_EPSILON_SECS)


def compute_decode_tok_s(completion_tokens: int | None, total_elapsed_ms: float | None) -> float | None:
    """Preserves the historical `decode_tok_s = completion_tokens /
    total_request_elapsed` definition unchanged."""
    if completion_tokens is None or total_elapsed_ms is None or total_elapsed_ms <= 0.0:
        return None
    return completion_tokens / (total_elapsed_ms / 1000.0)


def sse_data_payload(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def first_choice_delta_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) else None


def terminal_usage_completion_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("completion_tokens")
    return tokens if isinstance(tokens, int) else None


@dataclass
class StreamParseResult:
    completion_tokens: int | None
    ttft_ms: float | None
    malformed: bool = False


def parse_sse_stream(
    lines: Iterable[str],
    *,
    clock: Callable[[], float],
    started_at: float,
) -> StreamParseResult:
    """Parses an SSE chat-completion stream line-by-line, mirroring
    `streaming::parse_streaming_chat_response` in
    `crates/mesh-llm-commands/src/gpus/tune/benchmark/streaming.rs`:
    malformed individual chunks are skipped (not fatal), `[DONE]` or EOF
    ends the stream, TTFT is measured at the first non-empty content delta,
    and `completion_tokens` comes from the terminal `usage` object. A
    stream that never produces terminal usage returns `completion_tokens =
    None` (never zero) with `malformed = True`."""
    ttft_ms: float | None = None
    completion_tokens: int | None = None
    for raw_line in lines:
        payload = sse_data_payload(raw_line)
        if payload is None:
            continue
        if payload == "[DONE]":
            break
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if ttft_ms is None:
            content = first_choice_delta_content(value)
            if content:
                ttft_ms = (clock() - started_at) * 1000.0
        tokens = terminal_usage_completion_tokens(value)
        if tokens is not None:
            completion_tokens = tokens
    return StreamParseResult(
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        malformed=completion_tokens is None,
    )


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_chat_request_body(prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def send_streaming_chat_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout_secs: float,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> StreamParseResult:
    body = json.dumps(build_chat_request_body(prompt, max_tokens)).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = clock()
    with urllib.request.urlopen(request, timeout=timeout_secs) as response:
        lines = (raw.decode("utf-8", errors="replace") for raw in response)
        return parse_sse_stream(lines, clock=clock, started_at=started_at)


def wait_for_readiness(
    base_url: str,
    timeout_secs: float,
    poll_interval_secs: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    deadline = clock() + timeout_secs
    while clock() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=poll_interval_secs) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        sleep(poll_interval_secs)
    return False


@dataclass
class TrialResult:
    scenario: str
    pair_index: int
    status: str
    completion_tokens: int | None
    elapsed_ms: float | None
    decode_tok_s: float | None
    ttft_ms: float | None
    decode_only_tok_s: float | None
    setup_ms: float | None
    readiness_ms: float | None
    shutdown_ms: float | None
    error: str | None = None


def prompt_for_entry(entry: TrialPlanEntry) -> str:
    """A short, deterministic prompt derived from the entry's seeded value
    so every pair has a reproducible, comparable prompt without a fixture
    corpus dependency."""
    return f"Respond with a short factual sentence. token={entry.prompt_seed:016x}"


def execute_trial(
    binary: Path,
    model: str,
    mode: str,
    entry: TrialPlanEntry,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    readiness_timeout_secs: float = DEFAULT_READINESS_TIMEOUT_SECS,
    readiness_poll_interval_secs: float = DEFAULT_READINESS_POLL_INTERVAL_SECS,
    request_timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
    shutdown_timeout_secs: float = DEFAULT_SHUTDOWN_TIMEOUT_SECS,
) -> TrialResult:
    """Real trial execution: one fresh `<binary> --local-model-only`
    process launch, one readiness wait, one warmup request (excluded from
    metrics), one measured streaming request, one shutdown -- the exact
    trial unit `TRIAL_UNIT_DEFINITION` describes. NEVER called by this
    module's own test suite; `run_trial_plan` takes an injectable
    `trial_executor` so tests exercise manifest assembly without spawning a
    real process (see the paired test file)."""
    port = reserve_local_port()
    console_port = reserve_local_port()
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env[TRIAL_GATE_ENV_NAME] = "1"
    env[TRIAL_ENV_NAME] = resolve_trial_env_value(mode)
    argv = [
        str(binary),
        "serve",
        "--local-model-only",
        "--model",
        model,
        "--port",
        str(port),
        "--console",
        str(console_port),
        "--headless",
    ]
    prompt = prompt_for_entry(entry)
    setup_started = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 - trusted local binary under test
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    setup_ms = (time.monotonic() - setup_started) * 1000.0
    try:
        readiness_started = time.monotonic()
        ready = wait_for_readiness(base_url, readiness_timeout_secs, readiness_poll_interval_secs)
        readiness_ms = (time.monotonic() - readiness_started) * 1000.0
        if not ready:
            return TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                status="failed",
                completion_tokens=None,
                elapsed_ms=None,
                decode_tok_s=None,
                ttft_ms=None,
                decode_only_tok_s=None,
                setup_ms=setup_ms,
                readiness_ms=readiness_ms,
                shutdown_ms=None,
                error="readiness timeout",
            )
        with contextlib.suppress(urllib.error.URLError, OSError, TimeoutError):
            send_streaming_chat_request(base_url, prompt, max_tokens, request_timeout_secs)
        measured_started = time.monotonic()
        try:
            parsed = send_streaming_chat_request(base_url, prompt, max_tokens, request_timeout_secs)
            elapsed_ms = (time.monotonic() - measured_started) * 1000.0
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                status="failed",
                completion_tokens=None,
                elapsed_ms=None,
                decode_tok_s=None,
                ttft_ms=None,
                decode_only_tok_s=None,
                setup_ms=setup_ms,
                readiness_ms=readiness_ms,
                shutdown_ms=None,
                error=str(exc),
            )
    finally:
        shutdown_started = time.monotonic()
        process.terminate()
        try:
            process.wait(timeout=shutdown_timeout_secs)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=shutdown_timeout_secs)
        shutdown_ms = (time.monotonic() - shutdown_started) * 1000.0

    if parsed.malformed or parsed.completion_tokens is None:
        return TrialResult(
            scenario=entry.scenario,
            pair_index=entry.pair_index,
            status="failed",
            completion_tokens=None,
            elapsed_ms=elapsed_ms,
            decode_tok_s=None,
            ttft_ms=None,
            decode_only_tok_s=None,
            setup_ms=setup_ms,
            readiness_ms=readiness_ms,
            shutdown_ms=shutdown_ms,
            error="stream ended without terminal usage",
        )
    return TrialResult(
        scenario=entry.scenario,
        pair_index=entry.pair_index,
        status="succeeded",
        completion_tokens=parsed.completion_tokens,
        elapsed_ms=elapsed_ms,
        decode_tok_s=compute_decode_tok_s(parsed.completion_tokens, elapsed_ms),
        ttft_ms=parsed.ttft_ms,
        decode_only_tok_s=compute_decode_only_tok_s(parsed.completion_tokens, elapsed_ms, parsed.ttft_ms),
        setup_ms=setup_ms,
        readiness_ms=readiness_ms,
        shutdown_ms=shutdown_ms,
        error=None,
    )


TrialExecutor = Callable[[Path, str, str, TrialPlanEntry], TrialResult]


def run_trial_plan(
    binary: Path,
    model: str,
    mode: str,
    plan: Sequence[TrialPlanEntry],
    *,
    trial_executor: TrialExecutor = execute_trial,
) -> list[TrialResult]:
    return [trial_executor(binary, model, mode, entry) for entry in plan]


def summarize_health_expectations(mode: str, results: Sequence[TrialResult]) -> dict[str, int]:
    """The exact-count health expectation this manifest can prove without a
    live console API attached: under `event-disabled`, every attempted
    trial's underlying process runs with Progress/Diagnostic class
    submissions bypassed at the engine's single contract boundary, so the
    exact per-trial attempt count is the exact expected drop count for both
    classes. Under `production`, expected drops are zero -- a correctly
    sized reservation table coalesces progress and has ample diagnostic
    headroom for one benchmark trial's traffic."""
    attempted = len(results)
    if mode == "event-disabled":
        return {"expected_dropped_progress": attempted, "expected_dropped_diagnostic": attempted}
    return {"expected_dropped_progress": 0, "expected_dropped_diagnostic": 0}


def build_manifest(
    *,
    binary: Path,
    model: str,
    mode: str,
    seed: int,
    pairs_primary: int,
    pairs_scenario: int,
    scenarios: Sequence[str],
    results: Sequence[TrialResult],
    environ: Mapping[str, str],
    attempt: int = 1,
    generated_at: str,
    run_version: Callable[[Path], str | None] | None = None,
    thermal_state: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
    callback_ingress_p99_us: float | None = None,
    health: dict[str, int] | None = None,
) -> dict[str, Any]:
    expectations = summarize_health_expectations(mode, results)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metrics_schema": METRICS_SCHEMA,
        "mode": mode,
        "binary": capture_binary_identity(binary, run_version=run_version),
        "model": model,
        "seed": seed,
        "pairs_primary": pairs_primary,
        "pairs_scenario": pairs_scenario,
        "scenarios": list(scenarios),
        "attempt": attempt,
        "generated_at": generated_at,
        "host": host if host is not None else capture_host_classification(),
        "thermal_state": thermal_state if thermal_state is not None else capture_thermal_state(),
        "environment": capture_environment_snapshot(environ),
        "trial_unit": dict(TRIAL_UNIT_DEFINITION),
        "callback_ingress_p99_us": callback_ingress_p99_us,
        "health": health if health is not None else {},
        "expected_dropped_progress": expectations["expected_dropped_progress"],
        "expected_dropped_diagnostic": expectations["expected_dropped_diagnostic"],
        "trials": [asdict(result) for result in results],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-event-benchmark-matrix.py",
        description=(
            "Run one side (one binary, one trial mode) of the event-system "
            "paired benchmark matrix and write a deterministic manifest."
        ),
    )
    parser.add_argument("--binary", required=True, type=Path, help="Path to the mesh-llm binary under test.")
    parser.add_argument("--model", required=True, help="Approved deterministic local model reference.")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write the manifest and evidence into."
    )
    parser.add_argument(
        "--pairs-primary", required=True, type=int, help="Number of primary-comparison trial pairs (>=1)."
    )
    parser.add_argument(
        "--pairs-scenario", required=True, type=int, help="Number of trial pairs per named scenario (>=1)."
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Deterministic u64 seed; reuse the SAME seed across the production/event-disabled/baseline runs being compared.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=list(VALID_MODES),
        help="Hidden trial selector forwarded as MESH_LLM_EVENT_SYSTEM_TRIAL_MODE (also sets MESH_LLM_BENCHMARK_TUNE_TRIAL=1).",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        required=True,
        help="Named scenario; repeatable, at least one required.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_seed(args.seed)
        plan = build_trial_plan(args.seed, args.pairs_primary, args.pairs_scenario, args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    results = run_trial_plan(args.binary, args.model, args.mode, plan)
    manifest = build_manifest(
        binary=args.binary,
        model=args.model,
        mode=args.mode,
        seed=args.seed,
        pairs_primary=args.pairs_primary,
        pairs_scenario=args.pairs_scenario,
        scenarios=args.scenarios,
        results=results,
        environ=os.environ,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"manifest-{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_path": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
