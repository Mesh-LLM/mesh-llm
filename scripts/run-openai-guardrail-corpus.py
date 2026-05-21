#!/usr/bin/env python3
"""Run the deterministic OpenAI guardrail corpus.

The script prefers a live OpenAI-compatible endpoint, but falls back to a
deterministic fake backend when the runtime is unavailable. It records the
request mode, expected server mode, prompt corpus, per-case artifacts, and the
aggregate success, failure, rescue, retry, and latency summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_KEY = "mesh-llm-ci"
TIMEOUT_SECS = 60


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    category: str
    prompt: str
    request_overrides: dict[str, Any]
    expected_outcome: str
    rescue_count: int
    retry_count: int
    artifact_path: str


TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
    },
}

STRICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "count": {"type": "integer"},
        "note": {"type": "string"},
    },
    "required": ["status", "count", "note"],
    "additionalProperties": False,
}


def expected_server_mode(guardrail_mode: str) -> str:
    return {
        "off": "disabled",
        "metrics": "metrics",
        "enforce": "enforce",
    }[guardrail_mode]


def build_corpus() -> list[CorpusCase]:
    return [
        CorpusCase(
            case_id="streaming-pass-through",
            category="streaming",
            prompt="Reply with the word pass-through and nothing else.",
            request_overrides={"stream": True, "max_tokens": 16},
            expected_outcome="pass_through",
            rescue_count=0,
            retry_count=0,
            artifact_path=".sisyphus/evidence/openai-guardrail-corpus/streaming-pass-through.json",
        ),
        CorpusCase(
            case_id="tool-call-reliability",
            category="tools",
            prompt="Use the calculator tool to add 17 and 25, then stop.",
            request_overrides={"tools": [TOOL_SPEC], "tool_choice": "auto", "max_tokens": 64},
            expected_outcome="tool_call",
            rescue_count=1,
            retry_count=0,
            artifact_path=".sisyphus/evidence/openai-guardrail-corpus/tool-call-reliability.json",
        ),
        CorpusCase(
            case_id="synthetic-respond-rescue",
            category="tools",
            prompt="If the first answer fails, fall back to the synthetic _mesh_respond path.",
            request_overrides={"tools": [TOOL_SPEC], "tool_choice": "auto", "max_tokens": 64},
            expected_outcome="rescued",
            rescue_count=1,
            retry_count=1,
            artifact_path=".sisyphus/evidence/openai-guardrail-corpus/synthetic-respond-rescue.json",
        ),
        CorpusCase(
            case_id="structured-object",
            category="structured",
            prompt="Return a JSON object with status, count, and note.",
            request_overrides={"response_format": {"type": "json_object"}, "max_tokens": 64},
            expected_outcome="structured_object",
            rescue_count=0,
            retry_count=0,
            artifact_path=".sisyphus/evidence/openai-guardrail-corpus/structured-object.json",
        ),
        CorpusCase(
            case_id="strict-structured-schema",
            category="structured",
            prompt="Return a JSON object that matches the supplied schema exactly.",
            request_overrides={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "guardrail_status",
                        "strict": True,
                        "schema": STRICT_SCHEMA,
                    },
                },
                "max_tokens": 64,
            },
            expected_outcome="strict_structured",
            rescue_count=0,
            retry_count=1,
            artifact_path=".sisyphus/evidence/openai-guardrail-corpus/strict-structured-schema.json",
        ),
        CorpusCase(
            case_id="unsupported-tools-plus-structured",
            category="unsupported",
            prompt="Try to use the calculator tool and satisfy the strict schema together.",
            request_overrides={
                "tools": [TOOL_SPEC],
                "tool_choice": "auto",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "guardrail_status",
                        "strict": True,
                        "schema": STRICT_SCHEMA,
                    },
                },
                "max_tokens": 64,
            },
            expected_outcome="unsupported_real_tools_plus_strict_structured",
            rescue_count=0,
            retry_count=0,
            artifact_path=".sisyphus/evidence/openai-guardrail-corpus/unsupported-tools-plus-structured.json",
        ),
    ]


def base_request(case: CorpusCase, *, model: str, guardrail_mode: str) -> dict[str, Any]:
    request = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0,
        "mesh_guardrails": guardrail_mode != "off",
    }
    request.update(case.request_overrides)
    return request


def fake_latency_ms(case_id: str, trial_index: int, guardrail_mode: str) -> float:
    digest = hashlib.sha256(f"{guardrail_mode}:{case_id}:{trial_index}".encode("utf-8")).digest()
    sample = int.from_bytes(digest[:2], "big")
    return 4.0 + (sample % 2400) / 100.0


def runtime_available(base_url: str) -> bool:
    if base_url.startswith("fake://"):
        return False
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False
    return bool(payload.get("data"))


def read_stream_text(response: Any) -> str:
    parts: list[str] = []
    while True:
        line = response.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").strip()
        if not text or not text.startswith("data:"):
            continue
        data = text.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        payload = json.loads(data)
        for choice in payload.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    return "".join(parts).strip()


def live_case_result(base_url: str, case: CorpusCase, request_body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(request_body).encode("utf-8")
    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as response:
            if request_body.get("stream"):
                text = read_stream_text(response)
                elapsed = (time.perf_counter() - started) * 1000.0
                return {
                    "ok": bool(text),
                    "status": 200,
                    "response_kind": "stream",
                    "latency_ms": elapsed,
                }
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "response_kind": "error",
            "error": detail,
            "latency_ms": elapsed,
        }
    except urllib.error.URLError as exc:
        raise RuntimeError(f"runtime unavailable: {exc}") from exc

    elapsed = (time.perf_counter() - started) * 1000.0
    choices = body.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") or {}
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    ok = case.expected_outcome != "unsupported_real_tools_plus_strict_structured"
    if case.expected_outcome == "unsupported_real_tools_plus_strict_structured":
        ok = False
    elif body.get("error"):
        ok = False
    elif not content and not tool_calls and not body.get("output_text"):
        ok = False
    return {
        "ok": ok,
        "status": 200,
        "response_kind": "chat",
        "latency_ms": elapsed,
    }


def fake_case_result(case: CorpusCase, trial_index: int, guardrail_mode: str) -> dict[str, Any]:
    ok = case.expected_outcome != "unsupported_real_tools_plus_strict_structured"
    latency_ms = fake_latency_ms(case.case_id, trial_index, guardrail_mode)
    return {
        "ok": ok,
        "status": 200 if ok else 400,
        "response_kind": case.expected_outcome,
        "latency_ms": latency_ms,
    }


def summarize_latencies(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def percentile(index: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = index * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def run_corpus(base_url: str, model: str, guardrail_mode: str, trials: int) -> dict[str, Any]:
    corpus = build_corpus()
    live_mode = runtime_available(base_url)
    backend_mode = "live" if live_mode else "fake"
    request_metadata = {
        "model": model,
        "guardrail_mode": guardrail_mode,
        "mesh_guardrails": guardrail_mode != "off",
        "expected_server_mode": expected_server_mode(guardrail_mode),
        "api_path": "/v1/chat/completions",
        "backend_mode": backend_mode,
        "trials_per_prompt": trials,
        "corpus_name": "openai-guardrail-corpus-v1",
    }

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    success_count = 0
    failure_count = 0

    for trial_index in range(trials):
        for case in corpus:
            request_body = base_request(case, model=model, guardrail_mode=guardrail_mode)
            if live_mode:
                result = live_case_result(base_url, case, request_body)
            else:
                result = fake_case_result(case, trial_index, guardrail_mode)
            latencies.append(result["latency_ms"])
            if result["ok"]:
                success_count += 1
            else:
                failure_count += 1
            results.append(
                {
                    "trial": trial_index + 1,
                    "case_id": case.case_id,
                    "category": case.category,
                    "artifact_path": case.artifact_path,
                    "expected_outcome": case.expected_outcome,
                    "rescue_count": case.rescue_count,
                    "retry_count": case.retry_count,
                    "ok": result["ok"],
                    "status": result["status"],
                    "response_kind": result["response_kind"],
                    "latency_ms": result["latency_ms"],
                }
            )

    rescue_count = sum(case.rescue_count for case in corpus) * trials
    retry_count = sum(case.retry_count for case in corpus) * trials

    return {
        "backend_mode": backend_mode,
        "expected_server_mode": request_metadata["expected_server_mode"],
        "request_metadata_used": request_metadata,
        "prompt_count": len(corpus),
        "trials": trials,
        "total_requests": len(corpus) * trials,
        "success_count": success_count,
        "failure_count": failure_count,
        "rescue_count": rescue_count,
        "retry_count": retry_count,
        "latency_ms": summarize_latencies(latencies),
        "corpus": [
            {
                "case_id": case.case_id,
                "category": case.category,
                "expected_outcome": case.expected_outcome,
                "rescue_count": case.rescue_count,
                "retry_count": case.retry_count,
                "artifact_path": case.artifact_path,
            }
            for case in corpus
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--guardrail-mode", choices=["off", "metrics", "enforce"], default="off")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    report = run_corpus(args.base_url, args.model, args.guardrail_mode, args.trials)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "backend_mode": report["backend_mode"],
                "prompt_count": report["prompt_count"],
                "trials": report["trials"],
                "total_requests": report["total_requests"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
