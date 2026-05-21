#!/usr/bin/env python3
"""Probe mesh-llm's OpenAI tool-call surface for agent reliability."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, NamedTuple
import urllib.error
import urllib.request


TOOL_NAME = "lookup_fixture_fact"
FIXTURE_FACTS = {
    "codeword": "signal-7429",
    "checksum": "FS-319-DELTA",
}
DEFAULT_MODELS = "auto,mesh"
DEFAULT_BASE_URL = "http://127.0.0.1:9337/v1"
DEFAULT_OUTPUT = "target/agent-tool-call-reliability/results.jsonl"


class Probe(NamedTuple):
    model: str
    attempt: int


class ToolCall(NamedTuple):
    call_id: str
    key: str


class ProbeResult(NamedTuple):
    model: str
    attempt: int
    phase: str
    ok: bool
    detail: str
    elapsed_ms: int
    status_code: int | None = None


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


def build_plan(models: Iterable[str], attempts: int) -> list[Probe]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    return [
        Probe(model=model, attempt=attempt)
        for model in models
        for attempt in range(1, attempts + 1)
    ]


def render_plan(plan: Iterable[Probe], base_url: str) -> str:
    payload = {
        "name": "agent-tool-call-reliability",
        "endpoint": normalize_v1_base(base_url),
        "checks": [
            {
                "model": probe.model,
                "attempt": probe.attempt,
                "phases": [
                    "tool_call",
                    "tool_result",
                    "stream_tool_call",
                    "stream_tool_result",
                ],
            }
            for probe in plan
        ],
        "evidence": ["results.jsonl"],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Return one deterministic fact from the agent reliability fixture.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "enum": sorted(FIXTURE_FACTS),
                        },
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def initial_messages(attempt: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a strict OpenAI tool-call compatibility probe.",
        },
        {
            "role": "user",
            "content": (
                f"Attempt {attempt}: call the {TOOL_NAME} tool with key=codeword. "
                "Do not answer directly before the tool call."
            ),
        },
    ]


def build_tool_probe_request(model: str, attempt: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": initial_messages(attempt),
        "tools": tool_schema(),
        "tool_choice": {
            "type": "function",
            "function": {"name": TOOL_NAME},
        },
        "parallel_tool_calls": False,
        "stream": False,
        "max_tokens": 96,
        "temperature": 0,
    }


def build_stream_tool_probe_request(model: str, attempt: int) -> dict[str, Any]:
    request = build_tool_probe_request(model, attempt)
    request["stream"] = True
    return request


def extract_tool_call(response: dict[str, Any]) -> ToolCall:
    _require_tool_call_finish(response)
    message = _first_message(response)
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("response did not contain tool_calls")
    call = calls[0]
    if not isinstance(call, dict):
        raise ValueError("tool call is not an object")
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool call is missing an id")
    if not isinstance(function, dict):
        raise ValueError("tool call is missing a function object")
    if function.get("name") != TOOL_NAME:
        raise ValueError(f"unexpected tool name: {function.get('name')!r}")
    arguments = _decode_tool_arguments(function.get("arguments"))
    key = arguments.get("key")
    if key not in FIXTURE_FACTS:
        raise ValueError(f"unsupported fixture key: {key!r}")
    return ToolCall(call_id=call_id, key=key)


def extract_stream_tool_call(chunks: Iterable[dict[str, Any]]) -> ToolCall:
    parts: dict[int, dict[str, Any]] = {}
    saw_tool_finish = False
    for chunk in chunks:
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if choice.get("finish_reason") == "tool_calls":
                saw_tool_finish = True
            if not isinstance(delta, dict):
                continue
            for call_delta in delta.get("tool_calls") or []:
                if not isinstance(call_delta, dict):
                    continue
                index = call_delta.get("index", 0)
                if not isinstance(index, int):
                    raise ValueError(f"tool-call delta index is not an integer: {index!r}")
                entry = parts.setdefault(index, {"id": "", "name": "", "arguments": []})
                if isinstance(call_delta.get("id"), str):
                    entry["id"] = call_delta["id"]
                function = call_delta.get("function")
                if isinstance(function, dict):
                    if isinstance(function.get("name"), str):
                        entry["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        entry["arguments"].append(function["arguments"])
    if not parts:
        raise ValueError("stream did not contain tool_call deltas")
    if not saw_tool_finish:
        raise ValueError("stream did not finish with tool_calls")
    first = parts[min(parts)]
    return _tool_call_from_parts(
        call_id=first["id"],
        name=first["name"],
        arguments="".join(first["arguments"]),
    )


def build_tool_result_request(
    model: str,
    attempt: int,
    assistant_message: dict[str, Any],
    call: ToolCall,
) -> dict[str, Any]:
    expected = FIXTURE_FACTS[call.key]
    messages = initial_messages(attempt)
    messages.append(_sanitize_assistant_tool_message(assistant_message))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.call_id,
            "name": TOOL_NAME,
            "content": json.dumps(
                {"key": call.key, "value": expected},
                separators=(",", ":"),
            ),
        }
    )
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 64,
        "temperature": 0,
    }


def build_stream_tool_result_request(
    model: str,
    attempt: int,
    assistant_message: dict[str, Any],
    call: ToolCall,
) -> dict[str, Any]:
    request = build_tool_result_request(model, attempt, assistant_message, call)
    request["stream"] = True
    return request


def validate_final_answer(response: dict[str, Any], expected: str) -> None:
    message = _first_message(response)
    if message.get("tool_calls"):
        raise ValueError("continuation returned another tool call")
    content = message.get("content")
    validate_final_content(content, expected)


def validate_final_content(content: Any, expected: str) -> None:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("continuation returned empty content")
    if expected not in content:
        raise ValueError(f"missing expected fixture value: {expected}")


def extract_stream_content(chunks: Iterable[dict[str, Any]]) -> str:
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


def write_jsonl(path: Path, results: Iterable[ProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result._asdict(), sort_keys=True) + "\n")


def run_probe(
    base_url: str,
    probe: Probe,
    timeout: float,
    include_streaming: bool = True,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    results.extend(run_non_stream_probe(base_url, probe, timeout))
    if include_streaming:
        results.extend(run_stream_probe(base_url, probe, timeout))
    return results


def run_non_stream_probe(base_url: str, probe: Probe, timeout: float) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    tool_started = time.monotonic()
    try:
        request = build_tool_probe_request(probe.model, probe.attempt)
        response, status_code = post_json(base_url, "/chat/completions", request, timeout)
        call = extract_tool_call(response)
        results.append(
            _result(
                probe,
                "tool_call",
                True,
                f"{TOOL_NAME} key={call.key}",
                tool_started,
                status_code,
            )
        )
    except Exception as exc:
        results.append(_result(probe, "tool_call", False, str(exc), tool_started))
        return results

    final_started = time.monotonic()
    try:
        assistant_message = _first_message(response)
        request = build_tool_result_request(probe.model, probe.attempt, assistant_message, call)
        response, status_code = post_json(base_url, "/chat/completions", request, timeout)
        validate_final_answer(response, FIXTURE_FACTS[call.key])
        results.append(
            _result(
                probe,
                "tool_result",
                True,
                f"final answer included {FIXTURE_FACTS[call.key]}",
                final_started,
                status_code,
            )
        )
    except Exception as exc:
        results.append(_result(probe, "tool_result", False, str(exc), final_started))
    return results


def run_stream_probe(base_url: str, probe: Probe, timeout: float) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    tool_started = time.monotonic()
    try:
        request = build_stream_tool_probe_request(probe.model, probe.attempt)
        chunks, status_code = post_json_stream(base_url, "/chat/completions", request, timeout)
        call = extract_stream_tool_call(chunks)
        results.append(
            _result(
                probe,
                "stream_tool_call",
                True,
                f"{TOOL_NAME} key={call.key}",
                tool_started,
                status_code,
            )
        )
    except Exception as exc:
        results.append(_result(probe, "stream_tool_call", False, str(exc), tool_started))
        return results

    final_started = time.monotonic()
    try:
        assistant_message = assistant_message_from_tool_call(call)
        request = build_stream_tool_result_request(probe.model, probe.attempt, assistant_message, call)
        chunks, status_code = post_json_stream(base_url, "/chat/completions", request, timeout)
        content = extract_stream_content(chunks)
        validate_final_content(content, FIXTURE_FACTS[call.key])
        results.append(
            _result(
                probe,
                "stream_tool_result",
                True,
                f"final answer included {FIXTURE_FACTS[call.key]}",
                final_started,
                status_code,
            )
        )
    except Exception as exc:
        results.append(_result(probe, "stream_tool_result", False, str(exc), final_started))
    return results


def assistant_message_from_tool_call(call: ToolCall) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "arguments": json.dumps({"key": call.key}, separators=(",", ":")),
                },
            }
        ],
    }


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
            "authorization": "Bearer mesh-llm-agent-probe",
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


def post_json_stream(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[list[dict[str, Any]], int]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{normalize_v1_base(base_url)}{path}",
        data=data,
        headers={
            "content-type": "application/json",
            "authorization": "Bearer mesh-llm-agent-probe",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            chunks = list(
                parse_sse_lines(line.decode("utf-8", errors="replace") for line in response)
            )
            status_code = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    if not chunks:
        raise RuntimeError("stream returned no events")
    return chunks, status_code


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


def default_base_url() -> str:
    env_base = (
        os.environ.get("MESH_AGENT_TOOL_BASE_URL")
        or os.environ.get("MESH_AGENT_BASE_URL")
        or os.environ.get("MESH_OPENCODE_BASE_URL")
    )
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
    plan = build_plan(models, args.attempts)
    if args.print_plan:
        print(render_plan(plan, base_url))
        return 0

    results: list[ProbeResult] = []
    for probe in plan:
        results.extend(
            run_probe(
                base_url,
                probe,
                args.timeout,
                include_streaming=not args.skip_streaming,
            )
        )
    write_jsonl(Path(args.output), results)
    print_summary(results, Path(args.output))
    return 0 if all(result.ok for result in results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe OpenAI chat tool-call and tool-result continuation reliability.",
    )
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument(
        "--models",
        default=os.environ.get("MESH_AGENT_TOOL_MODELS", DEFAULT_MODELS),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=int(os.environ.get("MESH_AGENT_TOOL_ATTEMPTS", "1")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("MESH_AGENT_TOOL_TIMEOUT", "120")),
    )
    parser.add_argument("--output", default=os.environ.get("MESH_AGENT_TOOL_OUTPUT", DEFAULT_OUTPUT))
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args()


def print_summary(results: Iterable[ProbeResult], output: Path) -> None:
    rows = list(results)
    passed = sum(1 for row in rows if row.ok)
    print(f"agent tool-call reliability: {passed}/{len(rows)} phases passed")
    print(f"results: {output}")
    for row in rows:
        status = "PASS" if row.ok else "FAIL"
        print(f"{status} model={row.model} attempt={row.attempt} phase={row.phase}: {row.detail}")


def _first_message(response: dict[str, Any]) -> dict[str, Any]:
    first = _first_choice(response)
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("first choice did not contain a message object")
    return message


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response had no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("first choice is not an object")
    return first


def _require_tool_call_finish(response: dict[str, Any]) -> None:
    finish_reason = _first_choice(response).get("finish_reason")
    if finish_reason != "tool_calls":
        raise ValueError(f"tool-call turn finish_reason was not tool_calls: {finish_reason!r}")


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


def _tool_call_from_parts(call_id: str, name: str, arguments: str) -> ToolCall:
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool call is missing an id")
    if name != TOOL_NAME:
        raise ValueError(f"unexpected tool name: {name!r}")
    decoded = _decode_tool_arguments(arguments)
    key = decoded.get("key")
    if key not in FIXTURE_FACTS:
        raise ValueError(f"unsupported fixture key: {key!r}")
    return ToolCall(call_id=call_id, key=key)


def _sanitize_assistant_tool_message(message: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
    }
    return {key: value for key, value in sanitized.items() if value is not None}


def _result(
    probe: Probe,
    phase: str,
    ok: bool,
    detail: str,
    started: float,
    status_code: int | None = None,
) -> ProbeResult:
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return ProbeResult(
        model=probe.model,
        attempt=probe.attempt,
        phase=phase,
        ok=ok,
        detail=detail,
        elapsed_ms=elapsed_ms,
        status_code=status_code,
    )


if __name__ == "__main__":
    raise SystemExit(main())
