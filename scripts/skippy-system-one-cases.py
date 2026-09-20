#!/usr/bin/env python3
"""Drive `POST /v1/systemone` against a running embedded OpenAI frontend.

This is the request matrix behind `scripts/skippy-system-one-smoke.sh`. It is
kept separate from the orchestration shell so the contract can be exercised
against a stub server in `scripts/tests/test_skippy_system_one_smoke.py`
without a patched native runtime.

Two modes, with different preconditions:

* ``--mode contract`` assumes the server has a *non-DiffusionGemma* model
  loaded. It asserts the request/response contract and every fail-closed
  boundary that does not depend on the model architecture. Those boundaries
  live in the Rust frontend, so this mode is backend independent.
* ``--mode full-read`` assumes the server has a complete, single-lane
  DiffusionGemma model loaded. It asserts that real reads return finite,
  normalized label distributions over valid answer ranges, and that repeated
  and interleaved reads do not leak diffusion state between requests.

Exit status: 0 when every case passed, 1 when a case failed, 2 for a usage or
environment error.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import urllib.error
import urllib.request
from typing import Any

SCHEMA_VERSION = 1

# Transport-level tolerance for `f32` probabilities that made a JSON round trip
# through `serde_json` and Python floats.
PROBABILITY_TOLERANCE = 1e-3
# Two reads of clearly different states must not agree this closely; a
# degenerate constant response is the failure this guards against.
DISTINCT_READ_MIN_DELTA = 1e-2

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CaseFailure(AssertionError):
    """A single named assertion did not hold."""


class TransportError(RuntimeError):
    """The request could not be completed at the transport level."""


def post_json(url: str, payload: Any, timeout: float) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json"}, method="POST"
    )
    return _send(request, timeout)


def post_raw(url: str, body: bytes, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json"}, method="POST"
    )
    return _send(request, timeout)


def get(url: str, timeout: float) -> tuple[int, str]:
    return _send(urllib.request.Request(url, method="GET"), timeout)


def _send(request: urllib.request.Request, timeout: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")
    except (urllib.error.URLError, socket.timeout, OSError) as error:
        raise TransportError(f"{request.get_method()} {request.full_url}: {error}") from error


def parse_body(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise CaseFailure(f"response body is not JSON: {error}: {text[:400]!r}") from error


def expect_status(name: str, status: int, expected: int, body: Any) -> None:
    if status != expected:
        raise CaseFailure(f"expected HTTP {expected}, got {status}: {json.dumps(body)[:400]}")


def expect_error(
    name: str, status: int, expected_status: int, body: Any, expected_type: str, expected_code: str
) -> None:
    expect_status(name, status, expected_status, body)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        raise CaseFailure(f"expected an OpenAI error envelope, got: {json.dumps(body)[:400]}")
    if error.get("type") != expected_type:
        raise CaseFailure(f"expected error type {expected_type!r}, got {error.get('type')!r}")
    if error.get("code") != expected_code:
        raise CaseFailure(f"expected error code {expected_code!r}, got {error.get('code')!r}")
    if not str(error.get("message", "")).strip():
        raise CaseFailure("error envelope carries no message")


def noul_question(instructions: str = "Is this a billing issue?") -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def choice_question(count: int, instructions: str = "Which team should handle it?") -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {f"team_{index}": f"responsibility {index}" for index in range(count)},
    }


def score_question(count: int, instructions: str = "How urgent is this?") -> dict[str, Any]:
    return {
        "type": "score",
        "instructions": instructions,
        "criteria": [f"level {index}" for index in range(count)],
    }


def system_one_request(
    model: str, questions: dict[str, Any], state: Any = "I was charged twice this month.", **extra: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    payload.update(extra)
    return payload


def transport_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


# --------------------------------------------------------------------------
# contract mode
# --------------------------------------------------------------------------


def run_contract(base_url: str, model: str, timeout: float) -> list[dict[str, Any]]:
    endpoint = transport_url(base_url, "/systemone")
    results: list[dict[str, Any]] = []

    valid_questions = {"billing": noul_question()}

    def reject(
        name: str,
        expected_status: int,
        expected_type: str,
        expected_code: str,
        payload: dict[str, Any],
    ) -> None:
        status, text = post_json(endpoint, payload, timeout)
        body = parse_body(text)
        expect_error(name, status, expected_status, body, expected_type, expected_code)

    reject(
        "unknown-model",
        400,
        "invalid_request_error",
        "invalid_value",
        system_one_request("definitely-not-loaded", valid_questions),
    )
    reject(
        "empty-questions",
        400,
        "invalid_request_error",
        "invalid_value",
        system_one_request(model, {}),
    )
    reject(
        "sequential-read",
        400,
        "invalid_request_error",
        "unsupported_model_feature",
        system_one_request(model, valid_questions, sequential=True),
    )
    reject(
        "multiple-steps",
        400,
        "invalid_request_error",
        "unsupported_model_feature",
        system_one_request(model, valid_questions, steps=2),
    )
    reject(
        "multiple-samples",
        400,
        "invalid_request_error",
        "unsupported_model_feature",
        system_one_request(model, valid_questions, samples=2),
    )
    reject(
        "thinking-requested",
        400,
        "invalid_request_error",
        "unsupported_model_feature",
        system_one_request(model, valid_questions, think=1),
    )
    reject(
        "image-input",
        400,
        "invalid_request_error",
        "unsupported_model_feature",
        system_one_request(
            model, valid_questions, images=[{"type": "image_url", "image_url": {"url": "data:,"}}]
        ),
    )
    reject(
        "choice-one-option",
        400,
        "invalid_request_error",
        "invalid_value",
        system_one_request(model, {"team": choice_question(1)}),
    )
    reject(
        "choice-too-many-options",
        400,
        "invalid_request_error",
        "invalid_value",
        system_one_request(model, {"team": choice_question(len(CHOICE_LABELS) + 1)}),
    )
    reject(
        "score-one-level",
        400,
        "invalid_request_error",
        "invalid_value",
        system_one_request(model, {"urgency": score_question(1)}),
    )
    reject(
        "score-too-many-levels",
        400,
        "invalid_request_error",
        "invalid_value",
        system_one_request(model, {"urgency": score_question(11)}),
    )
    # A one-byte body is a JSON rejection at the axum extractor, before any
    # request field is read.
    status, text = post_raw(endpoint, b"{", timeout)
    expect_error(
        "malformed-json", status, 400, parse_body(text), "invalid_request_error", "invalid_value"
    )
    status, text = get(endpoint, timeout)
    expect_error(
        "method-not-allowed",
        status,
        405,
        parse_body(text),
        "invalid_request_error",
        "method_not_allowed",
    )
    # The model is a complete non-DiffusionGemma fixture, so a well-formed read
    # must be refused by the native boundary instead of answered.
    status, text = post_json(endpoint, system_one_request(model, valid_questions), timeout)
    body = parse_body(text)
    expect_error(
        "wrong-architecture-refused",
        status,
        502,
        body,
        "server_error",
        "service_unavailable",
    )
    message = str(body.get("error", {}).get("message", ""))
    if not any(token in message for token in ("DiffusionGemma", "System One", "system-one")):
        raise CaseFailure(
            f"architecture refusal does not name the System One capability: {message!r}"
        )

    results.append({"name": "contract-matrix", "status": "pass", "cases": 14})
    return results


# --------------------------------------------------------------------------
# full-read mode
# --------------------------------------------------------------------------


def finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CaseFailure(f"{label} is not a number: {value!r}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise CaseFailure(f"{label} is not finite: {number!r}")
    return number


def check_unit_interval(value: Any, label: str) -> float:
    number = finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise CaseFailure(f"{label} is outside [0, 1]: {number!r}")
    return number


def normalize_probabilities(probabilities: Any, label: str) -> dict[str, float]:
    if not isinstance(probabilities, dict) or not probabilities:
        raise CaseFailure(f"{label} is not a non-empty object: {probabilities!r}")
    values = {key: check_unit_interval(value, f"{label}.{key}") for key, value in probabilities.items()}
    total = sum(values.values())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise CaseFailure(f"{label} does not sum to 1: {total!r} ({values!r})")
    return values


def argmax(values: dict[str, float]) -> str:
    return max(sorted(values), key=lambda key: values[key])


def read(
    endpoint: str, payload: dict[str, Any], timeout: float, expected_model: str | None = None
) -> dict[str, Any]:
    status, text = post_json(endpoint, payload, timeout)
    body = parse_body(text)
    expect_status("read", status, 200, body)
    if not isinstance(body, dict):
        raise CaseFailure(f"read response is not an object: {json.dumps(body)[:400]}")
    if expected_model is not None and body.get("model") != expected_model:
        raise CaseFailure(f"expected model {expected_model!r}, got {body.get('model')!r}")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise CaseFailure(f"read response has no model string: {json.dumps(body)[:200]}")
    usage = body.get("usage")
    if not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens"}:
        raise CaseFailure(f"read response has an unexpected usage object: {usage!r}")
    if not isinstance(usage["input_tokens"], int) or usage["input_tokens"] <= 0:
        raise CaseFailure(f"usage.input_tokens must be a positive integer: {usage['input_tokens']!r}")
    if usage["output_tokens"] != 0:
        raise CaseFailure(f"a read must not report generated tokens: {usage['output_tokens']!r}")
    answers = body.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise CaseFailure(f"read response has no answers: {json.dumps(body)[:200]}")
    return body


def check_noul(answer: Any, label: str) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise CaseFailure(f"{label} is not a noul answer: {answer!r}")
    check_unit_interval(answer.get("noul"), f"{label}.noul")
    return answer


def check_choice(answer: Any, label: str, expected_choices: list[str]) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise CaseFailure(f"{label} is not a choice answer: {answer!r}")
    choice = answer.get("choice")
    if choice not in expected_choices:
        raise CaseFailure(f"{label}.choice {choice!r} is not one of {expected_choices}")
    probabilities = normalize_probabilities(answer.get("probabilities"), f"{label}.probabilities")
    if sorted(probabilities) != sorted(expected_choices):
        raise CaseFailure(
            f"{label}.probabilities keys {sorted(probabilities)} do not match {sorted(expected_choices)}"
        )
    check_unit_interval(answer.get("confidence"), f"{label}.confidence")
    if argmax(probabilities) != choice:
        raise CaseFailure(
            f"{label}.choice {choice!r} is not the most probable option ({argmax(probabilities)!r})"
        )
    return answer


def check_score(answer: Any, label: str, levels: int) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "score":
        raise CaseFailure(f"{label} is not a score answer: {answer!r}")
    score = finite(answer.get("score"), f"{label}.score")
    if not 0.0 <= score <= levels - 1:
        raise CaseFailure(f"{label}.score {score!r} is outside [0, {levels - 1}]")
    probabilities = normalize_probabilities(answer.get("probabilities"), f"{label}.probabilities")
    expected_keys = [str(index) for index in range(levels)]
    if sorted(probabilities, key=int) != expected_keys:
        raise CaseFailure(
            f"{label}.probabilities keys {sorted(probabilities)} do not match {expected_keys}"
        )
    legend = answer.get("legend")
    if not isinstance(legend, dict) or sorted(legend, key=int) != expected_keys:
        raise CaseFailure(f"{label}.legend keys do not match {expected_keys}: {legend!r}")
    check_unit_interval(answer.get("confidence"), f"{label}.confidence")
    # The reported score is the expectation over the label distribution.
    expected_score = sum(index * probabilities[str(index)] for index in range(levels))
    if abs(score - expected_score) > PROBABILITY_TOLERANCE:
        raise CaseFailure(
            f"{label}.score {score!r} is not the expectation of its distribution ({expected_score!r})"
        )
    return answer


def answer_signature(response: dict[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for key, answer in response["answers"].items():
        if not isinstance(answer, dict):
            signature[key] = answer
            continue
        entry: dict[str, Any] = {"type": answer.get("type")}
        for field in ("noul", "choice", "score", "confidence"):
            if field in answer:
                entry[field] = answer[field]
        if isinstance(answer.get("probabilities"), dict):
            entry["probabilities"] = {k: float(v) for k, v in answer["probabilities"].items()}
        signature[key] = entry
    return signature


def signatures_differ(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if sorted(left) != sorted(right):
        return True
    for key in left:
        first, second = left[key], right[key]
        if first.get("type") != second.get("type"):
            return True
        for field in ("choice",):
            if first.get(field) != second.get(field):
                return True
        for field in ("noul", "score"):
            if field in first and abs(float(first[field]) - float(second[field])) > DISTINCT_READ_MIN_DELTA:
                return True
        left_probabilities = first.get("probabilities", {})
        right_probabilities = second.get("probabilities", {})
        if sorted(left_probabilities) != sorted(right_probabilities):
            return True
        for label, value in left_probabilities.items():
            if abs(value - right_probabilities[label]) > DISTINCT_READ_MIN_DELTA:
                return True
    return False


def check_repeated_read_equal(first: dict[str, Any], second: dict[str, Any], label: str) -> None:
    left, right = answer_signature(first), answer_signature(second)
    if sorted(left) != sorted(right):
        raise CaseFailure(f"{label}: repeated read answered a different question set")
    for key in left:
        left_answer, right_answer = left[key], right[key]
        if left_answer.get("type") != right_answer.get("type"):
            raise CaseFailure(f"{label}: {key} changed answer type between identical reads")
        for field in ("choice",):
            if left_answer.get(field) != right_answer.get(field):
                raise CaseFailure(
                    f"{label}: {key}.{field} changed between identical reads "
                    f"({left_answer.get(field)!r} -> {right_answer.get(field)!r})"
                )
        for field in ("noul", "score"):
            if field in left_answer:
                if abs(float(left_answer[field]) - float(right_answer[field])) > PROBABILITY_TOLERANCE:
                    raise CaseFailure(
                        f"{label}: {key}.{field} changed between identical reads "
                        f"({left_answer[field]!r} -> {right_answer[field]!r})"
                    )
        left_probabilities = left_answer.get("probabilities", {})
        right_probabilities = right_answer.get("probabilities", {})
        for name, value in left_probabilities.items():
            if abs(value - right_probabilities.get(name, float("nan"))) > PROBABILITY_TOLERANCE:
                raise CaseFailure(
                    f"{label}: {key}.probabilities[{name!r}] changed between identical reads "
                    f"({value!r} -> {right_probabilities.get(name)!r})"
                )


def run_full_read(base_url: str, model: str, alias: str, timeout: float) -> list[dict[str, Any]]:
    endpoint = transport_url(base_url, "/systemone")
    results: list[dict[str, Any]] = []

    # noul
    answered = read(endpoint, system_one_request(model, {"billing": noul_question()}), timeout)
    check_noul(answered["answers"].get("billing"), "noul.billing")
    results.append({"name": "noul-read", "status": "pass"})

    # choice
    choices = [f"team_{index}" for index in range(4)]
    answered = read(endpoint, system_one_request(model, {"team": choice_question(4)}), timeout)
    check_choice(answered["answers"].get("team"), "choice.team", choices)
    results.append({"name": "choice-read", "status": "pass"})

    # score
    answered = read(endpoint, system_one_request(model, {"urgency": score_question(4)}), timeout)
    check_score(answered["answers"].get("urgency"), "score.urgency", 4)
    results.append({"name": "score-read", "status": "pass"})

    # interleaved question types in one read
    mixed = system_one_request(
        model,
        {
            "billing": noul_question(),
            "team": choice_question(3),
            "urgency": score_question(5),
        },
        state="The invoice shows two charges and the second one is unexplained.",
    )
    answered = read(endpoint, mixed, timeout)
    check_noul(answered["answers"].get("billing"), "mixed.billing")
    check_choice(
        answered["answers"].get("team"), "mixed.team", [f"team_{index}" for index in range(3)]
    )
    check_score(answered["answers"].get("urgency"), "mixed.urgency", 5)
    results.append({"name": "mixed-read", "status": "pass"})

    # The documented alias is accepted, and the response echoes the requested
    # model string rather than substituting a canonical name.
    answered = read(
        endpoint,
        system_one_request(alias, {"billing": noul_question()}),
        timeout,
        expected_model=alias,
    )
    check_noul(answered["answers"].get("billing"), "alias.billing")
    results.append({"name": "alias-read", "status": "pass"})

    # Repeat/interleave: the same state must read the same twice, with an
    # unrelated read in between. Leaked diffusion state would perturb it.
    first_request = system_one_request(
        model,
        {"billing": noul_question(), "team": choice_question(3)},
        state="I was charged twice this month.",
    )
    other_request = system_one_request(
        model,
        {
            "billing": noul_question("Is this a refund request?"),
            "team": choice_question(3, "Which queue should receive it?"),
        },
        state="The deployment tool crashes with a segmentation fault on startup.",
    )
    first = read(endpoint, first_request, timeout)
    other = read(endpoint, other_request, timeout)
    repeated = read(endpoint, first_request, timeout)
    check_repeated_read_equal(first, repeated, "interleaved-read")
    if not signatures_differ(answer_signature(first), answer_signature(other)):
        raise CaseFailure(
            "two clearly different states produced the same read; the endpoint may be "
            "answering from cached or leaked diffusion state"
        )
    results.append({"name": "interleaved-read-determinism", "status": "pass"})

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:9337/v1")
    parser.add_argument("--model", required=True, help="loaded model id")
    parser.add_argument("--alias", default="openjev-latest", help="documented System One alias")
    parser.add_argument("--mode", choices=("contract", "full-read"), required=True)
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    parser.add_argument("--json-out", help="write the structured result to this path")
    arguments = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": arguments.mode,
        "model": arguments.model,
        "cases": [],
        "status": "pass",
        "failures": [],
    }
    status = 0
    try:
        if arguments.mode == "contract":
            report["cases"] = run_contract(arguments.base_url, arguments.model, arguments.timeout)
        else:
            report["cases"] = run_full_read(
                arguments.base_url, arguments.model, arguments.alias, arguments.timeout
            )
    except CaseFailure as failure:
        report["status"] = "fail"
        report["failures"].append(str(failure))
        status = 1
    except TransportError as error:
        report["status"] = "error"
        report["failures"].append(str(error))
        status = 2

    for case in report["cases"]:
        print(f"system-one {arguments.mode}: {case['name']} ok", file=sys.stderr)
    for failure in report["failures"]:
        print(f"system-one {arguments.mode}: FAILED: {failure}", file=sys.stderr)
    if arguments.json_out:
        with open(arguments.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
