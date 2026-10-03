#!/usr/bin/env python3
"""Smoke the Decisions adapter through a running Mesh API and a real System One model.

Run: python3 scripts/skippy-decisions-smoke.py --base-url http://127.0.0.1:9337
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request


def get_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{url}: HTTP {error.code}: {error.read().decode(errors='replace')}") from error


def probability(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9337")
    parser.add_argument("--model", help="exact ID from /v1/models; defaults to first System One model")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    models = get_json(f"{base}/v1/models", timeout=args.timeout)["data"]
    capable = [
        model["id"]
        for model in models
        if model["id"] not in {"mesh", "auto"}
        and "system_one" in model.get("capabilities", [])
    ]
    model = args.model or next(iter(capable), None)
    if model not in capable:
        raise RuntimeError(f"model {model!r} does not advertise system_one in /v1/models")

    body = get_json(
        f"{base}/v1/decisions",
        {
            "model": model,
            "input": "I was charged twice. Please refund me today.",
            "questions": [
                {"type": "predicate", "name": "urgent", "instructions": "Does this need action today?"},
                {"type": "choice", "name": "team", "instructions": "Which team?", "choices": [
                    {"value": "billing", "description": "Payments and refunds"},
                    {"value": "support", "description": "Technical help"},
                ]},
                {"type": "score", "name": "frustration", "instructions": "How frustrated?", "levels": [
                    {"label": "0", "description": "Calm"},
                    {"label": "1", "description": "Frustrated"},
                ]},
            ],
        },
        timeout=args.timeout,
    )
    answers = body.get("answers", [])
    if body.get("model") != model or [(item.get("type"), item.get("name")) for item in answers] != [
        ("predicate", "urgent"), ("choice", "team"), ("score", "frustration")
    ]:
        raise RuntimeError(f"unexpected Decisions response shape: {body}")
    predicate, choice, score = answers
    if not probability(predicate.get("probability")):
        raise RuntimeError(f"invalid predicate probability: {predicate}")
    if choice.get("choice") not in {"billing", "support"} or not probability(choice.get("confidence")):
        raise RuntimeError(f"invalid choice answer: {choice}")
    if [item.get("value") for item in choice.get("probabilities", [])] != ["billing", "support"] or any(
        not probability(item.get("probability")) for item in choice["probabilities"]
    ):
        raise RuntimeError(f"invalid choice probabilities: {choice}")
    if not isinstance(score.get("score"), (int, float)) or not math.isfinite(score["score"]) or not probability(score.get("confidence")):
        raise RuntimeError(f"invalid score answer: {score}")
    if [(item.get("value"), item.get("label")) for item in score.get("probabilities", [])] != [(0, "0"), (1, "1")] or any(
        not probability(item.get("probability")) for item in score["probabilities"]
    ):
        raise RuntimeError(f"invalid score probabilities: {score}")
    usage = body.get("usage", {})
    if any(not isinstance(usage.get(key), int) or usage[key] < 0 for key in ("input_tokens", "output_tokens", "total_tokens")):
        raise RuntimeError(f"invalid usage: {usage}")
    print(f"Decisions live smoke passed: model={model}, questions=predicate,choice,score")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"Decisions live smoke failed: {error}", file=sys.stderr)
        sys.exit(1)
