#!/usr/bin/env python3
"""Select bounded llama-canary repair attempts without weakening certification."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def emit(**values: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")


def outputs(job: dict[str, Any]) -> dict[str, str]:
    value = job.get("outputs", {})
    return value if isinstance(value, dict) else {}


def resumable(source: dict[str, Any]) -> dict[str, str] | None:
    value = outputs(source)
    if value.get("repairable") != "true":
        return None
    required = ("package", "identity", "head", "feedback")
    if any(not value.get(key) for key in required):
        return None
    return {
        "state": "repairable",
        "repairable": "true",
        "resume_package": value["package"],
        "resume_identity": value["identity"],
        "resume_head": value["head"],
        "resume_feedback": value["feedback"],
        "failure_class": value.get("failure_class", "candidate"),
        "failure_stage": value.get("failure_stage", "family-certification"),
    }


def failed(source: dict[str, Any], stage: str) -> dict[str, str]:
    value = outputs(source)
    return {
        "state": "failed",
        "repairable": "false",
        "failure_class": value.get("failure_class", "infrastructure"),
        "failure_stage": value.get("failure_stage", stage),
    }


def green(source: dict[str, Any]) -> bool:
    return outputs(source).get("green") == "true"


def successful(source: dict[str, Any]) -> dict[str, str]:
    value = outputs(source)
    required = ("package", "identity", "head", "branch")
    if any(not value.get(key) for key in required):
        raise ValueError("green pass is missing immutable candidate outputs")
    return {
        "state": "green",
        "green": "true",
        "repairable": "false",
        **{key: value[key] for key in required},
    }


def select_attempt(changed: bool, repair: dict[str, Any], verification: dict[str, Any]) -> dict[str, str]:
    if not green(repair):
        resume = resumable(repair) if changed else None
        return resume or failed(repair, "candidate-build-or-family-certification")
    if not changed:
        return successful(repair)
    if green(verification):
        result = successful(verification)
        if outputs(repair).get("head") != result["head"]:
            return {
                "state": "failed",
                "repairable": "false",
                "failure_class": "candidate",
                "failure_stage": "independent-verification-identity",
            }
        return result
    return resumable(verification) or failed(verification, "independent-verification")


def select_final(certify: bool, changed: bool, mesh_source: str, preflight: str,
                 attempts: dict[str, Any]) -> dict[str, str]:
    if not certify:
        return {"state": "noop", "publish": "false"}
    if preflight != "success":
        raise ValueError("canary environment preflight failed; candidate source was not evaluated")
    available = [outputs(attempt) for _, attempt in sorted(attempts.items())
                 if outputs(attempt).get("state")]
    if not available:
        raise ValueError("no distributed canary attempt completed")
    selected = available[-1]
    if selected["state"] != "green":
        failure_class = selected.get("failure_class", "candidate")
        failure_stage = selected.get("failure_stage", "distributed-repair")
        raise ValueError(f"{failure_class} failure during {failure_stage}; publication denied")
    if mesh_source:
        if changed or selected.get("head") != mesh_source:
            raise ValueError("selected MeshLLM revision certification failed")
        return {"state": "green", "publish": "false"}
    if not changed:
        return {"state": "green", "publish": "false"}
    return {
        "state": "green",
        "publish": "true",
        **{key: selected[key] for key in ("package", "identity", "head", "branch")},
    }


def boolean(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"invalid boolean: {value}")
    return value == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    attempt = subparsers.add_parser("attempt")
    attempt.add_argument("--changed", required=True)
    attempt.add_argument("--repair-json", required=True)
    attempt.add_argument("--verification-json", required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--certify", required=True)
    final.add_argument("--changed", required=True)
    final.add_argument("--mesh-source", default="")
    final.add_argument("--preflight", required=True)
    final.add_argument("--attempts-json", required=True)
    args = parser.parse_args()
    if args.command == "attempt":
        result = select_attempt(boolean(args.changed), json.loads(args.repair_json),
                                json.loads(args.verification_json))
    else:
        result = select_final(boolean(args.certify), boolean(args.changed), args.mesh_source,
                              args.preflight, json.loads(args.attempts_json))
    emit(**result)


if __name__ == "__main__":
    main()
