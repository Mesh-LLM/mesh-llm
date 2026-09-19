#!/usr/bin/env python3
"""Verify an observed non-chat oracle result before battery certification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ORACLE_EXECUTABLE = {
    "embedding": "llama-server",
    "rerank": "llama-server",
    "encoder_decoder": "llama-completion",
    "ocr": "llama-server",
    "speech_synthesis": "llama-tts",
    "speech_recognition": "llama-server",
}


def sha256(path: Path) -> str:
    """Hash independently supplied artifact bytes for comparison with evidence."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(args: argparse.Namespace) -> None:
    """Reject missing prerequisites or evidence not bound to these exact inputs."""
    if args.model_class in {"ocr", "speech_synthesis", "speech_recognition"} and not args.projector_path:
        raise ValueError(f"{args.model_class} oracle evidence requires a projector path")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("oracle evidence must be an object")
    expected = {
        "status": "pass",
        "class": args.model_class,
        "smoke_lane": args.smoke_lane,
        "oracle_lane": args.oracle_lane,
        "model_id": args.model_id,
        "model_sha256": sha256(args.model_path),
        "projector_sha256": sha256(args.projector_path) if args.projector_path else None,
        "oracle_executable": ORACLE_EXECUTABLE[args.model_class],
        "oracle_executable_sha256": sha256(args.oracle_executable),
        "candidate_executable_sha256": sha256(args.candidate_executable),
        "pinned_patch_sha": args.pinned_patch_sha,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"oracle evidence {key} does not match this run")
    if args.oracle_executable.name != expected["oracle_executable"]:
        raise ValueError("wrong oracle executable for workload class")
    comparison = evidence.get("comparison")
    if not isinstance(comparison, str) or not comparison.startswith(
        f"{args.model_class} local-monolithic oracle passed: "
    ):
        raise ValueError("oracle evidence lacks an explicit comparator pass")
    if args.model_class == "speech_synthesis" and not isinstance(evidence.get("metrics"), dict):
        raise ValueError("TTS oracle evidence lacks PCM metrics")


def main() -> int:
    """Return success only when the recorded comparator pass matches this run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--class", dest="model_class", required=True, choices=ORACLE_EXECUTABLE)
    parser.add_argument("--smoke-lane", required=True)
    parser.add_argument("--oracle-lane", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--projector-path", type=Path)
    parser.add_argument("--candidate-executable", required=True, type=Path)
    parser.add_argument("--oracle-executable", required=True, type=Path)
    parser.add_argument("--pinned-patch-sha", required=True)
    args = parser.parse_args()
    try:
        verify(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"workload oracle evidence rejected: {error}", file=sys.stderr)
        return 1
    print(f"verified {args.model_class} local-monolithic oracle evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
