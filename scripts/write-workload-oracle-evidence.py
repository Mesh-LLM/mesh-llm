#!/usr/bin/env python3
"""Persist an explicit, identity-bound workload oracle pass for the battery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    """Hash a model sidecar or executable without buffering the whole artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence(args: argparse.Namespace) -> None:
    """Bind an observed comparator pass to its lane and executable identities."""
    if not args.smoke_lane.endswith("-smoke"):
        raise ValueError(f"smoke lane must end with '-smoke': {args.smoke_lane}")
    lines = [
        line.strip()
        for line in args.comparison_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines or not lines[-1].startswith(
        f"{args.model_class} local-monolithic oracle passed: "
    ):
        raise ValueError("oracle comparator did not emit an explicit class-specific pass")
    evidence: dict[str, object] = {
        "status": "pass",
        "class": args.model_class,
        "smoke_lane": args.smoke_lane,
        "oracle_lane": f"{args.smoke_lane.removesuffix('-smoke')}-oracle",
        "model_id": args.model_id,
        "model_sha256": args.model_sha256,
        "projector_sha256": sha256(args.projector_path) if args.projector_path else None,
        "oracle_executable": args.oracle_executable.name,
        "oracle_executable_sha256": sha256(args.oracle_executable),
        "candidate_executable_sha256": sha256(args.candidate_executable),
        "pinned_patch_sha": args.pinned_patch_sha,
        "comparison": lines[-1],
    }
    if args.model_class == "speech_synthesis":
        result = json.loads((args.work_dir / "tts-oracle-result.json").read_text(encoding="utf-8"))
        if result.get("status") != "pass" or result.get("pinned_patch_sha") != args.pinned_patch_sha:
            raise ValueError("TTS comparator result is missing or does not match the pinned patch")
        if not isinstance(result.get("metrics"), dict):
            raise ValueError("TTS comparator result lacks PCM metrics")
        evidence["metrics"] = result["metrics"]
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Persist a verified comparator result, returning failure for incomplete evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison-log", required=True, type=Path)
    parser.add_argument("--class", dest="model_class", required=True)
    parser.add_argument("--smoke-lane", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--projector-path", type=Path)
    parser.add_argument("--candidate-executable", required=True, type=Path)
    parser.add_argument("--oracle-executable", required=True, type=Path)
    parser.add_argument("--pinned-patch-sha", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        write_evidence(args)
    except (OSError, ValueError) as error:
        print(f"workload oracle evidence not written: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
