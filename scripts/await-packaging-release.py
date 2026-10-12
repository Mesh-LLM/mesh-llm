#!/usr/bin/env python3
"""Require the terminal, identity-bound mesh-packaging release receipt."""

import argparse
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def gh_api(path: str) -> bytes:
    result = subprocess.run(
        ["gh", "api", "--method", "GET", path],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return result.stdout


def find_run(repository: str, correlation_id: str) -> dict | None:
    matching = []
    for page in range(1, 6):
        payload = json.loads(
            gh_api(
                f"repos/{repository}/actions/workflows/images-release.yml/runs"
                f"?event=repository_dispatch&per_page=100&page={page}"
            )
        )
        runs = payload.get("workflow_runs", [])
        matching.extend(
            run for run in runs if run.get("display_title") == f"Packaging · {correlation_id}"
        )
        if len(runs) < 100:
            break
        if page == 5:
            raise ValueError("packaging run search exceeded the bounded history window")
    if len(matching) > 1:
        raise ValueError("more than one packaging run has the release correlation ID")
    return matching[0] if matching else None


def load_receipt(repository: str, run: dict) -> dict:
    run_id = run["id"]
    artifacts = []
    for page in range(1, 6):
        payload = json.loads(
            gh_api(f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100&page={page}")
        )
        current = payload.get("artifacts", [])
        artifacts.extend(
            artifact
            for artifact in current
            if artifact.get("name") == "packaging-readiness" and not artifact.get("expired")
        )
        if len(current) < 100:
            break
        if page == 5:
            raise ValueError("packaging artifact search exceeded the bounded artifact window")
    if len(artifacts) != 1:
        raise ValueError(f"packaging run {run_id} has {len(artifacts)} readiness artifacts")
    if artifacts[0].get("size_in_bytes", 0) > 1024 * 1024:
        raise ValueError("packaging readiness artifact is unexpectedly large")
    data = gh_api(f"repos/{repository}/actions/artifacts/{artifacts[0]['id']}/zip")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.namelist() != ["packaging-readiness.json"]:
            raise ValueError("packaging readiness archive has unexpected files")
        with archive.open("packaging-readiness.json") as source:
            return json.load(source)


def validate_receipt(receipt: dict, run: dict, args: argparse.Namespace) -> None:
    if receipt.get("schema") != "mesh-packaging-readiness-v1":
        raise ValueError("unsupported packaging readiness schema")
    if receipt.get("correlation_id") != args.correlation_id:
        raise ValueError("packaging receipt correlation ID mismatch")
    if receipt.get("packaging_run_id") != run["id"] or receipt.get("packaging_run_attempt") != run["run_attempt"]:
        raise ValueError("packaging receipt run identity mismatch")
    expected = {
        "repository": args.upstream_repository,
        "ref": args.upstream_ref,
        "sha": args.upstream_sha,
        "manifest_sha256": args.manifest_sha256,
    }
    if receipt.get("upstream") != expected:
        raise ValueError("packaging receipt upstream release identity mismatch")
    if receipt.get("requested") != {
        "publish_images": "true",
        "publish_release_assets": "true",
        "publish_npm": "true",
    }:
        raise ValueError("packaging receipt did not select all required release channels")
    if receipt.get("status") != "success" or run.get("conclusion") != "success":
        raise ValueError(f"packaging did not complete successfully: {receipt.get('results')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--correlation-id", required=True)
    parser.add_argument("--upstream-repository", required=True)
    parser.add_argument("--upstream-ref", required=True)
    parser.add_argument("--upstream-sha", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3 * 60 * 60)
    args = parser.parse_args()
    if args.repository != "Mesh-LLM/mesh-packaging" or args.upstream_repository != "Mesh-LLM/mesh-llm":
        parser.error("unexpected release repository")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", args.correlation_id):
        parser.error("invalid correlation ID")
    if not re.fullmatch(r"[0-9a-f]{40}", args.upstream_sha) or not re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256):
        parser.error("invalid release digest")

    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        run = find_run(args.repository, args.correlation_id)
        if run and run.get("status") == "completed":
            receipt = load_receipt(args.repository, run)
            args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validate_receipt(receipt, run, args)
            print(f"Verified packaging readiness from run {run['id']}/{run['run_attempt']}")
            return 0
        time.sleep(30)
    raise TimeoutError("correlated packaging run did not reach terminal readiness before the deadline")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TimeoutError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Packaging readiness failed: {error}", file=sys.stderr)
        raise SystemExit(1)
