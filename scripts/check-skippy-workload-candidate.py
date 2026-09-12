#!/usr/bin/env python3
"""Reject a stale statically linked candidate before an oracle comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(root: Path = ROOT) -> dict[str, str]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args])

    digest = hashlib.sha256(git("diff", "--binary", "HEAD", "--"))
    # Bind new source files as well as staged/unstaged tracked changes. Build
    # outputs are ignored by Git and cannot change this identity during a run.
    for name in sorted(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if name:
            digest.update(name + b"\0")
            digest.update(file_hash(root / os.fsdecode(name)).encode("ascii"))
    return {"head": git("rev-parse", "HEAD").decode().strip(), "worktree_sha256": digest.hexdigest()}


def producer_files(binary: Path, build_dir: Path, test_binary: Path) -> dict[str, Path]:
    return {
        "candidate": binary,
        "test_binary": test_binary,
        "model_package": binary.parent / "skippy-model-package",
        "correctness": binary.parent / "skippy-correctness",
        "native_stamp": build_dir / ".mesh-llm-build-stamp",
        **{name: build_dir / "bin" / name for name in ("llama-server", "llama-completion", "llama-tts")},
    }


def write_producer(output: Path, binary: Path, build_dir: Path, test_binary: Path, source_snapshot: Path) -> None:
    source = source_identity()
    if source != json.loads(source_snapshot.read_text(encoding="utf-8")):
        raise RuntimeError("repository source changed while building workload producers")
    check_candidate(binary, build_dir)
    check_candidate(test_binary, build_dir)
    files = producer_files(binary, build_dir, test_binary)
    payload = {
        "schema_version": 1,
        "source": source,
        "files": {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in files.items()},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_producer(manifest: Path, binary: Path, build_dir: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("source") != source_identity():
        raise RuntimeError("workload producer does not match the current repository head and worktree")
    records = payload["files"]
    files = producer_files(binary, build_dir, Path(records["test_binary"]["path"]))
    for name, path in files.items():
        record = records[name]
        if record["path"] != str(path.resolve()) or record["sha256"] != file_hash(path):
            raise RuntimeError(f"workload producer artifact changed: {name}")
    check_candidate(files["test_binary"], build_dir)


def check_candidate(binary: Path, build_dir: Path) -> None:
    stamp = build_dir / ".mesh-llm-build-stamp"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"candidate executable is missing: {binary}")
    if not stamp.is_file():
        raise RuntimeError(f"candidate native build stamp is missing: {stamp}")
    if binary.stat().st_mtime_ns <= stamp.stat().st_mtime_ns:
        raise RuntimeError(
            "candidate executable predates the stamped native ABI; "
            "rebuild skippy-server against the current LLAMA_STAGE_BUILD_DIR"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--native-build-dir", type=Path)
    producer = parser.add_mutually_exclusive_group()
    producer.add_argument("--write-producer", type=Path)
    producer.add_argument("--producer-manifest", type=Path)
    parser.add_argument("--test-binary", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--write-source-snapshot", type=Path)
    args = parser.parse_args()
    if args.write_source_snapshot:
        args.write_source_snapshot.write_text(json.dumps(source_identity()) + "\n", encoding="utf-8")
        return
    if not args.candidate_binary or not args.native_build_dir:
        parser.error("--candidate-binary and --native-build-dir are required")
    check_candidate(args.candidate_binary, args.native_build_dir)
    if args.write_producer:
        if not args.test_binary or not args.source_snapshot:
            parser.error("--write-producer requires --test-binary and --source-snapshot")
        write_producer(args.write_producer, args.candidate_binary, args.native_build_dir, args.test_binary, args.source_snapshot)
    elif args.producer_manifest:
        verify_producer(args.producer_manifest, args.candidate_binary, args.native_build_dir)


if __name__ == "__main__":
    main()
