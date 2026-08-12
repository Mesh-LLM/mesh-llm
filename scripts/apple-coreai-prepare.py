#!/usr/bin/env python3
"""Prepare a reproducible Core AI .aimodel bundle on a macOS target.

This is deliberately a preparation tool, not part of the serving path. It
downloads a pinned Hugging Face revision, runs Apple's pinned coreai-models
exporter in an isolated environment, writes a provenance manifest, and moves
the validated bundle into place atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EXPORTER_REPOSITORY = "https://github.com/apple/coreai-models.git"
DEFAULT_EXPORTER_REVISION = "f401272cd3b8574c27cf5071c56409ad772f91fb"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path) -> list[dict[str, int | str]]:
    return [
        {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(p for p in root.rglob("*") if p.is_file())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model id, for example Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit SHA")
    parser.add_argument("--output", required=True, type=Path, help="Final .aimodel resource directory")
    parser.add_argument("--exporter-revision", default=DEFAULT_EXPORTER_REVISION)
    parser.add_argument("--exporter-repository", default=DEFAULT_EXPORTER_REPOSITORY)
    parser.add_argument("--exporter-cache", type=Path, help="Persistent exporter checkout cache")
    parser.add_argument("--hf-cache", type=Path, help="Persistent Hugging Face cache")
    parser.add_argument("--context", type=int, help="Maximum exported context length")
    parser.add_argument("--compression", default="4bit", help="Core AI compression preset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_exporter(repository: str, revision: str, cache: Path) -> Path:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not (cache / ".git").is_dir():
        run(["git", "clone", "--filter=blob:none", repository, str(cache)])
    run(["git", "fetch", "--quiet", "origin", revision], cwd=cache)
    run(["git", "checkout", "--quiet", "--detach", revision], cwd=cache)
    actual = run(["git", "rev-parse", "HEAD"], cwd=cache).strip()
    if actual != revision:
        raise RuntimeError(f"exporter checkout mismatch: expected {revision}, got {actual}")
    return cache


def resolve_source(model: str, revision: str, hf_cache: Path) -> tuple[str, Path, list[dict[str, int | str]]]:
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import HfApi, snapshot_download\n"
        "model, revision, cache = sys.argv[1:]\n"
        "info = HfApi().model_info(model, revision=revision)\n"
        "sha = info.sha\n"
        "if sha != revision: raise SystemExit(f'revision resolved to {sha}, expected {revision}')\n"
        "path = Path(snapshot_download(model, revision=revision, cache_dir=str(Path(cache) / 'hub'), allow_patterns=[\n"
        " '*.json', 'model*.safetensors', '*.py', 'tokenizer.model', '*.tiktoken', 'tiktoken.model', '*.txt', '*.jsonl', '*.jinja']))\n"
        "refs = path.parents[1] / 'refs'; refs.mkdir(parents=True, exist_ok=True)\n"
        "(refs / 'main').write_text(sha + '\\n'); print(json.dumps({'sha': sha, 'path': str(path)}))\n"
    )
    env = os.environ.copy()
    env["HF_HOME"] = str(hf_cache)
    output = run(["uv", "run", "--with", "huggingface_hub", "python", "-c", script, model, revision, str(hf_cache)], env=env)
    result = json.loads(output.strip().splitlines()[-1])
    source = Path(result["path"])
    return result["sha"], source, file_inventory(source)


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    if sys.platform != "darwin" or platform.machine() not in {"arm64", "arm64e"}:
        raise SystemExit("Core AI preparation requires Apple silicon macOS")
    if len(args.revision) != 40:
        raise SystemExit("--revision must be a 40-character immutable Hugging Face commit SHA")
    if len(args.exporter_revision) != 40:
        raise SystemExit("--exporter-revision must be a 40-character immutable commit SHA")
    if args.context is not None and args.context <= 0:
        raise SystemExit("--context must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}; pass --overwrite to replace it")

    exporter_cache = (args.exporter_cache or Path.home() / ".cache/mesh-llm/apple/coreai-models").expanduser().resolve()
    hf_cache = (args.hf_cache or Path.home() / ".cache/mesh-llm/apple/huggingface").expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"model": args.model, "revision": args.revision, "exporter_revision": args.exporter_revision, "output": str(args.output), "compression": args.compression, "context": args.context, "exporter_cache": str(exporter_cache), "hf_cache": str(hf_cache)}, indent=2))
        return 0

    exporter = ensure_exporter(args.exporter_repository, args.exporter_revision, exporter_cache)
    resolved_revision, _source_dir, source_files = resolve_source(args.model, args.revision, hf_cache)
    if resolved_revision != args.revision:
        raise RuntimeError("Hugging Face revision changed during preparation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = args.output.parent / f".{args.output.name}.staging-{uuid.uuid4().hex}"
    staging_parent.mkdir()
    try:
        env = os.environ.copy()
        env["HF_HOME"] = str(hf_cache)
        command = ["uv", "run", "--project", str(exporter), "coreai.llm.export", args.model, "--platform", "macOS", "--compression", args.compression, "--output-dir", str(staging_parent), "--overwrite"]
        if args.context is not None:
            command.extend(["--max-context-length", str(args.context)])
        run(command, cwd=exporter, env=env)
        candidates = [path for path in staging_parent.iterdir() if path.is_dir()]
        if len(candidates) != 1:
            raise RuntimeError(f"export produced {len(candidates)} bundle directories, expected one")
        bundle = candidates[0]
        metadata_path = bundle / "metadata.json"
        aimodels = list(bundle.glob("*.aimodel"))
        if not metadata_path.is_file() or len(aimodels) != 1:
            raise RuntimeError("export did not produce a valid .aimodel resource bundle")
        metadata = json.loads(metadata_path.read_text())
        manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "source": {"repo": args.model, "revision": resolved_revision, "files": source_files}, "export": {"repository": args.exporter_repository, "revision": args.exporter_revision, "compression": args.compression, "context": args.context or metadata.get("language", {}).get("max_context_length"), "platform": "macOS"}, "toolchain": {"python": platform.python_version(), "xcode": subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True).stdout.strip(), "macos": platform.mac_ver()[0]}, "artifact": {"path": bundle.name, "metadata": metadata, "files": file_inventory(bundle)}}
        (bundle / "mesh-coreai-preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.output.exists():
            if not args.overwrite:
                raise RuntimeError(f"output appeared during preparation: {args.output}")
            shutil.rmtree(args.output)
        os.replace(bundle, args.output)
        print(json.dumps({"status": "prepared", "output": str(args.output), "manifest": str(args.output / "mesh-coreai-preparation.json")}, indent=2))
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
