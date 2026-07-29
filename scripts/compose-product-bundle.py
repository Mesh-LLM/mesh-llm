#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def compose_manifest(
    bundle: Path,
    host: Path,
    runtime: Path,
    version: str,
    backend: str,
) -> dict[str, object]:
    version = version.removeprefix("v")
    runtime_manifest_path = runtime / "manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    runtime_data = runtime_manifest["runtime"]
    runtime_id = runtime_data["id"]
    runtime_mesh_version = runtime_data["mesh_version"].removeprefix("v")
    if runtime_mesh_version != version:
        raise ValueError(
            f"native runtime {runtime_id} targets MeshLLM {runtime_mesh_version}, "
            f"expected {version}"
        )
    return {
        "schema_version": 2,
        "contract": "mesh-llm-product-v2",
        "mesh_version": version,
        "backend": backend,
        "host": {
            "path": host.relative_to(bundle).as_posix(),
            "sha256": file_sha256(host),
        },
        "runtime": {
            "id": runtime_id,
            "path": runtime.relative_to(bundle).as_posix(),
            "sha256": tree_sha256(runtime),
            "manifest_sha256": file_sha256(runtime_manifest_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--backend", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = compose_manifest(
        args.bundle, args.host, args.runtime, args.version, args.backend
    )
    (args.bundle / "product-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
