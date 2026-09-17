#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict


BACKEND_KIND_ALIASES = {"cuda-blackwell": "cuda", "hip": "rocm"}


class RuntimeBackend(TypedDict):
    kind: str


class RuntimeData(TypedDict):
    id: str
    mesh_version: str
    skippy_abi: str
    backend: RuntimeBackend


class BuildData(TypedDict):
    backend: str


class RuntimeManifest(TypedDict):
    runtime: RuntimeData
    build: NotRequired[BuildData]


def expected_backend_kind(backend: str) -> str:
    return BACKEND_KIND_ALIASES.get(backend, backend)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = (candidate for candidate in path.rglob("*") if candidate.is_file())
    for item in sorted(files, key=lambda candidate: candidate.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def validate_runtime_backend(
    runtime_id: str,
    runtime_manifest: RuntimeManifest,
    runtime_data: RuntimeData,
    requested_backend: str,
) -> None:
    expected_kind = expected_backend_kind(requested_backend)
    runtime_backend = runtime_data["backend"]
    runtime_kind = runtime_backend["kind"]
    if runtime_kind != expected_kind:
        raise ValueError(
            f"native runtime {runtime_id} backend mismatch: found {runtime_kind}, "
            f"expected {expected_kind} for requested backend {requested_backend}"
        )

    build_data = runtime_manifest.get("build")
    if build_data is not None and expected_backend_kind(build_data["backend"]) != expected_kind:
        raise ValueError(
            f"native runtime {runtime_id} build backend mismatch: found {build_data['backend']}, "
            f"expected runtime family {expected_kind} for requested backend {requested_backend}"
        )


def read_host_build_contract(host: Path) -> dict[str, object]:
    result = subprocess.run(
        [str(host.resolve()), "--log-format", "json", "--print-build-contract"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    contract = json.loads(result.stdout)
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("unsupported host build contract schema")
    for field in ("product_version", "runtime_release", "skippy_abi"):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise ValueError(f"host build contract is missing {field}")
    return contract


def compose_manifest(
    bundle: Path,
    host: Path,
    runtime: Path,
    version: str,
    backend: str,
) -> dict[str, object]:
    version = version.removeprefix("v")
    runtime_manifest_path = runtime / "manifest.json"
    runtime_manifest: RuntimeManifest = json.loads(
        runtime_manifest_path.read_text(encoding="utf-8")
    )
    runtime_data = runtime_manifest["runtime"]
    runtime_id = runtime_data["id"]
    validate_runtime_backend(runtime_id, runtime_manifest, runtime_data, backend)
    contract = read_host_build_contract(host)
    if contract["product_version"] != version:
        raise ValueError(
            f"host build contract product version {contract['product_version']} "
            f"does not match requested {version}"
        )
    if runtime_data["skippy_abi"] != contract["skippy_abi"]:
        raise ValueError(
            f"native runtime {runtime_id} ABI {runtime_data['skippy_abi']} "
            f"does not match host-required ABI {contract['skippy_abi']}"
        )
    return {
        "schema_version": 2,
        "contract": "mesh-llm-product-v2",
        "mesh_version": version,
        "backend": backend,
        "host": {
            "path": host.relative_to(bundle).as_posix(),
            "sha256": file_sha256(host),
            "required_skippy_abi": contract["skippy_abi"],
        },
        "runtime": {
            "id": runtime_id,
            "release_version": runtime_data["mesh_version"],
            "skippy_abi": runtime_data["skippy_abi"],
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing product manifest without rewriting it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = compose_manifest(
        args.bundle, args.host, args.runtime, args.version, args.backend
    )
    manifest_path = args.bundle / "product-manifest.json"
    if args.check:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"product manifest does not match composed bytes: {manifest_path}"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
