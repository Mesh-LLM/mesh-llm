#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import NotRequired, TypedDict


BACKEND_KIND_ALIASES = {"cuda-blackwell": "cuda", "hip": "rocm"}


class RuntimeBackend(TypedDict):
    kind: str


class RuntimeData(TypedDict):
    id: str
    mesh_version: str
    backend: RuntimeBackend


class BuildData(TypedDict):
    backend: str


class RuntimeManifest(TypedDict):
    runtime: RuntimeData
    build: NotRequired[BuildData]


class ProviderRuntimeData(TypedDict):
    id: str
    version: str
    provider_kind: str
    protocol_version: str


class ProviderRuntimeManifest(TypedDict):
    schema_version: int
    runtime: ProviderRuntimeData


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


def compose_manifest(
    bundle: Path,
    host: Path,
    runtime: Path,
    version: str,
    backend: str,
    provider_runtimes: list[Path],
) -> dict[str, object]:
    version = version.removeprefix("v")
    runtime_manifest_path = runtime / "manifest.json"
    runtime_manifest: RuntimeManifest = json.loads(
        runtime_manifest_path.read_text(encoding="utf-8")
    )
    runtime_data = runtime_manifest["runtime"]
    runtime_id = runtime_data["id"]
    runtime_mesh_version = runtime_data["mesh_version"].removeprefix("v")
    if runtime_mesh_version != version:
        raise ValueError(
            f"native runtime {runtime_id} targets MeshLLM {runtime_mesh_version}, "
            f"expected {version}"
        )
    validate_runtime_backend(runtime_id, runtime_manifest, runtime_data, backend)
    manifest: dict[str, object] = {
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
    if provider_runtimes:
        manifest["provider_runtimes"] = compose_provider_runtimes(
            bundle, provider_runtimes
        )
    return manifest


def validate_provider_runtime_manifest(
    manifest_path: Path, provider_manifest: ProviderRuntimeManifest, provider_runtime: Path
) -> None:
    runtime_data = provider_manifest["runtime"]
    required_fields = ["id", "version", "provider_kind", "protocol_version"]
    for field in required_fields:
        if field not in runtime_data:
            raise ValueError(f"provider runtime manifest {manifest_path} missing required field: runtime.{field}")
    if "entrypoint" not in runtime_data:
        raise ValueError(f"provider runtime manifest {manifest_path} missing required field: runtime.entrypoint")
    entrypoint = runtime_data["entrypoint"]
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ValueError(f"provider runtime manifest {manifest_path} has invalid entrypoint")
    entrypoint_path = provider_runtime / entrypoint
    if not entrypoint_path.is_file():
        raise ValueError(
            f"provider runtime {manifest_path} entrypoint {entrypoint} does not exist or is not a file"
        )


def compose_provider_runtimes(
    bundle: Path, provider_runtimes: list[Path]
) -> list[dict[str, str]]:
    composed: list[dict[str, str]] = []
    coordinates: set[tuple[str, str]] = set()
    for provider_runtime in sorted(provider_runtimes):
        manifest_path = provider_runtime / "provider-runtime.json"
        provider_manifest: ProviderRuntimeManifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if provider_manifest["schema_version"] != 1:
            raise ValueError(
                f"unsupported provider runtime schema in {manifest_path}: "
                f"{provider_manifest['schema_version']}"
            )
        runtime_data = provider_manifest["runtime"]
        validate_provider_runtime_manifest(manifest_path, provider_manifest, provider_runtime)
        runtime_id = runtime_data["id"]
        provider_kind = runtime_data["provider_kind"]
        coordinate = (runtime_id, runtime_data["version"])
        if coordinate in coordinates:
            raise ValueError(
                f"duplicate provider runtime coordinate {runtime_id}@{runtime_data['version']}"
            )
        coordinates.add(coordinate)
        expected_path = Path("provider-runtimes") / provider_kind / runtime_id
        actual_path = provider_runtime.relative_to(bundle)
        if actual_path != expected_path:
            raise ValueError(
                f"provider runtime {runtime_id} path mismatch: found "
                f"{actual_path.as_posix()}, expected {expected_path.as_posix()}"
            )
        composed.append(
            {
                "id": runtime_id,
                "version": runtime_data["version"],
                "provider_kind": provider_kind,
                "protocol_version": runtime_data["protocol_version"],
                "path": actual_path.as_posix(),
                "sha256": tree_sha256(provider_runtime),
                "manifest_sha256": file_sha256(manifest_path),
            }
        )
    return composed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument(
        "--provider-runtime",
        action="append",
        default=[],
        type=Path,
        help="Provider runtime bundle embedded in the composed product (repeatable).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing product manifest without rewriting it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = compose_manifest(
        args.bundle,
        args.host,
        args.runtime,
        args.version,
        args.backend,
        args.provider_runtime,
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
