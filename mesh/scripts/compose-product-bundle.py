#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict


BACKEND_KIND_ALIASES = {"cuda-blackwell": "cuda", "hip": "rocm"}

# Structural re-check of the standalone Skippy qualification receipt record bound
# into a Mesh product manifest. The record's provenance is established by its
# producer (scripts/ci-host-qualification-receipts.py); this rejects a record
# that is truncated, swapped or foreign before it reaches a composed product.
QUALIFICATION_SCHEMA_VERSION = 1
QUALIFICATION_PLATFORMS = ("linux", "macos", "windows")
# Filename the standalone qualification receipt record takes inside a composed
# product bundle. It is discovered automatically so a recomposition check of an
# extracted product binds the same record it was composed with.
QUALIFICATION_RECORD_NAME = "skippy-qualification-receipts.json"
QUALIFICATION_STATUSES = ("qualified", "hardware-unavailable")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class RuntimeBackend(TypedDict):
    kind: str


class RuntimeData(TypedDict):
    id: str
    release_version: str
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


def bind_qualification_record(bundle: Path, record_path: Path) -> dict[str, object]:
    """Bind the standalone qualification receipt record into the product manifest."""
    if not record_path.is_file():
        raise ValueError(f"qualification receipt record is missing: {record_path}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"qualification receipt record is not JSON: {record_path}"
        ) from error
    if not isinstance(record, dict):
        raise ValueError("qualification receipt record must be an object")
    if record.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("qualification receipt record has an unsupported schema")
    source_sha = record.get("source_sha")
    plan_digest = record.get("plan_digest")
    if not isinstance(source_sha, str) or GIT_SHA.fullmatch(source_sha) is None:
        raise ValueError("qualification receipt record has an invalid source revision")
    if not isinstance(plan_digest, str) or SHA256_HEX.fullmatch(plan_digest) is None:
        raise ValueError("qualification receipt record has an invalid plan digest")
    platform = record.get("platform")
    if platform not in QUALIFICATION_PLATFORMS:
        raise ValueError("qualification receipt record names an unknown platform")
    required_row = record.get("required_row")
    receipts = record.get("receipts")
    if not isinstance(required_row, str) or not required_row:
        raise ValueError("qualification receipt record names no required row")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("qualification receipt record carries no receipts")
    bound: list[dict[str, str]] = []
    statuses: dict[str, str] = {}
    for entry in receipts:
        if not isinstance(entry, dict):
            raise ValueError("qualification receipt record carries a malformed receipt")
        row_id = entry.get("row_id")
        status = entry.get("status")
        digest = entry.get("receipt_sha256")
        if not isinstance(row_id, str) or not row_id.startswith(f"{platform}-"):
            raise ValueError("qualification receipt record carries a foreign row")
        if status not in QUALIFICATION_STATUSES:
            raise ValueError(f"qualification receipt for {row_id} has an unknown status")
        if not isinstance(digest, str) or SHA256_HEX.fullmatch(digest) is None:
            raise ValueError(f"qualification receipt for {row_id} has an invalid digest")
        if row_id in statuses:
            raise ValueError(f"qualification receipt record repeats {row_id}")
        statuses[row_id] = status
        bound.append({"row_id": row_id, "status": status, "receipt_sha256": digest})
    if statuses.get(required_row) != "qualified":
        raise ValueError(
            f"required row {required_row} is not qualified in the receipt record"
        )
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "record_path": record_path.relative_to(bundle).as_posix(),
        "record_sha256": file_sha256(record_path),
        "source_sha": source_sha,
        "plan_digest": plan_digest,
        "platform": platform,
        "required_row": required_row,
        "receipts": sorted(bound, key=lambda entry: entry["row_id"]),
    }


def compose_manifest(
    bundle: Path,
    host: Path,
    runtime: Path,
    version: str,
    backend: str,
    qualification_receipts: Path | None = None,
) -> dict[str, object]:
    version = version.removeprefix("v")
    runtime_manifest_path = runtime / "manifest.json"
    runtime_manifest: RuntimeManifest = json.loads(
        runtime_manifest_path.read_text(encoding="utf-8")
    )
    if type(runtime_manifest.get("schema_version")) is not int or runtime_manifest["schema_version"] != 2:
        raise ValueError("native runtime manifest requires schema_version 2; import legacy caches explicitly")
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
    manifest: dict[str, object] = {
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
            "release_version": runtime_data["release_version"],
            "skippy_abi": runtime_data["skippy_abi"],
            "path": runtime.relative_to(bundle).as_posix(),
            "sha256": tree_sha256(runtime),
            "manifest_sha256": file_sha256(runtime_manifest_path),
        },
    }
    if qualification_receipts is not None:
        manifest["skippy_qualification"] = bind_qualification_record(
            bundle, qualification_receipts
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument(
        "--qualification-receipts",
        type=Path,
        default=None,
        help=(
            "Standalone Skippy qualification receipt record, inside the bundle, "
            "to bind into the product manifest."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing product manifest without rewriting it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qualification_receipts = args.qualification_receipts
    if qualification_receipts is None:
        discovered = args.bundle / QUALIFICATION_RECORD_NAME
        if discovered.is_file():
            qualification_receipts = discovered
    manifest = compose_manifest(
        args.bundle,
        args.host,
        args.runtime,
        args.version,
        args.backend,
        qualification_receipts=qualification_receipts,
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
