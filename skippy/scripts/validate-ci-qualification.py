#!/usr/bin/env python3
"""Validate a standalone Skippy execution receipt against its exact CI product."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
CORE_ROWS = {
    "linux-cpu", "linux-cuda", "linux-rocm", "linux-vulkan",
    "macos-metal", "windows-cpu", "windows-cuda", "windows-rocm",
    "windows-vulkan",
}
REQUIRED_AVAILABLE_ROWS = {"linux-cpu", "linux-cuda", "macos-metal", "windows-cpu"}
SUITE_CASES = {
    "packaging-runtime": {"archive-integrity", "imports", "abi", "discovery", "version", "no-driver"},
    "dense": {"load", "prefill-decode", "stream", "stop-cancel", "concurrent", "continuation", "restart", "staged-parity"},
    "recurrent": {"prefill-decode", "state-preservation", "repeated-restore", "reset-isolation", "restart"},
    "moe": {"expert-metadata", "expert-execution", "staged-parity", "repeated-restore", "suffix-continuation", "session-isolation"},
    "kv-cache": {"dense-prefix-hit", "recurrent-prefix-hit", "suffix-continuation", "divergent-prefix", "isolation", "eviction", "import-export", "persisted-restart", "corrupt-rejection", "restore-observed"},
    "system-one": {"laya-goldens", "reader-contract", "negative-cases", "lifecycle"},
    "decisions": {"endpoint-equivalence", "probability-contract", "negative-cases", "lifecycle"},
}
REQUIRED_SUITES = set(SUITE_CASES)
MODEL_TAGS = {
    "dense": ("dense",),
    "recurrent": ("hybrid", "recurrent"),
    "moe": ("moe",),
    "kv-cache": ("dense", "hybrid"),
    "system-one": ("system-one",),
    "decisions": ("decision",),
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tree_digest(directory: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "standalone product contains a symlink")
        if path.is_file():
            relative = path.relative_to(directory).as_posix().encode()
            hasher.update(len(relative).to_bytes(8, "big"))
            hasher.update(relative)
            hasher.update(bytes.fromhex(digest(path)))
    return hasher.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, entry in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = entry
    return value


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def catalog_row(row_id: str) -> dict:
    rows = json.loads((ROOT / "ci/slices.yml").read_text(encoding="utf-8"))["runtime_rows"]
    by_id = {row["id"]: row for row in rows}
    require(set(by_id) == CORE_ROWS and len(rows) == len(CORE_ROWS), "core row catalog drift")
    require(row_id in by_id, f"unknown core row: {row_id}")
    return by_id[row_id]


def validate_models(name: str, models: dict) -> None:
    registry = load_json(ROOT / "ci/model-artifacts/registry.json")
    artifacts = {item["id"]: item for item in registry["artifacts"]}
    require(isinstance(models, dict), f"{name} lacks model identities")
    if name == "packaging-runtime":
        require(models == {}, "packaging-only suite must not claim a model")
        return
    require(bool(models), f"{name} lacks model identities")
    tags_found: set[str] = set()
    for artifact_id, model_hash in models.items():
        require(artifact_id in artifacts, f"{name} names an unregistered model")
        artifact = artifacts[artifact_id]
        files = artifact.get("artifact", {}).get("files", [])
        require(len(files) == 1 and files[0].get("sha256") == model_hash, f"{name} model digest differs from pinned registry")
        tags_found.update(artifact.get("capability_tags", []))
    if name == "kv-cache":
        require("dense" in tags_found and ("hybrid" in tags_found or "recurrent" in tags_found), "KV suite requires dense and recurrent models")
    elif name == "recurrent":
        require("hybrid" in tags_found or "recurrent" in tags_found, "recurrent suite lacks a recurrent model")
    else:
        require(set(MODEL_TAGS[name]) <= tags_found, f"{name} lacks required model capability")


def product_file(product_dir: Path, name: object) -> Path:
    require(isinstance(name, str) and bool(name), "invalid standalone product path")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "standalone product path escapes bundle")
    require(all(not (product_dir / Path(*relative.parts[:index])).is_symlink()
                for index in range(1, len(relative.parts) + 1)),
            "standalone product path contains a symlink")
    path = product_dir / relative
    require(path.is_file() and not path.is_symlink(), f"missing standalone product file: {name}")
    return path


def validate_product_bytes(product: dict, product_dir: Path) -> None:
    cli = product.get("cli")
    runtime = product.get("runtime")
    require(isinstance(cli, dict) and isinstance(runtime, dict), "incomplete standalone product manifest")
    binary = product_file(product_dir, cli.get("path"))
    require(digest(binary) == cli.get("sha256"), "standalone CLI bytes differ from manifest")
    contract = product_file(product_dir, "build-contract.json")
    require(digest(contract) == cli.get("build_contract_sha256"), "standalone CLI contract bytes differ from manifest")
    imports = product_file(product_dir, "host-imports.json")
    require(digest(imports) == cli.get("host_imports_sha256"), "standalone CLI import report bytes differ from manifest")
    runtime_name = runtime.get("path")
    require(isinstance(runtime_name, str) and bool(runtime_name), "invalid standalone runtime path")
    runtime_relative = Path(runtime_name)
    require(not runtime_relative.is_absolute() and ".." not in runtime_relative.parts, "standalone runtime path escapes bundle")
    require(all(not (product_dir / Path(*runtime_relative.parts[:index])).is_symlink()
                for index in range(1, len(runtime_relative.parts) + 1)),
            "standalone runtime path contains a symlink")
    runtime_dir = product_dir / runtime_relative
    require(runtime_dir.is_dir() and not runtime_dir.is_symlink(), "standalone runtime directory is missing")
    manifest = product_file(runtime_dir, "manifest.json")
    require(digest(manifest) == runtime.get("manifest_sha256"), "standalone runtime manifest bytes differ")
    require(tree_digest(runtime_dir) == runtime.get("sha256"), "standalone runtime bytes differ from manifest")


def validate_receipt(
    receipt: dict, product: dict, product_manifest: Path, product_dir: Path,
    availability: dict, availability_path: Path, evidence_dir: Path, *, source_sha: str,
    plan_digest: str, row_id: str,
) -> None:
    row = catalog_row(row_id)
    require(GIT_SHA.fullmatch(source_sha) is not None, "invalid source SHA")
    require(SHA256.fullmatch(plan_digest) is not None, "invalid plan digest")
    require(product.get("schema_version") == 1 and product.get("contract") == "skippy-product-v1", "invalid Skippy product manifest")
    validate_product_bytes(product, product_dir)
    for key, expected in (("source_sha", source_sha), ("target", row["target"]), ("backend", row["backend"])):
        require(product.get(key) == expected, f"product {key} differs from selected row")
    require(receipt.get("schema_version") == 1, "unknown qualification receipt schema")
    for key, expected in (
        ("source_sha", source_sha), ("plan_digest", plan_digest),
        ("row_id", row_id), ("product_manifest_sha256", digest(product_manifest)),
        ("availability_sha256", digest(availability_path)),
    ):
        require(receipt.get(key) == expected, f"receipt {key} differs from selected product or plan")
    require(availability.get("schema_version") == 1, "unknown hardware availability schema")
    require(availability.get("source_sha") == source_sha and availability.get("plan_digest") == plan_digest, "availability belongs to another source or plan")
    require(availability.get("row_id") == row_id, "availability belongs to another row")
    require(availability.get("policy_source") == "protected-ci", "hardware availability is not from the protected policy")
    state = availability.get("state")
    require(state in {"available", "hardware-unavailable"}, "invalid hardware availability state")
    if row_id in REQUIRED_AVAILABLE_ROWS:
        require(state == "available", "required available row cannot be hardware-unavailable")
    if state == "hardware-unavailable":
        require(isinstance(availability.get("reason"), str) and bool(availability["reason"].strip()), "unavailable hardware needs a reason")
        require(receipt.get("status") == "hardware-unavailable", "unavailable row cannot claim qualification")
    else:
        require(isinstance(availability.get("runner"), str) and bool(availability["runner"].strip()), "available hardware needs an approved runner")
        require(receipt.get("status") == "qualified", "available row must be qualified")

    suites = receipt.get("suites")
    require(isinstance(suites, dict) and set(suites) == REQUIRED_SUITES, "missing, duplicate, or unknown qualification suite")
    for name, result in suites.items():
        require(isinstance(result, dict), f"{name} result must be an object")
        if state == "hardware-unavailable" and name != "packaging-runtime":
            require(result == {"status": "not-executed"}, f"{name} claims execution on unavailable hardware")
            continue
        require(result.get("status") == "passed", f"{name} did not pass")
        require(isinstance(result.get("cases"), list) and len(result["cases"]) > 0, f"{name} lacks executed cases")
        require(all(isinstance(case, str) and case.strip() for case in result["cases"]), f"{name} has an invalid case")
        require(len(result["cases"]) == len(set(result["cases"])), f"{name} repeats an executed case")
        require(SUITE_CASES[name] <= set(result["cases"]), f"{name} is missing required cases")
        evidence_name = result.get("evidence_file")
        require(isinstance(evidence_name, str) and evidence_name == f"{name}.json", f"{name} has an invalid evidence path")
        evidence_path = evidence_dir / evidence_name
        require(evidence_path.is_file() and not evidence_path.is_symlink(), f"{name} evidence is missing")
        require(result.get("evidence_sha256") == digest(evidence_path), f"{name} evidence digest differs from bytes")
        evidence = load_json(evidence_path)
        require(evidence.get("schema_version") == 1, f"{name} evidence has unknown schema")
        require(evidence.get("status") == "passed", f"{name} evidence did not pass")
        require(evidence.get("source_sha") == source_sha and evidence.get("row_id") == row_id, f"{name} evidence belongs to another source or row")
        require(evidence.get("suite") == name, f"{name} evidence belongs to another suite")
        require(evidence.get("product_manifest_sha256") == digest(product_manifest), f"{name} evidence belongs to another product")
        require(evidence.get("executed_cases") == result["cases"], f"{name} evidence cases differ from receipt")
        models = result.get("models")
        validate_models(name, models)
        require(evidence.get("models") == models, f"{name} evidence models differ from receipt")

    hardware = receipt.get("hardware")
    if state == "hardware-unavailable":
        require(hardware is None, "unavailable row cannot claim hardware use")
        return
    require(isinstance(hardware, dict), "qualified row lacks hardware evidence")
    require(hardware.get("actual_backend") == row["backend"], "actual backend differs from selected backend")
    require(hardware.get("runner") == availability["runner"], "execution used a different runner from the protected availability plan")
    require(isinstance(hardware.get("device"), str) and bool(hardware["device"].strip()), "missing actual device")
    if row["backend"] == "cpu":
        require(hardware["device"] == "CPU", "CPU row has an unexpected device")
        require(hardware.get("offloaded_layers", 0) == 0, "CPU row reports GPU offload")
    else:
        require(isinstance(hardware.get("driver"), str) and bool(hardware["driver"].strip()), "GPU row lacks driver identity")
        require(isinstance(hardware.get("runtime_version"), str) and bool(hardware["runtime_version"].strip()), "GPU row lacks runtime identity")
        require(isinstance(hardware.get("device_architecture"), str) and bool(hardware["device_architecture"].strip()), "GPU row lacks architecture identity")
        require(type(hardware.get("offloaded_layers")) is int and hardware["offloaded_layers"] > 0, "GPU row lacks positive offload evidence")
        require(hardware["device"].lower() != "cpu", "GPU row reports a CPU device")
        require(not any(word in hardware["device"].lower() for word in ("lavapipe", "llvmpipe", "software")), "software renderer cannot qualify GPU row")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--product-manifest", type=Path, required=True)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--plan-digest", required=True)
    parser.add_argument("--row-id", required=True)
    args = parser.parse_args()
    try:
        validate_receipt(
            load_json(args.receipt), load_json(args.product_manifest),
            args.product_manifest, args.product_dir, load_json(args.availability), args.availability,
            args.evidence_dir,
            source_sha=args.source_sha, plan_digest=args.plan_digest,
            row_id=args.row_id,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Skippy qualification rejected: {error}", file=sys.stderr)
        return 1
    print(digest(args.receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
