#!/usr/bin/env python3
"""Verify final composed Skippy product bytes and emit packaging suite evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def load_module(name: str, file: str):
    path = Path(__file__).with_name(file)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load_module("validate_ci_qualification", "validate-ci-qualification.py")
composer = load_module("compose_ci_product", "compose-ci-product.py")
cli_verifier = load_module("verify_cli_input", "verify-cli-input.py")


def verify_archive(archive: Path, product_dir: Path) -> str:
    expected = {
        path.relative_to(product_dir).as_posix(): path
        for path in product_dir.rglob("*") if path.is_file()
    }
    if any(path.is_symlink() for path in product_dir.rglob("*")):
        raise ValueError("standalone product contains a symlink")
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        actual = [item.name for item in members if item.isfile()]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError("standalone archive file census differs from composed product")
        for item in members:
            if not (item.isfile() or item.isdir()) or item.name.startswith("/") or ".." in Path(item.name).parts:
                raise ValueError("standalone archive contains an unsafe member")
            if item.isfile():
                source = package.extractfile(item)
                if source is None:
                    raise ValueError(f"standalone archive member is unreadable: {item.name}")
                hasher = hashlib.sha256()
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(block)
                if hasher.hexdigest() != contract.digest(expected[item.name]):
                    raise ValueError(f"standalone archive member differs from product: {item.name}")
    return contract.digest(archive)


def verify(product_dir: Path, archive: Path, *, source_sha: str, row_id: str) -> dict:
    row = contract.catalog_row(row_id)
    manifest_path = product_dir / "product-manifest.json"
    product = contract.load_json(manifest_path)
    contract.require(product.get("schema_version") == 1 and product.get("contract") == "skippy-product-v1",
                     "invalid standalone product manifest")
    for key, expected in (("source_sha", source_sha), ("target", row["target"]), ("backend", row["backend"])):
        contract.require(product.get(key) == expected, f"product {key} differs from selected row")
    contract.validate_product_bytes(product, product_dir)
    archive_sha256 = verify_archive(archive, product_dir)

    binary = product_dir / product["cli"]["path"]
    with tempfile.TemporaryDirectory(prefix="skippy-ci-cli-verify-") as temporary:
        cli_dir = Path(temporary)
        for name in (binary.name, "build-contract.json", "host-imports.json"):
            shutil.copy2(product_dir / name, cli_dir / name)
        (cli_dir / f"{binary.name}.sha256").write_text(
            f"{contract.digest(binary)}  {binary.name}\n", encoding="utf-8",
        )
        cli_build = cli_verifier.verify(cli_dir, source_sha)

    runtime_dir = product_dir / product["runtime"]["path"]
    runtime_manifest = contract.load_json(runtime_dir / "manifest.json")
    runtime = composer.validate_pair(
        cli_build, runtime_manifest, source_sha=source_sha,
        target=row["target"], backend=row["backend"],
    )
    composer.verify_discovery(binary, runtime_dir, runtime)
    return {
        "schema_version": 1,
        "status": "passed",
        "source_sha": source_sha,
        "row_id": row_id,
        "suite": "packaging-runtime",
        "product_manifest_sha256": contract.digest(manifest_path),
        "executed_cases": sorted(contract.SUITE_CASES["packaging-runtime"]),
        "models": {},
        "observations": {
            "archive_sha256": archive_sha256,
            "cli_sha256": contract.digest(binary),
            "runtime_sha256": contract.tree_digest(runtime_dir),
            "runtime_id": runtime["id"],
            "release_version": runtime["release_version"],
            "skippy_abi": runtime["skippy_abi"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--product-archive", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = verify(args.product_dir, args.product_archive,
                          source_sha=args.source_sha, row_id=args.row_id)
        args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired, tarfile.TarError) as error:
        print(f"standalone packaging qualification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
