#!/usr/bin/env python3
"""Compose the exact standalone CLI and native runtime produced by CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "x86_64-unknown-linux-gnu": ("linux", "x86_64"),
    "aarch64-unknown-linux-gnu": ("linux", "aarch64"),
    "aarch64-apple-darwin": ("macos", "aarch64"),
    "x86_64-pc-windows-msvc": ("windows", "x86_64"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        relative = path.relative_to(directory).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def validate_pair(
    cli_contract: dict[str, object], runtime_manifest: dict[str, object],
    *, source_sha: str, target: str, backend: str,
) -> dict[str, object]:
    if target not in TARGETS:
        raise ValueError(f"unsupported standalone target: {target}")
    if backend not in {"cpu", "cuda", "rocm", "vulkan", "metal"}:
        raise ValueError(f"unsupported standalone backend: {backend}")
    if runtime_manifest.get("schema_version") != 2:
        raise ValueError("native runtime manifest must use schema 2")
    runtime = runtime_manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("native runtime manifest is missing runtime")
    platform = runtime.get("platform")
    selected = runtime.get("backend")
    if not isinstance(platform, dict) or platform.get("target") != target:
        raise ValueError("native runtime target differs from selected row")
    if not isinstance(selected, dict) or selected.get("kind") != backend:
        raise ValueError("native runtime backend differs from selected row")
    expected_os, expected_arch = TARGETS[target]
    if platform.get("os") != expected_os or platform.get("arch") != expected_arch:
        raise ValueError("native runtime OS or architecture differs from selected row")
    if cli_contract.get("source_sha") != source_sha:
        raise ValueError("Skippy CLI source differs from selected source")
    if cli_contract.get("os") != expected_os or cli_contract.get("architecture") != expected_arch:
        raise ValueError("Skippy CLI target differs from native runtime target")
    if cli_contract.get("runtime_release") != runtime.get("release_version"):
        raise ValueError("native runtime release differs from Skippy CLI requirement")
    if cli_contract.get("skippy_abi") != runtime.get("skippy_abi"):
        raise ValueError("native runtime ABI differs from Skippy CLI requirement")
    build = runtime_manifest.get("build")
    if isinstance(build, dict) and build.get("backend") not in {backend, "hip" if backend == "rocm" else backend}:
        raise ValueError("native runtime build backend differs from selected row")
    return runtime


def verify_runtime_source(runtime_input: Path, source_sha: str) -> None:
    provenance = json.loads((runtime_input / "ci-source.json").read_text(encoding="utf-8"))
    if provenance != {"source_sha": source_sha}:
        raise ValueError("native runtime producer source differs from selected source")


def verification_bash(*, windows: bool = os.name == "nt") -> str:
    if not windows:
        return "bash"
    git = shutil.which("git")
    if git is not None:
        git_path = Path(git).resolve()
        for root in (git_path.parent.parent, git_path.parent.parent.parent):
            bash = root / "bin" / "bash.exe"
            if bash.is_file():
                return str(bash)
    raise FileNotFoundError("Git Bash is required to verify Windows native runtime packages")


def verify_discovery(binary: Path, runtime_dir: Path, runtime: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix="skippy-ci-runtime-cache-") as cache:
        result = subprocess.run(
            [str(binary), "--runtime-bundle", str(runtime_dir), "--runtime-cache", cache,
             "--runtime-release", str(runtime["release_version"]), "--output", "json",
             "runtime", "list", "--installed"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    installed = json.loads(result.stdout)
    if not isinstance(installed, list) or not any(
        isinstance(entry, dict)
        and entry.get("native_runtime_id") == runtime["id"]
        and entry.get("release_version") == runtime["release_version"]
        and isinstance(entry.get("path"), str)
        and Path(entry["path"]).resolve() == runtime_dir.resolve()
        for entry in installed
    ):
        raise ValueError("composed Skippy CLI did not discover the exact native runtime")


def compose(cli_input: Path, runtime_input: Path, output: Path, *, source_sha: str, target: str, backend: str) -> Path:
    for producer in (cli_input, runtime_input):
        if output == producer or output in producer.parents or producer in output.parents:
            raise ValueError("standalone product output overlaps a producer input")
    subprocess.run(
        [sys.executable, str(ROOT / "skippy/scripts/verify-cli-input.py"), str(cli_input), "--source-sha", source_sha],
        check=True,
    )
    verify_runtime_source(runtime_input, source_sha)
    cli_contract = json.loads((cli_input / "build-contract.json").read_text(encoding="utf-8-sig"))
    archives = sorted(runtime_input.rglob("*.tar.gz"))
    sidecars = sorted(runtime_input.rglob("*.tar.gz.sha256"))
    if len(archives) != 1 or sidecars != [Path(str(archives[0]) + ".sha256")]:
        raise ValueError("standalone product requires exactly one runtime archive and sidecar")
    archive = archives[0]
    archive_arg = archive.relative_to(ROOT).as_posix()
    bash = verification_bash()
    subprocess.run([bash, "scripts/verify-native-runtime-package.sh", archive_arg], cwd=ROOT, check=True)
    output.mkdir(parents=True, exist_ok=False)
    runtime_root = output / "native-runtimes"
    runtime_root.mkdir()
    subprocess.run([sys.executable, str(ROOT / "scripts/safe-extract-tar.py"), str(archive), str(runtime_root)], check=True)
    manifests = sorted(runtime_root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("runtime archive must contain exactly one runtime manifest")
    manifest_path = manifests[0]
    runtime_dir = manifest_path.parent
    runtime_arg = runtime_dir.relative_to(ROOT).as_posix()
    subprocess.run([bash, "scripts/verify-native-runtime-package.sh", runtime_arg], cwd=ROOT, check=True)
    runtime_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = validate_pair(cli_contract, runtime_manifest, source_sha=source_sha, target=target, backend=backend)
    binary_name = "skippy.exe" if target.startswith("x86_64-pc-windows") else "skippy"
    binary = output / binary_name
    shutil.copy2(cli_input / binary_name, binary)
    shutil.copy2(cli_input / "host-imports.json", output / "host-imports.json")
    shutil.copy2(cli_input / "build-contract.json", output / "build-contract.json")
    verify_discovery(binary, runtime_dir, runtime)
    product = {
        "schema_version": 1,
        "contract": "skippy-product-v1",
        "source_sha": source_sha,
        "target": target,
        "backend": backend,
        "cli": {
            "path": binary_name,
            "sha256": sha256(binary),
            "build_contract_sha256": sha256(output / "build-contract.json"),
            "host_imports_sha256": sha256(output / "host-imports.json"),
        },
        "runtime": {"id": runtime["id"], "path": runtime_dir.relative_to(output).as_posix(), "sha256": tree_sha256(runtime_dir), "manifest_sha256": sha256(manifest_path)},
    }
    (output / "product-manifest.json").write_text(json.dumps(product, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    product_archive = output.with_suffix(".tar.gz")
    with tarfile.open(product_archive, "w:gz") as tar:
        for item in sorted(output.rglob("*")):
            tar.add(item, arcname=item.relative_to(output).as_posix(), recursive=False)
    return product_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-input", type=Path, required=True)
    parser.add_argument("--runtime-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    try:
        print(compose(args.cli_input.resolve(), args.runtime_input.resolve(), args.output.resolve(), source_sha=args.source_sha, target=args.target, backend=args.backend))
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"Skippy product composition failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
