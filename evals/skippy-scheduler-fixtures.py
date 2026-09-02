#!/usr/bin/env python3
"""Validate and materialize the pinned Skippy scheduler workload fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO / "evals/skippy-scheduler-fixtures.json"
GENERATOR = REPO / "evals/skippy-agentic-prompt-manifest.py"
WORKLOAD_KEYS = (
    "rounds",
    "families",
    "requests_per_family",
    "prefix_blocks",
    "output_tokens",
    "ctx_size",
    "lanes",
    "admission_concurrency",
    "cache_entries",
    "stagger_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_catalog(document)
    return document


def require_keys(document: dict[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in document]
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(missing)}")


def derived_context_size(profile: dict[str, Any]) -> int | None:
    """Return the smallest power-of-two context that covers the pinned workload."""
    corpus = profile["corpus"]
    if corpus.get("kind") != "hf":
        return None
    workload = profile["workload"]
    prompt_tokens = sum(int(row["total_tokens"]) for row in corpus["rows"])
    prompt_tokens *= int(workload["requests_per_family"])
    output_tokens = int(workload["output_tokens"]) * (
        int(workload["families"]) * int(workload["requests_per_family"])
    )
    required_tokens = prompt_tokens + output_tokens
    return 1 << max(required_tokens - 1, 0).bit_length()


def validate_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("schema_version") != 1:
        raise ValueError("scheduler fixture schema_version must be 1")
    datasets = catalog.get("datasets")
    profiles = catalog.get("profiles")
    if not isinstance(datasets, dict) or not isinstance(profiles, dict) or not profiles:
        raise ValueError("scheduler fixture catalog needs datasets and profiles objects")
    for dataset_name, dataset in datasets.items():
        if not isinstance(dataset, dict):
            raise ValueError(f"dataset {dataset_name} must be an object")
        require_keys(
            dataset,
            ("repo_id", "repo_type", "revision", "files", "parquet_file", "provenance"),
            f"dataset {dataset_name}",
        )
        revision = dataset["revision"]
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"dataset {dataset_name} revision must be a 40-character commit")
        if dataset["repo_type"] != "dataset":
            raise ValueError(f"dataset {dataset_name} repo_type must be dataset")
        files = dataset["files"]
        if not isinstance(files, list) or dataset["parquet_file"] not in files:
            raise ValueError(f"dataset {dataset_name} files must include parquet_file")
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"profile {profile_name} must be an object")
        require_keys(
            profile,
            (
                "description",
                "model",
                "corpus",
                "workload",
                "ci_trace",
                "hardware_acceptance",
            ),
            f"profile {profile_name}",
        )
        model = profile["model"]
        require_keys(
            model,
            ("id", "repo", "revision", "filename", "sha256"),
            f"profile {profile_name} model",
        )
        if not isinstance(model["id"], str) or not model["id"]:
            raise ValueError(f"profile {profile_name} model id must be nonempty")
        if not isinstance(model["revision"], str) or len(model["revision"]) != 40:
            raise ValueError(f"profile {profile_name} model revision must be a commit")
        if not isinstance(model["sha256"], str) or len(model["sha256"]) != 64:
            raise ValueError(f"profile {profile_name} model needs a SHA-256")
        workload = profile["workload"]
        require_keys(workload, WORKLOAD_KEYS, f"profile {profile_name} workload")
        if any(float(workload[key]) <= 0 for key in WORKLOAD_KEYS):
            raise ValueError(f"profile {profile_name} workload values must be positive")
        expected_requests = int(workload["families"]) * int(workload["requests_per_family"])
        if int(workload["admission_concurrency"]) != expected_requests:
            raise ValueError(f"profile {profile_name} must admit its complete workload")
        if int(workload["lanes"]) < expected_requests:
            raise ValueError(f"profile {profile_name} needs at least one lane per request")
        trace = profile["ci_trace"]
        require_keys(
            trace,
            (
                "family_order",
                "prompt_tokens",
                "expected_fcfs_switches",
                "expected_dfs_switches",
            ),
            f"profile {profile_name} ci_trace",
        )
        family_order = trace["family_order"]
        if not isinstance(family_order, list) or len(family_order) != expected_requests:
            raise ValueError(f"profile {profile_name} ci_trace must cover every request")
        if set(family_order) != set(range(int(workload["families"]))):
            raise ValueError(f"profile {profile_name} ci_trace families do not match workload")
        corpus = profile["corpus"]
        if corpus.get("kind") == "hf":
            dataset_name = corpus.get("dataset")
            if dataset_name not in datasets:
                raise ValueError(f"profile {profile_name} references an unknown dataset")
            selection = corpus.get("selection")
            rows = corpus.get("rows")
            if not isinstance(selection, dict) or not isinstance(rows, list):
                raise ValueError(f"profile {profile_name} needs selection and rows")
            if len(rows) != int(workload["families"]):
                raise ValueError(f"profile {profile_name} must pin one row per family")
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(f"profile {profile_name} row {row_index} must be an object")
                require_keys(
                    row,
                    ("session_id", "source_dataset", "n_turns", "max_isl", "total_tokens"),
                    f"profile {profile_name} row {row_index}",
                )
                if int(row["total_tokens"]) <= 0:
                    raise ValueError(
                        f"profile {profile_name} row {row_index} total_tokens must be positive"
                    )
            if selection.get("families") != int(workload["families"]):
                raise ValueError(f"profile {profile_name} selection family count drifted")
            if selection.get("order") != "md5(session_id)":
                raise ValueError(f"profile {profile_name} selection order must be deterministic")
            manifest_hash = corpus.get("prompt_manifest_sha256")
            if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
                raise ValueError(f"profile {profile_name} needs a prompt manifest SHA-256")
            required_ctx_size = derived_context_size(profile)
            if required_ctx_size is None or int(workload["ctx_size"]) < required_ctx_size:
                raise ValueError(
                    f"profile {profile_name} ctx_size must cover its pinned row totals; "
                    f"need at least {required_ctx_size}"
                )
        elif corpus.get("kind") != "synthetic":
            raise ValueError(f"profile {profile_name} has an unsupported corpus kind")


def profile(catalog: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return catalog["profiles"][name]
    except KeyError as error:
        choices = ", ".join(sorted(catalog["profiles"]))
        raise ValueError(f"unknown profile {name!r}; choose one of: {choices}") from error


def dataset_for_profile(catalog: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    corpus = selected["corpus"]
    if corpus["kind"] != "hf":
        raise ValueError("the selected profile does not use a Hugging Face corpus")
    return catalog["datasets"][corpus["dataset"]]


def run_checked(
    command: list[str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        message = f"command failed ({error.returncode}): {' '.join(command)}{suffix}"
        raise RuntimeError(message) from error


def fetch_dataset(
    dataset: dict[str, Any],
    cache_dir: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    download = [
        "hf",
        "download",
        dataset["repo_id"],
        *dataset["files"],
        "--repo-type",
        dataset["repo_type"],
        "--revision",
        dataset["revision"],
    ]
    verify = [
        "hf",
        "cache",
        "verify",
        dataset["repo_id"],
        "--repo-type",
        dataset["repo_type"],
        "--revision",
        dataset["revision"],
        "--fail-on-missing-files",
    ]
    if cache_dir is not None:
        download.extend(("--cache-dir", str(cache_dir)))
        verify.extend(("--cache-dir", str(cache_dir)))
    snapshot = Path(run_checked(download, runner).stdout.strip())
    run_checked(verify, runner)
    parquet = snapshot / dataset["parquet_file"]
    if not parquet.is_file():
        raise RuntimeError(f"verified snapshot is missing {dataset['parquet_file']}: {snapshot}")
    return parquet


def load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("skippy_agentic_prompt_manifest", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def materialize_prompt_manifest(
    catalog: dict[str, Any],
    selected: dict[str, Any],
    dataset_file: Path,
    output: Path,
) -> str:
    dataset = dataset_for_profile(catalog, selected)
    corpus = selected["corpus"]
    selection = corpus["selection"]
    generator = load_generator()
    rows = generator.select_trajectories(
        dataset_file,
        selection["sources"],
        selection["families"],
        selection["min_isl"],
        selection["max_isl_exclusive"],
        selection["min_turns"],
    )
    pinned_rows = [
        {
            key: row[key]
            for key in ("session_id", "source_dataset", "n_turns", "max_isl", "total_tokens")
        }
        for row in rows
    ]
    if pinned_rows != corpus["rows"]:
        raise RuntimeError("selected HF rows do not match the pinned fixture provenance")
    document = generator.build_manifest(
        rows,
        selected["workload"]["requests_per_family"],
        {
            "dataset": dataset["repo_id"],
            "dataset_revision": dataset["revision"],
            "selection": selection,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    actual = sha256(temporary)
    expected = corpus["prompt_manifest_sha256"]
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"prompt manifest SHA-256 mismatch: expected {expected}, got {actual}")
    temporary.replace(output)
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the checked-in catalog without inference")
    show = subparsers.add_parser("show", help="print one resolved profile")
    show.add_argument("--profile", required=True)
    fetch = subparsers.add_parser("fetch", help="download and verify one pinned HF corpus")
    fetch.add_argument("--profile", required=True)
    fetch.add_argument("--cache-dir", type=Path)
    materialize = subparsers.add_parser(
        "materialize", help="build and hash-check a prompt manifest from a verified parquet"
    )
    materialize.add_argument("--profile", required=True)
    materialize.add_argument("--dataset-file", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare", help="fetch, verify, and materialize in one step")
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        catalog = load_catalog(args.catalog)
        if args.command == "validate":
            print(json.dumps({"profiles": sorted(catalog["profiles"]), "status": "valid"}))
            return 0
        selected = profile(catalog, args.profile)
        if args.command == "show":
            print(json.dumps(selected, indent=2))
            return 0
        dataset = dataset_for_profile(catalog, selected)
        if args.command == "fetch":
            print(fetch_dataset(dataset, args.cache_dir))
            return 0
        dataset_file = (
            args.dataset_file
            if args.command == "materialize"
            else fetch_dataset(dataset, args.cache_dir)
        )
        if not dataset_file.is_file():
            raise ValueError(f"dataset parquet not found: {dataset_file}")
        actual = materialize_prompt_manifest(catalog, selected, dataset_file, args.output)
        print(json.dumps({"output": str(args.output), "sha256": actual}))
        return 0
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
