#!/usr/bin/env python3
"""Atomically promote a staged Hugging Face layer-package snapshot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable


def snapshot_paths(manifest: dict[str, Any]) -> list[str]:
    entries = manifest.get("artifact_catalog", {}).get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest artifact catalog is empty")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if len(paths) != len(entries) or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("manifest artifact catalog contains an invalid path")
    paths.append("model-package.json")
    if len(paths) != len(set(paths)):
        raise ValueError("manifest snapshot paths are not unique")
    return paths


def prepare_snapshot(api: Any, repo_id: str, source_revision: str, token: str) -> tuple[str, str]:
    parent = api.model_info(repo_id, revision="main").sha
    if not isinstance(parent, str) or not re.fullmatch(r"[0-9a-f]{40}", parent):
        raise ValueError("target main did not resolve to an immutable commit")
    safe_token = re.sub(r"[^A-Za-z0-9._-]", "-", token).strip("-")
    if not safe_token:
        raise ValueError("snapshot token is empty")
    revision = f"automation/republish-{source_revision[:12]}-{safe_token}"
    api.create_branch(repo_id, branch=revision, revision=parent, repo_type="model")
    return revision, parent


def promote_snapshot(
    api: Any,
    copy_operation: Callable[..., Any],
    repo_id: str,
    manifest_path: Path,
    staging_revision: str,
    parent_commit: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operations = [
        copy_operation(
            src_path_in_repo=path,
            path_in_repo=path,
            src_revision=staging_revision,
        )
        for path in snapshot_paths(manifest)
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        parent_commit=parent_commit,
        operations=operations,
        commit_message=f"Atomically promote layer package from {staging_revision}",
    )
    try:
        api.delete_branch(repo_id, branch=staging_revision, repo_type="model")
    except Exception as error:  # Promotion is complete; branch cleanup is best effort.
        print(f"WARNING: could not delete staging branch {staging_revision}: {error}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo", required=True)
    prepare.add_argument("--source-revision", required=True)
    prepare.add_argument("--token", required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--repo", required=True)
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--staging-revision", required=True)
    promote.add_argument("--parent-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import CommitOperationCopy, HfApi

    api = HfApi()
    if args.command == "prepare":
        revision, parent = prepare_snapshot(api, args.repo, args.source_revision, args.token)
        print(revision)
        print(parent)
        return
    promote_snapshot(
        api,
        CommitOperationCopy,
        args.repo,
        args.manifest,
        args.staging_revision,
        args.parent_commit,
    )


if __name__ == "__main__":
    main()
