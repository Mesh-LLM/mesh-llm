#!/usr/bin/env python3
"""Collect raw GitHub and git evidence for a MeshLLM release delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(*args: str) -> str:
    result = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{detail}")
    return result.stdout


def git(*args: str) -> str:
    return run("git", *args).strip()


def gh_json(*args: str):
    return json.loads(run("gh", *args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect an unclassified release-delta evidence manifest."
    )
    parser.add_argument("--repo", default="Mesh-LLM/mesh-llm")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--release-tag", help="Override the latest published release tag")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    return parser.parse_args()


def latest_release(repo: str, override_tag: str | None) -> dict:
    fields = "tagName,name,publishedAt,url,isPrerelease,isDraft,body,targetCommitish,assets"
    command = ["release", "view"]
    if override_tag:
        command.append(override_tag)
    command.extend(["--repo", repo, "--json", fields])
    return gh_json(*command)


def require_local_tag(tag: str) -> str:
    try:
        return git("rev-parse", f"{tag}^{{commit}}")
    except RuntimeError as error:
        raise RuntimeError(
            f"release tag {tag!r} is not available locally; fetch tags from the intended "
            "remote and rerun"
        ) from error


# scripts/release.sh and RELEASE.md's manual publish path both write this exact
# subject for the commit that bakes built/generated release artifacts (console
# dist, SwiftPM manifest, version bump) onto the tag. The workflow_dispatch path
# tags that prepared-source commit directly without pushing it to origin/main, so
# every release tag -- prerelease or stable -- sits one documented commit ahead
# of its main-line ancestor by design.
def release_prepare_subject(tag: str) -> str:
    return f"{tag}: prepare release source"


def require_release_base(origin_main: str, tag: str, tag_sha: str) -> str:
    """Return the main-line commit a release tag was prepared from.

    Fails closed unless every commit between that point and the tag itself is
    the documented release-prepare commit.
    """
    release_base = git("merge-base", origin_main, tag_sha)
    if release_base != tag_sha:
        commits = collect_commits(release_base, tag_sha)
        expected_subject = release_prepare_subject(tag)
        if len(commits) != 1 or commits[0]["subject"] != expected_subject:
            detail = "; ".join(f"{c['sha'][:12]} {c['subject']!r}" for c in commits)
            raise RuntimeError(
                f"release {tag!r} ({tag_sha}) diverges from origin/main (base "
                f"{release_base}); expected zero commits or exactly one commit with "
                f"subject {expected_subject!r}, found: {detail or 'no commits'}"
            )
    return release_base


def canonical_candidate_tag(origin_main: str, head_ref: str, head_sha: str) -> str | None:
    """Resolve the tag that names an off-main release-prepare commit."""
    if git("merge-base", origin_main, head_sha) == head_sha:
        return None

    explicit_tag = head_ref.removeprefix("refs/tags/")
    try:
        explicit_tag_sha = git("rev-parse", "--verify", f"refs/tags/{explicit_tag}^{{commit}}")
    except RuntimeError:
        explicit_tag_sha = None
    if explicit_tag_sha == head_sha:
        return explicit_tag

    tags = [tag for tag in git("tag", "--points-at", head_sha).splitlines() if tag]
    if len(tags) != 1:
        detail = ", ".join(repr(tag) for tag in tags) or "none"
        raise RuntimeError(
            f"candidate {head_ref!r} ({head_sha}) is off origin/main and must be named "
            "by an explicit release tag or have exactly one tag pointing at it; "
            f"found: {detail}"
        )
    return tags[0]


def require_release_provenance(
    base_tag: str, base_sha: str, head_ref: str, head_sha: str
) -> str | None:
    origin_main = git("rev-parse", "--verify", "origin/main^{commit}")
    base_release_base = require_release_base(origin_main, base_tag, base_sha)
    candidate_tag = canonical_candidate_tag(origin_main, head_ref, head_sha)
    head_release_base = require_release_base(
        origin_main, candidate_tag or head_ref, head_sha
    )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_release_base, head_release_base],
        check=False,
    ).returncode != 0:
        raise RuntimeError(
            f"previous release commit {base_sha} (base {base_release_base}) is not an "
            f"ancestor of candidate {head_sha} (base {head_release_base})"
        )
    return candidate_tag


def collect_commits(base: str, head: str) -> list[dict]:
    output = run(
        "git",
        "log",
        "-z",
        "--reverse",
        "--format=%H%x00%aI%x00%an%x00%ae%x00%s%x00%b",
        f"{base}..{head}",
    )
    fields = output.split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    if not fields:
        return []
    field_names = ("sha", "authored_at", "author", "author_email", "subject", "body")
    if len(fields) % len(field_names) != 0:
        raise RuntimeError("could not parse NUL-delimited git log records")

    commits = []
    for offset in range(0, len(fields), len(field_names)):
        commit_fields = fields[offset : offset + len(field_names)]
        commit_fields[-1] = commit_fields[-1].rstrip("\n")
        if len(commit_fields) != len(field_names):
            raise RuntimeError("unexpected NUL-delimited git log record length")
        commits.append(
            dict(zip(field_names, commit_fields, strict=True))
        )
    return commits


def collect_changed_files(base: str, head: str) -> list[dict]:
    output = run("git", "diff", "--name-status", "--find-renames", base, head)
    changed = []
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0]
        entry = {"status": status, "path": fields[-1]}
        if status.startswith("R") and len(fields) == 3:
            entry["old_path"] = fields[1]
        changed.append(entry)
    return changed


def collect_prs(repo: str, published_at: str, base_sha: str, head_sha: str) -> list[dict]:
    prs = gh_json(
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "merged",
        "--search",
        f"merged:>={published_at}",
        "--limit",
        "1000",
        "--json",
        "number,title,url,body,mergedAt,mergeCommit,labels,author,baseRefName,headRefName",
    )
    for pr in prs:
        oid = (pr.get("mergeCommit") or {}).get("oid")
        if not oid:
            pr["merge_commit_in_range"] = None
            continue
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", oid, head_sha], check=False
        )
        after_base = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base_sha, oid], check=False
        )
        pr["merge_commit_in_range"] = result.returncode == 0 and after_base.returncode == 0
    return prs


def hash_untracked_files(hasher: "hashlib._Hash") -> None:
    paths = run("git", "ls-files", "--others", "--exclude-standard", "-z")
    for raw_path in paths.split("\0"):
        if not raw_path:
            continue
        path = Path(raw_path)
        hasher.update(b"untracked\0")
        hasher.update(raw_path.encode())
        hasher.update(b"\0")
        if path.is_symlink():
            hasher.update(path.readlink().as_posix().encode())
        else:
            hasher.update(path.read_bytes())
        hasher.update(b"\0")


def dirty_state() -> dict:
    porcelain = run("git", "status", "--porcelain=v1", "--untracked-files=all")
    diff = run("git", "diff", "--binary", "HEAD")
    staged = run("git", "diff", "--binary", "--cached", "HEAD")
    hasher = hashlib.sha256()
    hasher.update(diff.encode())
    hasher.update(staged.encode())
    hasher.update(porcelain.encode())
    hash_untracked_files(hasher)
    return {
        "is_dirty": bool(porcelain.strip()),
        "porcelain": porcelain.splitlines(),
        "evidence_sha256": hasher.hexdigest(),
    }


def main() -> int:
    args = parse_args()
    try:
        release = latest_release(args.repo, args.release_tag)
        base_sha = require_local_tag(release["tagName"])
        head_sha = git("rev-parse", f"{args.head}^{{commit}}")
        candidate_tag = require_release_provenance(
            release["tagName"], base_sha, args.head, head_sha
        )
        manifest = {
            "schema_version": 1,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "repository": args.repo,
            "remote_urls": git("remote", "-v").splitlines(),
            "candidate": {
                "ref": args.head,
                "tag": candidate_tag,
                "sha": head_sha,
                "branch": git("branch", "--show-current"),
                "dirty": dirty_state(),
            },
            "previous_release": release,
            "previous_release_commit": base_sha,
            "comparison": {
                "range": f"{release['tagName']}..{head_sha}",
                "merge_base": git("merge-base", base_sha, head_sha),
                "commits": collect_commits(base_sha, head_sha),
                "changed_files": collect_changed_files(base_sha, head_sha),
                "merged_pr_query_limit": 1000,
            },
            "classification_note": (
                "Raw evidence only. A validator must reconcile and classify atomic "
                "release claims; commit and PR counts are not release-item counts."
            ),
        }
        prs = collect_prs(args.repo, release["publishedAt"], base_sha, head_sha)
        manifest["comparison"]["merged_prs_since_release"] = prs
        manifest["comparison"]["merged_pr_query_may_be_truncated"] = len(prs) == 1000
    except (RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
