#!/usr/bin/env python3
"""Build a release-notes plan deterministically from canonical git commits.

Squash-merge commit subjects carry a Conventional Commits type, and the type
maps to exactly one Keep a Changelog section. This produces the plan that
scripts/release-notes-regroup.py renders, with no model in the loop.

Entries whose commit subject is not conventional cannot be classified and land
in "Other changes" rather than being guessed at. An optional agent review pass
may reclassify those afterwards.

Usage:
    release-notes-classify.py --body BODY.md --range v0.75.1..v0.76.0 \
        --version 0.76.0 --date 2026-09-10 --out plan.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

_CHECKER_PATH = Path(__file__).resolve().parent / "check-conventional-commit.py"
_SPEC = importlib.util.spec_from_file_location("conventional_commit", _CHECKER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to import {_CHECKER_PATH}")
CONVENTIONAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CONVENTIONAL)

BODY_ENTRY_RE = re.compile(r"^\* .*/pull/(\d+)\s*$")
PR_SUFFIX_RE = re.compile(r"\(#(\d+)\)$")
TRAILER_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z -]*):\s*(?P<value>.+)$")

SECTION_ORDER = [
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Other changes",
]

# Internal entries group by the type that produced them.
INTERNAL_GROUPS = OrderedDict(
    [
        ("CI and release engineering", ["ci"]),
        ("Build and dependencies", ["build", "deps", "chore"]),
        ("Tests", ["test"]),
        ("Refactors, docs, and hygiene", ["refactor", "style", "docs"]),
    ]
)

# Work on the repository's own tooling is internal whatever its type: a
# fix(ci) repairs CI, it does not fix the product.
INTERNAL_SCOPES = {
    "ci": "CI and release engineering",
    "release": "CI and release engineering",
    "build": "Build and dependencies",
    "deps": "Build and dependencies",
    "xtask": "Build and dependencies",
    "bench": "Tests",
    "just": "Refactors, docs, and hygiene",
}

INTERNAL_SUMMARY = (
    "CI, build, test, and repository work with no user-facing behavior change"
)

# Below this, a flat section reads better than sub-headings.
SUBGROUP_THRESHOLD = 20

# A sub-heading needs enough entries to earn its line; smaller scopes merge.
MIN_GROUP = 4


def read_body_prs(path):
    """Return PR numbers in release-body order."""
    prs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("## New Contributors") or line.startswith(
                "**Full Changelog**"
            ):
                break
            match = BODY_ENTRY_RE.match(line.rstrip("\n"))
            if match:
                prs.append(int(match.group(1)))
    return prs


def read_commits(git_range, repo_root=None):
    """Map PR number -> {subject, trailers} from the canonical commits."""
    result = subprocess.run(
        ["git", "log", "--format=%s%x1f%b%x1e", git_range],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    commits = {}
    for record in result.stdout.split("\x1e"):
        if not record.strip():
            continue
        subject, _, body = record.strip("\n").partition("\x1f")
        match = PR_SUFFIX_RE.search(subject.strip())
        if not match:
            continue
        trailers = {}
        for line in body.splitlines():
            found = TRAILER_RE.match(line.strip())
            if found:
                trailers[found.group("key").strip().lower()] = found.group("value").strip()
            if line.strip().startswith("BREAKING CHANGE:"):
                trailers["breaking change"] = line.split(":", 1)[1].strip()
        commits[int(match.group(1))] = {"subject": subject.strip(), "trailers": trailers}
    return commits


def scope_of(commit):
    """Return the conventional scope of a subject, or None if it has none."""
    authored = CONVENTIONAL.TRAILING_PR_RE.sub("", commit["subject"])
    match = CONVENTIONAL.SUBJECT_RE.match(authored)
    return match.group("scope") if match else None


def type_of(commit):
    """Return the conventional type of a subject, or None if it has none."""
    authored = CONVENTIONAL.TRAILING_PR_RE.sub("", commit["subject"])
    match = CONVENTIONAL.SUBJECT_RE.match(authored)
    return match.group("type") if match else None


def classify(commit):
    """Return (section, group_key) for one commit, or (None, None) if unknown."""
    if commit is None:
        return None, None

    trailers = commit["trailers"]
    override = trailers.get("release-notes")
    if override:
        title = override.strip().title()
        if title in SECTION_ORDER or title == "Internal":
            return title, scope_of(commit)
        # A misspelled override ("Secuirty") must not quietly fall through to
        # type classification and land a security fix in Fixed.
        return None, None

    authored = CONVENTIONAL.TRAILING_PR_RE.sub("", commit["subject"])
    match = CONVENTIONAL.SUBJECT_RE.match(authored)
    if not match:
        return None, None

    kind = match.group("type")
    if kind not in CONVENTIONAL.TYPES:
        return None, None
    scope = match.group("scope")

    if "deprecated" in trailers:
        return "Deprecated", scope
    if "removed" in trailers:
        return "Removed", scope
    if "security" in trailers:
        return "Security", scope
    if match.group("breaking") or "breaking change" in trailers:
        return "Changed", scope

    if scope in INTERNAL_SCOPES:
        return "Internal", scope

    return CONVENTIONAL.TYPES[kind], scope


def subgroups(by_scope):
    """Split one section by scope, keeping only sub-headings that earn a line.

    Scopes below MIN_GROUP entries merge into a trailing "Other" group, and a
    split that yields fewer than two headings is not worth making at all.
    """
    named, leftover = [], []
    for scope, prs in sorted(by_scope.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if scope != "General" and len(prs) >= MIN_GROUP:
            named.append({"title": scope, "prs": prs})
        else:
            leftover.extend(prs)
    if len(named) < 2:
        return None
    if leftover:
        named.append({"title": "Other", "prs": sorted(leftover)})
    return named


def build_plan(prs, commits, version, date):
    sections = defaultdict(list)
    scopes = defaultdict(lambda: defaultdict(list))
    internal = defaultdict(list)
    unclassified = 0

    for pr in prs:
        commit = commits.get(pr)
        section, scope = classify(commit)
        if section is None:
            sections["Other changes"].append(pr)
            unclassified += 1
            continue
        if section == "Internal":
            # A Release-Notes: Internal override can arrive on a subject with no
            # conventional type at all, so fall back rather than assume one.
            kind = type_of(commit)
            title = INTERNAL_SCOPES.get(scope) or next(
                (name for name, kinds in INTERNAL_GROUPS.items() if kind in kinds),
                "Refactors, docs, and hygiene",
            )
            internal[title].append(pr)
            continue
        sections[section].append(pr)
        scopes[section][scope or "General"].append(pr)

    plan = {"version": version, "date": date, "sections": []}
    for title in SECTION_ORDER:
        prs_in_section = sections.get(title)
        if not prs_in_section:
            continue
        groups = None
        if len(prs_in_section) > SUBGROUP_THRESHOLD and title != "Other changes":
            groups = subgroups(scopes[title])
        if groups:
            plan["sections"].append({"title": title, "groups": groups})
        else:
            plan["sections"].append({"title": title, "prs": prs_in_section})

    if internal:
        plan["internal"] = {
            "summary": INTERNAL_SUMMARY,
            "groups": [
                {"title": title, "prs": internal[title]}
                for title in INTERNAL_GROUPS
                if internal.get(title)
            ],
        }
    return plan, unclassified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", required=True, help="published release body")
    parser.add_argument("--range", required=True, help="git range, e.g. v0.75.1..v0.76.0")
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    prs = read_body_prs(args.body)
    if not prs:
        sys.exit("error: no PR entries found in the release body")
    commits = read_commits(args.range, args.repo_root)
    plan, unclassified = build_plan(prs, commits, args.version, args.date)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")

    classified = len(prs) - unclassified
    print(
        f"classified {classified}/{len(prs)} entries deterministically "
        f"({unclassified} in 'Other changes')"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
