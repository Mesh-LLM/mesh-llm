#!/usr/bin/env python3
"""Validate a commit message against Conventional Commits v1.0.0.

Release notes are regrouped deterministically from the canonical squash-merge
commit subjects, so the type/scope prefix on a subject is release metadata, not
decoration. See .agents/skills/release-notes/SKILL.md.

Usage:
    check-conventional-commit.py <commit-message-file>
    check-conventional-commit.py --message "fix(skippy): restore prefix reuse"
    check-conventional-commit.py --range origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Closed set: every type maps to exactly one Keep a Changelog section.
TYPES = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Changed",
    "security": "Security",
    "revert": "Changed",
    "refactor": "Internal",
    "style": "Internal",
    "test": "Internal",
    "build": "Internal",
    "deps": "Internal",
    "ci": "Internal",
    "chore": "Internal",
    "docs": "Internal",
}

SUBJECT_RE = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[a-z0-9][a-z0-9._/-]*)\))?"
    r"(?P<breaking>!)?"
    r": (?P<description>.+)$"
)

# Git and the release workflow author these; they are never release entries.
EXEMPT_RE = re.compile(
    r"^(Merge |Revert \"|fixup! |squash! |amend! )"
    r"|^v?\d+\.\d+\.\d+[^:]*: prepare release source$"
)

MAX_SUBJECT = 100
TRAILING_PR_RE = re.compile(r"\s*\(#\d+\)$")

# Attribution trailers for agents, bots, and relay identities. GitHub adds
# these automatically when squashing a PR whose commits carry them, so they
# have to be kept out of the branch commits in the first place.
TRAILER_RE = re.compile(
    r"^(?P<token>[A-Za-z][A-Za-z-]*-by)\s*:\s*(?P<name>[^<]*?)\s*(?:<(?P<email>[^>]*)>)?\s*$",
    re.IGNORECASE,
)

# Any address at these domains, including subdomains.
DENIED_DOMAINS = ("buzz.xyz",)

DENIED_ADDRESSES = ("noreply@anthropic.com", "noreply@coderabbit.ai")

# Matched against whole name tokens, so "Sol" is denied but "Solomon" is not.
DENIED_NAMES = (
    "claude",
    "anthropic",
    "chatgpt",
    "openai",
    "codex",
    "copilot",
    "sisyphus",
    "astra",
    "sol",
    "luna",
    "terra",
    "coderabbit",
    "coderabbitai",
    "devin",
    "cursor",
)

NAME_TOKEN_RE = re.compile(r"[a-z0-9]+")


def denied_identity(name, email):
    """Return why this trailer identity is denied, or None if it is allowed."""
    email = (email or "").strip().lower()
    name = (name or "").strip()

    if email:
        if email in DENIED_ADDRESSES:
            return f"'{email}' is an agent attribution address"
        domain = email.rpartition("@")[2]
        for denied in DENIED_DOMAINS:
            if domain == denied or domain.endswith("." + denied):
                return f"'{domain}' is a relay identity domain"
        if "[bot]" in email:
            return f"'{email}' is a bot account"

    tokens = set(NAME_TOKEN_RE.findall(name.lower()))
    hits = tokens & set(DENIED_NAMES)
    if hits:
        return f"'{name}' names an agent or bot ({', '.join(sorted(hits))})"
    if "[bot]" in name.lower():
        return f"'{name}' is a bot account"
    return None


def check_trailers(lines):
    """Return a list of problems with the attribution trailers in a message."""
    problems = []
    for line in lines:
        match = TRAILER_RE.match(line.strip())
        if not match:
            continue
        reason = denied_identity(match.group("name"), match.group("email"))
        if reason:
            problems.append(f"drop '{line.strip()}': {reason}")
    return problems


def check_subject(subject):
    """Return a list of problems with one commit subject."""
    if not subject.strip():
        return ["empty commit subject"]
    if EXEMPT_RE.search(subject):
        return []

    # GitHub appends "(#1234)" when squash-merging; judge the authored part.
    authored = TRAILING_PR_RE.sub("", subject)
    match = SUBJECT_RE.match(authored)
    if not match:
        return [
            "subject is not Conventional Commits v1.0.0",
            "  expected: <type>(<optional scope>)<optional !>: <description>",
            f"  received: {subject}",
            f"  types:    {', '.join(sorted(TYPES))}",
        ]

    problems = []
    kind = match.group("type")
    if kind not in TYPES:
        problems.append(
            f"unknown type '{kind}'; use one of: {', '.join(sorted(TYPES))}"
        )
    description = match.group("description")
    if description.endswith("."):
        problems.append("description must not end with a period")
    if description[0].isupper() and not description.split()[0].isupper():
        problems.append("description should start lowercase unless it is a proper noun")
    if len(authored) > MAX_SUBJECT:
        problems.append(
            f"subject is {len(authored)} characters; keep it under {MAX_SUBJECT}"
        )
    return problems


def report(subject, problems, stream=sys.stderr):
    print(f"commit message rejected: {subject}", file=stream)
    for problem in problems:
        print(f"  {problem}", file=stream)
    print(
        "\nConventional Commits: https://www.conventionalcommits.org/en/v1.0.0/\n"
        "The type decides which release-notes section the change lands in.\n"
        "Add a 'Release-Notes: <Section>' trailer to override, or\n"
        "'BREAKING CHANGE: <what>' for a breaking change.\n"
        "Agent, bot, and relay attribution trailers are not kept in this\n"
        "history. GitHub re-adds them when squashing a PR whose commits carry\n"
        "them, so remove them from the branch commits.\n"
        "Bypass once with --no-verify if you know the commit is not a release entry.",
        file=stream,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("file", nargs="?", help="path to a commit message file")
    source.add_argument("--message", help="validate this subject directly")
    source.add_argument("--range", help="validate every commit in a git range")
    args = parser.parse_args()

    if args.range:
        out = subprocess.run(
            ["git", "log", "--format=%B%x1e", "--no-merges", args.range],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        failed = False
        for record in out.split("\x1e"):
            lines = [line for line in record.strip("\n").splitlines() if line.strip()]
            if not lines:
                continue
            problems = check_subject(lines[0]) + check_trailers(lines[1:])
            if problems:
                report(lines[0], problems)
                failed = True
        return 1 if failed else 0

    if args.message is not None:
        lines = args.message.splitlines() if args.message.strip() else []
    else:
        with open(args.file, encoding="utf-8") as handle:
            lines = [
                line
                for line in handle.read().splitlines()
                if not line.startswith("#")
            ]
    subject = lines[0] if lines else ""

    problems = check_subject(subject) + check_trailers(lines[1:])
    if problems:
        report(subject, problems)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
