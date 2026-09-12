#!/usr/bin/env python3
"""Link release commits to the pull requests that carried them.

The classifier reaches a commit through the "(#N)" suffix that a squash merge
leaves on the subject, and GitHub's generated body credits one entry per pull
request merged into the release branch. Two routes break that pairing:

  * a commit that reached the branch without the suffix has no link back to
    its entry, so the classifier cannot read its type and a perfectly ordinary
    "test(ci): ..." lands in "Other changes";
  * a batch of pull requests merged into a staging branch that then reached
    the release by rebase is credited once, to the roll-up pull request, and
    every fix inside it disappears from the notes. v0.76.1 shipped sixteen
    fixes behind a single "Fixed" entry that way.

This step repairs both before classification. It resolves the missing links
through the GitHub API, writes an augmented body that credits the recovered
pull requests, and hands the classifier the commit records git alone could not
key. It only adds entries: an entry GitHub published is never dropped or
rewritten here.

Every API call is best-effort. A pull request it cannot resolve keeps the
behaviour it has today rather than failing the release.

Usage:
    release-notes-link.py --body BODY.md --range v0.76.0..v0.76.1 \
        --repo Mesh-LLM/mesh-llm --out-body body.linked.md --out-links links.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import OrderedDict

BODY_ENTRY_RE = re.compile(r"^\* .*/pull/(\d+)\s*$")
PR_SUFFIX_RE = re.compile(r"\s*\(#(\d+)\)$")
TRAILER_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z -]*):\s*(?P<value>.+)$")
TAIL_RE = re.compile(r"^(## New Contributors|\*\*Full Changelog\*\*)")

# A release with hundreds of entries must not turn into hundreds of API calls.
# Past the budget the remaining commits keep the behaviour they have today.
DEFAULT_API_BUDGET = 200


class Gh:
    """Best-effort `gh` caller with a call budget and a per-call timeout."""

    def __init__(self, budget=DEFAULT_API_BUDGET, timeout=30):
        self.remaining = budget
        self.timeout = timeout
        self.exhausted = False
        self.failures = 0

    def json(self, args):
        """Run `gh` and return parsed JSON, or None if it did not work out."""
        if self.remaining <= 0:
            if not self.exhausted:
                print(
                    "release-notes-link: API budget exhausted; "
                    "remaining commits keep their published entries",
                    file=sys.stderr,
                )
            self.exhausted = True
            return None
        self.remaining -= 1
        try:
            result = subprocess.run(
                ["gh", *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True,
            )
            return json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.failures += 1
            print(f"release-notes-link: gh {' '.join(args)} failed: {error}", file=sys.stderr)
            return None


def read_body(path):
    """Return (lines, credited PRs in order, index just past the last entry)."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    tail_at = next((i for i, line in enumerate(lines) if TAIL_RE.match(line)), len(lines))
    credited, last_entry = [], None
    for index, line in enumerate(lines[:tail_at]):
        match = BODY_ENTRY_RE.match(line)
        if match:
            credited.append(int(match.group(1)))
            last_entry = index
    return lines, credited, tail_at if last_entry is None else last_entry + 1


def read_commits(git_range, repo_root=None):
    """Return the range's commits, oldest first, as {sha, subject, trailers}."""
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H%x1f%s%x1f%b%x1e", git_range],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    commits = []
    for record in result.stdout.split("\x1e"):
        if not record.strip():
            continue
        sha, _, rest = record.strip("\n").partition("\x1f")
        subject, _, body = rest.partition("\x1f")
        trailers = {}
        for line in body.splitlines():
            found = TRAILER_RE.match(line.strip())
            if found:
                trailers[found.group("key").strip().lower()] = found.group("value").strip()
            if line.strip().startswith("BREAKING CHANGE:"):
                trailers["breaking change"] = line.split(":", 1)[1].strip()
        commits.append(
            {"sha": sha.strip(), "subject": subject.strip(), "trailers": trailers}
        )
    return commits


def resolve_pull_requests(commits, repo, gh):
    """Give every commit a pull request number where one can be established.

    The "(#N)" suffix is authoritative and free, so only the commits without
    one cost an API call. A commit the API cannot place is left unlinked
    rather than guessed at.
    """
    for commit in commits:
        match = PR_SUFFIX_RE.search(commit["subject"])
        if match:
            commit["pr"] = int(match.group(1))
            continue
        commit["pr"] = None
        numbers = gh.json(
            ["api", f"repos/{repo}/commits/{commit['sha']}/pulls", "--jq", "[.[].number]"]
        )
        if numbers:
            commit["pr"] = int(numbers[0])
            commit["linked_by_api"] = True
    return sum(1 for commit in commits if commit.get("linked_by_api"))


def entry_line(repo, pr, title, author):
    return f"* {title} by @{author} in https://github.com/{repo}/pull/{pr}"


def build_entry(pr, commit, repo, gh):
    """Render the entry GitHub would have published for an uncredited PR."""
    details = gh.json(["pr", "view", str(pr), "--repo", repo, "--json", "title,author"]) or {}
    author = (details.get("author") or {}).get("login")
    if not author:
        print(
            f"release-notes-link: cannot credit #{pr} without its author; skipping it",
            file=sys.stderr,
        )
        return None
    title = details.get("title") or PR_SUFFIX_RE.sub("", commit["subject"]).strip()
    return entry_line(repo, pr, title, author)


def recover_entries(commits, credited, repo, gh):
    """Return (insertions, trailing) entry lines for uncredited pull requests.

    An uncredited pull request is attributed to the next credited one in commit
    order: that is the roll-up that carried it into the release, and the
    recovered entries read best directly beneath it. Anything left over at the
    end of the range had no carrier and is appended instead, so a pull request
    is never dropped for want of somewhere tidy to put it.
    """
    known, seen = set(credited), set()
    insertions, pending = OrderedDict(), []
    for commit in commits:
        pr = commit.get("pr")
        if pr is None or pr in seen:
            continue
        seen.add(pr)
        if pr in known:
            if pending:
                insertions.setdefault(pr, []).extend(pending)
                pending = []
            continue
        line = build_entry(pr, commit, repo, gh)
        if line:
            pending.append(line)
    return insertions, pending


def augment(lines, insert_at, insertions, trailing):
    """Splice recovered entries into the published body."""
    out, leftover = [], list(trailing)
    for index, line in enumerate(lines):
        if index == insert_at and leftover:
            out.extend(leftover)
            leftover = []
        out.append(line)
        if index < insert_at:
            match = BODY_ENTRY_RE.match(line)
            if match:
                out.extend(insertions.get(int(match.group(1)), []))
    out.extend(leftover)
    return "\n".join(out).rstrip() + "\n"


def commit_links(commits):
    """Commit records the classifier cannot key from the subject alone."""
    links = {}
    for commit in commits:
        if not commit.get("linked_by_api"):
            continue
        links.setdefault(
            str(commit["pr"]),
            {"subject": commit["subject"], "trailers": commit["trailers"]},
        )
    return links


def run(args):
    lines, credited, insert_at = read_body(args.body)
    commits = read_commits(args.range, args.repo_root)
    gh = Gh(budget=args.api_budget)

    linked = resolve_pull_requests(commits, args.repo, gh)
    insertions, trailing = recover_entries(commits, credited, args.repo, gh)
    recovered = sum(len(entries) for entries in insertions.values()) + len(trailing)

    with open(args.out_body, "w", encoding="utf-8") as handle:
        handle.write(augment(lines, insert_at, insertions, trailing))
    with open(args.out_links, "w", encoding="utf-8") as handle:
        json.dump(commit_links(commits), handle, indent=2)
        handle.write("\n")

    print(
        f"linked {linked} commit(s) to a pull request through the API; "
        f"recovered {recovered} entry(ies) GitHub did not credit "
        f"({len(credited)} published)"
    )
    if gh.failures:
        print(f"release-notes-link: {gh.failures} API call(s) failed", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", required=True, help="published release body")
    parser.add_argument("--range", required=True, help="git range, e.g. v0.76.0..v0.76.1")
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--out-body", required=True, help="write the augmented body here")
    parser.add_argument("--out-links", required=True, help="write the commit records here")
    parser.add_argument("--api-budget", type=int, default=DEFAULT_API_BUDGET)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
