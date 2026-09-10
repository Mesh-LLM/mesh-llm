#!/usr/bin/env python3
"""Regroup a GitHub release body into Keep a Changelog sections.

Entry lines are moved verbatim: the subject, the author, and the PR link are
never rewritten, so contributor credit survives the reformat. The script fails
loudly when a plan would drop, duplicate, or invent an entry.

Usage:
    regroup_release_notes.py --body BODY.md --list
    regroup_release_notes.py --body BODY.md --plan PLAN.json --out NEW.md
    regroup_release_notes.py --body BODY.md --plan PLAN.json --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys

ENTRY_RE = re.compile(r"^\* .*/pull/(\d+)\s*$")
TAIL_RE = re.compile(r"^(## New Contributors|\*\*Full Changelog\*\*)")
DEFAULT_INTRO = (
    "The notable changes in this release, grouped per "
    "[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)."
)


def parse_body(text):
    """Split a release body into (entries, ordered PRs, trailing lines)."""
    lines = text.splitlines()
    tail_at = next(
        (i for i, line in enumerate(lines) if TAIL_RE.match(line)), len(lines)
    )
    entries, order = {}, []
    for line in lines[:tail_at]:
        match = ENTRY_RE.match(line)
        if not match:
            continue
        pr = int(match.group(1))
        if pr in entries:
            sys.exit(f"error: PR #{pr} appears twice in the source body")
        entries[pr] = line
        order.append(pr)
    if not entries:
        sys.exit("error: no '* ... /pull/<n>' entry lines found in the body")
    return entries, order, lines[tail_at:]


def plan_groups(plan):
    """Yield (bucket, group_title, prs) for every group in a plan."""
    for section in plan.get("sections", []):
        title = section["title"]
        if "prs" in section:
            yield title, None, section["prs"]
        for group in section.get("groups", []):
            yield title, group["title"], group["prs"]
    internal = plan.get("internal")
    if internal:
        for group in internal.get("groups", []):
            yield "Internal", group["title"], group["prs"]


def validate(plan, entries, order):
    assigned, seen, dupes = [], set(), []
    for _, _, prs in plan_groups(plan):
        for pr in prs:
            if pr in seen:
                dupes.append(pr)
            seen.add(pr)
            assigned.append(pr)
    missing = [pr for pr in order if pr not in seen]
    unknown = [pr for pr in assigned if pr not in entries]

    problems = []
    if dupes:
        problems.append("assigned to more than one section:")
        problems += [f"  #{pr}  {entries.get(pr, '(unknown PR)')}" for pr in dupes]
    if missing:
        problems.append("missing from the plan:")
        problems += [f"  #{pr}  {entries[pr]}" for pr in missing]
    if unknown:
        problems.append("in the plan but not in the release body:")
        problems += [f"  #{pr}" for pr in unknown]
    if problems:
        sys.exit("error: plan does not cover the body exactly\n" + "\n".join(problems))
    return len(assigned)


def render(plan, entries, tail):
    out = [plan.get("intro", DEFAULT_INTRO), ""]
    if plan.get("version"):
        heading = f"## [{plan['version']}]"
        if plan.get("date"):
            heading += f" - {plan['date']}"
        out += [heading, ""]

    for section in plan.get("sections", []):
        out += [f"### {section['title']}", ""]
        if "prs" in section:
            out += [entries[pr] for pr in section["prs"]] + [""]
        for group in section.get("groups", []):
            out += [f"#### {group['title']}", ""]
            out += [entries[pr] for pr in group["prs"]] + [""]

    internal = plan.get("internal")
    if internal:
        count = sum(len(g["prs"]) for g in internal["groups"])
        out += ["### Internal", "", "<details>"]
        out += [f"<summary>{internal['summary']} ({count} changes)</summary>", ""]
        for group in internal["groups"]:
            out += [f"#### {group['title']}", ""]
            out += [entries[pr] for pr in group["prs"]] + [""]
        out += ["</details>", ""]

    out += tail
    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", required=True, help="release body markdown")
    parser.add_argument("--plan", help="JSON plan mapping PR numbers to sections")
    parser.add_argument("--out", help="write the regrouped body here")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every entry as '<pr>\\t<subject>' and exit",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate the plan without rendering"
    )
    args = parser.parse_args()

    entries, order, tail = parse_body(open(args.body, encoding="utf-8").read())

    if args.list:
        for pr in order:
            subject = entries[pr][2:].split(" by @")[0]
            print(f"{pr}\t{subject}")
        print(f"\n{len(order)} entries", file=sys.stderr)
        return

    if not args.plan:
        parser.error("--plan is required unless --list is given")
    plan = json.load(open(args.plan, encoding="utf-8"))
    count = validate(plan, entries, order)

    if args.check:
        print(f"ok: plan covers all {count} entries exactly once")
        return
    if not args.out:
        parser.error("--out is required unless --check is given")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(render(plan, entries, tail))
    print(f"ok: regrouped {count} entries -> {args.out}")


if __name__ == "__main__":
    main()
