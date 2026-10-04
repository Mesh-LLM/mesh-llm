#!/usr/bin/env python3
"""Give the repair agent a bounded index into verified failed-family evidence."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def summary(directory: Path) -> str:
    feedback = json.loads((directory / "feedback.json").read_text())
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for family in feedback["candidate_failures"]:
        family_dir = directory / family
        decoder = json.JSONDecoder()
        raw = (family_dir / "results.jsonl").read_text()
        rows = []
        while raw.strip():
            row, end = decoder.raw_decode(raw.lstrip())
            rows.append(row)
            raw = raw.lstrip()[end:]
        failed = [outcome for row in rows for outcome in row.get("outcomes", [])
                  if outcome.get("status") == "fail"]
        first = failed[0] if failed else {}
        lane = str(first.get("name", "unclassified"))
        detail = str(first.get("note", "")).strip()
        if not detail:
            for path in sorted(family_dir.rglob("*.log")):
                for line in path.read_text(errors="replace").splitlines():
                    if re.search(r"error|fail|mismatch|assert|panic", line, re.I):
                        detail = f"{path.relative_to(family_dir)}: {line.strip()}"
                        break
                if detail:
                    break
        groups[lane].append((family, detail[:300] or "see family evidence"))
    lines = ["Grouped candidate failures (first useful trace per family):"]
    for lane, entries in sorted(groups.items()):
        lines.append(f"- {lane} ({len(entries)}):")
        lines.extend(f"  {family}: {trace}" for family, trace in entries)
    lines.append("Run the affected family or a reduced reproducer first; the trusted wrapper still runs every final gate.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(Path(sys.argv[1])))
