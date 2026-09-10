#!/usr/bin/env python3
"""Validate and export the agentic-replay matrix parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODE_MAP = {"checkpoint": "checkpoints", "final": "final", "all": "all"}
POSITIVE_FIELDS = (
    "trajectories_per_framework",
    "passes",
    "warmup_turns",
    "max_output_tokens",
)


def load_replay(matrix_path: Path) -> tuple[dict[str, object], str, list[int]]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    replay = matrix.get("replay")
    if not isinstance(replay, dict):
        raise SystemExit("matrix replay block is missing")

    mode = replay.get("mode")
    if not isinstance(mode, str) or mode not in MODE_MAP:
        raise SystemExit(
            f"replay mode must be one of checkpoint, final, all (got {mode!r})"
        )

    for key in POSITIVE_FIELDS:
        value = replay.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SystemExit(f"{key} must be a positive integer")

    concurrency = replay.get("concurrency")
    if (
        not isinstance(concurrency, list)
        or not concurrency
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in concurrency
        )
        or len(set(concurrency)) != len(concurrency)
    ):
        raise SystemExit(
            "concurrency must be a non-empty list of unique positive integers"
        )

    return replay, MODE_MAP[mode], concurrency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--github-env", type=Path)
    parser.add_argument("--print-shell", action="store_true")
    args = parser.parse_args()

    replay, replay_mode, concurrency = load_replay(args.matrix)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(replay, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.github_env:
        with args.github_env.open("a", encoding="utf-8") as env:
            env.write(f"AGENTIC_REPLAY_MODE={replay_mode}\n")
            for key in POSITIVE_FIELDS:
                env.write(f"AGENTIC_REPLAY_{key.upper()}={replay[key]}\n")
            env.write(
                f"AGENTIC_REPLAY_CONCURRENCY={','.join(map(str, concurrency))}\n"
            )
    if args.print_shell:
        print(
            replay_mode,
            *(replay[key] for key in POSITIVE_FIELDS),
            ",".join(map(str, concurrency)),
            sep="\t",
        )


if __name__ == "__main__":
    main()
