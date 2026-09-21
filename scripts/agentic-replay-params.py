#!/usr/bin/env python3
"""Validate and export the agentic-replay matrix parameters."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODE_MAP = {"all": "all"}
POSITIVE_FIELDS = (
    "sessions_per_concurrency",
    "minimum_worker_waves",
    "minimum_context_tokens",
    "minimum_session_prompt_tokens",
    "min_isl",
    "max_isl",
    "min_turns",
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
        raise SystemExit(f"replay mode must be all (got {mode!r})")

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

    if replay["sessions_per_concurrency"] < replay["minimum_worker_waves"] * max(
        concurrency
    ):
        raise SystemExit("session count does not cover the required worker waves")
    if replay["sessions_per_concurrency"] < 3:
        raise SystemExit("session count must cover all three frameworks")
    if replay["minimum_context_tokens"] < 131072:
        raise SystemExit("nightly requires at least 128K effective context")
    if replay["max_isl"] <= replay["min_isl"]:
        raise SystemExit("invalid selection window")
    if replay.get("temperature") != 0 or replay.get("seed") != 42:
        raise SystemExit("replay sampling must be pinned to temperature 0 and seed 42")
    if (
        replay.get("backend") != "metal"
        or replay.get("selection_algorithm") != "balanced-md5-v2"
    ):
        raise SystemExit("unsupported backend or selection algorithm")
    return replay, MODE_MAP[mode], concurrency


def replay_command(matrix_path, family, refs, dataset_file, output):
    replay, mode, concurrency = load_replay(matrix_path)
    matrix = json.loads(matrix_path.read_text())
    matches = [model for model in matrix["models"] if model["family"] == family]
    if len(matches) != 1:
        raise ValueError("family must identify exactly one pinned model")
    model = matches[0]
    if model.get("native_context_tokens", 0) < replay["minimum_context_tokens"]:
        raise ValueError("model native context is below the required window")
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "evals/agentic-replay.py"),
        "run",
        "--model",
        f"{model['repo']}@{model['revision']}/{model['file']}",
        "--backend",
        replay["backend"],
        "--replay-mode",
        mode,
        "--expected-model-sha256",
        model["sha256"],
        "--dataset-file",
        str(dataset_file),
        "--output",
        str(output),
    ]
    for ref in refs:
        command.extend(("--ref", ref))
    for key in POSITIVE_FIELDS:
        command.extend(("--" + key.replace("_", "-"), str(replay[key])))
    for level in concurrency:
        command.extend(("--concurrency", str(level)))
    if model["class"] == "hybrid-recurrent":
        command.append("--require-recurrent-restores")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--github-env", type=Path)
    parser.add_argument("--print-shell", action="store_true")
    parser.add_argument("--run-family")
    parser.add_argument("--ref", action="append", default=[])
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--output", type=Path)
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
            env.write(f"AGENTIC_REPLAY_CONCURRENCY={','.join(map(str, concurrency))}\n")
    if args.run_family:
        if not args.ref or args.dataset_file is None or args.output is None:
            parser.error("--run-family needs --ref, --dataset-file and --output")
        subprocess.run(
            replay_command(
                args.matrix, args.run_family, args.ref, args.dataset_file, args.output
            ),
            check=True,
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
