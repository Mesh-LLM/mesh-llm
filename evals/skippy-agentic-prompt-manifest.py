#!/usr/bin/env python3
"""Build a deterministic scheduler prompt manifest from an HF-cached corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_trajectories(
    dataset_file: Path,
    sources: list[str],
    families: int,
    min_isl: int,
    max_isl: int,
    min_turns: int,
) -> list[dict[str, Any]]:
    try:
        import duckdb
    except ModuleNotFoundError as error:
        raise RuntimeError("DuckDB is required to read the cached parquet") from error
    source_placeholders = ", ".join("?" for _ in sources)
    query = f"""
        WITH eligible AS (
            SELECT
                session_id,
                source_dataset,
                messages_json,
                n_turns,
                max_isl,
                total_tokens,
                row_number() OVER (
                    PARTITION BY session_id
                    ORDER BY max_isl DESC, total_tokens DESC
                ) AS occurrence
            FROM read_parquet(?)
            WHERE max_isl >= ?
              AND max_isl < ?
              AND n_turns >= ?
              AND source_dataset IN ({source_placeholders})
        )
        SELECT
            session_id,
            source_dataset,
            messages_json,
            n_turns,
            max_isl,
            total_tokens
        FROM eligible
        WHERE occurrence = 1
        ORDER BY md5(session_id)
        LIMIT ?
    """
    parameters: list[Any] = [
        str(dataset_file),
        min_isl,
        max_isl,
        min_turns,
        *sources,
        families,
    ]
    columns = (
        "session_id",
        "source_dataset",
        "messages_json",
        "n_turns",
        "max_isl",
        "total_tokens",
    )
    rows = duckdb.execute(query, parameters).fetchall()
    if len(rows) != families:
        raise ValueError(f"selected {len(rows)} trajectories, expected {families}")
    return [dict(zip(columns, row, strict=True)) for row in rows]


def flatten_messages(messages_json: str) -> str:
    messages = json.loads(messages_json)
    if not isinstance(messages, list) or not messages:
        raise ValueError("trajectory messages_json must contain a nonempty list")
    sections = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"trajectory message {index} must be an object")
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, ensure_ascii=False)
        tool_calls = message.get("tool_calls_json")
        if tool_calls:
            content = f"{content}\n<tool_calls>{tool_calls}</tool_calls>"
        sections.append(f"<{role}>\n{content}")
    return "\n\n".join(sections)


def build_manifest(
    trajectories: list[dict[str, Any]],
    requests_per_family: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    family_prefixes = [flatten_messages(row["messages_json"]) for row in trajectories]
    prompts = []
    for request_index in range(requests_per_family):
        for family_index, prefix in enumerate(family_prefixes):
            prompts.append(
                {
                    "family": f"trajectory-{family_index}",
                    "prompt": (
                        f"{prefix}\n\n<user>\nBenchmark branch {request_index}: "
                        "summarize the latest repository state in one sentence."
                    ),
                }
            )
    metadata = {
        **metadata,
        "requests_per_family": requests_per_family,
        "rows": [
            {
                key: row[key]
                for key in (
                    "session_id",
                    "source_dataset",
                    "n_turns",
                    "max_isl",
                    "total_tokens",
                )
            }
            for row in trajectories
        ],
    }
    return {"metadata": metadata, "prompts": prompts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--requests-per-family", type=int, default=2)
    parser.add_argument("--min-isl", type=int, default=8192)
    parser.add_argument("--max-isl", type=int, default=12000)
    parser.add_argument("--min-turns", type=int, default=20)
    parser.add_argument(
        "--source-dataset",
        action="append",
        dest="sources",
        default=[],
        help="repeat to allow more than one source",
    )
    args = parser.parse_args()
    if min(args.families, args.requests_per_family, args.min_isl, args.min_turns) <= 0:
        parser.error("families, requests, minimum ISL, and minimum turns must be positive")
    if args.max_isl <= args.min_isl:
        parser.error("maximum ISL must exceed minimum ISL")
    if not args.dataset_file.is_file():
        parser.error(f"dataset parquet not found: {args.dataset_file}")
    if not args.sources:
        parser.error("at least one --source-dataset is required")

    trajectories = select_trajectories(
        args.dataset_file,
        args.sources,
        args.families,
        args.min_isl,
        args.max_isl,
        args.min_turns,
    )
    manifest = build_manifest(
        trajectories,
        args.requests_per_family,
        {
            "dataset": "thoughtworks/agentic-coding-trajectories",
            "dataset_revision": args.dataset_revision,
            "selection": {
                "sources": args.sources,
                "families": args.families,
                "min_isl": args.min_isl,
                "max_isl_exclusive": args.max_isl,
                "min_turns": args.min_turns,
                "order": "md5(session_id)",
            },
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
