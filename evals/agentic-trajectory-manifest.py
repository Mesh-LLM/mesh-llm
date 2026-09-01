#!/usr/bin/env python3
"""Select deterministic, disjoint, ordered agent trajectories from parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def assistant_turn_count(messages: Sequence[dict[str, Any]]) -> int:
    return sum(message.get("role") == "assistant" for message in messages)


def validate_messages(messages_json: str, session_id: str) -> list[dict[str, Any]]:
    messages = json.loads(messages_json)
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"trajectory {session_id} has no messages")
    validated: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"trajectory {session_id} message {index} is not an object")
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(
                f"trajectory {session_id} message {index} has unsupported role {role!r}"
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError(
                f"trajectory {session_id} message {index} has non-string content"
            )
        tool_calls_json = message.get("tool_calls_json")
        if tool_calls_json:
            tool_calls = json.loads(tool_calls_json)
            if not isinstance(tool_calls, list):
                raise ValueError(
                    f"trajectory {session_id} message {index} tool calls are not a list"
                )
        validated.append(
            {
                "role": role,
                "content": content or "",
                "tool_calls_json": tool_calls_json,
                "tool_call_id": message.get("tool_call_id"),
            }
        )
    if assistant_turn_count(validated) == 0:
        raise ValueError(f"trajectory {session_id} has no assistant turns")
    return validated


def build_cohorts(
    rows: Iterable[dict[str, Any]],
    cohort_names: Sequence[str],
    frameworks: Sequence[str],
    trajectories_per_framework: int,
) -> dict[str, list[dict[str, Any]]]:
    if not cohort_names or not frameworks or trajectories_per_framework <= 0:
        raise ValueError("cohorts, frameworks, and trajectory count must be positive")
    cohorts = {name: [] for name in cohort_names}
    by_framework = {framework: [] for framework in frameworks}
    required = len(cohort_names) * trajectories_per_framework
    seen: set[str] = set()
    for row in rows:
        session_id = row["session_id"]
        framework = row["agent_framework"]
        if (
            framework not in by_framework
            or len(by_framework[framework]) >= required
            or session_id in seen
        ):
            continue
        seen.add(session_id)
        messages = validate_messages(row["messages_json"], session_id)
        turns = assistant_turn_count(messages)
        by_framework[framework].append(
            {
                "session_id": session_id,
                "source_dataset": row["source_dataset"],
                "agent_framework": row["agent_framework"],
                "recorded_model": row["recorded_model"],
                "n_turns": row["n_turns"],
                "max_isl": row["max_isl"],
                "total_tokens": row["total_tokens"],
                "assistant_turns": turns,
                "messages": messages,
            }
        )
        if all(len(by_framework[item]) >= required for item in frameworks):
            break
    for framework in frameworks:
        available = by_framework[framework]
        if len(available) < required:
            raise ValueError(
                f"framework {framework} has {len(available)} eligible trajectories, "
                f"but {required} are required"
            )
        for cohort_index, cohort in enumerate(cohort_names):
            start = cohort_index * trajectories_per_framework
            end = start + trajectories_per_framework
            cohorts[cohort].extend(available[start:end])
    return cohorts


def select_rows(
    dataset_file: Path,
    sources: Sequence[str],
    min_isl: int,
    max_isl: int,
    min_turns: int,
) -> Iterable[dict[str, Any]]:
    try:
        import duckdb
    except ModuleNotFoundError as error:
        raise RuntimeError("DuckDB is required to read the cached parquet") from error
    placeholders = ", ".join("?" for _ in sources)
    query = f"""
        WITH eligible AS (
            SELECT
                session_id,
                source_dataset,
                agent_framework,
                recorded_model,
                messages_json,
                n_turns,
                max_isl,
                total_tokens,
                row_number() OVER (
                    PARTITION BY session_id
                    ORDER BY max_isl DESC, total_tokens DESC, md5(messages_json)
                ) AS occurrence
            FROM read_parquet(?)
            WHERE max_isl >= ?
              AND max_isl < ?
              AND n_turns >= ?
              AND source_dataset IN ({placeholders})
        )
        SELECT
            session_id,
            source_dataset,
            agent_framework,
            recorded_model,
            messages_json,
            n_turns,
            max_isl,
            total_tokens
        FROM eligible
        WHERE occurrence = 1
        ORDER BY agent_framework, md5(session_id)
    """
    columns = (
        "session_id",
        "source_dataset",
        "agent_framework",
        "recorded_model",
        "messages_json",
        "n_turns",
        "max_isl",
        "total_tokens",
    )
    cursor = duckdb.execute(
        query,
        [str(dataset_file), min_isl, max_isl, min_turns, *sources],
    )
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        yield dict(zip(columns, row))


def manifest_document(
    cohorts: dict[str, list[dict[str, Any]]], metadata: dict[str, Any]
) -> dict[str, Any]:
    cohort_metadata = {}
    for name, trajectories in cohorts.items():
        framework_trajectories: dict[str, int] = {}
        framework_turns: dict[str, int] = {}
        for item in trajectories:
            framework = item["agent_framework"]
            framework_trajectories[framework] = framework_trajectories.get(framework, 0) + 1
            framework_turns[framework] = framework_turns.get(framework, 0) + item[
                "assistant_turns"
            ]
        cohort_metadata[name] = {
            "trajectory_count": len(trajectories),
            "assistant_turns": sum(item["assistant_turns"] for item in trajectories),
            "framework_trajectories": framework_trajectories,
            "framework_assistant_turns": framework_turns,
            "session_ids_sha256": hashlib.sha256(
                "\n".join(item["session_id"] for item in trajectories).encode()
            ).hexdigest(),
        }
    return {
        "schema_version": 1,
        "metadata": {**metadata, "cohorts": cohort_metadata},
        "cohorts": cohorts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", action="append", required=True)
    parser.add_argument("--framework", action="append", required=True)
    parser.add_argument("--trajectories-per-framework", type=int, required=True)
    parser.add_argument("--min-isl", type=int, default=8192)
    parser.add_argument("--max-isl", type=int, default=16384)
    parser.add_argument("--min-turns", type=int, default=5)
    parser.add_argument("--source-dataset", action="append", dest="sources", default=[])
    args = parser.parse_args()
    if args.trajectories_per_framework <= 0 or args.min_isl <= 0 or args.min_turns <= 0:
        parser.error("trajectory count, minimum ISL, and minimum turns must be positive")
    if args.max_isl <= args.min_isl:
        parser.error("maximum ISL must exceed minimum ISL")
    if len(set(args.cohort)) != len(args.cohort) or len(set(args.framework)) != len(
        args.framework
    ):
        parser.error("cohort and framework names must be unique")
    if not args.sources:
        parser.error("at least one --source-dataset is required")
    if not args.dataset_file.is_file():
        parser.error(f"dataset parquet not found: {args.dataset_file}")

    cohorts = build_cohorts(
        select_rows(
            args.dataset_file,
            args.sources,
            args.min_isl,
            args.max_isl,
            args.min_turns,
        ),
        args.cohort,
        args.framework,
        args.trajectories_per_framework,
    )
    document = manifest_document(
        cohorts,
        {
            "dataset": "thoughtworks/agentic-coding-trajectories",
            "dataset_revision": args.dataset_revision,
            "selection": {
                "sources": args.sources,
                "frameworks": args.framework,
                "trajectories_per_framework_per_cohort": args.trajectories_per_framework,
                "min_isl": args.min_isl,
                "max_isl_exclusive": args.max_isl,
                "min_turns": args.min_turns,
                "order": "agent_framework, md5(session_id)",
                "whole_trajectories": True,
                "cohorts_disjoint": True,
            },
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
