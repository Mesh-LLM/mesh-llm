#!/usr/bin/env python3
"""Collect dependency-free GitHub Actions timing summaries."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import pathlib
import subprocess
import sys
from typing import Any


RUN_FIELDS = ",".join(
    (
        "databaseId",
        "attempt",
        "workflowName",
        "displayTitle",
        "event",
        "status",
        "conclusion",
        "createdAt",
        "startedAt",
        "updatedAt",
        "url",
        "headSha",
        "headBranch",
    )
)
SKIPPED = "skipped"


def pick(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid timestamp {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    parsed = parsed.astimezone(dt.timezone.utc)
    return parsed if parsed.year > 1970 else None


def elapsed(start: dt.datetime | None, end: dt.datetime | None) -> float | None:
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds >= 0 else None


def percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def summarize(values: list[float | None]) -> dict[str, float | int | None]:
    samples = sorted(value for value in values if value is not None)
    if not samples:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    def rounded(value: float) -> float:
        return round(value, 3)

    return {
        "count": len(samples),
        "min": rounded(samples[0]),
        "mean": rounded(sum(samples) / len(samples)),
        "p50": rounded(percentile(samples, 0.50)),
        "p90": rounded(percentile(samples, 0.90)),
        "p95": rounded(percentile(samples, 0.95)),
        "max": rounded(samples[-1]),
    }


def normalize_step(raw: dict[str, Any]) -> dict[str, Any]:
    started = timestamp(pick(raw, "started_at", "startedAt"))
    completed = timestamp(pick(raw, "completed_at", "completedAt"))
    return {
        "name": str(pick(raw, "name", default="unknown step")),
        "number": pick(raw, "number"),
        "conclusion": str(pick(raw, "conclusion", default="")),
        "started": started,
        "completed": completed,
        "duration_seconds": elapsed(started, completed),
    }


def normalize_job(raw: dict[str, Any]) -> dict[str, Any]:
    labels = pick(raw, "labels", "runner_labels", default=[])
    raw_steps = pick(raw, "steps", default=[])
    steps = []
    if isinstance(raw_steps, list):
        steps = [
            normalize_step(step)
            for step in raw_steps
            if isinstance(step, dict)
        ]
    return {
        "id": pick(raw, "id", "databaseId", "database_id"),
        "name": str(pick(raw, "name", default="unknown job")),
        "conclusion": str(pick(raw, "conclusion", default="")),
        "created": timestamp(pick(raw, "created_at", "createdAt")),
        "started": timestamp(pick(raw, "started_at", "startedAt")),
        "completed": timestamp(pick(raw, "completed_at", "completedAt")),
        "url": str(pick(raw, "html_url", "url", default="")),
        "labels": [str(label) for label in labels]
        if isinstance(labels, list)
        else [],
        "steps": steps,
    }


def normalize_run(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw.get("jobs"), list):
        run_id = pick(raw, "id", "databaseId", "database_id", default="unknown")
        raise ValueError(
            f"run {run_id} has no jobs array; use --raw-out or save "
            "'gh run view --json ...,jobs' output"
        )
    raw_attempt = pick(raw, "attempt", "run_attempt", default=1)
    try:
        attempt = int(raw_attempt)
    except (TypeError, ValueError):
        attempt = 1
    return {
        "id": pick(raw, "id", "databaseId", "database_id"),
        "attempt": max(attempt, 1),
        "workflow": str(pick(raw, "workflow_name", "workflowName", default="")),
        "title": str(pick(raw, "title", "displayTitle", default="")),
        "event": str(pick(raw, "event", default="")),
        "status": str(pick(raw, "status", default="")),
        "conclusion": str(pick(raw, "conclusion", default="")),
        "created": timestamp(pick(raw, "created_at", "createdAt")),
        "started": timestamp(pick(raw, "started_at", "startedAt")),
        "updated": timestamp(pick(raw, "updated_at", "updatedAt")),
        "url": str(pick(raw, "html_url", "url", default="")),
        "sha": str(pick(raw, "head_sha", "headSha", default="")),
        "branch": str(pick(raw, "head_branch", "headBranch", default="")),
        "jobs": [normalize_job(job) for job in raw["jobs"]],
    }


def load_runs(path: str) -> list[dict[str, Any]]:
    if path == "-":
        data = json.load(sys.stdin)
    else:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        data = data["runs"]
    elif isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(run, dict) for run in data):
        raise ValueError("input must be a run object, run array, or object.runs")
    return data


def gh_json(arguments: list[str]) -> Any:
    command = ["gh", *arguments]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("gh is required for live collection") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "unknown error"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}") from error
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{' '.join(command)} returned invalid JSON") from error


def fetch_jobs(repository: str, run_id: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    page = 1
    while True:
        response = gh_json(
            [
                "api",
                "--method",
                "GET",
                f"repos/{repository}/actions/runs/{run_id}/jobs",
                "-f",
                "filter=latest",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
            ]
        )
        if not isinstance(response, dict) or not isinstance(response.get("jobs"), list):
            raise RuntimeError(f"invalid jobs response for run {run_id}")
        page_jobs = response["jobs"]
        jobs.extend(page_jobs)
        total = response.get("total_count")
        if len(page_jobs) < 100 or (isinstance(total, int) and len(jobs) >= total):
            return jobs
        page += 1


def fetch_exact_run(repository: str, run_id: int) -> dict[str, Any]:
    run = gh_json(
        [
            "run",
            "view",
            str(run_id),
            "--repo",
            repository,
            "--json",
            RUN_FIELDS,
        ]
    )
    if not isinstance(run, dict):
        raise RuntimeError(f"invalid run response for {run_id}")
    run["jobs"] = fetch_jobs(repository, run_id)
    return run


def fetch_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.run_id:
        return [fetch_exact_run(args.repo, run_id) for run_id in args.run_id]
    command = [
        "run",
        "list",
        "--repo",
        args.repo,
        "--workflow",
        args.workflow,
        "--limit",
        str(args.limit),
        "--json",
        RUN_FIELDS,
    ]
    for flag, value in (
        ("--status", None if args.status == "all" else args.status),
        ("--branch", args.branch),
        ("--event", args.event),
        ("--created", args.created),
    ):
        if value:
            command.extend([flag, value])
    runs = gh_json(command)
    if not isinstance(runs, list):
        raise RuntimeError("invalid run list response")
    for run in runs:
        run_id = run.get("databaseId") if isinstance(run, dict) else None
        if not isinstance(run_id, int):
            raise RuntimeError("run list contains an invalid run")
        run["jobs"] = fetch_jobs(args.repo, run_id)
    return runs


def runner_dimensions(labels: list[str]) -> dict[str, str | None]:
    """Derive only dimensions encoded by the selected runner label.

    Depot labels encode architecture and vCPU size. Hosted and legacy labels
    identify a provider/architecture in the checked-in runner contract, but do
    not expose a comparable Depot size, so their runner_size remains null.
    """

    normalized = {label.lower() for label in labels}
    depot_label = next(
        (label for label in sorted(normalized) if label.startswith("depot-")),
        None,
    )
    if depot_label:
        suffix = depot_label.rsplit("-", 1)[-1]
        return {
            "provider": "depot",
            "architecture": "arm64" if "-arm" in depot_label else "amd64",
            "runner_size": suffix if suffix in {"4", "8", "16"} else "default",
        }

    if "mesh-llm-arm64" in normalized or "ubuntu-24.04-arm" in normalized:
        architecture = "arm64"
    elif "macos-15" in normalized:
        architecture = "arm64"
    elif (
        "macos-15-intel" in normalized
        or "windows-2022" in normalized
        or "ubuntu-24.04" in normalized
        or "x64" in normalized
        or "amd64" in normalized
    ):
        architecture = "amd64"
    else:
        architecture = None

    if "self-hosted" in normalized or any(
        label.startswith("mesh-llm-") for label in normalized
    ):
        provider = "self-hosted"
    elif architecture is not None or any(
        label.startswith(("ubuntu-", "macos-", "windows-"))
        for label in normalized
    ):
        provider = "github-hosted"
    else:
        provider = "unknown"

    return {
        "provider": provider,
        "architecture": architecture,
        "runner_size": None,
    }


def observation(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["id"],
        "run_attempt": run["attempt"],
        "run_url": run["url"],
        "job_id": job["id"],
        "job_url": job["url"],
        "name": job["name"],
        "conclusion": job["conclusion"],
        "duration_seconds": elapsed(job["started"], job["completed"]),
        "queue_seconds": elapsed(job["created"], job["started"]),
        "start_delay_seconds": (
            elapsed(run["created"], job["started"])
            if run["attempt"] == 1
            else None
        ),
        "runner_labels": job["labels"],
        "runner_dimensions": runner_dimensions(job["labels"]),
    }


def included(run: dict[str, Any], requested: str) -> tuple[bool, str]:
    if run["status"] != "completed":
        return False, "not_completed"
    if requested in ("all", "completed") or run["conclusion"] == requested:
        return True, ""
    return False, f"conclusion_{run['conclusion'] or 'missing'}"


def analyze(
    runs: list[dict[str, Any]],
    requested_status: str,
    top: int,
    source: dict[str, Any],
    labels: dict[str, str],
) -> dict[str, Any]:
    selected = []
    skipped = collections.Counter()
    for run in runs:
        matches, reason = included(run, requested_status)
        if matches:
            selected.append(run)
        else:
            skipped[reason] += 1
    if not selected:
        raise ValueError("no completed workflow runs matched the requested status")

    observations: list[dict[str, Any]] = []
    run_reports = []
    terminal_counts = collections.Counter()
    runner_groups: dict[
        tuple[str | None, str | None, str | None], list[dict[str, Any]]
    ] = collections.defaultdict(list)
    step_observations: list[dict[str, Any]] = []
    wall_times = []
    queue_times = []
    workflow_timing_excluded_reruns = 0
    for run in selected:
        jobs = [job for job in run["jobs"] if job["conclusion"] != SKIPPED]
        samples = [observation(run, job) for job in jobs]
        observations.extend(samples)
        for sample, job in zip(samples, jobs, strict=True):
            dimensions = sample["runner_dimensions"]
            runner_groups[
                (
                    dimensions["provider"],
                    dimensions["architecture"],
                    dimensions["runner_size"],
                )
            ].append(sample)
            for step in job["steps"]:
                step_observations.append(
                    {
                        "run_id": run["id"],
                        "job_id": job["id"],
                        "job_name": job["name"],
                        "step_name": step["name"],
                        "conclusion": step["conclusion"],
                        "duration_seconds": step["duration_seconds"],
                        "runner_dimensions": dimensions,
                    }
                )
        completed = [job for job in jobs if job["completed"] is not None]
        terminal = (
            max(completed, key=lambda job: job["completed"]) if completed else None
        )
        if terminal:
            terminal_counts[terminal["name"]] += 1
        longest = max(
            (sample for sample in samples if sample["duration_seconds"] is not None),
            key=lambda sample: sample["duration_seconds"],
            default=None,
        )
        workflow_timing_eligible = run["attempt"] == 1
        wall = (
            elapsed(run["created"], run["updated"])
            if workflow_timing_eligible
            else None
        )
        queue = (
            elapsed(run["created"], run["started"])
            if workflow_timing_eligible
            else None
        )
        if workflow_timing_eligible:
            wall_times.append(wall)
            queue_times.append(queue)
        else:
            workflow_timing_excluded_reruns += 1
        run_reports.append(
            {
                "id": run["id"],
                "attempt": run["attempt"],
                "url": run["url"],
                "workflow": run["workflow"],
                "title": run["title"],
                "event": run["event"],
                "conclusion": run["conclusion"],
                "head_sha": run["sha"],
                "head_branch": run["branch"],
                "wall_seconds": wall,
                "queue_seconds": queue,
                "workflow_timing_excluded": not workflow_timing_eligible,
                "executed_job_count": len(jobs),
                "skipped_job_count": len(run["jobs"]) - len(jobs),
                "terminal_job": terminal["name"] if terminal else None,
                "longest_job": longest["name"] if longest else None,
                "longest_job_seconds": longest["duration_seconds"] if longest else None,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for sample in observations:
        groups[sample["name"]].append(sample)
    by_name = [
        {
            "name": name,
            "sample_count": len(samples),
            "duration_seconds": summarize(
                [sample["duration_seconds"] for sample in samples]
            ),
            "queue_seconds": summarize(
                [sample["queue_seconds"] for sample in samples]
            ),
            "start_delay_seconds": summarize(
                [sample["start_delay_seconds"] for sample in samples]
            ),
            "terminal_count": terminal_counts[name],
            "conclusions": dict(
                sorted(collections.Counter(s["conclusion"] for s in samples).items())
            ),
        }
        for name, samples in groups.items()
    ]
    by_name.sort(
        key=lambda item: (
            item["duration_seconds"]["p95"] or -1,
            item["duration_seconds"]["mean"] or -1,
        ),
        reverse=True,
    )
    slowest = sorted(
        (
            sample
            for sample in observations
            if sample["duration_seconds"] is not None
        ),
        key=lambda sample: sample["duration_seconds"],
        reverse=True,
    )[:top]
    critical = [
        {
            "name": name,
            "terminal_count": count,
            "share": round(count / len(selected), 4),
        }
        for name, count in terminal_counts.most_common(top)
    ]
    by_runner = []
    for (provider, architecture, runner_size), samples in runner_groups.items():
        by_runner.append(
            {
                "provider": provider,
                "architecture": architecture,
                "runner_size": runner_size,
                "sample_count": len(samples),
                "job_names": sorted({sample["name"] for sample in samples}),
                "duration_seconds": summarize(
                    [sample["duration_seconds"] for sample in samples]
                ),
                "queue_seconds": summarize(
                    [sample["queue_seconds"] for sample in samples]
                ),
            }
        )
    by_runner.sort(
        key=lambda item: (
            item["duration_seconds"]["p95"] or -1,
            item["duration_seconds"]["mean"] or -1,
        ),
        reverse=True,
    )

    step_groups: dict[
        tuple[
            str,
            str,
            str | None,
            str | None,
            str | None,
        ],
        list[dict[str, Any]],
    ] = collections.defaultdict(list)
    for sample in step_observations:
        dimensions = sample["runner_dimensions"]
        step_groups[
            (
                sample["job_name"],
                sample["step_name"],
                dimensions["provider"],
                dimensions["architecture"],
                dimensions["runner_size"],
            )
        ].append(sample)
    by_step = [
        {
            "job_name": job_name,
            "step_name": step_name,
            "provider": provider,
            "architecture": architecture,
            "runner_size": runner_size,
            "sample_count": len(samples),
            "duration_seconds": summarize(
                [sample["duration_seconds"] for sample in samples]
            ),
            "conclusions": dict(
                sorted(collections.Counter(s["conclusion"] for s in samples).items())
            ),
        }
        for (
            job_name,
            step_name,
            provider,
            architecture,
            runner_size,
        ), samples in step_groups.items()
    ]
    by_step.sort(
        key=lambda item: (
            item["duration_seconds"]["p95"] or -1,
            item["duration_seconds"]["mean"] or -1,
        ),
        reverse=True,
    )
    return {
        "schema_version": 2,
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": source,
        "benchmark_labels": labels,
        "definitions": {
            "workflow_wall_seconds": (
                "first-attempt run created_at to updated_at; reruns excluded"
            ),
            "workflow_queue_seconds": (
                "first-attempt run created_at to started_at; reruns excluded"
            ),
            "job_duration_seconds": "job started_at to completed_at",
            "job_queue_seconds": (
                "job created_at to started_at; unavailable without job created_at"
            ),
            "job_start_delay_seconds": (
                "first-attempt workflow created_at to job started_at; includes "
                "dependency wait; reruns excluded"
            ),
            "terminal_job": (
                "last non-skipped job to finish; a critical-path candidate"
            ),
            "runner_dimensions": (
                "provider, architecture, and Depot runner size derived from labels"
            ),
            "step_duration_seconds": "step started_at to completed_at",
        },
        "selection": {
            "requested_status": requested_status,
            "seen_run_count": len(runs),
            "included_run_count": len(selected),
            "workflow_timing_excluded_reruns": workflow_timing_excluded_reruns,
            "skipped_runs": dict(sorted(skipped.items())),
        },
        "workflow": {
            "wall_seconds": summarize(wall_times),
            "queue_seconds": summarize(queue_times),
        },
        "jobs": {
            "sample_count": len(observations),
            "duration_seconds": summarize(
                [sample["duration_seconds"] for sample in observations]
            ),
            "queue_seconds": summarize(
                [sample["queue_seconds"] for sample in observations]
            ),
            "start_delay_seconds": summarize(
                [sample["start_delay_seconds"] for sample in observations]
            ),
            "by_name": by_name,
            "by_runner": by_runner,
            "critical_finish_candidates": critical,
            "slowest_observations": slowest,
        },
        "steps": {
            "sample_count": len(step_observations),
            "by_name": by_step,
        },
        "runs": run_reports,
    }


def human(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any], top: int) -> str:
    workflow = report["workflow"]
    jobs = report["jobs"]
    lines = [
        "# CI timing summary",
        "",
        (
            f"Analyzed **{report['selection']['included_run_count']}** completed "
            f"run(s) from `{markdown_escape(report['source']['description'])}`."
        ),
        "",
    ]
    excluded_reruns = report["selection"]["workflow_timing_excluded_reruns"]
    if excluded_reruns:
        lines += [
            (
                f"> Excluded workflow wall, workflow queue, and job start-delay "
                f"timing from **{excluded_reruns}** rerun attempt(s). GitHub "
                "retains the original run timestamps when jobs are rerun."
            ),
            "",
        ]
    lines += [
        "## Workflow timing",
        "",
        "| Timing | Samples | p50 | p90 | p95 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("Wall time", "wall_seconds"), ("Queue", "queue_seconds")):
        stats = workflow[key]
        lines.append(
            f"| {label} | {stats['count']} | {human(stats['p50'])} | "
            f"{human(stats['p90'])} | {human(stats['p95'])} | "
            f"{human(stats['max'])} |"
        )
    lines += [
        "",
        "## Runner dimensions",
        "",
        "| Provider | Architecture | Depot size | Jobs | Duration p50 | Duration p95 | Queue p95 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in jobs["by_runner"][:top]:
        lines.append(
            f"| {markdown_escape(item['provider'])} | "
            f"{markdown_escape(item['architecture'] or 'n/a')} | "
            f"{markdown_escape(item['runner_size'] or 'n/a')} | "
            f"{item['sample_count']} | "
            f"{human(item['duration_seconds']['p50'])} | "
            f"{human(item['duration_seconds']['p95'])} | "
            f"{human(item['queue_seconds']['p95'])} |"
        )
    lines += [
        "",
        "## Step timing",
        "",
    ]
    if report["steps"]["by_name"]:
        lines += [
            "| Provider | Architecture | Runner size | Job | Step | Samples | Duration p50 | Duration p95 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for item in report["steps"]["by_name"][:top]:
            lines.append(
                f"| {markdown_escape(item['provider'] or 'n/a')} | "
                f"{markdown_escape(item['architecture'] or 'n/a')} | "
                f"{markdown_escape(item['runner_size'] or 'n/a')} | "
                f"{markdown_escape(item['job_name'])} | "
                f"{markdown_escape(item['step_name'])} | {item['sample_count']} | "
                f"{human(item['duration_seconds']['p50'])} | "
                f"{human(item['duration_seconds']['p95'])} |"
            )
    else:
        lines.append(
            "No job step timestamps were available; cache, context-upload, and "
            "export/import phases are not inferred from logs."
        )
    lines += [
        "",
        "## Slow job families",
        "",
        "| Job | Samples | Duration p50 | Duration p95 | Queue p50 | Queue p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in jobs["by_name"][:top]:
        duration, queue = item["duration_seconds"], item["queue_seconds"]
        lines.append(
            f"| {markdown_escape(item['name'])} | {item['sample_count']} | "
            f"{human(duration['p50'])} | {human(duration['p95'])} | "
            f"{human(queue['p50'])} | {human(queue['p95'])} |"
        )
    lines += [
        "",
        "## Terminal jobs (critical-path candidates)",
        "",
        "| Job | Runs finishing last | Share |",
        "| --- | ---: | ---: |",
    ]
    for item in jobs["critical_finish_candidates"][:top]:
        lines.append(
            f"| {markdown_escape(item['name'])} | {item['terminal_count']} | "
            f"{item['share']:.1%} |"
        )
    lines += [
        "",
        "## Slowest observations",
        "",
        "| Run | Job | Duration | Queue |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in jobs["slowest_observations"][:top]:
        run_id = item["run_id"]
        run = f"[{run_id}]({item['run_url']})" if item["run_url"] else str(run_id)
        lines.append(
            f"| {run} | {markdown_escape(item['name'])} | "
            f"{human(item['duration_seconds'])} | {human(item['queue_seconds'])} |"
        )
    lines += [
        "",
        (
            "_Job queue uses GitHub's job creation-to-start interval. Offline "
            "`gh run view` data may omit creation times; those samples are n/a. "
            "Workflow and start-delay timing excludes rerun attempts because "
            "their run timestamps belong to the original attempt. Terminal jobs "
            "are candidates only; the DAG is not reconstructed._"
        ),
        "",
    ]
    return "\n".join(lines)


def write(path: str, content: str) -> None:
    if path == "-":
        sys.stdout.write(content)
        return
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def labels(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, label = value.partition("=")
        if not key or not separator:
            raise ValueError(f"benchmark label must be KEY=VALUE, got {value!r}")
        result[key] = label
    return dict(sorted(result.items()))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect read-only GitHub Actions timing and runner metrics."
    )
    parser.add_argument("--repo", default="Mesh-LLM/mesh-llm")
    parser.add_argument("--workflow")
    parser.add_argument("--run-id", type=int, action="append", default=[])
    parser.add_argument("--input", help="Detailed run JSON, --raw-out JSON, or -")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--status", default="success")
    parser.add_argument("--branch")
    parser.add_argument("--event")
    parser.add_argument("--created", help="GitHub date filter, e.g. >=2026-07-01")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--raw-out", help="Save detailed inputs for offline analysis")
    args = parser.parse_args(argv)
    if args.input and (args.workflow or args.run_id):
        parser.error("--input cannot be combined with --workflow or --run-id")
    if args.workflow and args.run_id:
        parser.error("--workflow cannot be combined with --run-id")
    if not args.input and not args.workflow and not args.run_id:
        parser.error("one of --input, --workflow, or --run-id is required")
    if args.limit < 1 or args.top < 1:
        parser.error("--limit and --top must be at least 1")
    outputs = (args.json_out, args.markdown_out, args.raw_out)
    if sum(path == "-" for path in outputs) > 1:
        parser.error("only one output may use stdout (-)")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.input:
            raw_runs = load_runs(args.input)
            source = {"kind": "file", "description": args.input}
        else:
            raw_runs = fetch_runs(args)
            target = ",".join(map(str, args.run_id)) if args.run_id else args.workflow
            source = {
                "kind": "github",
                "description": f"{args.repo}:{target}",
                "repository": args.repo,
                "workflow": args.workflow,
            }
        runs = [normalize_run(run) for run in raw_runs]
        report = analyze(runs, args.status, args.top, source, labels(args.label))
        if args.raw_out:
            write(
                args.raw_out,
                json.dumps({"schema_version": 1, "runs": raw_runs}, indent=2) + "\n",
            )
        if args.json_out:
            write(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
        summary = render_markdown(report, args.top)
        if args.markdown_out:
            write(args.markdown_out, summary)
        if not args.json_out and not args.markdown_out:
            sys.stdout.write(summary)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ci metrics error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
