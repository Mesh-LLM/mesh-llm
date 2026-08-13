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
QUEUE_WARN_SECONDS = 60.0
QUEUE_CONTAMINATION_SECONDS = 300.0
MIN_HEURISTIC_SAMPLES = 3


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


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


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
        # The jobs API does not expose the time at which `needs` became ready.
        # Preserve an optional instrumented value without treating an absent
        # value as zero dependency wait.
        "dependency_ready": timestamp(
            pick(
                raw,
                "dependency_ready_at",
                "needs_completed_at",
                "ready_at",
            )
        ),
        "dependency_wait_seconds": _number(
            pick(raw, "dependency_wait_seconds", "needs_wait_seconds")
        ),
        "needs": (
            [str(item) for item in pick(raw, "needs", default=[])]
            if isinstance(pick(raw, "needs", default=[]), list)
            else []
        ),
        "runner_role": pick(raw, "runner_role", "role"),
        "operating_system": pick(
            raw,
            "operating_system",
            "runner_os",
            "os",
        ),
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
        "plan_profile": pick(raw, "plan_profile", "profile"),
        "plan_digest": pick(raw, "plan_digest", "planDigest"),
        "change_class": pick(raw, "change_class", "changeClass"),
        "runner_image": pick(raw, "runner_image", "runnerImage"),
        "toolchain_epoch": pick(raw, "toolchain_epoch", "toolchainEpoch"),
        "cache_mode": pick(raw, "cache_mode", "cacheMode"),
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


def _canonical_os(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().lower()
    if value in {"linux", "ubuntu"} or value.startswith("linux-"):
        return "linux"
    if value in {"macos", "mac", "darwin"} or value.startswith("macos-"):
        return "macos"
    if value in {"windows", "win"} or value.startswith("windows-"):
        return "windows"
    return value


def _label_dimension(labels: set[str], prefixes: tuple[str, ...]) -> str | None:
    for label in sorted(labels):
        for prefix in prefixes:
            if label.startswith(prefix) and len(label) > len(prefix):
                return label[len(prefix) :]
    return None


def runner_dimensions(
    labels: list[str],
    runner_role: Any = None,
    operating_system: Any = None,
) -> dict[str, str | None]:
    """Derive only dimensions encoded by the selected runner label.

    Depot labels encode architecture and vCPU size. Hosted and legacy labels
    identify a provider/architecture in the checked-in runner contract, but do
    not expose a comparable Depot size, so their runner_size remains null.
    """

    normalized = {label.lower() for label in labels}
    role = (
        str(runner_role).strip()
        if isinstance(runner_role, str) and runner_role.strip()
        else _label_dimension(normalized, ("runner-role:", "role:"))
    )
    depot_label = next(
        (label for label in sorted(normalized) if label.startswith("depot-")),
        None,
    )
    if depot_label:
        suffix = depot_label.rsplit("-", 1)[-1]
        depot_platform = depot_label[len("depot-") :]
        if depot_platform.startswith(("macos-", "mac-")):
            depot_os = "macos"
        elif depot_platform.startswith(("windows-", "win-")):
            depot_os = "windows"
        else:
            depot_os = "linux"
        return {
            "provider": "depot",
            "architecture": (
                "arm64"
                if "-arm" in depot_label or depot_os == "macos"
                else "amd64"
            ),
            "runner_size": suffix if suffix in {"4", "8", "16"} else "default",
            "operating_system": _canonical_os(operating_system) or depot_os,
            "runner_role": role,
        }

    if "mesh-llm-arm64" in normalized or "ubuntu-24.04-arm" in normalized:
        architecture = "arm64"
    elif "macos-15-intel" in normalized:
        architecture = "amd64"
    elif "macos-15" in normalized:
        architecture = "arm64"
    elif (
        "windows-2022" in normalized
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
        "operating_system": _canonical_os(operating_system)
        or (
            "linux"
            if any(label.startswith("ubuntu-") for label in normalized)
            else "macos"
            if any(label.startswith("macos-") for label in normalized)
            else "windows"
            if any(label.startswith("windows-") for label in normalized)
            else None
        ),
        "runner_role": role,
    }


def observation(run: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    duration = elapsed(job["started"], job["completed"])
    dependency_wait = job["dependency_wait_seconds"]
    if dependency_wait is None:
        dependency_wait = elapsed(job["created"], job["dependency_ready"])
    runner_queue = elapsed(
        job["dependency_ready"] or job["created"],
        job["started"],
    )
    dimensions = runner_dimensions(
        job["labels"], job["runner_role"], job["operating_system"]
    )
    return {
        "run_id": run["id"],
        "run_attempt": run["attempt"],
        "run_url": run["url"],
        "job_id": job["id"],
        "job_url": job["url"],
        "name": job["name"],
        "run_conclusion": run["conclusion"],
        "conclusion": job["conclusion"],
        "created_at": job["created"].isoformat() if job["created"] else None,
        "started_at": job["started"].isoformat() if job["started"] else None,
        "completed_at": (
            job["completed"].isoformat() if job["completed"] else None
        ),
        "dependency_ready_at": (
            job["dependency_ready"].isoformat()
            if job["dependency_ready"]
            else None
        ),
        # `duration_seconds` remains the v2 name. `execution_seconds` is the
        # explicit name used by cohort comparisons.
        "duration_seconds": duration,
        "execution_seconds": duration,
        "wall_seconds": elapsed(job["created"], job["completed"]),
        "queue_seconds": elapsed(job["created"], job["started"]),
        "runner_queue_seconds": runner_queue,
        "dependency_wait_seconds": dependency_wait,
        "dependency_wait_observed": dependency_wait is not None,
        "capacity_contaminated": (
            elapsed(job["created"], job["started"]) or 0
        )
        >= QUEUE_CONTAMINATION_SECONDS,
        "start_delay_seconds": (
            elapsed(run["created"], job["started"])
            if run["attempt"] == 1
            else None
        ),
        "runner_labels": job["labels"],
        "runner_dimensions": dimensions,
    }


def included(run: dict[str, Any], requested: str) -> tuple[bool, str]:
    if run["status"] != "completed":
        return False, "not_completed"
    if requested in ("all", "completed") or run["conclusion"] == requested:
        return True, ""
    return False, f"conclusion_{run['conclusion'] or 'missing'}"


def is_comparison_executor(name: str) -> bool:
    """Exclude hosted orchestration and credentialed smoke jobs from provider cohorts."""

    normalized = name.strip().lower()
    if normalized in {"changes", "summary"}:
        return False
    if normalized.startswith(("plan ", "pr /")):
        return False
    if "/ ci /" in normalized or "lane plan" in normalized:
        return False
    if "select " in normalized and " runner" in normalized:
        return False
    if "runner and cache contract" in normalized:
        return False
    if "smoke" in normalized:
        return False
    return True


def _job_timestamp(sample: dict[str, Any], field: str) -> dt.datetime | None:
    return timestamp(sample.get(field))


def capacity_metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize allocated time and bounded runner overlap.

    Job intervals are the only portable evidence available from the Actions
    jobs API. Missing timestamps are omitted rather than treated as zero.
    """

    runner_seconds = sum(
        sample["execution_seconds"]
        for sample in observations
        if sample["execution_seconds"] is not None
    )
    cancelled_seconds = sum(
        sample["execution_seconds"]
        for sample in observations
        if sample["execution_seconds"] is not None
        and (
            sample["run_conclusion"] in {"cancelled", "canceled"}
            or sample["conclusion"] in {"cancelled", "canceled"}
        )
    )
    events: dict[tuple[str | None, str | None, str | None], list[tuple[dt.datetime, int]]] = (
        collections.defaultdict(list)
    )
    for sample in observations:
        started = _job_timestamp(sample, "started_at")
        completed = _job_timestamp(sample, "completed_at")
        if started is None or completed is None:
            continue
        dimensions = sample["runner_dimensions"]
        key = (
            dimensions.get("provider"),
            dimensions.get("operating_system"),
            dimensions.get("runner_role"),
        )
        events[key].extend(((started, 1), (completed, -1)))

    peaks: dict[str, int] = {}
    for key, points in events.items():
        active = 0
        peak = 0
        # End events sort before starts at the same instant. Two adjacent jobs
        # therefore do not count as concurrent workers.
        for _, delta in sorted(points, key=lambda point: (point[0], point[1])):
            active += delta
            peak = max(peak, active)
        provider, operating_system, role = key
        name = "/".join(value or "unknown" for value in (provider, operating_system, role))
        peaks[name] = peak

    return {
        "runner_minutes": round(runner_seconds / 60, 3),
        "cancelled_runner_minutes": round(cancelled_seconds / 60, 3),
        "timestamped_job_count": sum(
            1
            for sample in observations
            if _job_timestamp(sample, "started_at") is not None
            and _job_timestamp(sample, "completed_at") is not None
        ),
        "peak_workers": {
            "total": max(peaks.values(), default=None),
            "by_provider_os_role": dict(sorted(peaks.items())),
        },
    }


def queue_heuristics(report: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic, date-independent provider rollout signals."""

    # Without dependency-ready instrumentation runner queue falls back to the
    # raw creation-to-start interval; the report still marks dependency wait as
    # unavailable so this fallback cannot be mistaken for a clean decomposition.
    job_queue = report["jobs"].get(
        "runner_queue_seconds", report["jobs"]["queue_seconds"]
    )
    terminal_queue = report["workflow"].get(
        "terminal_job_runner_queue_seconds",
        report["workflow"].get("terminal_job_queue_seconds", {}),
    )
    queue_p95 = job_queue.get("p95")
    terminal_p95 = terminal_queue.get("p95")
    contaminated = any(
        value is not None and value >= QUEUE_CONTAMINATION_SECONDS
        for value in (queue_p95, terminal_p95)
    )
    sample_count = int(job_queue.get("count") or 0)
    if sample_count < MIN_HEURISTIC_SAMPLES:
        state = "insufficient_sample"
    elif contaminated:
        state = "rollback"
    elif queue_p95 is None or queue_p95 > QUEUE_WARN_SECONDS:
        state = "hold"
    else:
        state = "eligible"
    return {
        "thresholds_seconds": {
            "queue_warn": QUEUE_WARN_SECONDS,
            "queue_capacity_contamination": QUEUE_CONTAMINATION_SECONDS,
            "minimum_samples": MIN_HEURISTIC_SAMPLES,
        },
        "sample_count": sample_count,
        "queue_p95_seconds": queue_p95,
        "terminal_job_queue_p95_seconds": terminal_p95,
        "capacity_contaminated": contaminated,
        "state": state,
        "interpretation": {
            "eligible": "queue p95 <= warn threshold and no capacity contamination",
            "hold": "queue p95 exceeds warn threshold or is unavailable",
            "rollback": "queue p95 reaches the capacity-contamination threshold",
            "insufficient_sample": "collect at least the minimum sample count",
        }[state],
    }


def _comparison_cohort(report: dict[str, Any]) -> dict[str, Any]:
    return report["jobs"].get("comparison_cohort", report["jobs"])


def _providers(report: dict[str, Any]) -> set[str]:
    cohort = _comparison_cohort(report)
    if "providers" in cohort:
        return {
            str(provider)
            for provider in cohort["providers"]
            if provider and provider != "unknown"
        }
    return {
        str(item["provider"])
        for item in report["jobs"].get("by_runner", [])
        if item.get("provider") and item.get("provider") != "unknown"
    }


def _runner_dimension_values(report: dict[str, Any], key: str) -> set[str]:
    cohort = _comparison_cohort(report)
    if "dimensions" in cohort:
        return {str(value) for value in cohort["dimensions"].get(key, []) if value}
    return {
        str(item[key])
        for item in report["jobs"].get("by_runner", [])
        if item.get(key)
    }


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare two reports without conflating provider or capacity effects."""

    baseline_providers = _providers(baseline)
    candidate_providers = _providers(candidate)
    provider_separated = bool(baseline_providers and candidate_providers) and (
        baseline_providers.isdisjoint(candidate_providers)
    )
    baseline_cohort = _comparison_cohort(baseline)
    candidate_cohort = _comparison_cohort(candidate)
    baseline_names = set(baseline_cohort.get("job_names", [])) or {
        item["name"] for item in baseline["jobs"].get("by_name", [])
    }
    candidate_names = set(candidate_cohort.get("job_names", [])) or {
        item["name"] for item in candidate["jobs"].get("by_name", [])
    }
    common_names = sorted(baseline_names & candidate_names)
    dimension_overlap = {
        key: sorted(
            _runner_dimension_values(baseline, key)
            & _runner_dimension_values(candidate, key)
        )
        for key in ("operating_system", "architecture", "runner_role")
    }
    comparable_dimensions = all(
        not _runner_dimension_values(baseline, key)
        or not _runner_dimension_values(candidate, key)
        or bool(values)
        for key, values in dimension_overlap.items()
    )
    base_queue_summary = baseline_cohort.get("runner_queue_seconds", {})
    candidate_queue_summary = candidate_cohort.get("runner_queue_seconds", {})
    base_queue = base_queue_summary.get("p95")
    candidate_queue = candidate_queue_summary.get("p95")
    base_execution = baseline_cohort.get("execution_seconds", {}).get("p95")
    candidate_execution = candidate_cohort.get("execution_seconds", {}).get("p95")
    baseline_count = int(base_queue_summary.get("count") or 0)
    candidate_count = int(candidate_queue_summary.get("count") or 0)
    sufficient_evidence = (
        baseline_count >= MIN_HEURISTIC_SAMPLES
        and candidate_count >= MIN_HEURISTIC_SAMPLES
    )
    if candidate_queue is not None and candidate_queue >= QUEUE_CONTAMINATION_SECONDS:
        recommendation = "rollback"
    elif (
        not sufficient_evidence
        or not provider_separated
        or not common_names
        or not comparable_dimensions
    ):
        recommendation = "hold"
    elif candidate_queue is None or candidate_queue > QUEUE_WARN_SECONDS:
        recommendation = "hold"
    else:
        recommendation = "eligible"
    return {
        "baseline_source": baseline.get("source", {}).get("description"),
        "candidate_source": candidate.get("source", {}).get("description"),
        "provider_cohort_separation": {
            "baseline_providers": sorted(baseline_providers),
            "candidate_providers": sorted(candidate_providers),
            "disjoint": provider_separated,
            "status": "pass" if provider_separated else "fail",
        },
        "common_job_families": common_names,
        "dimension_overlap": dimension_overlap,
        "comparable_dimensions": comparable_dimensions,
        "sample_counts": {
            "baseline_jobs": baseline_count,
            "candidate_jobs": candidate_count,
            "minimum_each": MIN_HEURISTIC_SAMPLES,
            "sufficient": sufficient_evidence,
        },
        "p95_seconds": {
            "queue": {
                "baseline": base_queue,
                "candidate": candidate_queue,
                "delta": (
                    round(candidate_queue - base_queue, 3)
                    if candidate_queue is not None and base_queue is not None
                    else None
                ),
            },
            "execution": {
                "baseline": base_execution,
                "candidate": candidate_execution,
                "delta": (
                    round(candidate_execution - base_execution, 3)
                    if candidate_execution is not None and base_execution is not None
                    else None
                ),
            },
        },
        "recommendation": recommendation,
        "interpretation": {
            "eligible": "provider-separated comparable cohorts with candidate queue heuristics eligible",
            "hold": "collect more comparable evidence or resolve provider/capacity mismatch",
            "rollback": "candidate queue p95 crossed the deterministic contamination threshold",
        }[recommendation],
    }


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
        tuple[
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
        ],
        list[dict[str, Any]],
    ] = collections.defaultdict(list)
    step_observations: list[dict[str, Any]] = []
    wall_times = []
    queue_times = []
    dependency_wait_times = []
    execution_times = []
    job_wall_times = []
    terminal_queue_times = []
    terminal_runner_queue_times = []
    terminal_dependency_wait_times = []
    terminal_execution_times = []
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
                    dimensions["operating_system"],
                    dimensions["architecture"],
                    dimensions["runner_role"],
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
            terminal_sample = observation(run, terminal)
            terminal_queue_times.append(terminal_sample["queue_seconds"])
            terminal_runner_queue_times.append(
                terminal_sample["runner_queue_seconds"]
            )
            terminal_dependency_wait_times.append(
                terminal_sample["dependency_wait_seconds"]
            )
            terminal_execution_times.append(terminal_sample["execution_seconds"])
        else:
            terminal_sample = None
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
                "plan_profile": run["plan_profile"],
                "plan_digest": run["plan_digest"],
                "change_class": run["change_class"],
                "runner_image": run["runner_image"],
                "toolchain_epoch": run["toolchain_epoch"],
                "cache_mode": run["cache_mode"],
                "wall_seconds": wall,
                "queue_seconds": queue,
                "workflow_timing_excluded": not workflow_timing_eligible,
                "executed_job_count": len(jobs),
                "skipped_job_count": len(run["jobs"]) - len(jobs),
                "terminal_job": terminal["name"] if terminal else None,
                "longest_job": longest["name"] if longest else None,
                "longest_job_seconds": longest["duration_seconds"] if longest else None,
                "terminal_job_queue_seconds": (
                    terminal_sample["queue_seconds"] if terminal_sample else None
                ),
                "terminal_job_runner_queue_seconds": (
                    terminal_sample["runner_queue_seconds"]
                    if terminal_sample
                    else None
                ),
                "terminal_job_dependency_wait_seconds": (
                    terminal_sample["dependency_wait_seconds"]
                    if terminal_sample
                    else None
                ),
                "terminal_job_execution_seconds": (
                    terminal_sample["execution_seconds"]
                    if terminal_sample
                    else None
                ),
                "capacity_contaminated": any(
                    sample["runner_queue_seconds"] is not None
                    and sample["runner_queue_seconds"]
                    >= QUEUE_CONTAMINATION_SECONDS
                    for sample in samples
                )
                or (
                    terminal_sample is not None
                    and terminal_sample["runner_queue_seconds"] is not None
                    and terminal_sample["runner_queue_seconds"]
                    >= QUEUE_CONTAMINATION_SECONDS
                ),
            }
        )
        dependency_wait_times.extend(
            sample["dependency_wait_seconds"] for sample in samples
        )
        execution_times.extend(sample["execution_seconds"] for sample in samples)
        job_wall_times.extend(sample["wall_seconds"] for sample in samples)

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
            "execution_seconds": summarize(
                [sample["execution_seconds"] for sample in samples]
            ),
            "wall_seconds": summarize(
                [sample["wall_seconds"] for sample in samples]
            ),
            "queue_seconds": summarize(
                [sample["queue_seconds"] for sample in samples]
            ),
            "runner_queue_seconds": summarize(
                [sample["runner_queue_seconds"] for sample in samples]
            ),
            "dependency_wait_seconds": summarize(
                [sample["dependency_wait_seconds"] for sample in samples]
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
    for (
        provider,
        operating_system,
        architecture,
        runner_role,
        runner_size,
    ), samples in runner_groups.items():
        by_runner.append(
            {
                "provider": provider,
                "operating_system": operating_system,
                "architecture": architecture,
                "runner_role": runner_role,
                "runner_size": runner_size,
                "sample_count": len(samples),
                "job_names": sorted({sample["name"] for sample in samples}),
                "duration_seconds": summarize(
                    [sample["duration_seconds"] for sample in samples]
                ),
                "execution_seconds": summarize(
                    [sample["execution_seconds"] for sample in samples]
                ),
                "queue_seconds": summarize(
                    [sample["queue_seconds"] for sample in samples]
                ),
                "runner_queue_seconds": summarize(
                    [sample["runner_queue_seconds"] for sample in samples]
                ),
                "dependency_wait_seconds": summarize(
                    [sample["dependency_wait_seconds"] for sample in samples]
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
    comparison_observations = [
        sample for sample in observations if is_comparison_executor(sample["name"])
    ]
    comparison_dimensions = {
        key: sorted(
            {
                str(sample["runner_dimensions"][key])
                for sample in comparison_observations
                if sample["runner_dimensions"].get(key)
            }
        )
        for key in ("operating_system", "architecture", "runner_role")
    }
    comparison_cohort = {
        "scope": "build-test executors; orchestration and credentialed smoke jobs excluded",
        "sample_count": len(comparison_observations),
        "job_names": sorted({sample["name"] for sample in comparison_observations}),
        "providers": sorted(
            {
                str(sample["runner_dimensions"]["provider"])
                for sample in comparison_observations
                if sample["runner_dimensions"].get("provider") not in {None, "unknown"}
            }
        ),
        "dimensions": comparison_dimensions,
        "runner_queue_seconds": summarize(
            [sample["runner_queue_seconds"] for sample in comparison_observations]
        ),
        "execution_seconds": summarize(
            [sample["execution_seconds"] for sample in comparison_observations]
        ),
    }

    step_groups: dict[
        tuple[
            str,
            str,
            str | None,
            str | None,
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
                dimensions["operating_system"],
                dimensions["architecture"],
                dimensions["runner_role"],
                dimensions["runner_size"],
            )
        ].append(sample)
    by_step = [
        {
            "job_name": job_name,
            "step_name": step_name,
            "provider": provider,
            "operating_system": operating_system,
            "architecture": architecture,
            "runner_role": runner_role,
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
            operating_system,
            architecture,
            runner_role,
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
    report = {
        "schema_version": 3,
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
            "job_execution_seconds": "same interval as job_duration_seconds",
            "job_wall_seconds": "job created_at to completed_at",
            "job_queue_seconds": (
                "job created_at to started_at; unavailable without job created_at"
            ),
            "runner_queue_seconds": (
                "job started_at minus dependency-ready timestamp when present; "
                "otherwise falls back to creation-to-start for compatibility"
            ),
            "dependency_wait_seconds": (
                "dependency-ready timestamp to job created_at, or an explicit "
                "instrumented duration; unavailable in standard jobs API data"
            ),
            "job_start_delay_seconds": (
                "first-attempt workflow created_at to job started_at; includes "
                "dependency wait; reruns excluded"
            ),
            "terminal_job": (
                "last non-skipped job to finish; a critical-path candidate"
            ),
            "runner_dimensions": (
                "provider, operating system, architecture, semantic runner role, "
                "and Depot runner size derived from labels or optional metadata"
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
            "dependency_wait_seconds": summarize(dependency_wait_times),
            "execution_seconds": summarize(execution_times),
            "job_wall_seconds": summarize(job_wall_times),
            "terminal_job_queue_seconds": summarize(terminal_queue_times),
            "terminal_job_runner_queue_seconds": summarize(
                terminal_runner_queue_times
            ),
            "terminal_job_dependency_wait_seconds": summarize(
                terminal_dependency_wait_times
            ),
            "terminal_job_execution_seconds": summarize(terminal_execution_times),
        },
        "jobs": {
            "sample_count": len(observations),
            "duration_seconds": summarize(
                [sample["duration_seconds"] for sample in observations]
            ),
            "execution_seconds": summarize(
                [sample["execution_seconds"] for sample in observations]
            ),
            "wall_seconds": summarize(
                [sample["wall_seconds"] for sample in observations]
            ),
            "queue_seconds": summarize(
                [sample["queue_seconds"] for sample in observations]
            ),
            "runner_queue_seconds": summarize(
                [sample["runner_queue_seconds"] for sample in observations]
            ),
            "dependency_wait_seconds": summarize(
                [sample["dependency_wait_seconds"] for sample in observations]
            ),
            "start_delay_seconds": summarize(
                [sample["start_delay_seconds"] for sample in observations]
            ),
            "by_name": by_name,
            "by_runner": by_runner,
            "comparison_cohort": comparison_cohort,
            "critical_finish_candidates": critical,
            "slowest_observations": slowest,
        },
        "steps": {
            "sample_count": len(step_observations),
            "by_name": by_step,
        },
        "runs": run_reports,
    }
    report["capacity"] = capacity_metrics(observations)
    report["heuristics"] = queue_heuristics(report)
    return report


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
    for label, key in (
        ("Wall time", "wall_seconds"),
        ("Queue", "queue_seconds"),
        ("Dependency wait", "dependency_wait_seconds"),
        ("Execution", "execution_seconds"),
    ):
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
        "| Provider | OS | Architecture | Role | Depot size | Jobs | Execution p50 | Execution p95 | Runner queue p95 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in jobs["by_runner"][:top]:
        lines.append(
            f"| {markdown_escape(item['provider'])} | "
            f"{markdown_escape(item['operating_system'] or 'n/a')} | "
            f"{markdown_escape(item['architecture'] or 'n/a')} | "
            f"{markdown_escape(item['runner_role'] or 'n/a')} | "
            f"{markdown_escape(item['runner_size'] or 'n/a')} | "
            f"{item['sample_count']} | "
            f"{human(item['execution_seconds']['p50'])} | "
            f"{human(item['execution_seconds']['p95'])} | "
            f"{human(item['runner_queue_seconds']['p95'])} |"
        )
    lines += [
        "",
        "## Capacity and rollout heuristics",
        "",
        (
            f"Runner-minutes: **{report['capacity']['runner_minutes']}**; "
            f"cancelled runner-minutes: **{report['capacity']['cancelled_runner_minutes']}**; "
            f"peak workers: **{report['capacity']['peak_workers']['total'] or 'n/a'}**."
        ),
        (
            f"Queue p95: **{human(report['heuristics']['queue_p95_seconds'])}**; "
            f"terminal queue p95: **{human(report['heuristics']['terminal_job_queue_p95_seconds'])}**; "
            f"capacity contaminated: **{report['heuristics']['capacity_contaminated']}**; "
            f"state: **{report['heuristics']['state']}**."
        ),
        "",
        "## Step timing",
        "",
    ]
    if report["steps"]["by_name"]:
        lines += [
            "| Provider | OS | Architecture | Role | Runner size | Job | Step | Samples | Duration p50 | Duration p95 |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for item in report["steps"]["by_name"][:top]:
            lines.append(
                f"| {markdown_escape(item['provider'] or 'n/a')} | "
                f"{markdown_escape(item['operating_system'] or 'n/a')} | "
                f"{markdown_escape(item['architecture'] or 'n/a')} | "
                f"{markdown_escape(item['runner_role'] or 'n/a')} | "
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
        "| Job | Samples | Duration p50 | Duration p95 | Runner queue p50 | Runner queue p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in jobs["by_name"][:top]:
        duration, queue = item["duration_seconds"], item["runner_queue_seconds"]
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
    ]
    if report.get("comparison"):
        comparison = report["comparison"]
        separation = comparison["provider_cohort_separation"]
        lines += [
            "## Provider cohort comparison",
            "",
            (
                f"Provider separation: **{separation['status']}** "
                f"({', '.join(separation['baseline_providers']) or 'n/a'} → "
                f"{', '.join(separation['candidate_providers']) or 'n/a'}); "
                f"recommendation: **{comparison['recommendation']}**."
            ),
            "",
        ]
    lines += [
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
    parser.add_argument(
        "--compare-input",
        help="Detailed/raw run JSON for a historical baseline cohort",
    )
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
        if args.compare_input:
            baseline_raw = load_runs(args.compare_input)
            baseline_runs = [normalize_run(run) for run in baseline_raw]
            baseline = analyze(
                baseline_runs,
                args.status,
                args.top,
                {
                    "kind": "file",
                    "description": args.compare_input,
                },
                {"cohort": "baseline"},
            )
            report["comparison"] = compare_reports(baseline, report)
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
