#!/usr/bin/env python3
"""Measure MTP N-gram proposer arms against staged OpenAI endpoints.

The runner drives endpoints that are already serving the same MTP-capable model
on equivalent >=2-stage splits. It writes per-sample artifacts and fails when a
requested suffix activation arm never produces an N-gram tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

EDIT_FILE = """def parse_config(path):
    with open(path) as handle:
        raw = handle.read()
    data = json.loads(raw)
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = value.strip()
        else:
            result[key] = value
    return result"""

BUILT_IN_WORKLOADS: dict[str, str] = {
    "edit": (
        "Here is a Python function:\n\n```python\n"
        + EDIT_FILE
        + "\n```\n\nRe-emit the entire function verbatim, changing only the name "
        "`parse_config` to `load_config`. Output just the code."
    ),
    "chat": (
        "Explain, in two short paragraphs, why generating a token is more "
        "expensive than verifying one in speculative decoding."
    ),
}


@dataclass(frozen=True)
class Workload:
    name: str
    prompt: str


@dataclass(frozen=True)
class Sample:
    arm: str
    workload: str
    run: int
    wall_seconds: float
    wall_tok_s: float
    server_tok_s: float
    predicted_n: int
    draft_n: int
    draft_accepted: int
    ngram_proposer: str
    ngram_tokens: int
    ngram_accepted_tokens: int
    proposer_attempts: int
    proposer_hits: int
    proposer_match_length_max: int
    proposer_candidates_examined: int
    proposer_appended_tokens: int
    proposer_rebuilds: int
    proposer_sync_us: int
    proposer_lookup_us: int
    finish_reason: str
    output_sha256: str


def load_workloads(corpus: Path | None) -> list[Workload]:
    workloads = [Workload(name, prompt) for name, prompt in BUILT_IN_WORKLOADS.items()]
    corpus = corpus or HERE / "skippy-coding-agent-loop.jsonl"
    if not corpus.exists():
        return workloads
    with corpus.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{corpus}:{line_number} has no non-empty prompt")
            name = str(row.get("id") or f"corpus-{line_number}")
            workloads.append(Workload(name, prompt))
    return workloads


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    if not isinstance(body, dict):
        raise ValueError(f"{url} returned a non-object response")
    return body


def timing_int(timings: dict[str, Any], name: str) -> int:
    return int(timings.get(name, 0) or 0)


def response_text(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("chat response contains no choices")
    choice = choices[0]
    content = (choice.get("message") or {}).get("content") or ""
    return str(content), str(choice.get("finish_reason") or "unknown")


def run_once(
    arm: str,
    base_url: str,
    workload: Workload,
    run: int,
    model: str,
    max_tokens: int,
    timeout: float,
) -> Sample:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": workload.prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    started = time.perf_counter()
    response = http_json(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    wall_seconds = time.perf_counter() - started
    timings = response.get("timings") or {}
    if not isinstance(timings, dict):
        raise ValueError(f"{arm}/{workload.name} returned invalid timings")
    predicted_n = timing_int(timings, "predicted_n")
    server_tok_s = float(timings.get("predicted_per_second", 0.0) or 0.0)
    if predicted_n <= 0 or server_tok_s <= 0.0:
        raise ValueError(
            f"{arm}/{workload.name} has unusable timings: "
            f"predicted_n={predicted_n} predicted_per_second={server_tok_s}"
        )
    content, finish_reason = response_text(response)
    return Sample(
        arm=arm,
        workload=workload.name,
        run=run,
        wall_seconds=wall_seconds,
        wall_tok_s=predicted_n / wall_seconds,
        server_tok_s=server_tok_s,
        predicted_n=predicted_n,
        draft_n=timing_int(timings, "draft_n"),
        draft_accepted=timing_int(timings, "draft_n_accepted"),
        ngram_proposer=str(timings.get("native_mtp_ngram_proposer", "unknown")),
        ngram_tokens=timing_int(timings, "native_mtp_hybrid_ngram_tokens"),
        ngram_accepted_tokens=timing_int(
            timings, "native_mtp_hybrid_accepted_tail_tokens"
        ),
        proposer_attempts=timing_int(timings, "native_mtp_ngram_proposer_attempts"),
        proposer_hits=timing_int(timings, "native_mtp_ngram_proposer_hits"),
        proposer_match_length_max=timing_int(
            timings, "native_mtp_ngram_proposer_match_length_max"
        ),
        proposer_candidates_examined=timing_int(
            timings, "native_mtp_ngram_proposer_candidates_examined"
        ),
        proposer_appended_tokens=timing_int(
            timings, "native_mtp_ngram_proposer_appended_tokens"
        ),
        proposer_rebuilds=timing_int(timings, "native_mtp_ngram_proposer_rebuilds"),
        proposer_sync_us=timing_int(timings, "native_mtp_ngram_proposer_sync_us"),
        proposer_lookup_us=timing_int(timings, "native_mtp_ngram_proposer_lookup_us"),
        finish_reason=finish_reason,
        output_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def median(samples: list[Sample], field: str) -> float:
    return statistics.median(float(getattr(sample, field)) for sample in samples)


def summarize(samples: list[Sample], baseline_arm: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.arm, sample.workload), []).append(sample)

    rows: list[dict[str, Any]] = []
    for (arm, workload), group in sorted(grouped.items()):
        hybrid_tokens = sum(sample.ngram_tokens for sample in group)
        standalone_tokens = sum(sample.draft_n for sample in group)
        proposal_tokens = hybrid_tokens or standalone_tokens
        accepted = (
            sum(sample.ngram_accepted_tokens for sample in group)
            if hybrid_tokens
            else sum(sample.draft_accepted for sample in group)
        )
        rows.append(
            {
                "arm": arm,
                "workload": workload,
                "samples": len(group),
                "wall_tok_s_median": median(group, "wall_tok_s"),
                "server_tok_s_median": median(group, "server_tok_s"),
                "server_tok_s_stdev": statistics.stdev(
                    sample.server_tok_s for sample in group
                )
                if len(group) > 1
                else 0.0,
                "proposal_mode": "mtp-hybrid" if hybrid_tokens else "standalone",
                "ngram_tokens": proposal_tokens,
                "ngram_accepted_tokens": accepted,
                "ngram_acceptance": accepted / proposal_tokens if proposal_tokens else 0.0,
                "proposer_match_length_max": max(
                    sample.proposer_match_length_max for sample in group
                ),
                "proposer_candidates_examined": sum(
                    sample.proposer_candidates_examined for sample in group
                ),
                "proposer_lookup_us": sum(sample.proposer_lookup_us for sample in group),
            }
        )

    mismatches: list[dict[str, Any]] = []
    by_identity = {(sample.arm, sample.workload, sample.run): sample for sample in samples}
    for sample in samples:
        baseline = by_identity.get((baseline_arm, sample.workload, sample.run))
        if baseline and baseline.output_sha256 != sample.output_sha256:
            mismatches.append(
                {
                    "arm": sample.arm,
                    "workload": sample.workload,
                    "run": sample.run,
                    "baseline_sha256": baseline.output_sha256,
                    "output_sha256": sample.output_sha256,
                }
            )
    return {"rows": rows, "output_hash_mismatches": mismatches}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Suffix proposer benchmark",
        "",
        "| Arm | Workload | N | Wall tok/s median | Server tok/s median | N-gram acceptance | Max match |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['arm']} | {row['workload']} | {row['samples']} | "
            f"{row['wall_tok_s_median']:.2f} | {row['server_tok_s_median']:.2f} | "
            f"{row['ngram_acceptance']:.3f} | {row['proposer_match_length_max']} |"
        )
    lines.extend(
        [
            "",
            f"Output hash mismatches: {len(summary['output_hash_mismatches'])}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("target/skippy-suffix-proposer-bench") / stamp


def parse_arms(values: list[str]) -> dict[str, str]:
    arms: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid arm {value!r}; expected NAME=URL")
        name, url = value.split("=", 1)
        if not name or not url or name in arms:
            raise ValueError(f"invalid or duplicate arm {value!r}")
        arms[name] = url
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=URL")
    parser.add_argument("--model", required=True, help="Served model id from /v1/models")
    parser.add_argument("--corpus", type=Path, help="Additional JSONL prompts")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--baseline-arm", default="off")
    parser.add_argument("--require-drafts-arm", default="suffix")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    args = parser.parse_args()
    if args.warmups < 0 or args.runs <= 0 or args.max_tokens <= 0:
        parser.error("warmups must be non-negative; runs and max-tokens must be positive")

    arms = parse_arms(args.arm)
    if args.baseline_arm not in arms:
        parser.error(f"baseline arm {args.baseline_arm!r} is not configured")
    if args.require_drafts_arm and args.require_drafts_arm not in arms:
        parser.error(f"required activation arm {args.require_drafts_arm!r} is not configured")
    workloads = load_workloads(args.corpus)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "arms": arms,
        "workloads": [workload.name for workload in workloads],
        "warmups": args.warmups,
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "baseline_arm": args.baseline_arm,
        "require_drafts_arm": args.require_drafts_arm,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    rng = random.Random(args.seed)
    arm_names = list(arms)
    for _ in range(args.warmups):
        rng.shuffle(arm_names)
        for workload in workloads:
            for arm in arm_names:
                run_once(
                    arm,
                    arms[arm],
                    workload,
                    -1,
                    args.model,
                    args.max_tokens,
                    args.timeout,
                )

    samples: list[Sample] = []
    results_path = args.output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results:
        for run in range(args.runs):
            rng.shuffle(arm_names)
            for workload in workloads:
                for arm in arm_names:
                    sample = run_once(
                        arm,
                        arms[arm],
                        workload,
                        run,
                        args.model,
                        args.max_tokens,
                        args.timeout,
                    )
                    samples.append(sample)
                    results.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
                    results.flush()
                    print(
                        f"{arm:12} {workload.name:24} run={run + 1:<2} "
                        f"wall={sample.wall_tok_s:7.2f} tok/s "
                        f"server={sample.server_tok_s:7.2f} tok/s "
                        f"ngram={sample.ngram_accepted_tokens}/{sample.ngram_tokens}"
                    )

    summary = summarize(samples, args.baseline_arm)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, args.output_dir / "summary.md")

    required = [sample for sample in samples if sample.arm == args.require_drafts_arm]
    wrong_source = [
        sample
        for sample in required
        if sample.ngram_proposer != args.require_drafts_arm
    ]
    if wrong_source:
        observed = sorted({sample.ngram_proposer for sample in wrong_source})
        raise SystemExit(
            f"{args.require_drafts_arm} activation failed: proposer timing labels were {observed}"
        )
    activated = any(
        sample.proposer_hits > 0
        if sample.ngram_proposer in {"cache", "suffix"}
        else sample.draft_n > 0
        for sample in required
    )
    if required and not activated:
        raise SystemExit(
            f"{args.require_drafts_arm} activation failed: proposer reported no lookup hits"
        )
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
