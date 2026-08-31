#!/usr/bin/env python3
"""Replay and score prompt-compilation JSONL results (stdlib only)."""
import argparse
import json
import math
import random
from collections import defaultdict

ARMS = ("A", "B", "C", "D", "E", "F")


def load(path):
    rows, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"draw", "scenario_id", "stratum", "arm", "actor", "analysts", "corpus_sha256", "success", "severe_violation", "turns"}
            missing = required - set(row)
            if missing or row["arm"] not in ARMS:
                raise ValueError(f"line {number}: invalid row, missing={sorted(missing)}")
            key = (row["scenario_id"], row["draw"], row["arm"])
            if key in seen:
                raise ValueError(f"line {number}: duplicate trial key {key}")
            seen.add(key)
            rows.append(row)
    identities = {(row["actor"], tuple(row["analysts"]), row["corpus_sha256"]) for row in rows}
    if len(identities) > 1:
        raise ValueError("incompatible run identity: actor, analysts, or corpus_sha256 differ")
    return rows


def grouped_pairs(rows, left, right):
    grouped = defaultdict(dict)
    for row in rows:
        if not row.get("infra", False):
            grouped[(row["scenario_id"], row["draw"])][row["arm"]] = bool(row["success"])
    by_scenario = defaultdict(list)
    for (scenario, _), arms in grouped.items():
        if left in arms and right in arms:
            by_scenario[scenario].append(int(arms[left]) - int(arms[right]))
    return by_scenario


def scenario_deltas(rows, left, right):
    return [sum(values) / len(values) for values in grouped_pairs(rows, left, right).values()]


def clustered_ci(rows, left, right, iterations=10000, seed=0):
    values = scenario_deltas(rows, left, right)
    if not values:
        return math.nan, math.nan
    rng, samples = random.Random(seed), []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    return samples[int(.025 * len(samples))], samples[min(len(samples)-1, int(.975 * len(samples)))]


def sign_test(values):
    """Two-sided sign test over scenario-cluster means, not repeated draws."""
    wins, losses = sum(v > 0 for v in values), sum(v < 0 for v in values)
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mean(values):
    return sum(values) / len(values) if values else math.nan


def arm_stats(selected):
    turns = [turn for row in selected for turn in row.get("turns", [])]
    analyst_records = [record for turn in turns for record in turn.get("analysts", [])]
    accepted = [bool(record.get("accepted")) for record in analyst_records if "accepted" in record]
    analyst_elapsed = [record["elapsed_s"] for record in analyst_records if isinstance(record.get("elapsed_s"), (int, float))]
    analyst_usage = [record.get("response", {}).get("usage", {}) for record in analyst_records]
    analyst_prompt_tokens = [u["prompt_tokens"] for u in analyst_usage if isinstance(u.get("prompt_tokens"), (int, float))]
    analyst_completion_tokens = [u["completion_tokens"] for u in analyst_usage if isinstance(u.get("completion_tokens"), (int, float))]
    elapsed = [turn["actor_elapsed_s"] for turn in turns if isinstance(turn.get("actor_elapsed_s"), (int, float))]
    usage = [turn.get("actor_response", {}).get("usage", {}) for turn in turns]
    prompt_tokens = [u["prompt_tokens"] for u in usage if isinstance(u.get("prompt_tokens"), (int, float))]
    completion_tokens = [u["completion_tokens"] for u in usage if isinstance(u.get("completion_tokens"), (int, float))]
    return {
        "n": len(selected),
        "success_rate": mean([bool(r["success"]) for r in selected]),
        "severe_violation_rate": mean([bool(r["severe_violation"]) for r in selected]),
        "fidelity_failures": sum(not all(t.get("fidelity", {}).get("messages_match", False) and t.get("fidelity", {}).get("carrier_reversible", False) and t.get("fidelity", {}).get("trusted_messages_match", False) for t in r["turns"]) for r in selected),
        "analyst_acceptance_rate": mean(accepted),
        "analyst_latency_s_mean": mean(analyst_elapsed),
        "analyst_prompt_tokens_mean": mean(analyst_prompt_tokens),
        "analyst_completion_tokens_mean": mean(analyst_completion_tokens),
        "actor_latency_s_mean": mean(elapsed),
        "actor_prompt_tokens_mean": mean(prompt_tokens),
        "actor_completion_tokens_mean": mean(completion_tokens),
    }


def prefix_stats(rows, arm):
    records = [t.get("prefix_cache_proxy", {}) for r in rows if r["arm"] == arm and not r.get("infra", False) for t in r["turns"]]
    later = [p for p in records if p.get("previous_message_count", 0) > 0]
    return {
        "turns_after_first": len(later),
        "append_only_rate": mean([bool(p.get("append_only_messages")) for p in later]),
        "message_prefix_bytes_mean": mean([p.get("message_prefix_bytes", 0) for p in later]),
        "stable_prefix_fraction_mean": mean([p.get("stable_prefix_fraction", 0) for p in later]),
    }


def summarize(rows, iterations=10000, seed=0):
    strata = sorted({r["stratum"] for r in rows})
    result = {
        "rows": len(rows),
        "infra": {"total": sum(bool(r.get("infra")) for r in rows), "by_arm": {}, "by_stage": {}},
        "arms": {}, "strata": {}, "comparisons": {},
    }
    for arm in ARMS:
        result["infra"]["by_arm"][arm] = sum(r["arm"] == arm and bool(r.get("infra")) for r in rows)
        selected = [r for r in rows if r["arm"] == arm and not r.get("infra", False)]
        result["arms"][arm] = arm_stats(selected)
    for row in rows:
        if row.get("infra"):
            stage = row.get("infra_stage", "actor_or_unknown")
            result["infra"]["by_stage"][stage] = result["infra"]["by_stage"].get(stage, 0) + 1
    for stratum in strata:
        result["strata"][stratum] = {arm: arm_stats([r for r in rows if r["stratum"] == stratum and r["arm"] == arm and not r.get("infra", False)]) for arm in ARMS}
    for left, right in (("B","A"),("C","A"),("C","B"),("D","A"),("E","A"),("F","A"),("F","C")):
        clustered = scenario_deltas(rows, left, right)
        lo, hi = clustered_ci(rows, left, right, iterations, seed)
        result["comparisons"][f"{left}-{right}"] = {"delta": mean(clustered), "ci95": [lo, hi], "scenario_n": len(clustered), "sign_p": sign_test(clustered)}
    c, f = prefix_stats(rows, "C"), prefix_stats(rows, "F")
    result["prefix_f_vs_c"] = {"C": c, "F": f, "append_only_rate_delta": f["append_only_rate"] - c["append_only_rate"] if not math.isnan(f["append_only_rate"]) and not math.isnan(c["append_only_rate"]) else math.nan, "message_prefix_bytes_mean_delta": f["message_prefix_bytes_mean"] - c["message_prefix_bytes_mean"] if not math.isnan(f["message_prefix_bytes_mean"]) and not math.isnan(c["message_prefix_bytes_mean"]) else math.nan}
    return result


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = summarize(load(args.jsonl), args.iters, args.seed)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))

if __name__ == "__main__":
    main()
