#!/usr/bin/env python3
"""Analyze the scaled MoA actor-ablation study.

Consumes the per-trial JSONL written by `ablation_scaled_study`
(eval_openrouter.rs) and reports the paired net-uplift of references over
actor-alone, with a hierarchical bootstrap CI.

Arms per (draw, task):
  A = actor alone
  B = actor + real references     (production tool path)
  C = actor + shuffled references (advice from a different task)

Primary statistic: net uplift = P(rescue) - P(harm), where
  rescue = A fail & B pass,   harm = A pass & B fail,
on trials where BOTH A and B were scored (infra errors excluded).

The bootstrap is HIERARCHICAL and PAIRED to respect the design:
  - resample TASKS within each category (stratified), then
  - resample DRAWS within each resampled task,
so the CI reflects generalization across tasks, not just draw noise.

The C arm is the control: if B uplift ~ C uplift, the gain is "extra tokens +
a decision prompt", not advice content. We report B-vs-A and C-vs-A side by
side, and the differential (B_uplift - C_uplift).

Usage:
    python3 analyze_ablation.py /tmp/moa_ablation.jsonl [--iters 10000] [--seed 0]

Deterministic given (jsonl, iters, seed). Stdlib only.
"""

import argparse
import json
import random
import sys
from collections import defaultdict


def load(path):
    """-> trials[(task_id)] = {'category', 'draws': {draw: {arm: outcome}}}"""
    tasks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = tasks.setdefault(
                r["task_id"], {"category": r["category"], "draws": defaultdict(dict)}
            )
            t["draws"][r["draw"]][r["arm"]] = r["outcome"]
    return tasks


def paired_counts(draws, arm):
    """Rescue/harm/scored counts for `arm` vs A over this task's draws."""
    rescue = harm = scored = 0
    for d in draws.values():
        a = d.get("A")
        x = d.get(arm)
        if a in ("pass", "fail") and x in ("pass", "fail"):
            scored += 1
            if a == "fail" and x == "pass":
                rescue += 1
            elif a == "pass" and x == "fail":
                harm += 1
    return rescue, harm, scored


def task_uplift(draws, arm):
    """Net uplift for one task (mean over its scored draws), or None."""
    rescue, harm, scored = paired_counts(draws, arm)
    if scored == 0:
        return None
    return (rescue - harm) / scored


def point_estimate(tasks, arm):
    """Equal-weight mean of per-task uplift (per expert: task is the unit)."""
    vals = [u for t in tasks.values() if (u := task_uplift(t["draws"], arm)) is not None]
    return sum(vals) / len(vals) if vals else float("nan"), len(vals)


def bootstrap_ci(tasks, arm, iters, seed):
    """Hierarchical paired bootstrap: resample tasks within category, then
    draws within task. Returns (lo, hi) 95% CI for mean per-task uplift."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for tid, t in tasks.items():
        by_cat[t["category"]].append(tid)

    # Precompute per-task draw lists so resampling draws is cheap.
    draw_lists = {tid: list(t["draws"].values()) for tid, t in tasks.items()}

    samples = []
    cats = sorted(by_cat)
    for _ in range(iters):
        vals = []
        for cat in cats:
            ids = by_cat[cat]
            for _ in range(len(ids)):
                tid = ids[rng.randrange(len(ids))]
                dl = draw_lists[tid]
                if not dl:
                    continue
                # resample draws within this task, with replacement
                res = [dl[rng.randrange(len(dl))] for _ in range(len(dl))]
                rescue = harm = scored = 0
                for d in res:
                    a = d.get("A")
                    x = d.get(arm)
                    if a in ("pass", "fail") and x in ("pass", "fail"):
                        scored += 1
                        if a == "fail" and x == "pass":
                            rescue += 1
                        elif a == "pass" and x == "fail":
                            harm += 1
                if scored:
                    vals.append((rescue - harm) / scored)
        if vals:
            samples.append(sum(vals) / len(vals))
    samples.sort()
    if not samples:
        return float("nan"), float("nan")
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples)) - 1]
    return lo, hi


def arm_pass_rate(tasks, arm):
    p = n = 0
    for t in tasks.values():
        for d in t["draws"].values():
            o = d.get(arm)
            if o in ("pass", "fail"):
                n += 1
                p += o == "pass"
    return p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = load(args.jsonl)
    if not tasks:
        print("no trials found", file=sys.stderr)
        sys.exit(1)

    n_tasks = len(tasks)
    cats = defaultdict(int)
    for t in tasks.values():
        cats[t["category"]] += 1
    infra = 0
    total = 0
    for t in tasks.values():
        for d in t["draws"].values():
            for arm in ("A", "B", "C"):
                if arm in d:
                    total += 1
                    infra += d[arm] == "infra"

    print(f"tasks={n_tasks}  strata={dict(sorted(cats.items()))}")
    print(f"trials={total}  infra_excluded={infra} ({100*infra/max(total,1):.1f}%)")
    print()
    for arm in ("A", "B", "C"):
        p, n = arm_pass_rate(tasks, arm)
        label = {"A": "actor alone", "B": "actor + real", "C": "actor + shuffled"}[arm]
        print(f"  {arm} {label:18} pass {p}/{n} ({100*p/max(n,1):.0f}%)")
    print()

    b_pt, b_k = point_estimate(tasks, "B")
    c_pt, c_k = point_estimate(tasks, "C")
    b_lo, b_hi = bootstrap_ci(tasks, "B", args.iters, args.seed)
    c_lo, c_hi = bootstrap_ci(tasks, "C", args.iters, args.seed)

    print("  net uplift = P(rescue) - P(harm), equal-weight mean over tasks")
    print(f"  B (real)     uplift {b_pt:+.3f}   95% CI [{b_lo:+.3f}, {b_hi:+.3f}]  (tasks n={b_k})")
    print(f"  C (shuffled) uplift {c_pt:+.3f}   95% CI [{c_lo:+.3f}, {c_hi:+.3f}]  (tasks n={c_k})")
    print(f"  differential B-C: {b_pt - c_pt:+.3f}  (content effect beyond token/prompt effect)")
    print()

    # Verdicts (directional; the CI is what matters for a claim).
    if b_lo > 0:
        print("  => references HELP: B net uplift CI is entirely > 0")
    elif b_hi < 0:
        print("  => references HARM: B net uplift CI is entirely < 0")
    else:
        print("  => inconclusive: B net uplift CI spans 0")
    if b_pt - c_pt > 0 and b_lo > 0:
        print("  => and the gain is CONTENT (B > C), not just extra tokens/prompt")
    elif abs(b_pt - c_pt) < 0.02:
        print("  => gain (if any) is NOT content-specific (B ~ C)")


if __name__ == "__main__":
    main()
