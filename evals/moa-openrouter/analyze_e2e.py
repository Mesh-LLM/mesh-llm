#!/usr/bin/env python3
"""Score an e2e MoA run: per-judge win/tie/loss, sign test, judge agreement,
length-controlled subset.

Usage: python3 analyze_e2e.py run.jsonl [run2.jsonl ...]

Why each number is here:

- **Sign test** on decided trials only. Ties carry no directional information,
  so including them in the denominator understates the effect and inflates n.
- **Second-judge columns** come from the *same* answer texts, so a disagreement
  is a judge effect, not a resampling effect. A verdict that only one judge
  sees is not bankable.
- **Shorter-MoA subset** is the length control. This judge has shown
  r(length, verdict) up to +0.465, so when MoA output is longer than solo the
  bias runs *with* the winner; the subset where MoA is shorter is the arm where
  a win cannot be explained by length.
"""

import json
import math
import sys
from collections import Counter


def sign_p(w, l):
    """Two-sided exact binomial sign test at p=0.5 on decided trials."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def wtl(rows, field):
    c = Counter(r.get(field) for r in rows)
    return c.get(1, 0), c.get(0, 0), c.get(-1, 0)


def report(path):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    print(f"\n=== {path}  n={len(rows)} ===")

    for label, field in (("judge1", "b_vs_a"), ("judge2", "b_vs_a_j2")):
        if field == "b_vs_a_j2" and not any(r.get(field) is not None for r in rows):
            continue
        w, t, l = wtl(rows, field)
        print(
            f"  {label}: win {w:3}  tie {t:3}  loss {l:3}"
            f"   decided {w + l:3}   sign p = {sign_p(w, l):.5f}"
        )

    both = [r for r in rows if r.get("b_vs_a_j2") is not None]
    if both:
        agree = sum(1 for r in both if r["b_vs_a_j2"] == r["b_vs_a"])
        # Disagreement only matters where at least one judge decided; two
        # independent ties agree trivially and would inflate the rate.
        decided = [r for r in both if r["b_vs_a"] != 0 or r["b_vs_a_j2"] != 0]
        d_agree = sum(1 for r in decided if r["b_vs_a_j2"] == r["b_vs_a"])
        opposed = sum(
            1 for r in decided if r["b_vs_a"] * r["b_vs_a_j2"] == -1
        )
        print(
            f"  agreement: {agree}/{len(both)} overall;"
            f" {d_agree}/{len(decided)} where either judge decided;"
            f" {opposed} directly opposed"
        )
        # Conjunction: a win only both judges see. The strictest reading.
        cw = sum(1 for r in both if r["b_vs_a"] == 1 and r["b_vs_a_j2"] == 1)
        cl = sum(1 for r in both if r["b_vs_a"] == -1 and r["b_vs_a_j2"] == -1)
        print(
            f"  BOTH judges agree: win {cw}  loss {cl}"
            f"   sign p = {sign_p(cw, cl):.5f}"
        )

    shorter = [r for r in rows if r["len_b"] < r["len_a"]]
    if shorter:
        w, t, l = wtl(shorter, "b_vs_a")
        print(
            f"  length control (MoA shorter, n={len(shorter)}):"
            f" win {w} tie {t} loss {l}  sign p = {sign_p(w, l):.5f}"
        )

    la = sum(r["len_a"] for r in rows) / max(len(rows), 1)
    lb = sum(r["len_b"] for r in rows) / max(len(rows), 1)
    print(f"  mean chars: solo {la:.0f}  MoA {lb:.0f}")

    strata = sorted({r["category"] for r in rows})
    for s in strata:
        sub = [r for r in rows if r["category"] == s]
        w, t, l = wtl(sub, "b_vs_a")
        print(f"    {s:22} {w}-{l} (n={len(sub)})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
