"""Turn recorded OpenRouter traces into a Rust test fixture.

Reads agentic.jsonl (multi-step tool traces) and corpus.jsonl (single-shot
fan-out, including truncated responses) and emits a compact JSON fixture
consumed by crates/mesh-mixture-of-agents/tests/sim_real_traces.rs.

The fixture keeps *structured* tool_calls and finish_reason — the two
things that get lost if you flatten worker responses to text. Latencies
are preserved as recorded so ordering-sensitive behaviour (early exit,
first-answer grace, strong patience) replays realistically; the Rust side
scales them down so tests stay fast.
"""

import json

OUT = "../../crates/mesh-mixture-of-agents/tests/fixtures/real_traces.json"


def worker_row(w):
    return {
        "model": w["model"],
        "tier": w["tier"],
        "elapsed_ms": int(round((w["elapsed"] or 0) * 1000)),
        "finish_reason": w.get("finish_reason"),
        "content": w.get("text"),
        "tool_calls": w.get("tool_calls"),
        "error": w.get("error"),
    }


cases = []

# ── Multi-step agentic traces ────────────────────────────────────────
for line in open("agentic.jsonl"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    cases.append(
        {
            "id": f"{r['scenario']}__d{r['draw']}_s{r['step']}",
            "source": "agentic",
            "scenario": r["scenario"],
            "draw": r["draw"],
            "step": r["step"],
            "has_tools": True,
            "messages": r["messages"],
            "workers": [worker_row(w) for w in r["workers"]],
        }
    )

# ── Single-shot fan-out, incl. truncated responses ───────────────────
for line in open("corpus.jsonl"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    cases.append(
        {
            "id": f"{r['case']}__d{r['draw']}",
            "source": "corpus",
            "scenario": r["case"],
            "draw": r["draw"],
            "step": 0,
            "has_tools": r["has_tools"],
            "messages": r["messages"],
            "workers": [worker_row(w) for w in r["workers"]],
        }
    )

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=1, sort_keys=True)

# ── Report what the fixture actually contains ────────────────────────
n_trunc = sum(
    1 for c in cases for w in c["workers"] if w["finish_reason"] == "length"
)
n_tool_steps = sum(
    1 for c in cases if any(w["tool_calls"] for w in c["workers"])
)
n_mixed = 0
n_unanimous_name_diff_args = 0
for c in cases:
    ok = [w for w in c["workers"] if not w["error"]]
    tools = [w for w in ok if w["tool_calls"]]
    text = [w for w in ok if not w["tool_calls"]]
    if tools and text:
        n_mixed += 1
    if tools and not text:
        names = {t["function"]["name"] for w in tools for t in w["tool_calls"]}
        args = {
            (t["function"]["name"], t["function"].get("arguments") or "{}")
            for w in tools
            for t in w["tool_calls"]
        }
        if len(names) == 1 and len(args) > 1:
            n_unanimous_name_diff_args += 1

print(f"wrote {OUT}")
print(f"  cases:                            {len(cases)}")
print(f"  cases with >=1 tool call:         {n_tool_steps}")
print(f"  mixed tool+text cases:            {n_mixed}")
print(f"  name-unanimous, args differ:      {n_unanimous_name_diff_args}")
print(f"  truncated worker responses:       {n_trunc}")
print(f"  distinct models:                  {len({w['model'] for c in cases for w in c['workers']})}")
