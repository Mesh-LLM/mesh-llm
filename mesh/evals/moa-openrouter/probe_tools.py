"""Probe: does Together-style MoA survive agentic tool calling?

Runs three shapes against real open-weight models on OpenRouter:

  A. Fan-out with tools, then aggregate Together-style (numbered
     plaintext concat of worker text into a system prompt).
  B. The same, but reporting what mesh-llm's arbiter would see
     (structured tool proposals + consensus).
  C. A tool-result turn: feed a real tool result back and aggregate.

Everything is recorded to fanout.jsonl so it can seed a replay corpus.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import orclient as oc

# Real agentic tool schemas, close to what goose/opencode send.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return stdout",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Command to run"}},
                "required": ["cmd"],
            },
        },
    },
]

# Together's aggregator system prompt, verbatim from their utils.py.
TOGETHER_AGG_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

RECORD = open("fanout.jsonl", "a")


def record(kind, **kw):
    RECORD.write(json.dumps({"kind": kind, **kw}) + "\n")
    RECORD.flush()


def fan_out(messages, tools, max_tokens=384):
    """Parallel fan-out across the pool. Never raises — a dead worker is
    a recorded failure, not a dead turn (unlike Together's asyncio.gather).
    """

    def one(model):
        resp, elapsed = oc.chat(model, messages, tools=tools, max_tokens=max_tokens, temperature=0.8)
        text, tcs = oc.first_choice(resp)
        return {
            "model": model,
            "tier": oc.tier(model),
            "elapsed": round(elapsed, 2),
            "error": resp.get("error"),
            "text": text,
            "tool_calls": tcs,
        }

    with ThreadPoolExecutor(max_workers=len(oc.POOL)) as ex:
        return list(ex.map(one, oc.POOL))


def summarize(results, label):
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    for r in results:
        if r["error"]:
            print(f"  {r['model']:40s} [{r['tier']:5s}] FAILED: {str(r['error'])[:60]}")
            continue
        tcs = r["tool_calls"] or []
        if tcs:
            calls = ", ".join(
                f"{c['function']['name']}({c['function']['arguments']})" for c in tcs
            )
            print(f"  {r['model']:40s} [{r['tier']:5s}] {r['elapsed']:5.1f}s TOOL: {calls}")
        else:
            print(
                f"  {r['model']:40s} [{r['tier']:5s}] {r['elapsed']:5.1f}s TEXT: "
                f"{(r['text'] or '')[:70]!r}"
            )
    return results


def together_aggregate(user_prompt, results, aggregator, with_tools):
    """Together's aggregation: numbered plaintext concat of worker *text*.

    Note what necessarily happens to tool calls here: the references are
    built from `.text`, because that is all their `inject_references_to_
    messages` knows how to read. Structured tool_calls have nowhere to go.
    """
    refs = []
    for r in results:
        if r["error"]:
            continue
        refs.append(r["text"] or "")  # <-- tool_calls are dropped on the floor

    system = TOGETHER_AGG_PROMPT
    for i, ref in enumerate(refs):
        system += f"\n{i + 1}. {ref}"

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    resp, elapsed = oc.chat(
        aggregator, msgs, tools=TOOLS if with_tools else None, max_tokens=384, temperature=0.3
    )
    text, tcs = oc.first_choice(resp)
    return {"text": text, "tool_calls": tcs, "elapsed": round(elapsed, 2), "error": resp.get("error")}


def main():
    aggregator = "qwen/qwen3-32b"

    # ── A. Agentic fan-out: does every worker propose a tool? ────────
    prompt_a = (
        "I need to understand the error handling in this Rust project. "
        "Start by looking at what's in the src directory."
    )
    msgs_a = [{"role": "user", "content": prompt_a}]
    res_a = summarize(fan_out(msgs_a, TOOLS), "A. FAN-OUT WITH TOOLS (agentic first turn)")
    record("fanout_tools", prompt=prompt_a, results=res_a)

    proposals = {}
    for r in res_a:
        for c in r["tool_calls"] or []:
            proposals.setdefault(c["function"]["name"], []).append(r["model"])
    print(f"\n  tool proposals: {json.dumps(proposals, indent=2)}")
    n_tool = sum(1 for r in res_a if r["tool_calls"])
    n_text = sum(1 for r in res_a if not r["tool_calls"] and not r["error"])
    print(f"  -> {n_tool} workers proposed tools, {n_text} answered with text only")

    # ── A2. Aggregate Together-style, tools available ────────────────
    agg = together_aggregate(prompt_a, res_a, aggregator, with_tools=True)
    print(f"\n  Together-style aggregation (aggregator={aggregator}):")
    print(f"    elapsed={agg['elapsed']}s error={agg['error']}")
    print(f"    tool_calls={agg['tool_calls']}")
    print(f"    text={(agg['text'] or '')[:200]!r}")
    record("aggregate_tools", prompt=prompt_a, aggregator=aggregator, result=agg)

    # What the aggregator actually received as "references":
    refs_seen = [(r["model"], (r["text"] or "")[:40], bool(r["tool_calls"])) for r in res_a if not r["error"]]
    print("\n  what Together's aggregator SAW (text only, per model):")
    for m, t, had_tool in refs_seen:
        flag = "  <-- had a tool_call that was DROPPED" if had_tool else ""
        print(f"    {m:40s} {t!r}{flag}")

    # ── B. Tool-result turn (agentic step 2) ─────────────────────────
    tool_call_id = "call_probe_1"
    msgs_b = [
        {"role": "user", "content": prompt_a},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": '{"path": "src"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": "main.rs\nlib.rs\nerror.rs\nconfig.rs\nnetwork/\ninference/",
        },
    ]
    res_b = summarize(fan_out(msgs_b, TOOLS), "B. TOOL-RESULT TURN (agentic step 2)")
    record("fanout_tool_result", results=res_b)
    n_follow = sum(1 for r in res_b if r["tool_calls"])
    print(f"\n  -> {n_follow} workers chained a follow-up tool call after seeing the result")

    # ── C. Plain chat fan-out for comparison (no tools) ──────────────
    prompt_c = "What are 3 fun things to do in SF?"
    res_c = summarize(
        fan_out([{"role": "user", "content": prompt_c}], None),
        "C. PLAIN CHAT FAN-OUT (no tools — Together's home turf)",
    )
    agg_c = together_aggregate(prompt_c, res_c, aggregator, with_tools=False)
    print(f"\n  aggregated: {(agg_c['text'] or '')[:240]!r}")
    record("fanout_chat", prompt=prompt_c, results=res_c, aggregate=agg_c)

    print("\n" + "=" * 74)
    print("recorded to fanout.jsonl")


if __name__ == "__main__":
    sys.exit(main())
