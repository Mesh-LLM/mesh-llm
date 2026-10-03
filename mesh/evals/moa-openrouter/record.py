"""Record real open-model fan-out responses into a replay corpus.

Writes corpus.jsonl: one line per (prompt, draw), each holding every
worker's full response + observed latency. Feeds a Rust ScriptedBackend.

Workers get reasoning_effort=none (mirrors mesh-llm's
effective_enable_thinking_for_moa default) so reasoning models don't
burn the whole budget thinking.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import orclient as oc
from probe_tools import TOOLS

DRAWS = 2


def fan_out(messages, tools, temperature=0.8):
    def one(model):
        resp, elapsed = oc.chat(
            model,
            messages,
            tools=tools,
            max_tokens=512,
            temperature=temperature,
            no_think=True,
        )
        text, tcs = oc.first_choice(resp)
        ch = (resp.get("choices") or [{}])[0]
        return {
            "model": model,
            "tier": oc.tier(model),
            "elapsed": round(elapsed, 2),
            "error": resp.get("error"),
            "finish_reason": ch.get("finish_reason"),
            "text": text,
            "tool_calls": tcs,
        }

    with ThreadPoolExecutor(max_workers=len(oc.POOL)) as ex:
        return list(ex.map(one, oc.POOL))


# ── Cases: agentic tool turns + plain chat ───────────────────────────
CASES = []

CASES.append(
    {
        "id": "agentic_explore",
        "tools": True,
        "messages": [
            {
                "role": "user",
                "content": "I need to understand error handling in this Rust project. "
                "Start by looking at what's in the src directory.",
            }
        ],
    }
)

CASES.append(
    {
        "id": "agentic_tool_result_chain",
        "tools": True,
        "messages": [
            {"role": "user", "content": "Understand error handling in this Rust project."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": '{"path": "src"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "main.rs\nlib.rs\nerror.rs\nconfig.rs\nnetwork/\ninference/",
            },
        ],
    }
)

CASES.append(
    {
        "id": "agentic_ambiguous",
        "tools": True,
        "messages": [
            {"role": "user", "content": "Is this project's test suite passing?"}
        ],
    }
)

CASES.append(
    {
        "id": "agentic_no_tool_needed",
        "tools": True,
        "messages": [
            {"role": "user", "content": "What does the Rust `?` operator do?"}
        ],
    }
)

CASES.append(
    {
        "id": "chat_factual",
        "tools": False,
        "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
    }
)

CASES.append(
    {
        "id": "chat_arithmetic",
        "tools": False,
        "messages": [
            {"role": "user", "content": "A train leaves at 14:35 and arrives 2h50m later. What time?"}
        ],
    }
)

# Pull a few real MT-Bench prompts if available.
try:
    with open("../moe/prompts/mt-bench-8.jsonl") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # This corpus uses OpenAI-shaped `messages`; older MT-Bench
            # dumps use `turns`. Accept either.
            first_user = None
            for m in row.get("messages") or []:
                if m.get("role") == "user":
                    first_user = m.get("content")
                    break
            if first_user is None:
                turns = row.get("turns") or []
                first_user = turns[0] if turns else None
            if not first_user:
                continue
            CASES.append(
                {
                    # `id` is unique per row; `category` is NOT (all 8 rows
                    # in this corpus are "writing") so it must not be the key.
                    "id": f"mtbench_{row.get('id') or i}",
                    "tools": False,
                    "messages": [{"role": "user", "content": first_user}],
                }
            )
except FileNotFoundError:
    print("(mt-bench not found, skipping)", file=sys.stderr)


def main():
    out = open("corpus.jsonl", "w")
    for case in CASES:
        for draw in range(DRAWS):
            results = fan_out(case["messages"], TOOLS if case["tools"] else None)
            out.write(
                json.dumps(
                    {
                        "case": case["id"],
                        "draw": draw,
                        "has_tools": case["tools"],
                        "messages": case["messages"],
                        "workers": results,
                    }
                )
                + "\n"
            )
            out.flush()

            ok = [r for r in results if not r["error"]]
            tools_n = sum(1 for r in ok if r["tool_calls"])
            empty_n = sum(1 for r in ok if not r["tool_calls"] and not (r["text"] or "").strip())
            names = {
                c["function"]["name"] for r in ok for c in (r["tool_calls"] or [])
            }
            lat = sorted(r["elapsed"] for r in ok)
            print(
                f"{case['id']:28s} draw={draw} ok={len(ok)}/{len(results)} "
                f"tool={tools_n} empty={empty_n} distinct_tools={sorted(names)} "
                f"lat={lat[0] if lat else 0}..{lat[-1] if lat else 0}s"
            )
    out.close()
    print("\nwrote corpus.jsonl")


if __name__ == "__main__":
    main()
