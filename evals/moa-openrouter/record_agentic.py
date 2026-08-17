"""Record real multi-step agentic traces from open models on OpenRouter.

Unlike record.py (single-shot fan-out per case), this walks a scripted
agentic loop: fan out with tools, take the consensus tool call, feed a
canned tool result back, fan out again. That produces the multi-turn
shapes mesh-llm's MoA actually sees from goose/opencode.

Output: agentic.jsonl — one line per (scenario, step, draw) with every
worker's full response, finish_reason, latency, and structured
tool_calls preserved (never flattened to text).

Thinking is disabled for every worker (mesh-llm MoA policy). Where an
endpoint rejects that (minimax: "Reasoning is mandatory"), orclient
retries without the flags so the worker still contributes.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import orclient as oc

DRAWS = 2
MAX_TOKENS = 700  # headroom so long answers aren't truncated by default

# ── Agentic tool schemas, close to goose / opencode shapes ───────────
TOOLS = [
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
            "name": "search",
            "description": "Search the repository for a regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex to search for"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
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
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a string in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                },
                "required": ["path", "before", "after"],
            },
        },
    },
]


def fan_out(messages, tools, temperature=0.8, max_tokens=MAX_TOKENS):
    """Parallel fan-out. A dead worker is a recorded failure, never a dead turn."""

    def one(model):
        resp, elapsed = oc.chat(
            model,
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=True,
        )
        text, tcs = oc.first_choice(resp)
        ch = (resp.get("choices") or [{}])[0]
        usage = resp.get("usage") or {}
        return {
            "model": model,
            "tier": oc.tier(model),
            "elapsed": round(elapsed, 2),
            "error": resp.get("error"),
            "finish_reason": ch.get("finish_reason"),
            "text": text,
            "tool_calls": tcs,
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
        }

    with ThreadPoolExecutor(max_workers=len(oc.POOL)) as ex:
        return list(ex.map(one, oc.POOL))


def consensus_tool(workers):
    """Most-proposed (name, arguments) across workers, or None."""
    counts = {}
    for w in workers:
        for c in w.get("tool_calls") or []:
            key = (c["function"]["name"], c["function"].get("arguments") or "{}")
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ── Scenarios: (id, opening user message, canned tool results) ───────
# `results` maps a tool name to the observation fed back when a worker
# calls it. Keeps the loop deterministic without a real filesystem.
SCENARIOS = [
    {
        "id": "explore_error_handling",
        "user": "I need to understand error handling in this Rust project. "
        "Start by looking at what's in the src directory.",
        "results": {
            "list_dir": "main.rs\nlib.rs\nerror.rs\nconfig.rs\nnetwork/\ninference/",
            "read_file": (
                "use thiserror::Error;\n\n"
                "#[derive(Error, Debug)]\npub enum MeshError {\n"
                '    #[error("connection failed: {0}")]\n    Connection(String),\n'
                '    #[error("model not found: {0}")]\n    ModelNotFound(String),\n'
                '    #[error("timeout after {0}ms")]\n    Timeout(u64),\n}\n'
            ),
            "search": "src/error.rs:5:pub enum MeshError\nsrc/network/mod.rs:88:MeshError::Timeout",
            "run_command": "(no output)",
        },
        "steps": 3,
    },
    {
        "id": "failing_test_triage",
        "user": "The test suite is failing. Find out which test fails and why.",
        "results": {
            "run_command": (
                "running 42 tests\n"
                "test routing::tests::picks_local_first ... FAILED\n\n"
                "failures:\n---- routing::tests::picks_local_first stdout ----\n"
                "thread 'main' panicked at src/routing.rs:214:\n"
                "assertion `left == right` failed\n  left: Remote(peer-2)\n right: Local(9337)\n"
            ),
            "read_file": (
                "pub fn pick_target(targets: &[Target]) -> Target {\n"
                "    // BUG: sorts remote-first\n"
                "    let mut t = targets.to_vec();\n"
                "    t.sort_by_key(|x| matches!(x, Target::Local(_)));\n"
                "    t[0].clone()\n}\n"
            ),
            "search": "src/routing.rs:214:    assert_eq!(pick_target(&targets), Target::Local(9337));",
            "list_dir": "routing.rs\nmain.rs\nlib.rs",
        },
        "steps": 3,
    },
    {
        "id": "add_feature_edit",
        "user": "Add a `--verbose` flag to the CLI parser in src/cli.rs. Read it first.",
        "results": {
            "read_file": (
                "#[derive(Parser)]\npub struct Cli {\n"
                '    #[arg(long)]\n    pub port: u16,\n'
                '    #[arg(long)]\n    pub model: Option<String>,\n}\n'
            ),
            "list_dir": "cli.rs\nmain.rs\nlib.rs",
            "edit_file": "edited src/cli.rs (1 replacement)",
            "search": "src/cli.rs:2:pub struct Cli",
            "run_command": "   Compiling mesh-llm v0.1.0\n    Finished dev profile",
        },
        "steps": 3,
    },
    {
        "id": "ambiguous_is_it_passing",
        "user": "Is this project's test suite passing?",
        "results": {
            "run_command": "test result: ok. 42 passed; 0 failed; 0 ignored",
            "read_file": "[package]\nname = \"mesh-llm\"\n",
            "list_dir": "Cargo.toml\nsrc/\ntests/",
            "search": "tests/integration.rs:1:#[test]",
        },
        "steps": 2,
    },
    {
        "id": "no_tool_needed_concept",
        "user": "What does the Rust `?` operator do? Just explain, don't look at files.",
        "results": {},
        "steps": 1,
    },
    {
        "id": "multi_tool_choice",
        "user": "Find every place MeshError::Timeout is constructed in this repo.",
        "results": {
            "search": "src/network/mod.rs:88:  MeshError::Timeout(elapsed)\n"
            "src/inference/pipeline.rs:301:  MeshError::Timeout(ms)",
            "run_command": "src/network/mod.rs:88\nsrc/inference/pipeline.rs:301",
            "list_dir": "network/\ninference/\nerror.rs",
            "read_file": "// see MeshError::Timeout usage",
        },
        "steps": 2,
    },
]


def main():
    out = open("agentic.jsonl", "w")
    for sc in SCENARIOS:
        for draw in range(DRAWS):
            messages = [{"role": "user", "content": sc["user"]}]
            for step in range(sc["steps"]):
                workers = fan_out(messages, TOOLS)
                out.write(
                    json.dumps(
                        {
                            "scenario": sc["id"],
                            "draw": draw,
                            "step": step,
                            "has_tools": True,
                            "messages": messages,
                            "workers": workers,
                        }
                    )
                    + "\n"
                )
                out.flush()

                ok = [w for w in workers if not w["error"]]
                n_tool = sum(1 for w in ok if w["tool_calls"])
                n_trunc = sum(1 for w in ok if w["finish_reason"] == "length")
                lat = sorted(w["elapsed"] for w in ok) or [0]
                chosen = consensus_tool(ok)
                print(
                    f"{sc['id']:26s} d{draw} s{step} ok={len(ok)}/{len(workers)} "
                    f"tool={n_tool} trunc={n_trunc} lat={lat[0]}..{lat[-1]}s "
                    f"chose={chosen[0] if chosen else None}",
                    flush=True,
                )

                if not chosen:
                    break  # everyone answered in text; scenario is done

                name, args = chosen
                observation = sc["results"].get(name, "(no output)")
                messages = messages + [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"call_{step}",
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{step}",
                        "content": observation,
                    },
                ]
    out.close()
    print("\nwrote agentic.jsonl")


if __name__ == "__main__":
    sys.exit(main())
