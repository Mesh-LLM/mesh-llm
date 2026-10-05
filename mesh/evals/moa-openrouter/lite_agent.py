#!/usr/bin/env python3
"""Lite goose-style agentic harness against a mesh-llm OpenAI endpoint.

Drives a real tool-call loop through `model=mesh` (or any model) to prove the
MoA gateway emits usable `tool_calls` end to end on a live 2-node mesh — the
path OpenRouter standins could not exercise (gossip, dedup, self-fill, the
actual proxy).

Tools are canned/filesystem-lite so the loop is deterministic and offline
apart from the model calls. Prints a worklog of every step.

Usage:
  python3 lite_agent.py --base http://localhost:9337/v1 --model mesh
  python3 lite_agent.py --base http://localhost:9337/v1 --model mesh --task find_symbol
"""
import argparse
import json
import os
import time
import urllib.request

# --- canned workspace the agent can act on (deterministic tool results) ---
WORKSPACE = {
    "src/lib.rs": "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\npub fn timeout() -> MeshError { MeshError::Timeout }\n",
    "src/error.rs": "pub enum MeshError { Timeout, Reset }\n",
    "README.md": "# demo\nA tiny crate. Run `cargo test`.\n",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files under a directory path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file by path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the codebase for a regex or substring.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TASKS = {
    "explore": "List the files under src, then read src/error.rs and tell me what error variants exist.",
    "find_symbol": "Find every place MeshError::Timeout is used in this repo, then summarise where.",
    "explain": "What does the add function in src/lib.rs do? Read it first.",
}


def run_tool(name, args):
    """Execute a canned tool against WORKSPACE. Returns a string result."""
    if name == "list_dir":
        p = (args.get("path") or "").strip("/")
        hits = sorted(f for f in WORKSPACE if f.startswith(p))
        return "\n".join(hits) if hits else f"(no files under {p!r})"
    if name == "read_file":
        p = (args.get("path") or "").lstrip("./")
        return WORKSPACE.get(p, f"(no such file: {p!r})")
    if name == "search":
        q = args.get("query") or ""
        # substring match, tolerate regex-ish word-boundary noise
        needle = q.replace("\\b", "").replace("\\", "")
        hits = [f"{f}: {ln}" for f, body in WORKSPACE.items() for ln in body.splitlines() if needle and needle in ln]
        return "\n".join(hits) if hits else f"(no matches for {q!r})"
    return f"(unknown tool {name!r})"


def chat(base, model, messages, api_key=None, timeout=180):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:9337/v1")
    ap.add_argument("--model", default="mesh")
    ap.add_argument("--task", default="explore", choices=list(TASKS))
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    messages = [
        {"role": "system", "content": "You are a coding agent. Use the provided tools to inspect the "
                                       "repository before answering. Call one tool at a time."},
        {"role": "user", "content": TASKS[args.task]},
    ]

    print(f"=== lite-agent: model={args.model} task={args.task} base={args.base} ===")
    print(f"USER: {TASKS[args.task]}\n")

    tool_calls_made = 0
    for step in range(1, args.max_steps + 1):
        try:
            resp, dt = chat(args.base, args.model, messages)
        except Exception as e:  # noqa: BLE001
            print(f"[step {step}] REQUEST FAILED: {e}")
            return 2
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        moa_workers = resp.get("usage", {})  # x-moa headers arrive as headers; body usage is fine to show
        if calls:
            messages.append(msg)
            for c in calls:
                fn = c["function"]["name"]
                try:
                    fa = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    fa = {}
                tool_calls_made += 1
                result = run_tool(fn, fa)
                print(f"[step {step}] ({dt:.1f}s) TOOL CALL: {fn}({json.dumps(fa)})")
                print(f"           -> {result.splitlines()[0] if result else '(empty)'}"
                      + (" ..." if result.count("\n") else ""))
                messages.append({
                    "role": "tool",
                    "tool_call_id": c.get("id", f"call_{step}"),
                    "content": result,
                })
        else:
            content = (msg.get("content") or "").strip()
            print(f"[step {step}] ({dt:.1f}s) FINAL ANSWER:\n{content}\n")
            print(f"=== done: {tool_calls_made} tool call(s) over {step} step(s) ===")
            return 0 if tool_calls_made > 0 else 1

    print(f"=== hit max steps ({args.max_steps}); {tool_calls_made} tool call(s) made ===")
    return 0 if tool_calls_made > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
