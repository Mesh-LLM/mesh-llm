#!/usr/bin/env python3
"""Generate a realistic agent-harness-style prompt prefix for KV cache
benchmarks.

The acceptance criteria for the L3 tier ask for warm-up measured against a
real ~19K-token agent prefix (opencode/goose/pi/buzz shape), not hello-world.
When a captured harness prefix is available, use that instead — this
generator produces a stand-in with the same texture: a long system prompt,
JSON tool schemas, environment/repo context, and multi-turn scaffolding,
deterministic for a given target size.

Usage: generate-agent-prefix.py [target_tokens] [out_file]
Token count is estimated at ~3.4 characters/token for English-plus-JSON.
"""

import json
import sys

CHARS_PER_TOKEN = 3.4

TOOLS = [
    ("read_file", "Read a file from the workspace", {"path": "string", "offset": "integer", "limit": "integer"}),
    ("write_file", "Create or overwrite a file", {"path": "string", "content": "string"}),
    ("edit_file", "Replace an exact string in a file", {"path": "string", "old_string": "string", "new_string": "string"}),
    ("bash", "Execute a shell command and return stdout/stderr", {"command": "string", "timeout_ms": "integer"}),
    ("grep", "Search file contents with a regular expression", {"pattern": "string", "path": "string", "glob": "string"}),
    ("list_dir", "List directory entries", {"path": "string", "depth": "integer"}),
    ("web_fetch", "Fetch a URL and return the page text", {"url": "string", "prompt": "string"}),
    ("task_update", "Update the shared task list", {"task_id": "string", "status": "string", "note": "string"}),
]

SYSTEM = """You are a coding agent operating inside a repository checkout. You work
autonomously: plan briefly, then act with tools. Prefer small verifiable
steps; run the test suite after any change that could affect behavior.
Never fabricate file contents or command output — read before you write,
and quote real paths. When a task is ambiguous, choose the interpretation
that is reversible and state the assumption in your summary. Keep diffs
minimal and idiomatic to the surrounding code. Cite files as path:line.
"""

FILES = [
    ("src/scheduler/engine.rs", "iteration planner: batches prefill and decode work per lane"),
    ("src/cache/radix.rs", "unified radix cache over resident and exact-state components"),
    ("src/transport/frames.rs", "length-prefixed binary framing for stage links"),
    ("src/api/openai.rs", "OpenAI-compatible ingress: completions and chat endpoints"),
    ("src/runtime/session.rs", "session lifecycle over the native runtime FFI"),
    ("docs/ARCHITECTURE.md", "system overview: stages, lanes, cache tiers, planners"),
]


def tool_block() -> str:
    blocks = []
    for name, description, params in TOOLS:
        blocks.append(json.dumps({
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in params.items()},
                "required": list(params)[:1],
            },
        }, indent=2))
    return "Available tools:\n" + "\n".join(blocks)


def context_round(index: int) -> str:
    file_path, summary = FILES[index % len(FILES)]
    return (
        f"\n[context {index}] {file_path} — {summary}. "
        f"Recent change {index}: refactored the {summary.split(':')[0]} to "
        f"separate policy from mechanism; follow-ups tracked as item {index} "
        f"in the task list. Invariants to preserve: ordering of admitted "
        f"work, bounded memory per lane, and deterministic replay of the "
        f"decision log. Test entry points: cargo test -p module-{index % 7}, "
        f"plus the integration smoke behind the `slow` feature.\n"
        + "".join(
            f"  {file_path}:{20 * line + index}: fn handler_{line}_{index}"
            f"(state: &mut State, event: Event) -> Result<Action>\n"
            for line in range(6)
        )
    )


def main() -> None:
    target_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 19_000
    out_path = sys.argv[2] if len(sys.argv) > 2 else "agent-prefix.txt"
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    parts = [SYSTEM, tool_block()]
    index = 0
    size = sum(len(part) for part in parts)
    while size < target_chars:
        part = context_round(index)
        parts.append(part)
        size += len(part)
        index += 1
    prefix = "".join(parts)[:target_chars]
    with open(out_path, "w") as handle:
        handle.write(prefix)
    print(f"wrote ~{int(len(prefix) / CHARS_PER_TOKEN)} tokens ({len(prefix)} chars) to {out_path}")


if __name__ == "__main__":
    main()
