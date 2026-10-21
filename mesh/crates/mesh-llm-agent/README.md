# mesh-llm-agent

A minimal one-shot agent behind `mesh-llm agent -t "..."`.

It runs goose's unrolled loop ([`goose-agent`](https://crates.io/crates/goose-agent))
with goose's OpenAI provider pointed at the local mesh API, and two tools:

- `shell` — one `sh -c` command with a timeout; long output is cut to head and
  tail, with the full output saved to a temp file.
- `load_skill` — loads a `SKILL.md` (or a supporting file inside it) found in
  the same roots goose searches (`.agents/skills`, `~/.agents/skills`,
  `~/.config/goose/skills`, ...).

The system prompt is deliberately short (OS, working directory, one tool call at
a time, the skill index and `AGENTS.md`), because small mesh models follow a few
lines better than a full agent prompt.

It owns no sessions, MCP extensions, permissions or compaction, and does not
depend on the `goose` crate. Its consumer is `mesh-llm-commands`.
