# MoA trace recording (OpenRouter)

Records real fan-out responses from open-weight models and turns them into a
deterministic replay fixture for the MoA test suite.

The point: fake worker backends can't tell you whether the arbiter handles
*real* model behaviour — divergent tool arguments, truncated answers, malformed
tool-call text, 60x latency spreads. These scripts capture that behaviour once,
so tests can replay it forever without GPUs, network, or nondeterminism.

## Why the raw response shape is preserved

Both recorders store the **full OpenAI-shaped response** per worker —
structured `tool_calls`, `content`, `finish_reason`, usage — never flattened to
text. That matters because the two things lost by flattening are exactly the two
things the tests exist to protect:

- `tool_calls` — empty `content` on an agentic turn. Together's MoA aggregator
  reads `.content`, so its synthesis step receives a list of blank strings on
  every tool turn.
- `finish_reason` — `"length"` means the backend cut the response off. Partial
  text parses as a normal answer, so without this field a half-finished
  sentence can be returned verbatim.

## Scripts

| Script | Purpose |
|---|---|
| `orclient.py` | Minimal stdlib-only OpenRouter client (no `requests`). Owns the worker pool and tier mapping. |
| `probe_tools.py` | One-off probe: fan out with tools, then aggregate Together-style to show what their design does with tool calls. Writes `fanout.jsonl`. |
| `record.py` | Single-shot fan-out over assorted prompts incl. MT-Bench. Writes `corpus.jsonl`. |
| `record_agentic.py` | Multi-step agentic loop: fan out → take consensus tool call → feed a canned tool result → fan out again. Writes `agentic.jsonl`. |
| `make_fixture.py` | Merges both corpora into `crates/mesh-mixture-of-agents/tests/fixtures/real_traces.json`. |

## Usage

```bash
export OPENROUTER_API_KEY=...        # required
python3 record.py                    # -> corpus.jsonl
python3 record_agentic.py            # -> agentic.jsonl
python3 make_fixture.py              # -> tests/fixtures/real_traces.json
```

Stdlib only, no install step. A full re-record is a few hundred calls against
cheap open models — on the order of a couple of dollars.

## Worker pool

Nine open-weight tool-capable models, chosen so tiers line up with how
`mesh-llm` classifies names (single-digit-B ⇒ small tier):

- small: `qwen3-8b`, `qwen3.5-9b`, `ministral-8b`, `ministral-3b`
- big: `qwen3-14b`, `qwen3-32b`, `qwen3-30b-a3b`, `minimax-m2.5`,
  `mistral-small-3.2-24b`

Of 367 models on OpenRouter, 80 are open-weight *and* tool-capable, so the pool
can be widened without changing any code but the list in `orclient.py`.

## Two endpoint behaviours worth knowing

Both were found by recording, and both are now handled in `HttpBackend`:

1. **Thinking-disable flags are not universally accepted.** `minimax-m2.5`
   returns `HTTP 400: Reasoning is mandatory for this endpoint` and failed
   12/12 requests until the flags were dropped and the call retried.
2. **Reasoning models starve on a short budget.** With thinking on,
   `qwen3-32b` spent 408 reasoning tokens against a 384-token cap and returned
   `finish_reason=length` with `content: null` — 1620 characters of reasoning
   and no answer. This is why MoA forces thinking off for every worker.

## Caveats

- Workers run at `temperature: 0.8`, so each row is **one draw** from a
  stochastic process. `DRAWS = 2` per case; treat the corpus as a regression
  baseline, not ground truth about live behaviour.
- Fixtures build a `GatewayConfig` directly, so they **bypass**
  `build_moa_config` / `canonical_base_name`. Model-dedup bugs cannot be caught
  here and need their own unit tests.
- `record_agentic.py` feeds canned tool results, not a real filesystem. It
  exercises MoA's arbitration over real model output, not end-to-end agent
  correctness.
