# Prompt-compilation workflow evaluation

This is an **eval-only**, stdlib-only harness for testing whether narrow typed analyses improve a fixed sub-10B actor over complete OpenAI tool workflows. It does not change the production MoA gateway. Live OpenRouter calls require both `--live` and `OPENROUTER_API_KEY`.

## Arms

All arms use one pinned actor, identical sampling/token budgets, original ordered messages, and unchanged tool JSON.

- **A** — neutral user-level carrier: no observations.
- **B** — current-style tool-free prose advice in the same carrier.
- **C** — strict extractive analyses merged deterministically.
- **D** — valid compiled brief for the next scenario (negative control for generated-context bias).
- **E** — hand-authored oracle brief (mechanism ceiling).

The fallible brief is never placed in `system` or `developer`. It is appended inside the **latest existing user message**, even after tool results, so native `user → assistant(tool_calls) → tool` ordering stays valid. The transformation is reversible. Every actor turn logs hashes of original messages, actor messages after removing the carrier, and tools, plus booleans proving role order and trusted messages were unchanged.

## Grounding and schema

Each analyst sees a bounded, role-specific map of stable source IDs and an explicit JSON template; validation is against that exact map, not the global transcript catalog. Fact kinds are role-restricted: only the intent analyst may classify authoritative system/developer/user extracts as possible goals, constraints, or output contracts; state and tool analysts cannot promote assistant/tool text into those kinds. Exact quotes are mechanically verified, while model-assigned goal/constraint/state/risk classifications and all recommended actions are visibly labeled unverified semantic hypotheses. Candidate tools must be empty for non-tool modes and known for tool modes. Argument fields must exist in a candidate schema; values must occur in the quoted span or equal the single allowlisted `derive exactly from quoted source` token. Invalid analyses are logged with raw errors, while the actor brief receives only a fixed diagnostic code. Analyst transport/HTTP failures make the trial infrastructure-failed rather than capability-failed. Analysts never receive executable authority.

`prompt_compile.py` owns strict validation, source-span checks, deterministic merging, reversible carrier injection, hashes, corpus validation, and oracle action scoring. `run.py` owns opt-in calls and the sandbox loop. Malformed native tool arguments are protocol failures. Provider-only assistant fields are discarded before continuing the canonical OpenAI transcript. Severe safety scoring covers both prohibited tool names and state-level tool/argument predicates. HTTP 429/5xx and transport failures are retried and then recorded as `infra`, which the scorer excludes from capability comparisons. Replay loading rejects duplicate trial keys and mixed actor/analyst/corpus identities; missing metrics serialize as standards-compliant JSON `null`.

## Offline validation

```bash
python3 -m unittest discover -s evals/moa-prompt-compile -p 'test_*.py'
python3 -m py_compile evals/moa-prompt-compile/*.py
python3 evals/moa-prompt-compile/run.py
```

The last command validates and hashes the fixture but makes no network request.

## Live pilot and replay

```bash
export OPENROUTER_API_KEY=...
python3 evals/moa-prompt-compile/run.py --live --draws 3 \
  --output /tmp/moa-prompt-compile.jsonl
python3 evals/moa-prompt-compile/score.py /tmp/moa-prompt-compile.jsonl
```

The initial corpus has six representative 1–4-turn state machines: trusted constraint recovery, list/read dependency, missing-file recovery, answer from completed evidence, exact historical path, and loop/instruction-in-tool-output pressure. Expand to the preregistered 24 scenarios (four per stratum) before making a product claim. Suggested go bar: C−A ≥8 points with clustered 95% CI excluding zero, C−B positive, no more than +1 point severe violations, then replication with another sub-10B actor family.

## Prefix/KV reuse arm

**F** is the cacheability arm: compile once on turn 0, keep that user-level carrier byte-stable, and append canonical assistant tool-call and tool-result messages without mutating prior messages. **C** recompiles and replaces the reversible latest-user carrier each turn. Every turn logs the exact common prefix at message boundaries—message count, summed canonical bytes of those complete shared messages, hash, explicit append-only equality against the prior actor request, unchanged model/tools-field hash, and stable-prefix fraction—as a tokenizer-independent cacheability proxy. The scorer reports explicit F-vs-C deltas for append-only rate and common-prefix bytes. This is not a whole-request byte metric or a claim about backend KV hits; a production experiment would additionally collect backend `cached_tokens`.

The deterministic boundary checks are intentionally split: analyst pre-call validation checks schemas, exact provenance spans, candidate tools, and proposed argument fields; a separate actor-call validator checks known tool, object arguments, required fields, and supported unknown-field/type/enum constraints before any canned result; post-result classification labels success/failure/empty output and exposes recovery/loop behavior to scoring without granting any helper execution authority.

This first instrument conditionally measures capability when helpers return; it excludes helper-infrastructure trials and does **not** execute an actor-alone fallback after helper timeout. That fallback remains a required degraded-helper/fail-open experiment before any operational recommendation.

## Follow-up: authority-aware cognitive mesh

This first instrument tests a prompt compiler, not a scheduler. The follow-up target is broader: one authoritative, append-only event timeline and one moderate actor that alone receives native tools and may execute them. Optional small-model observers receive bounded, versioned snapshots and return analysis tickets; they never mutate history, elevate authority, or execute a tool.

Two artifact lanes must remain explicit:

1. **Mechanically verified evidence** — exact extractions, copied scalar values, schema facts, source spans, and deterministic protocol checks. Acceptance means the mechanical predicate passed; it does not establish semantic entailment.
2. **Unverified semantic hypotheses** — interpretations, reasons, tool suggestions, critiques, alternative plans, and recovery ideas. These may be useful, but are visibly fallible and retain citations rather than being laundered into facts.

Candidate observer tickets are: intent/constraints; simple-list tool scouting; new-result interpretation; failure recovery; pre-call critique; and evidence/final-answer critique. The first factorial study should compare actor-only against one-observer-at-a-time additions, then a preregistered small set of interactions. It must also cross useful observers with degraded-helper conditions: timeout/missing, malformed schema, stale snapshot, valid-but-task-shuffled output, and confidently wrong semantic hypothesis. A monolithic all-observers arm may be included for operational cost, but cannot identify marginal utility.

### Proposed ticket and event contract

Every authoritative event has `event_id`, monotonic `timeline_version`, canonical payload hash, role/type, and creation time. An observer ticket records `ticket_id`, `observer_role`, `snapshot_version`, ordered source event IDs and hashes, requested artifact lane, deadline, and byte/count budget. A result repeats those identifiers and adds model/config identity, start/finish times, status, mechanically verified artifacts, unverified hypotheses, validation diagnostics, usage, and latency. Admission requires matching ticket and source hashes, completion before the deadline, and a snapshot version still accepted by policy. Late, stale, malformed, missing, or oversized results are logged and ignored; the actor proceeds alone. Accumulation is bounded by per-ticket and per-epoch limits.

The KV-oriented shape is a compile-once stable checkpoint followed by append-only observation deltas. Prior actor-visible bytes are not rewritten during an epoch. Intentional compaction starts a new, explicitly identified epoch with a new checkpoint and provenance back to the covered event range; measurements must not describe that boundary as a cache hit. This PR does not implement scheduling, deadlines, or compaction—it preserves seams for later version/provenance fields and establishes strict bounded admission first.

### Follow-up matrix and outcomes

Report workflow success and severe violations by scenario stratum, observer, degradation, and interaction cell. Also report helper acceptance/rejection/staleness, actor and helper latency/usage, ticket deadline misses, accumulated bytes, compaction epochs, and message-boundary prefix proxies. Primary comparisons are scenario-clustered. The causal questions are: each observer's marginal task utility; whether semantic hypotheses help beyond verified evidence; whether degraded helpers are safely no worse than actor-only; and whether compile-once plus deltas improves reusable-prefix proxies over turn-by-turn recompilation. Production work is gated on those results rather than assumed from this initial composition.
