# llama-spec-bench

Local target/draft speculative decoding checker and benchmark.

`llama-spec-bench` compares a target GGUF model with a draft GGUF model on a
prompt set. It checks tokenizer compatibility, verifies speculative output
against baseline target decoding, measures acceptance behavior, and reports
the measured target and draft costs of the serial diagnostic loop.

## Architecture Role

This crate runs both models locally through `skippy-runtime`. It is a
preflight tool for deciding whether a draft model is safe and useful before it
is wired into mesh-owned stage serving, `skippy-correctness` diagnostics, or
benchmark launchers.
The target and draft are opened as complete local models without tensor
filtering; this keeps draft compatibility focused on tokenizer agreement and
full-model decode behavior instead of requiring every candidate draft
architecture to support staged tensor filtering.

```mermaid
flowchart LR
    Corpus["prompt(s)<br/>inline or JSONL corpus"] --> Bench["llama-spec-bench"]
    Target["target model<br/>full runtime session"] --> Bench
    Draft["draft model<br/>full runtime session"] --> Bench
    Bench --> Base["baseline target decode"]
    Bench --> Spec["draft proposals<br/>target verification"]
    Base --> Compare["token equality check"]
    Spec --> Compare
    Compare --> Report["human summary<br/>optional JSON report"]
    Report --> Mesh["mesh/skippy plan<br/>draft opt-in evidence"]
```

## Verification Loop

```mermaid
sequenceDiagram
    participant B as bench
    participant T as target
    participant D as draft

    B->>T: prefill prompt
    B->>D: prefill prompt
    loop until max_new_tokens
        B->>D: draft speculative window
        D-->>B: proposed tokens
        B->>T: verify proposals
        T-->>B: accepted prefix or rejection
        B->>B: update acceptance stats
    end
    B->>B: compare against baseline target tokens
```

## Commands

```bash
llama-spec-bench \
  --target-model-path target.gguf \
  --draft-model-path draft.gguf \
  --prompt "Write a short Rust function." \
  --max-new-tokens 128 \
  --speculative-window 4

llama-spec-bench \
  --target-model-path target.gguf \
  --draft-model-path draft.gguf \
  --prompt-corpus skippy/crates/skippy-bench/corpora/kv_mixed_prompts.jsonl \
  --prompt-limit 20 \
  --json-out /tmp/spec-bench.json
```

Use `--allow-mismatch` only while investigating failures; by default, any
speculative output mismatch makes the command fail.

`--dflash` treats the draft as a DFlash or DFlash2 block drafter. The draft is
attached to the target model instead of running its own session: it reads
target hidden states, drafts up to `--speculative-window` tokens per block, and
the target verifies each block in one batched decode.

```bash
llama-spec-bench \
  --target-model-path Qwen3-4B-Q4_K_M.gguf \
  --draft-model-path Qwen3-4B-DFlash.Q8_0.gguf \
  --dflash \
  --speculative-window 15
```

Batched verification can round differently from single-token decode, so a long
generation may legitimately diverge from the baseline at a near-tie token.

## Report Contents

- prompt/token counts
- tokenizer compatibility
- accepted/rejected draft token counts
- acceptance rate and mean accepted tokens per window
- baseline and speculative decode timing
- measured target verification and draft proposal costs
- measured speculative throughput and speedup versus target-only decoding
- per-prompt text previews and mismatch index

The default corpus path is
`skippy/crates/skippy-bench/corpora/kv_mixed_prompts.jsonl`.
