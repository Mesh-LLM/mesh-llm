# Speculative Decoding Wiring Plan

## Goal

Every speculative decoding setting exposed in `config.toml`, the CLI, and the
configuration schema must either work end-to-end or be removed/renamed before it
can mislead users. The first implementation pass should keep existing persisted
TOML compatible, but it must stop treating schema-only fields as complete.

Benchmark tuning must then treat speculative decoding as a first-class sweep
dimension. `mesh-llm benchmark tune` should automatically discover viable
speculative modes, try them in the order most likely to improve decode tok/s,
and write trial configs that use the same resolver path as normal serving.

## Current Status And Gaps

- `mode = "ngram"` is accepted by config validation, resolved for staged
  serving, translated into embedded OpenAI args, and included in benchmark tune
  auto fallback candidates.
- Draft-model speculation wires local `draft_model`, `draft_max_tokens`,
  `draft_min_tokens`, `draft_gpu_layers`, and `pairing_fault`.
- Draft HF source fields are validated but runtime rejects them.
- Draft runtime placement fields (`draft_device`, `draft_threads`,
  `draft_cache_type_k`, `draft_cache_type_v`) are validated but runtime rejects
  them.
- Draft acceptance controls (`draft_acceptance_threshold`,
  `draft_split_probability`) are validated but runtime rejects them.
- `spec_default = true` is validated but runtime rejects it.
- `strategy = "auto"` enables native MTP when package generation metadata or
  direct GGUF inspection proves native MTP support.
- Native MTP tuning controls are environment variables rather than first-class
  config fields.
- `mesh-llm benchmark tune` sweeps model fit, KV cache, `mmap`, `mlock`, native
  MTP, local draft-model candidates, ngram candidates, and disabled baselines.
- Trial output does not summarize speculative acceptance, rejection, draft
  window, or native-MTP verification telemetry, making a low-throughput MTP run
  hard to diagnose.

## Implementation Order

### 1. Lock Current Failures With Tests

Add failing tests before implementation:

- Resolver tests proving each field in `SpeculativeConfig` reaches a resolved
  runtime representation instead of producing "unsupported by embedded runtime".
- Direct GGUF MTP detection tests using a synthetic GGUF metadata/tensor fixture
  or a factored detection function that proves `.nextn.` tensors enable
  `strategy = "auto"`.
- Translation tests proving resolved speculative settings reach:
  - `EmbeddedOpenAiArgs`
  - `SkippyModelLoadOptions`
  - stage `StageConfig`
  - binary transport load requests where applicable
- Config validation/schema contract tests for any new enum values or renamed
  fields.
- CLI parsing tests for new runtime flags and benchmark tune flags.

### 2. Expand Resolved Runtime Types

Update `ResolvedSpeculativeConfig` to represent all supported fields instead of
dropping or rejecting them:

- `strategy`
- `native_mtp_enabled`
- `mode`
- `draft_model`
- `draft_hf_repo`
- `draft_hf_file`
- `draft_selection_policy`
- `pairing_fault`
- `draft_max_tokens`
- `draft_min_tokens`
- `draft_acceptance_threshold`
- `draft_split_probability`
- `draft_gpu_layers`
- `draft_device`
- `draft_threads`
- `draft_cache_type_k`
- `draft_cache_type_v`
- `ngram_min`
- `ngram_max`
- `spec_default`
- Native MTP runtime knobs promoted from env:
  - `native_mtp_batched_verify`
  - `native_mtp_reject_cooldown_tokens`
  - `native_mtp_defer_reject_trim`
  - `native_mtp_suppress_cooldown_drafts`
  - `native_mtp_suppress_cooldown_draft_limit`

Use explicit structs/enums where possible. Avoid stringly typed runtime
internals after config validation.

### 3. Native MTP Auto Detection

Make `strategy = "auto"` enable native MTP for direct GGUF MTP models:

- Reuse package generation metadata when present.
- For direct GGUFs, detect `.nextn.` tensors or model metadata through an
  isolated GGUF inspection helper.
- Keep `strategy = "disabled"` authoritative.
- Keep `strategy = "mtp"` fail-closed when MTP support cannot be
  proven.
- Add telemetry/log output that reports why native MTP is enabled or disabled.

### 4. Draft Model Source Resolution

Make `draft_hf_repo` and `draft_hf_file` work:

- Resolve/download draft GGUFs through existing model resolver/HF artifact code.
- Preserve explicit local `draft_model` as the highest-priority source.
- Implement `draft_selection_policy = "auto"` as a bounded local/catalog search
  for a compatible draft when no explicit source exists.
- Keep `pairing_fault` semantics:
  - `warn_disable`: disable draft mode with a warning
  - `fail_open`: allow the pairing
  - `fail_closed`: reject launch

### 5. Draft Runtime Options

Thread draft runtime controls into the actual draft runner:

- `draft_device`
- `draft_threads`
- `draft_cache_type_k`
- `draft_cache_type_v`
- `draft_gpu_layers`

Update `DraftRunner::open` and the lower llama/skippy load calls as needed. If
llama.cpp lacks a direct option for a field, keep the config field but surface a
structured warning and document the limitation; do not silently ignore it.

### 6. Draft Acceptance Controls

Implement:

- `draft_min_tokens`
- `draft_acceptance_threshold`
- `draft_split_probability`

These should map to the same semantics as llama.cpp speculative parameters:
minimum draft tokens, minimum acceptance probability, and split probability.
Add tests that change generation behavior or resolved request settings.

### 7. N-Gram Speculation

Wire `mode = "ngram"` into embedded generation:

- Start with `ngram-mod` because it is currently the most useful llama.cpp
  default for repetitive agent/code workloads.
- Map `ngram_min` and `ngram_max` to ngram minimum/maximum drafted tokens.
- Add a match/window setting if llama.cpp/skippy requires one; expose it in
  config/schema only when implemented.
- Ensure ngram can be benchmarked without a draft model.

### 8. Spec Default

Make `spec_default = true|false|auto` effective:

- `true`: enable the runtime's recommended default speculative mode.
- `false`: disable speculative defaults unless another explicit setting is
  present.
- `auto`: use safe runtime defaults based on model capability and config.

### 9. CLI Surface

Add or update CLI flags for serve/benchmark paths:

- `--spec-mode`
- `--spec-strategy`
- `--spec-draft-model`
- `--spec-draft-hf`
- `--spec-draft-hf-file`
- `--spec-draft-max-tokens`
- `--spec-draft-min-tokens`
- `--spec-draft-acceptance-threshold`
- `--spec-draft-split-probability`
- `--spec-draft-gpu-layers`
- `--spec-draft-device`
- `--spec-draft-threads`
- `--spec-draft-cache-type-k`
- `--spec-draft-cache-type-v`
- `--spec-ngram-min`
- `--spec-ngram-max`
- `--spec-default`

CLI overrides must feed the same resolver path as config TOML so behavior
cannot diverge.

### 10. Documentation

Update:

- `docs/CLI.md`
- `docs/USAGE.md`
- `docs/design/TESTING.md`
- Configuration docs/schema descriptions
- `.agents/skills/benchmark-tune/SKILL.md`

Document which speculative modes are supported by Mesh runtime, how automatic
MTP/draft selection works, and which telemetry proves speculation is active.

### 11. Benchmark Tune Speculative Surface

Add speculative tuning after the runtime resolver has effective settings or
structured unsupported diagnostics.

Candidate detection order:

1. Native MTP (`strategy = "mtp"`) when the target GGUF/package proves
   native MTP tensors are present. This must be tried before draft-model and
   ngram modes because it avoids a second draft model and should be the most
   hands-off path for MTP checkpoints.
2. Other proven in-model speculative strategies such as DSpArk/DFLASH/EAGLE
   only after the runtime can resolve and launch them. Until wired, report them
   as detected-but-unsupported rather than silently skipping them.
3. Draft-model speculation when an explicit draft source exists or an automatic
   local/catalog match can be found with compatible tokenizer/family metadata.
4. Ngram speculation when no model-native strategy is available, or as a
   comparison mode for repetitive/code prompts.
5. Disabled speculation as the baseline for every target.

Candidate settings:

- Native MTP: start with `auto`, `disabled`, and forced `mtp` where
  support can be proven. If native MTP runtime knobs become config fields, tune
  should sweep only small safe sets first: batched verification on/off, reject
  cooldown `0/1/2`, deferred reject trim on/off, and cooldown draft suppression
  on/off.
- Draft model: sweep `draft_max_tokens` first, then `draft_min_tokens`,
  acceptance threshold, split probability, draft GPU layers, draft device,
  draft threads, and draft KV cache types. Use the explicit configured draft
  first; then try auto-discovered candidates only when compatibility can be
  established.
- Ngram: sweep `ngram_min`/`ngram_max` in bounded pairs and keep a baseline
  disabled candidate.

Benchmark CLI:

- Default behavior should be automatic: `mesh-llm benchmark tune` includes
  viable speculative candidates without requiring extra flags.
- Add opt-in controls to bound the search, for example:
  `--speculative-types auto,disabled,mtp,draft,ngram`,
  `--spec-draft-max-tokens 1,2,4,8`,
  `--spec-draft-min-tokens 0,1,2`, `--spec-ngram-min 2,3`, and
  `--spec-ngram-max 4,6`.
- Add `--no-speculative-tune` to reproduce the old fit-only sweep.
- Reject unsupported requested modes early with actionable diagnostics.

Benchmark output:

- Include speculative fields in `TuneBenchmarkCandidate`.
- Render `[models.speculative]` in each trial config.
- Include speculative telemetry in each trial when available:
  draft tokens, accepted tokens, rejected tokens, acceptance rate, native-MTP
  verification counts, ngram drafted tokens, and active strategy/mode.
- Selection should keep using decode tok/s with throughput tolerance, but ties
  should prefer larger context and then simpler/lower-risk speculative settings
  in this order: native MTP, ngram, draft model, disabled, unless the measured
  throughput difference exceeds tolerance.

Benchmark TDD:

- Add candidate-generation tests proving MTP candidates are produced before
  draft/ngram/disabled for MTP-capable models.
- Add trial TOML tests proving `[models.speculative]` is written for native MTP,
  draft, ngram, and disabled candidates.
- Add CLI parsing tests for each new benchmark tune flag.
- Add selection tests for equal-throughput speculative candidates and context
  tradeoffs.
- Add JSON output tests proving speculative candidate and telemetry fields are
  serialized.
- Add docs/skill tests or examples showing the canonical command and evidence
  fields agents should collect.

## Required Test Gates

Run serially:

```bash
cargo test -p mesh-llm-config --lib speculative
cargo test -p mesh-llm-host-runtime --lib speculative
cargo test -p mesh-llm-host-runtime --lib native_mtp
cargo test -p mesh-llm-cli --lib benchmark
cargo test -p mesh-llm-commands --lib tune
cargo check -p mesh-llm
cargo clippy -p mesh-llm-config -p mesh-llm-host-runtime -p mesh-llm-cli -p mesh-llm-commands -p mesh-llm --all-targets -- -D warnings
```

Run UI checks if schema output changes:

```bash
cd crates/mesh-llm-ui
npm test -- --run src/features/configuration/api/config-adapter.test.ts
npm run typecheck
```
