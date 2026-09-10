# llama.cpp changed-pin canary repair runbook (agent instructions)

You are running on the `family-certify` self-hosted runner inside a mesh-llm
checkout. The deterministic wrapper owns one
`prepare -> build -> certify -> publish` state machine for the candidate SHA in
`.deps/llama-canary-target-sha`. It has handed you one failed phase to repair.
Keep all work in the current checkout and leave commits, branches, pushes, PRs,
and comments to the wrapper.

**Before touching the queue, read the repo skills and follow them:**
`.agents/skills/llama-patch-changes/SKILL.md` (queue edits, upstream pin,
prepare/build flow, patch ownership boundaries) and, when a patch changes the
stage ABI, `.agents/skills/llama-stage-patch-changes/SKILL.md`. The boundaries
in those skills are hard requirements for this repair, not suggestions.

1. **Reproduce the failed phase.** The wrapper has already written the candidate
   to `third_party/llama.cpp/upstream.txt`; prepare it with
   `scripts/prepare-llama.sh pinned`. Inspect the supplied failure tail and run
   only the focused build or certification commands needed to identify the root
   cause. If patch application left `.git/rebase-apply`, use
   `git am --show-current-patch` and `git am --3way --continue`/`--abort` to
   inspect the conflict.

2. **Fix the queue — follow `llama-patch-changes`, do not loop on `git am`.**
   If a patch fails to apply, `git am --3way` retry alone is not an acceptable
   resolution: a conflict means upstream refactored code a patch owns, and the
   skill's deliberate queue rewrite is the required path. Resolve the conflict
   on a llama.cpp branch (base on upstream `ggml-org/llama.cpp` `master` at the
   canary target SHA), reconstruct capability-owned commits, verify the
   reconstructed head is tree-identical to the intended final checkout, then
   regenerate the series with `git format-patch` per the skill. Keep the series
   ordered, keep every patch that still applies unchanged, and make the minimal
   semantic fix in the broken ones. Regenerate the series so
   `scripts/prepare-llama.sh` runs clean end to end.

   Model-builder stage controls live in the single generated family patch.
   Run the Clang rewriter and `scripts/generate-skippy-family-patch.py`; do not
   hand-edit per-family stage-filter or `begin_block`/`end_block` patches. A
   conventional builder must be regenerated from its proven source shape. An
   irregular builder remains unchanged with the rewriter's precise
   `unsupported_shape` reason until a sound general rule exists.

3. **Use focused verification while repairing.** The wrapper restarts from
   prepare after every agent turn, runs the complete patched llama.cpp and Rust
   build gates, and only then runs certification. Do not spend the remaining
   wrapper deadline duplicating the complete battery unless the failure itself
   requires a focused battery reproduction.

4. **Preserve every gate.** Do not weaken, skip, narrow, or mark a failing lane
   unsupported to make the run green. Repair the patch queue, ABI mirrors,
   manifests, or runtime code that owns the failure. The loop ends only when
   the wrapper's own complete certification passes or its phase turn/time bound
   is exhausted.

5. **Leave the result local.** Do not switch or create a branch in the mesh-llm
   checkout, commit there, push, open or edit a PR, comment on GitHub, or use
   GitHub credentials. Temporary llama.cpp reconstruction branches and
   worktrees required by the patch-queue skill remain local. The wrapper
   commits the final mesh-llm tree, publishes one run-specific branch,
   generates the PR body with the upstream summary, and opens either an exact
   certified PR or an uncertified draft at terminal failure.

Notes:
- Models come from the runner's pre-warmed HF cache (`HF_CACHE`); `hf download`
  is only a miss backstop. Never add GitHub Actions model caching.
- The deterministic wrapper owns the sole upstream selector,
  `third_party/llama.cpp/upstream.txt`. It writes it to the repair target and
  validates the queue through `scripts/prepare-llama.sh pinned`; do not edit
  the pin file yourself.
- Do not modify files outside `third_party/llama.cpp/patches/` unless the
  Rust ABI mirrors in `crates/` genuinely need to track a patch ABI change
  (bump `PREPARE_SCHEMA`/ABI version together in that case). Existing
  model-manifest rows may be corrected when a battery failure proves they are
  stale. Do not add a checkpoint merely because an upstream builder is new;
  the source rewriter supplies structural stage-control coverage.

## New upstream model families

Run the source rewriter across every `src/models/*.cpp` translation unit. A
new conventional builder should appear in the consolidated generated patch
without a family-specific rule or checkpoint. If it reports
`unsupported_shape`, preserve the refusal and add a general AST rule only when
the activation loop and ownership edits can be proved. Never use an
architecture name or tensor spelling as the eligibility predicate.

The family manifests remain the executable numeric battery. New source
coverage does not automatically create a new model row and does not trigger a
download. Add or change a row only when there is an independent numeric,
backend, or state-pattern reason and immutable artifact evidence is available.

The canary continues to run `scripts/skippy-canary-live-matrix.sh` after the smoke
gates: every runnable `model_pin` row must resolve its pinned GGUF, pass
size/sha256 verification, package as source-complete package-v2, pass
independent `verify-package-v2`, and pass the two-node split smoke. Its
per-row evidence lives under `target/family-battery/<run>/live-matrix/` in
the uploaded battery artifact; a failed existing row routes here for a
semantic patch or manifest repair backed by that evidence.
