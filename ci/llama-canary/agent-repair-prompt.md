# llama.cpp canary patch-queue repair runbook (agent instructions)

You are running on the `family-certify` self-hosted runner inside a mesh-llm
checkout. The nightly llama-upstream canary either failed to apply our patch
queue in `third_party/llama.cpp/patches/` onto the new upstream pin
(patch-queue mode) or applied the queue but failed a certification lane
(battery mode). Your job:

**Before touching the queue, read the repo skills and follow them:**
`.agents/skills/llama-patch-changes/SKILL.md` (queue edits, upstream pin,
prepare/build flow, patch ownership boundaries) and, when a patch changes the
stage ABI, `.agents/skills/llama-stage-patch-changes/SKILL.md`. The boundaries
in those skills are hard requirements for this repair, not suggestions.

1. **Reproduce.** Run `scripts/prepare-llama.sh "$(cat .deps/llama-canary-target-sha)"`
   and capture which patch fails to apply (`git -C .deps/llama.cpp am --3way ...`).
   A `.git/rebase-apply` state may be left behind; use `git am --show-current-patch`
   and `git am --3way --continue`/`--abort` to inspect the conflict.

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

3. **Build.** `scripts/build-llama.sh` then
   `cargo check -p skippy-ffi -p skippy-runtime -p skippy-server`.

4. **Certify.** `scripts/skippy-family-battery.sh --skip-build`.
   All lanes must pass. Do not weaken a failing lane; if a model is genuinely
   broken by upstream, revert to fixing our patches or flag it in the PR body.
   The wrapper re-runs the battery itself after your turn; if lanes fail you
   will get the failure output in a follow-up repair turn — the loop only
   ends when the wrapper's own battery run passes.

5. **Commit locally; the wrapper owns the PR.** Work on branch
   `llama-canary/patch-queue-fix`. Commit the patch-queue changes with a
   `fix(llama): rebase patch queue onto upstream <short-sha>` message. You
   have no GitHub credentials: the deterministic wrapper that drives you
   commits any remaining work, pushes the branch, and creates/updates the
   repair PR itself. The wrapper separately asks you to write the full PR
   description (key upstream changes, how the patch queue evolved, risks for
   reviewers) — when that turn arrives, write the finished Markdown to the
   file it names and touch nothing else. After the wrapper's own battery run
   passes, a separate review agent — not you — gets one fresh-context turn
   to review the certified repair and fix any dropped intent or rebase
   leftovers it finds; its changes land as their own `review(llama):`
   commit, and the next canary re-certifies everything after the merge.

Notes:
- Models come from the runner's pre-warmed HF cache (`HF_CACHE`); `hf download`
  is only a miss backstop. Never add GitHub Actions model caching.
- Do not modify files outside `third_party/llama.cpp/patches/` unless the
  Rust ABI mirrors in `crates/` genuinely need to track a patch ABI change
  (bump `PREPARE_SCHEMA`/ABI version together in that case).
