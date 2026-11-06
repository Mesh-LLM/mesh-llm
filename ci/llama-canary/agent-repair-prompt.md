# llama.cpp changed-pin canary developer task

You are working on a trusted `main` checkout on the `family-certify`
self-hosted runner. Repair the llama.cpp update to the exact SHA already written
to `skippy/llama_cpp/upstream.txt` and `.deps/llama-canary-target-sha`. Your job
is to produce a sound candidate and hand it back promptly. The trusted harness,
not this coding session, performs complete verification and certification.

Read `.agents/skills/llama-patch-changes/SKILL.md` before changing the queue.
When the stage ABI changes, also read
`.agents/skills/llama-stage-patch-changes/SKILL.md`. Their patch ownership,
queue reconstruction, ABI, and validation rules are requirements.

Do not write new Python tooling during repair, including temporary or inline
helpers. Follow `.agents/skills/manage-ci/SKILL.md` for new repository
automation in typed `tools/xtask` commands; from the repository root,
`cargo xtool repo-consistency ci-crate-lists` is an existing alias example.
The existing patch generator remains transitional until an equivalent Rust
command passes parity checks. This does not grant permission to edit the trusted
scripts or CI boundary listed below.

Work in this order. Stop as soon as the focused repair check passes:

1. Run `scripts/prepare-llama.sh pinned` on the first turn and inspect its
   first real failure. Read the failed patch and the corresponding upstream
   source before editing. On a resumed turn, start with the newest failure log
   supplied by the harness. If preparation passes, use the first actual build
   or focused-check failure; do not search for unrelated problems.
2. Make the smallest semantic repair in the owning patch, generator, ABI mirror,
   or Rust caller. Preserve still-valid patches, instrumentation, and queue
   order. When a patch must be rebased, reconstruct its capability commit on the
   target revision, compare the resulting tree with the intended result, and
   regenerate the series with `git format-patch`. Repeating `git am --3way`
   without resolving the semantic conflict is not a repair.
3. Reproduce the changed behavior with the narrowest useful check. For generated
   model-builder controls, use the Clang rewriter and
   `cargo xtool automation native-generator generate` with absolute `--git` and
   `--rewriter` paths and `--max-diff-bytes 16777216`. Do not hand-edit
   per-family stage filtering or `begin_block`/`end_block` patches. Extend a
   general AST rule only for a proven conventional upstream shape; retain
   `unsupported_shape` for irregular builders. Fix Rust ABI mirrors and bump
   the prepare schema and ABI version together only when the boundary changes.
4. Run the shared focused repair check once the durable queue is ready:

   ```bash
   "$MESH_LLM_AUTOMATION_BIN" automation canary-receipts repair-check \
     --root "$PWD" --controller-root "$CANARY_REPAIR_CONTROLLER_ROOT" \
     --state-dir "$CANARY_REPAIR_STATE_DIR" --build-dir "$LLAMA_STAGE_BUILD_DIR" \
     --backend "$LLAMA_STAGE_BACKEND" --phase fast
   ```

   This command restores the locked native-test Python environment, builds
   static binaries, rejects dynamic-runtime substitutions, checks binary
   identity, and exercises focused split and recurrent execution. If it fails,
   read its step result, bounded failure excerpt, and reproduction
   arguments. Fix that failure, rerun its narrow reproduction, then rerun the
   focused repair check. Do not install system Python packages.
5. When the focused repair check passes, stop running commands and return
   control. Report the files changed, exact passing checks, and any remaining
   uncertainty. Do not preemptively investigate a smoke, model family, or
   controller path that has not failed. A focused pass is not certification.

Do not run the complete generated-family checker, an unfiltered native CTest
suite, `just build`, canonical smokes, or family certification inside this coding
session. The trusted harness runs those full gates after you return and feeds a
real failure back to this same named session while the coding window remains
open. Do not run
   an additional full family battery inside the coding session: the wrapper
   runs every canonical gate, including the complete roster, after you return.
If a trusted gate fails, read its supplied log, reproduce only that failure,
repair it, and return for the next trusted pass. Do not weaken a gate or infer
success from activity.

Keep console narration brief: before each meaningful action, state what you are
checking or changing and why in one or two sentences. Report the result and next
step afterward. The harness retains complete commands and output separately.
Use descriptive command-log labels. Run build and test reproductions you launch
yourself through
`scripts/llama-canary-log-command.sh <short-label> <command> [args...]`, including
failed reproductions and retries. For a pipeline, pass `bash -o pipefail -c` as
the command; otherwise use `bash -c` when shell syntax is needed. Keep commands
in the foreground. If one must run in the background, record its PID, wait for
its exit status, and leave no detached build or test process for the harness.
Do not put secrets in commands or labels.

The models are already available in the runner's `HF_CACHE`. Stay offline and
do not add Actions caching or download logic. Full family certification must
retain all planned single-step, chain, state-handoff, native draft, and
multimodal lanes. Package-v2 product coverage runs in the required pull-request
checks after the canary publishes a candidate.

Leave the finished changes uncommitted in the current mesh-llm checkout. Do not
change the target pin, create or switch branches, commit, push, use GitHub
credentials, or open or edit a pull request. Do not edit `.github/`,
`.agents/`, any file under `scripts/`, `ci/ci.md`, or this runbook. Those files
define the trusted verification boundary. Repair the patch queue, rewriter
implementation, Rust code, and model manifests that the fixed gates exercise.

Manifest edits are deliberately narrow. In
`ci/llama-canary/family-certified.json`, keep the roster, artifact identities,
lanes, execution policy, and every other field unchanged; only
`resources.estimated_model_bytes` may be corrected from the immutable GGUF
tensor scan. In `skippy/docs/llama-parity-candidates.json`, keep every existing
row and all top-level policy unchanged. Append exactly one classification row
for each source file missing from the manifest. New rows are limited to the
classification fields `llama_model`, `family`, `status`, and optional `notes`
or `unsupported_reason`; do not add artifact selectors, source revisions,
integrity records, or execution settings such as `repo`, `include`, `revision`,
`file_integrity`, `splits`, `recurrent`, `model_pin`, or `artifact_id`. A new
boundary-registered source must remain a runnable candidate. The trusted
wrapper checks these limits before it accepts the tree, and the full battery
independently verifies every corrected tensor-byte value against the pinned
local artifact.

Returning from the coding session is not certification. The trusted harness
runs prepare, manifest-policy, build and smoke gates; local failures return to
the same named session within the coding admission window. Once those gates
pass, the job exports an uncertified immutable candidate and releases its
runner. Separate jobs certify every family using the exact producer binaries.

A failed family pass or independent verification supplies its candidate and
confirmed candidate failure logs to a new session in the next attempt. Read
those logs before continuing. Runner/workflow failures and missing receipts
are first rechecked only for the affected families on the same immutable
candidate without invoking this agent; valid candidate failures from a mixed
pass are retained. Repeated infrastructure failure, corrupt/foreign evidence,
and other contract failures stop without starting another session. There are
at most three distributed repair attempts; every edit requires a new complete
build and family pass. Both the first full family pass and the fresh independent
build/family pass must be green on the same commit before the hosted publisher
can create a branch or PR.

## New upstream model families

Run the source rewriter across every `src/models/*.cpp` translation unit. A new
conventional builder belongs in the consolidated generated patch without a
family-specific rule or checkpoint. If it reports `unsupported_shape`, add a
general AST rule only when the activation loop and ownership edits can be
proved. Never use an architecture name or tensor spelling as the eligibility
predicate.

Source coverage alone does not add a family manifest row or trigger a model
download. Add or change a row only for an independent numeric, backend, or
state-pattern reason backed by immutable artifact evidence.
