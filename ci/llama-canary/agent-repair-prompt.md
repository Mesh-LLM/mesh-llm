# llama.cpp changed-pin canary developer task

You are working on a trusted `main` checkout on the `family-certify`
self-hosted runner. Complete the llama.cpp upstream update as one developer
task. The harness has written the exact target SHA to
`third_party/llama.cpp/upstream.txt` and `.deps/llama-canary-target-sha`.

Read `.agents/skills/llama-patch-changes/SKILL.md` before changing the queue.
When the stage ABI changes, also read
`.agents/skills/llama-stage-patch-changes/SKILL.md`. Their patch ownership,
queue reconstruction, ABI, and validation rules are requirements.

Own the repair end to end:

1. Run `scripts/prepare-llama.sh pinned` and inspect the first real failure. If
   patch application stops, use the failed patch and upstream source to
   understand the refactor. Reconstruct the affected capability commits on the
   target llama.cpp revision, prove the reconstructed tree has the intended
   result, and regenerate the series with `git format-patch`. Repeating
   `git am --3way` without resolving the semantic conflict is not a repair.
2. Preserve every still-valid patch and make the smallest semantic changes to
   the broken patches. Keep the queue ordered. Do not delete instrumentation or
   weaken a gate to get a build through.
3. Generate model-builder stage controls through the Clang rewriter and
   `scripts/generate-skippy-family-patch.py`. Do not hand-edit per-family stage
   filtering or `begin_block`/`end_block` patches. Extend general AST rules for
   conventional upstream shapes. Preserve an exact `unsupported_shape` refusal
   for irregular builders until a sound general rule exists.
4. Fix Rust ABI mirrors, manifests, or runtime code when the upstream change
   requires it. Bump the prepare schema and ABI version together where the
   repository skills require that. A model row may change only when runtime
   evidence shows that its immutable manifest data is stale.
5. Run the canonical path repeatedly until it is green: prepare, the complete
   patched llama.cpp build with upstream tests, the generated-family check, all
   four Rust package builds, Skippy smoke tests, parity validation, the
   `llama-bump` family plan, the live package-v2 matrix, and the full family
   battery. Inspect failures and continue repairing rather than stopping after
   the first partial pass.

The models are already available in the runner's `HF_CACHE`. Stay offline and
do not add Actions caching or download logic. Every runnable `model_pin` row
must still resolve its exact GGUF bytes, package as source-complete package-v2,
pass independent verification, and pass the two-node split smoke. Full family
certification must retain all planned single-step, chain, state-handoff, native
draft, and multimodal lanes.

Leave the finished changes uncommitted in the current mesh-llm checkout. Do not
change the target pin, create or switch branches, commit, push, use GitHub
credentials, or open or edit a pull request. Do not edit `.github/`,
`.agents/`, any file under `scripts/`, `ci/ci.md`, or this runbook. Those files
define the trusted verification boundary. Repair the patch queue, rewriter
implementation, Rust code, and model manifests that the fixed gates exercise.

Manifest edits are deliberately narrow. In
`ci/llama-canary/family-certified.json`, keep the roster, artifact identities,
cadences, lanes, execution policy, and every other field unchanged; only
`resources.estimated_model_bytes` may be corrected from the immutable GGUF
tensor scan. In `docs/skippy/llama-parity-candidates.json`, keep every existing
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

Returning from the coding session is not a success signal. The trusted harness
runs the complete gate sequence on the working tree. If a gate is red, it
returns the current logs to this same session and you continue the task within
the shared repair-and-test deadline. Only a green repair pass may create the
local candidate commit. A separate job then independently reruns the same
sequence on that exact tree before a later success-gated step owns GitHub
publication.

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
