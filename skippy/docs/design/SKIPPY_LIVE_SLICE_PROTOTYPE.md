# Live graph slicing cutover

Status: replay-free implementation, rebased onto main; targeted rebase
qualification passes. The CI repair qualification is recorded below separately
from the original rebase results. Earlier final95 results are historical evidence. No runtime
fallback or backward-compatibility selector remains. Broad release certification
still requires the immutable candidate canary, including CUDA and real GLM-DSA.

The ordinary architecture builder computes the unsplit graph at the actual batch
shape. `skippy_realize_stage_graph` validates dependencies and typed frontiers,
then retains the admitted stage. Existing context graph reuse still applies:
this does not rebuild the graph for every token when reuse is valid.

This corrects an assumption in [Graph Filter V2](SKIPPY_GRAPH_FILTER_V2.md):
metadata-only planning does not encode every runtime view, stride, rollback slot,
or convolution window. Those values belong to the normal builder. The splitter
uses the resulting graph instead of maintaining a second geometry interpreter.

## October 15 review and main integration

Both open review findings on PR #2414 were confirmed in the current source.
Split-edge admission now rejects unsupported transport types in every activation
part before matching descriptors, including parts following dynamic geometry.
The payload-size hint uses the same validation, so an early dynamic-dimension
result cannot hide an unsupported later part. F32, F16, BF16 and I32 frontiers
remain supported.

Compact GGUF metadata now selects `general.architecture` before assigning any
structural fields. Only exact fields under that namespace can supply model
dimensions, attention/recurrent state or MTP metadata; competing namespaces and
auxiliary subgraphs cannot overwrite them. Missing architecture is rejected.
This reads header metadata only, never model tensor payloads. Regression tests
cover both field orders and architecture declarations before and after fields.

The branch is rebased onto main `11f371be0`, retaining its single native-test
pass, qualification ownership and glibc detection changes. The documentation
conflict preserves both the single-test-pass rule and the longer whole-source
rewriter deadline. Native patches and the split-certification digest are
unchanged by this integration. [Canary 38596029602](https://github.com/Mesh-LLM/mesh-llm/actions/runs/38596029602)
completed successfully after the manual Gemma4 rerun for candidate
`cf727d2d3ce3ed884bbf6ebf00dc921edac9537e`. Its earlier automatic retry passed
Gemma4 but failed evidence reconciliation (`retry artifact root contains
undeclared files`); the manual rerun used the ordinary family-receipt path.
That separate retry-evidence defect is not repaired here. The successful canary
does not certify the new Rust changes.

Local review validation passes: 70 model-artifact tests, the 279-test runtime
suite plus the new dynamic-payload regression, and 3,937 Mesh host tests (the
existing one runtime and 24 host ignores remain). Checks and warning-denying
Clippy cover all three changed crates and the shipped Mesh binary. Formatting
and the complete `just ci-validate` gate pass, including 2,338 automation unit
tests, 346 runner contracts, 1,148 lifecycle tests, the native-test ownership
contracts and repository consistency checks.

## October 14 multimodal canary investigation

Candidate `20e91082ccf3f10948188b0877c376feb2482920` passed single-step,
chain and state-handoff for both Mistral 3 and Muse Glimmer in
[canary 38519421738](https://github.com/Mesh-LLM/mesh-llm/actions/runs/38519421738).
Both failed the same three image-smoke assertions: the 16-token response ended
with `length`, nonempty reasoning and no final answer, in local and split execution.
This is not completed multimodal qualification.

The pinned templates were extracted with the GGUF metadata reader, without
hashing NAS payloads, and rendered through the candidate's native chat library.
Ministral-3-14B-Reasoning-2512 at `fe3b038f30334729263d860d5dadbaa34e0f2a18`
ignores all tested reasoning controls. Muse-Glimmer-30B at
`70bf1b61ac09f91b24d39038091b41c582bc5d7a` distinguishes effort via the native
`reasoning_strength` alias, but its original on/off prompts matched. Both native parsers
report reasoning support. Neither observation proves generation obeys an effort
or can be made reasoning-free.

Local Mistral HTTP reproduction matched the 16-token canary output with default
reasoning and explicit `reasoning.enabled=false`. At a 512-token ceiling it
returned a final image description and `stop` after 285 generated tokens.
The test's image fixture and model/projector revisions were unchanged. Automatic
compaction was disabled to isolate this comparison: the existing compactor
counted encoded image bytes as text and inserted a warning even when it removed
nothing. That system message displaced Mistral's template-default instructions.

Comparing direct library renders with the actual C ABI found an additional
bridge bug: with thinking disabled, the bridge overwrote explicit string
`reasoning_effort: "none"` with numeric zero after merging request kwargs.
Numeric zero does not activate the string alias, so Muse rendered its default
`Reasoning strength: high`. The bridge now supplies canonical `"none"` for every
disabled-thinking request, including defaults and contradictory positive efforts.
The native capability layer retains numeric aliases for templates that need them.
An actual C ABI regression
fails on the old bridge and passes on the repaired one. The repair is folded
into core patch 0005; core 0006 updates the matching template expectation, and
the Inkling patch is refreshed only for adjacent test context. Cold application
changes only `common/stage-chat.cpp` and `tests/test-chat.cpp` in the native tree.

On the rebuilt packaged Metal runtime, Muse now advertises `none` and renders
`Reasoning strength: none`. It still emits reasoning: at 16 tokens the response
ends by length, while a 1,024-token ceiling produces a final image answer and
`stop` after 198 tokens. The final default-off correction produces that same
198-token answer with ordinary defaults, instead of the earlier 428-token answer
at the template's high strength. The probe now observes an on/off prompt change.
These are observations for the pinned image fixture, not proof that the model
always obeys an effort control. Mistral's HTTP check with ordinary automatic
compaction now preserves the original 162-token prompt and reaches the same
285-token final answer without an injected compaction warning.

The repaired native-linked image harness passes 10/10 cases for each pinned
model, including local/split image serving, length interruption and lane reuse,
the real parser probe, and the actual C ABI off-effort regression. The ordinary
`just skippy metal` product build passes. Rust validation passes 17 FFI, 279
runtime, 998 serving, 126 API, 300 inference-API, 18 guardrail, 3,926 Mesh host,
and 37 Mesh binary tests. One runtime, nine serving, and 24 Mesh host cases
remain explicitly ignored; the real image fixtures above were exercised, not
skipped. Checks and warning-denying Clippy cover the affected crates and Mesh;
formatting, no-console-print, and the regenerated roster check pass. Native-linked
Skippy tests run separately from packages selecting dynamic FFI defaults:
combining them leaves four low-level tests without a loaded library.
The production-linked CPU native suite passed 278/278, without skips or crashes,
before the final default-effort correction; its graph code is unchanged. The
final correction was rechecked through native-boundary tests, both real image
suites, Rust checks/Clippy, and the rebuilt packaged Metal server.
These local checks do not certify the full family roster or resolve the
separate Gemma runner sccache failure.

Later receipts from the same old candidate exposed a related Mistral4 failure:
the pinned template accepts only `"none"` or `"high"`, so numeric zero caused a
Jinja exception before generation. The default-off correction fixes that exact
rendering failure. The pinned template from revision
`bd93c721735aa32c035c0f19e738cb3371fd56ff` passes default/off rendering through
the actual server C ABI using a local renderer host and a template override;
this is template validation, not a new Mistral4 real-weight inference result.
The regression covers absent effort, explicit `"none"`, and conflicting `"high"`.
MiMo2 separately passed native MTP and state handoff but hit a 60-second stage
readiness limit during parity, followed by a Metal teardown assertion.
MiniMax-M2 was rejected by memory admission before tests. Those two failures
remain outside this image/reasoning repair and must be resolved separately.

The repair separates native parser support from an observed, unobserved or
unknown toggle; preserves the ordinary default-off request; and checks real
template parser evidence in the image harness. Length-limited smoke responses
may contain reasoning or final content; normal completion must contain a final
answer. Empty output, zero generated tokens and unexpected tool calls still
fail. Compaction excludes media payloads from its text estimate and leaves the
prompt unchanged when it cannot remove anything. No family-specific runtime
branch or reasoning-forcing prompt is introduced. The next candidate run must
qualify the repaired behavior; the failed run does not become a pass retroactively.

## October 15 startup and runner repairs

The completed old run has 88 passing families and seven failures. Alongside the
three image/template failures above, MiMo2 and MiniMax-M3 failed stage startup,
MiniMax-M2 failed memory admission, and Gemma3 failed compiler-cache bootstrap.

The correctness harness waited with the configured startup budget but omitted
`--startup-timeout-secs` from its child `skippy serve` command. The child therefore
exited after its default 60 seconds, even when the harness allowed 790 or 900.
A shared child-command owner now forwards that budget for every local stage
launch, and remote MTP launches forward it too. No family-specific timeout or
sleep was added. MiniMax-M3's retained two-stage report and state handoff passed;
its three-stage startup hit the same 60-second failure as MiMo2. The subsequent
Metal teardown assertion is retained as historical failure evidence, not a
successful graph comparison.

MiniMax-M3 already ran on a 256 GiB host with the pinned UD-IQ1_M artifact.
Its lowest sampled available memory was 63,161,401,344 bytes, above the
27,487,790,695-byte reserve. This failure does not justify reducing its quant.
MiniMax-M2's 123,489,073,904-byte estimated peak barely fits the nominal 128 GiB
tier budget and failed actual available-memory admission. Its source-owned
resource requirement now selects the existing 256 GiB tier. The 10% reserve,
available-memory checks, immutable model identity and runtime remain unchanged.

Gemma3 failed before inference because bootstrap stopped and restarted sccache
on the host's shared default endpoint. The macOS action now exports a unique,
short Unix socket before compiler probes, preserving the disk-cache directory
without stopping other jobs' servers. Canary toolchain verification leaves
server startup to that owner. The controller repair landed on protected main in
[PR #2439](https://github.com/Mesh-LLM/mesh-llm/pull/2439), commit
`b51116042f5a3926de2095c89fde059c0a27466b`, before the next candidate dispatch.
The maintainer authorized its immediate merge without waiting for remote CI.

Local validation passes all 53 correctness unit tests and 84 integration
contracts (247 existing real-model cases remain opt-in), warning-denying Clippy,
and two-/three-stage SmolLM2 parity through the packaged Skippy server with a
790-second launch budget. The macOS action regression executes its real shell
steps with finite tool fixtures, checks unique sockets and propagated cache
settings, and rejects any stop-server call. Two real local sccache servers also
started independently; stopping one left the other available. These checks do
not substitute for the next full immutable family run. A real C compilation
also verified persistent reuse: the first server recorded one miss, then a
fresh server using a different socket and the same disk directory recorded one
hit and zero misses for the identical compilation.

The full local CI invocation passed 2,338 automation unit tests, all 346 runner
contracts, and 1,148 lifecycle tests (eight existing lifecycle ignores), then
caught a stale generated family manifest. The memory requirement was moved into
the canonical model registry and all projections regenerated. All 26 model
manifest checks, affected planner checks, every remaining automation target,
legacy contract gate, and repository consistency check pass on resume. No
passing model result or test expectation was relaxed to address that failure.

## Deleted implementation

- All `src/skippy/stage_program*` source/header modules (14 on main; 15 on the
  earlier final95 base), including capture,
  extraction, installation, input binding, geometry realization, and replay.
- Program storage and dispatch in `llama_context`, `llama_model`, model loading,
  and external decode; installation counters and replay-only test accessors.
- Every `skippy_stage_geometry_note` recipe and its environment selector.
- Unused family-specific stage-input adapters and cache-storage lookup helpers
  whose final consumer was the extracted program.
- Replay-only tests and temporary skip code. Their retained responsibilities
  now have live build, numerical, state, and MTP assertions.

The main-based native source change removes roughly 7,500 net lines (the earlier
final95 cutover removed roughly 8,500). The owning six core
patches were rebuilt directly from the upstream pin. Later repair patches no
longer introduce the retired code and then delete it. Normal model builders,
generated block/alias/effect semantics, typed activation transport, exact
parameter loading, graph admission, and state ownership remain required.
No model-family branch or allowlist was added to shared splitting machinery.

## Reproduction identity

The current candidate is rebased onto main `f5273fd1e`, retaining its
llama.cpp pin `8212c7802455255460ab8e18fc34754560031b34`, scheduler/KV fixes,
family registry, and Rust automation. Native live-slice author tree: `20fa8db76`.
This does not import final95's upstream update or unrelated branch changes.
Rebase evidence uses `main-*` filenames in the report directory.

The main-based native suite passes 281/281 with no skips. All five rows below
also pass both two-stage and three-stage parity on the main-based production
runtime. Kimi state/cache roundtrips and synthetic dense, GLM-DSA, and
three-head MTP probes pass. The packaged Metal runtime passes a two-stage
OpenAI serving smoke. Generated semantic annotations cover 154 builders and
pass idempotence; cold queue application at `070b23c68233` reproduces the tested
native tree exactly.

Main-based Rust checks and warning-denying Clippy pass for the affected crates
and Mesh. Unit suites pass for Mesh (37), host runtime (3,919), Skippy API (126),
model artifacts (69), hub (211), native runtime (279), and serving (1,000), plus
the benchmark/correctness tools. Native runtime and serving tests run separately
with `--no-default-features`, avoiding dynamic-runtime feature unification from
Mesh. Existing ignored tests retain their declared prerequisites. CI cache
fixture timestamps require `TZ=UTC` for their expected ordering on this host.
The full CI gate also requires `MESH_PYTHON_SDK_SOURCE` pointing to the external
SDK revision declared in `ci/required-sdk-python/sdk-source.json`; source
admission and the affected SDK fixtures pass with that exact checkout.

The original rebase's `just ci-validate` was not green on this host. That run passed 2,325 unit
contracts and reached 1,155 passing lifecycle fixtures with one failure:
`actual_custom_driver_deadline_and_causal_cancel_reap_owned_sleep_without_success`
expects its child-start marker within a 200 ms deadline. That unchanged main
fixture passes in isolation but fails in the full run here. Its deadline was
not relaxed. The full command stops there, so later automation suites are not
claimed as passing. The targeted split, native, product, and generator results
above are independent completed checks.

### CI repair qualification

The first candidate canary stopped in protected-main bootstrap before testing
models: disk-only sccache had no `SCCACHE_DIR`. The shared macOS setup action now
initializes that directory both for explicit disk-only callers and the existing
missing-GHA-endpoint fallback. Executable fixtures cover both and preserve an
explicit cache directory. This controller fix requires a separate main rollout.

Linux's growing-prefix smoke exposed backend sampler attachment on an
activation-export context. Such contexts output every token, while backend
sampling permits at most one output per sequence. The generic attachment guard
now checks the context's all-row embedding capability. No architecture dispatch
or replay path is involved. The change is folded into core patch 0002; patch 0006
exercises sampling configuration and repeated multi-token prefill/reset on the
synthetic split families.

The retained CI requests reproduce the exact `rc=-1` failure at position 658 on
the original runtime. All six pass with the repaired packaged runtime, including
cached-token counts 0, 664, 658, 808, 802, and 936. The focused synthetic suite
passes 18/18 including fixtures. Author tree `91012bf49` and cold-applied queue
`2ef154c37965` match exactly; neither the queue nor applied source contains
stage-program modules, geometry recipes, or the experimental selector.

Quality's pooled-sampler contract now inspects its current owning patches and
the current reuse lifecycle. The process-deadline fixture uses a two-second
startup budget to reach its held child before testing deadline cleanup; causal
cancellation remains separately exercised. The full lifecycle target passes
1,156 tests with eight existing ignored cases. All automation contract targets
and repository consistency gates pass; the run resumed after installing the
local `gtimeout` prerequisite and admitting the pinned external research source.
Formatting, ShellCheck, xtask check/Clippy, Skippy API tests (124), API check/Clippy,
and Mesh check/Clippy pass. The PR records the complete native and hosted results
separately from this focused repair evidence.

Main's Rust generator initially rejected the C++ producer's null refusal reason
on successful builders. The parser now accepts that documented nullable field;
regressions retain rejection of unsupported builders and non-string values.
Stale family-to-builder ownership entries were also corrected in certification
metadata; this adds no runtime family dispatch.

The following identities and results describe the earlier final95 qualification:

- Mesh base: `5e4da11b886b2cc05416a3ad2b88f09e8c1080ef`
  (`origin/scammed/final95-geometry`). The candidate identity is the Git commit
  containing this ledger and the replay-free patch queue.
- llama.cpp upstream: `5e03bdd8700948b9c41c54dd1b00f28a2aebc03f`.
- Replay-free native tree: `0b2f2759f` in the local native author repo.
- Cold-applied final queue: `592f03d6a66fda0ebf7eb412d2777a10d7fb6fd5`,
  with a byte-identical tracked tree to the native author checkout.
- Production build: `just skippy-release metal`, backend-neutral CLI and packaged
  Metal runtime. Static production primitives were also built for parity tools;
  private test instrumentation was disabled in both production runtimes.
- Reports: `/private/tmp/live-slice-final95`; working checkout:
  `/private/tmp/skippy-live-slice-final95`. These temporary paths are not durable
  release receipts and must be archived with the final candidate identity.

## Qualification ledger

| Model / family | Earlier replay single / chain | Replay-free single / chain |
| --- | --- | --- |
| SmolLM2-135M Q8, dense | pass / pass | pass / pass |
| Ling-3.0-tiny BF16, bailingmoe3 | pass / pass | pass / pass |
| GLM-4.7-Flash Q4_K_M, deepseek2 | pass / pass | pass / pass |
| Kimi-Linear IQ2_XS | pass / pass | pass / pass |
| Synthetic GLM-DSA, shared index | startup rejected integer boundary | pass / pass |

Final production binaries ran without the removed gate. `core-parity` compares
selected tokens against unsplit execution; these rows do not claim exhaustive
logit or long-conversation equality. Two-stage cuts are respectively 15/30,
12/24, 24/47, 14/27, and 1/3. Three-stage cuts are 10/20/30, 8/16/24, 16/32/47,
9/18/27, and 1/2/3.

The GLM-DSA startup failure was a generic Rust width assumption: the primary
frontier part can be I32. `primary_token_width` now describes shape independently
of dtype; each typed descriptor still determines its own payload bytes. Serving,
correctness, and benchmark consumers follow that contract. No tensor reordering
or GLM-specific admission exception was introduced.

The native suite passes **281/281, no skips or crashes**. The final95 gate
experiment had 282 tests; one test exclusively for the deleted interpreter was
removed. Replacement checks include independent graph builds across input,
sequence, output-selection, and batch profiles, exact slice/dependency equality,
fresh-versus-reused logits, and bytewise serialized sequence-state comparisons.
Initial accounting failures were missing generated Gemma/MoE fixtures; after
creating those fixtures the entire suite passed.

Kimi's OpenAI serving probes cover 25- and 99-token prompts followed by 64 decode
tokens, including a chunked prefill transition. Longer synthetic GLM-DSA prompts
also pass both two- and three-stage parity. Ling, GLM-4.7-Flash, and Kimi pass
state roundtrip, suffix-prefill, and repeated cache-hit checks. Synthetic GLM-DSA
is not a real-weight serving certification: its fixture contains non-finite
auxiliary metadata rejected by the public model catalog, so its phase probes
use native admission through the correctness harness.

Synthetic qwen35 (one MTP head), GLM-DSA (one), and hy_v3 (three) every-head
probes pass using the final binaries. Every declared head was proposed and
verified, with target tokens matching an independently loaded MTP-disabled
baseline. These CPU probes establish tested target-state behavior after
teacher-forced proposals, not useful draft acceptance. Native tests additionally
check out-of-range head rejection and persistent-state equality.

## Build and repository validation

- The final native suite passes 281 tests without skips. The independently
  regenerated tree passes all 272 selected graph/runtime checks. Its first run
  could not find the runtime-events probe; building that target and rerunning
  the affected tests passed. The checker now explicitly builds that target.
- Generator validation and idempotence pass for 157 builders. Regenerated shards
  exactly match the checked-in series. Cold application reproduces the intended
  final native tree, and no replay-program or geometry-recipe symbols remain in
  the patch queue.
- Rust checks and warning-denying Clippy pass for all touched crates and
  `mesh-llm`. Unit tests pass: runtime 252, API 124, serving 928, correctness 49,
  bench 85, HF hub 200, model artifact 69, Mesh host 3,857, and Mesh CLI 34.
  Correctness integration adds 84 passes. Existing ignored/download-dependent
  Rust tests are not counted as executed certification.
- Formatting, no-console-print, and `just ci-validate` pass; the latter includes
  1,998 legacy contract tests with 15 declared skips.
- `just skippy-release metal` completes. Its backend-neutral CLI loads the
  packaged runtime, plans two stages, starts both, and serves an OpenAI request
  successfully. This checks the actual shipped runtime composition.

## Admission and reuse

`require_equal`, exact dependency closure, effect ownership, and profile admission
remain enforced. Unknown or incompatible frontiers fail closed. No per-domain
admission redesign was needed by the observed Kimi and GLM-DSA cases. If future
prefill/token-generation domains have incompatible frontiers, admission must
represent and select those domains using explicit graph inputs before execution;
a family-name exception or a resident-weight union is not sufficient evidence.

MTP uses its ordinary auxiliary builder and validates the whole auxiliary graph.
It is not decoder-sliced. The head offset is validated against declared metadata
before changing context state. Its old whole-graph replay path is deleted too.

No shape-keyed slice cache was added. Normal context reuse and computation
invalidation remain responsible for reuse. Any future cache must include the
full graph-reuse identity, admitted domain, and storage lifetime; token count
alone is insufficient.

Activation output sizing no longer calculates an index width by model family.
Preflight reads dynamic dimensions from the scheduler's reserved graph before
state changes. Output copying emits dimensions from the completed live graph.
An undersized caller buffer rejects before decode; a live output exceeding the
reserved bound fails closed and requires session reset rather than retrying a
state-changing decode. Tests compare serialized state before and after rejected
preflight and explicitly exercise the reservation-overflow failure.

The planner uses ordinary context capability resolution instead of a MiniMax
Flash Attention branch. Declared MTP heads use ordinary builder admission instead
of a GLM-specific silent opt-out. Architecture-owned computation and existing
model-specific telemetry/speculation behavior remain outside this deletion.

Rust metadata scanning also keeps nested auxiliary layer counts, embedding sizes,
and indexer head counts from overwriting decoder structural fields. The regression
uses an unknown architecture namespace and tests both metadata orders.

## Historical same-checkout A/B

The measurements below used the earlier development gate at native tree
`321ad75b8809c0e3d0cc809b54e747fe948a4eb7`, before deletion. Both modes used the
same checkout and executable. They preserve the requested replay comparison;
the final binary has no replay mode.

## Timing method

Use the same release executables, model bytes, admitted split, 8,192-token
context, Metal backend, raw F32 activation transport, 64-token prefill chunks,
and one request lane in both modes. Disable prefix caching and speculation.
SkippyBench sends a fixed JSONL corpus, requests 64 output tokens at temperature
zero, and exports both client results and metrics-server telemetry. The first
request at each input size is warmup; report the median of the other three.
Retain individual observations and actual token counts. Three observations do
not support reliable tail-latency claims.

Report client TTFT separately from server `predicted_per_second`; the latter
uses generated-token accounting, not SSE chunk count. Cold process startup is
outside the request timer. Do not run compilers, other model probes, or native
tests during timed sweeps. These measurements describe local two-stage serving,
not networked lab performance or an isolated slicer microbenchmark.

### Same-checkout results

Hardware: local Apple M5 Max, 128 GiB unified memory. GLM serving split at
23/47 and Ling at 12/24. Initial order was GLM replay/live, Ling live/replay;
the repeat reversed each order and added the longest prompt. Values are
median client TTFT in milliseconds / server decode tokens per second.

| Model | Actual input tokens | Initial replay | Initial live | Repeat replay | Repeat live |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-4.7-Flash | 770 | 1072 / 67.0 | 1162 / 67.6 | 1170 / 72.3 | 1323 / 69.6 |
| GLM-4.7-Flash | 1880 | 2731 / 68.0 | 4267 / 25.4 | 3093 / 65.9 | 2819 / 65.7 |
| GLM-4.7-Flash | 3360 | 4966 / 66.2 | 7441 / 25.6 | 6239 / 58.7 | 5314 / 61.7 |
| GLM-4.7-Flash | 6352 | — | — | 14712 / 45.9 | 10170 / 50.6 |
| Ling | 780 | 1088 / 44.4 | 871 / 90.3 | 784 / 78.8 | 924 / 73.4 |
| Ling | 1876 | 1221 / 92.6 | 1374 / 88.8 | 1645 / 72.5 | 1709 / 68.7 |
| Ling | 3336 | 1890 / 70.8 | 2074 / 79.5 | 2705 / 73.3 | 3082 / 71.2 |
| Ling | 6288 | — | — | 4717 / 75.9 | 5240 / 72.7 |

All eight benchmark runs had zero request errors, 64 output tokens per request,
successful metrics exports, and zero reported telemetry loss. However, GLM's
direction reverses between sweeps. A subsequent host snapshot showed background
`mediaanalysisd` consuming about 388% CPU and `sysmond` about 51%; this is a
confounder, not proof of the cause of any individual observation. No unrelated
host processes were stopped. These data do not isolate slicing cost or support
a stable speedup/regression claim. No shape cache was added. Repeat interleaved
A/B on a quiescent host and instrument build versus slice before optimizing.

Raw reports and configs are in
`/private/tmp/live-slice-final95/live-slice-final95-{glm,ling}-{0,1}[-repeat]/`.
Keep the corpus, per-request observations, and metrics exports together when
moving this temporary evidence to durable storage.

## 2026-10-14 canary diagnosis and rebase

[Canary 38473877840](https://github.com/Mesh-LLM/mesh-llm/actions/runs/38473877840)
attempted to certify source `91d48ce44917b7fae40afbbcf4b8cf9db7dc3121` through protected
controller `6ca629425d4f6f661742855f427943a820e20815`. Source resolution and NAS
preflight passed. The studio54 native build passed all 339 tests, without skips,
in 354.49 seconds. The following first graph-rewriter pass hit its 60-second
subprocess deadline after reaching translation unit 87 of 157. No family
certification ran. The retained build artifact establishes this failure;
the previous run's missing diagnostics do not establish its precise cause.

The Rust generator now budgets 600 seconds for each complete rewriter pass,
separately from the default 60-second Git command budget. An explicit timeout
still overrides both. Tests cover both passes exceeding the Git budget,
deadline cleanup without publication, and readable retained diagnostics.
The default failure message previously printed captured bytes as a large
single-line debug array, which the outer log collector suppressed. It now
prints the phase, outcome, status, elapsed time, cleanup result, and readable
sanitized output. Generation, idempotence, and all graph gates remain required.

The pre-rebase local generator check passed validation and idempotence for all
154 builders, exact generated-shard comparison, compilation, and 272 graph
tests. That faster local host did not expose the runner's 60-second deadline.

The candidate is rebased onto main `8b848ead56609a00166c89f339b9139db97d73c9`,
including #2412's token-history restoration and native ABI 0.1.68. Its five
native source changes were merged into the replay-free tree, folded into the
existing patch owners, and cold-applied. The resulting tree exactly matches
the intended combined source. Qualification of that combined candidate is
separate from the older 339-test and model-parity evidence above. All 345 native
tests have passing results: the initial full run passed 344, and the upstream
tokenizer fixture test passed after hydrating its Git LFS data. Skippy FFI,
runtime, serving, and Mesh checks and warning-denying Clippy passed. The
generator's 33 tests and API documentation check also passed. The later main
commits changed Rust response/runtime selection code and ownership metadata,
without changing the validated native queue.

## Four-lane startup reproduction and repair

[Canary 38489296604](https://github.com/Mesh-LLM/mesh-llm/actions/runs/38489296604)
selected `1adc7a4c07b067b677c334e5275341a3b5e423a2`. Its 345 native tests and
complete generator gate passed. The build then failed starting the repeated-prompt
cache smoke server, before family fan-out. The captured native diagnostic was
truncated before the useful terminal error.

Local reproduction with that revision, the pinned Qwen3-0.6B Q8_0 fixture, four
lanes, an 8192-token context, and disabled flash attention exposed the error:
`prewarm embedded OpenAI runtime sessions: RuntimeError: no skippy execution lane is available`.
Binary startup had filled the idle pool with four sessions; graph warmup then
asked the native model for a fifth session. Leaving one lane free confirmed the
diagnosis by allowing the same binary to start. Current main `8b848ead5` retains
the same startup sequence; its panic-recovery repairs do not cover this case.

Warmup now borrows a clean idle session and returns it only after a successful
decode and reset. Failures discard the session and reclaim its lane index;
temporary sessions release their capacity as before. Retained-prefix sessions
are not selected for warmup. This is a shared Rust lifecycle repair, with no
model-family branch or native patch change.

Local validation passed three explicitly enabled model-backed regressions:
repeated warmup with full one-/four-lane pools followed by decoding on every
lane, warmup with idle retention disabled, and failure cleanup of a dirty native
session followed by capacity reuse. The ordinary serving suite passed 1007 tests
with eight explicit ignored model-backed tests (including the three separately
run regressions). Serving and Mesh compile/Clippy, formatting, and no-console
checks passed. The original four-lane server now reaches OpenAI readiness.

The subsequent cache miss is now traced to the smoke configuration. Its typed
generator requests resident KV but omits `kv_graph_state`. Main's graph-derived
payload selection (#2252) therefore correctly falls back to asynchronous
full-state snapshots. A fresh-server telemetry reproduction queued a 44,045,496-byte
checkpoint, then the immediate repeat missed. A model-backed regression observed
zero published entries and two pending records after the seed response. The
immediate repeat missed; after the existing capture/recorder completion barriers
drained, the next repeat reused 384 tokens. No drops or worker failures occurred.
The drain after the immediate request took approximately 527 ms in this run.

A controlled fresh-server comparison used `serve --print-effective-config` to
derive genuine graph evidence through the ordinary planner. The local copy's
source digest matched the original NAS fixture before its admitted graph-state
field was transferred to the otherwise unchanged smoke config. Selection became
`ResidentKv` with zero fallbacks. The unchanged smoke passed on its first run:
515 cached tokens for the identical repeat, 512 for growing chat, and 542 for
the growing-chat repeat. No sleeps, request retries, or family-name conditions
were added. The publication-boundary regression passed along with serving
compile/Clippy, formatting, and no-console checks.

That diagnosis exposed an unnecessary configuration boundary. The replacement
removes operator-authored graph-state labels and cache payload selection
entirely. The native context intersects state coverage across its reserved
graphs with the loaded memory's complete exporters. The first capability query
freezes this admission; subsequent reservations and live graph builds reject
state coverage that would invalidate it. Graph reuse retains the proof for the
graph being reused. Unclassified or additional persistent state selects complete
snapshots; an absent loaded runtime does not authorize a cache.

The public `kv_graph_state`, cache `payload`, Mesh `payload_mode`,
`--prefix-cache-payload`, and both payload environment overrides are removed.
Retired inputs fail explicitly, without aliases or compatibility behavior.
Planner descriptors retain graph-state diagnostics, but those strings no longer
authorize serving state. Cache enablement, memory budgets, token thresholds,
and numeric precision remain operator controls.

All three standalone serving smokes now start `skippy serve --model-path`.
The handmade OpenAI/cache and System One config generators are deleted. Split
parity still uses real planner-produced contracts and binary stage servers.
The cache smoke keeps its explicit 64-token threshold through the public CLI:
its first HTTP fixture has 215 tokens, below the normal 256-token default.
It does not select a representation, retry the request, or sleep for cache
publication.

This changes the semantics of
`uint32_t skippy_model_memory_cache_capabilities(const struct skippy_model *)`
without changing its signature. Native and Rust ABI versions advance together
from 0.1.68 to 0.1.69. No native declarations are added or removed. The changes
remain folded into the six owning core patches; there is no replay-era fixup
or fallback patch.

Local validation of this cleanup passed the 287-test full native suite, followed
by added cache-admission regressions for attention, recurrent, unclassified,
and malformed graph state. Public headers compile as C11/C++17. The final
production native rebuild, serving suite (995 passed, nine ignored), CLI suite
(63 passed), Mesh host suite (3,926 passed, 24 ignored), Mesh binary library
(37 passed), ABI tests (17 passed), and UI configuration adapters (52 passed)
passed. The ignored resident-cache regression was explicitly enabled against
the pinned NAS Qwen model: native selection was `ResidentKv`, reason
`loaded_graph_and_memory`, and the immediate repeat reused 515 tokens.

The complete standalone smoke passed against NAS Qwen3-0.6B Q8 and Falcon-H1
Q4_K_M, with dense two-/three-stage parity and dense full-state handoff enabled.
The optional dense suffix comparison needed two lanes because it holds an
independent reference beside the restored session; the old one-lane invocation
failed before that comparison and is corrected. All OpenAI assertions and the
unchanged repeat/growing-chat checks passed (515, 512, and 542 cached tokens).
The System One contract passed through normal server startup; its full-model
Metal read remains explicitly unqualified. The 1,156-test automation lifecycle
target passed 1,148 tests with eight existing ignores. A later packaging test
caught its stale expected ABI 0.1.68; the expectation is updated to 0.1.69.
The failed target and every remaining CI contract/repository gate then passed.
The standalone OpenAI smoke also passed after updating stale expectations:
`typical_p` is supported, `adaptive` is still rejected, and normal server
startup emits request/decode telemetry rather than the staged scheduler's
iteration event.

A separate planner-generated two-stage Qwen server, using the arbitrary model
ID `local/split-cache-proof`, selected `ResidentKv` with reason
`loaded_graph_and_memory` on both stages. The cold request reused zero tokens;
the immediate repeat reused 516 tokens. No representation fields or graph-state
labels were injected into the generated stage configs.

`just build metal` completed both products, including the packaged native
runtime and Mesh console. The packaged dynamic Metal runtime then served the
same NAS model with all layers requested on GPU and reused 515 tokens on an
immediate repeat. The local Rust toolchain emitted a non-fatal debug-stripping
warning because its `rust-objcopy` could not find `libLLVM.dylib`; compilation,
packaging, ABI checks, and the packaged serving check still succeeded.

## Follow-up audit of automatic cache admission

Canary [38507110517, attempt 2](https://github.com/Mesh-LLM/mesh-llm/actions/runs/38507110517/job/115677108735)
failed after the disk cleanup, for two independent reasons. Nomic and Jina
completed inference and their independent oracles, but the parent evidence
verifier looked for the deleted `skippy-serving` binary instead of `skippy`.
T5 failed during encoding because a valid stateless encoder graph returned
zero cache capabilities and invalidated the decoder reservation. Neither
failure was a KV cache miss or another disk-space failure.

The wider audit found a correctness issue beyond that assertion: native memory
snapshots do not serialize the context-owned encoder embeddings and sequence
IDs used by cross-attention. Falling back from zero capabilities to a full
snapshot would therefore be unsafe. ABI 0.1.70 adds the explicit
`SKIPPY_MEMORY_CACHE_FULL_STATE` bit; zero now disables prefix caching. Both
cross-attention graph-input owners declare their unserialized context state.
This is a graph-input contract, with no family-name dispatch. A valid stateless
phase imposes no additional restriction, while malformed graphs admit nothing.
RAM and disk tiers select only an explicitly admitted complete exporter.

The non-chat canary now starts ordinary `skippy serve --model-path`, deleting
the fourth handwritten smoke-config generator. Behavioral contracts check the
actual startup arguments and every workload class's parent evidence verifier,
including failure propagation. The remaining benchmark payload override is
removed. The API documentation generator now writes to its owning website
directory after the workspace move.

Expanded real-model testing also exposed two fixture mistakes. OCR, speech
synthesis, and speech recognition can contain causal decoders and legitimately
warm a generation graph; their API class does not imply encode-only execution.
The OCR HTTP smoke had used a colored image containing no text while requiring
a nonempty transcription. It now uses the manifest's independently reviewed
`ocr-mesh-42.png` input for backend, HTTP, and oracle checks. A regression
executes the media selector and checks it against that manifest.

On the repaired CPU runtime, Nomic embedding, Jina reranking, T5 translation,
PaddleOCR recognition, Qwen3-TTS synthesis, and Ultravox recognition pass their complete local battery
paths, including normal HTTP serving, independent oracle, and parent receipt
verification. T5 explicitly admits no prefix-cache representation and passes
A/B/A translation with different A/B output and identical repeated A output.
The TTS oracle reports relative RMS error 0 and cosine similarity 1 over 32,640
samples at 24 kHz. The corrected OCR input returns exactly `MESH 42` through
normal HTTP; its corrected complete certification also passes, including the
independent oracle and parent receipt verifier.

Dense full-state handoff and hybrid KV/recurrent handoff pass with real models.
Normal Qwen HTTP cache checks reuse 515, 512, and 542 tokens for immediate
repeat, growing chat, and growing-chat repeat, without sleeps or request
retries. The static native-linked Rust suites pass: 18 ABI, 279 runtime, and
996 serving tests, with one existing runtime ignore and nine serving ignores.
The full Metal native suite passes 287/287 with no skips or crashes, including
MTP and the new state-admission regressions. The six owning core patches
cold-apply to the tested native tree; no extracted
stage-program or geometry-recipe symbols remain in its source or headers.
This follow-up supersedes the earlier assumption that a full snapshot was
always a safe fallback. The first full CI invocation passed 1,147 lifecycle
tests but found one stale ASR startup fixture still expecting the deleted
generated config. That fixture now checks the real model/projector CLI and
CPU controls; all nine related media-comparison tests pass. Every remaining
automation target and CI repository gate passes on resume, including the new
media selector and all-class parent-verifier regressions. Final formatting,
warning-denying Clippy (including automation-contract tests and the Mesh
binary), generated API docs, split roster, and ShellCheck pass. The previous
packaged Metal serving evidence above is from ABI 0.1.69; this follow-up has
production CPU execution and the complete ABI 0.1.70 native Metal suite, with
the packaged Metal rebuild still pending. The candidate is ready for a
certify-only canary; that dispatch does not merge or deploy the change.

## Canary and release decision

| Decision | Result | Remaining obligation |
| --- | --- | --- |
| Remove extracted programs and geometry recipes | done | No fallback or selector remains |
| Local dense, bailingmoe3, deepseek2, Kimi parity | pass | Full immutable roster canary |
| GLM-DSA graph, integer frontier, and phase behavior | synthetic pass | Real-weight canary |
| MTP and persistent-state behavior | tested probes pass | Full roster/head coverage |
| Native and Rust validation; packaged Metal serving | pass | Remote clean candidate build |
| Complete main-based local CI gate | resumed checks pass | Original run hit the 200 ms fixture; its existing fix and every remaining gate passed |
| GitHub Metal family canary | historical run failed; repaired full local smoke passes | Dispatch and finish certification of the final candidate |
| Stable performance conclusion | inconclusive | Quiescent interleaved A/B |
| Universal or cross-backend support claim | no-go | Full roster plus separate CUDA lane |

The local evidence supports running the replay-free candidate through the full
canary. It does not establish universal model support or a stable performance
improvement. Keep the existing immutable family manifest, complete expected-row
reconciliation, independent verification, state/cache obligations, and every-head
MTP coverage. Synthetic live build-at-shape evidence replaces interpreter
extraction evidence; it does not replace real-model parity.

The current PR targets main. The requested qualification uses GitHub's protected-main
`llama-upstream-canary.yml` controller with `mesh_ref=<full candidate SHA>` and
`force_certify=true`. This certifies the selected source's existing pin and
patches without repairing or publishing it. Runner placement follows the
controller's family-certify and memory-tier labels; it is not a micstudio-only
dispatch. Link the resulting run from the PR and distinguish scheduling from
successful certification.

An alternative Jianyang route was prepared for micstudio before the GitHub
dispatch was selected; it is not needed for this request.
The installed MeshLLM templates currently lack a canary entry. A complete
`skippy-canary` template is prepared for the owner (`general-inference/jianyang`,
`pilot/templates/mesh-llm.yaml`); `jy repos validate` passes with no warnings.
It preserves the existing templates, runs two independent candidate builds,
reconciles the complete roster, and archives failure receipts. It is not yet
installed or submitted. Publish the candidate and template, sync the repository,
then run `jy repos run mesh-llm skippy-canary --ref <full-sha> --node micstudio`.
Do not dispatch
an old commit, upload an uncommitted checkout, or relabel a decode benchmark as
full family certification. Preserve candidate, upstream, queue digest, backend,
model identities, reports, and failure receipts in the resulting job artifacts.

Repeat GLM/Ling timing on a quiescent host. A Metal canary does not certify CUDA;
run the existing CUDA lane before claiming cross-backend completion. Deployment
rollback is a previous binary, not an old execution path inside the new binary.
