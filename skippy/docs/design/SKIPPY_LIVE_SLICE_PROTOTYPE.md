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

Cache qualification remains open: the first repeated-prompt workload reported
zero cached tokens; a later invocation against the same server passed all three
cache-hit checks with 384 cached tokens. This suggests asynchronous publication
timing but does not establish its cause or a reliable cold-start pass. Do not
replace that failure with the later success or dispatch another canary as if the
whole smoke gate has passed. No family certification is claimed from this run.

## Canary and release decision

| Decision | Result | Remaining obligation |
| --- | --- | --- |
| Remove extracted programs and geometry recipes | done | No fallback or selector remains |
| Local dense, bailingmoe3, deepseek2, Kimi parity | pass | Full immutable roster canary |
| GLM-DSA graph, integer frontier, and phase behavior | synthetic pass | Real-weight canary |
| MTP and persistent-state behavior | tested probes pass | Full roster/head coverage |
| Native and Rust validation; packaged Metal serving | pass | Remote clean candidate build |
| Complete main-based local CI gate | resumed checks pass | Original run hit the 200 ms fixture; its existing fix and every remaining gate passed |
| GitHub Metal family canary | native/generator passed; smoke startup failed | Startup repaired locally; resolve cold-start cache smoke before redispatch |
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
