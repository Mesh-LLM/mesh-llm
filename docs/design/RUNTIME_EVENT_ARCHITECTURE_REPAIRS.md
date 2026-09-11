# Runtime event architecture repairs

This repair follows the correctness review of PR #1667 at `aed70700f`.
The ratified specification is tracked at `.omo/specs/event-system.md`; its
original hash remains pinned in the event inventory. The implementation and
tests, rather than earlier plan completion marks, determine which guarantees
are established.

## Design and acceptance

| Area | Required behavior | Regression evidence |
| --- | --- | --- |
| Publication | One serialized reducer/publication owner; replay sequences increase using the existing ingress sequence. Producer enqueue and drain capture cannot leave an earlier accepted state or terminal behind a later publication. | Deferred progress across operations and drain passes; concurrent submit/drain; terminal follows retained state. |
| Stream attachment | Capture the published frontier, state, health, replay, and live registration at one publication boundary. No asynchronous writes while holding that boundary. | Publish at attachment boundaries; no missing or duplicate events; state/replay/health agree. |
| Cursors | Sequence zero represents an empty snapshot. Real inputs start at one. After the initial snapshot checkpoint, live health acknowledges only events delivered to its connection; it never regresses below that checkpoint. | Initial connection before first input; reconnect from zero; health while live events remain queued. |
| Shutdown | Close admission and submission, stop and join the driver, synthesize all unsettled reserved operations, then perform the exclusive drain using a shared cooperative deadline and work budget. Report every unsettled remainder. | Held guards, concurrent producers, child reservations, repeated shutdown, and zero budget. |
| State pressure | Preserve retained accepted transitions. Capacity exhaustion reports loss and degraded observability explicitly; it cannot silently evict another operation's state. | More than 4,096 distinct pending keys; same-key coalescing; recovery and health. |
| Request lifetime | Only the request root terminal removes its in-flight row. | Backend child finishes while the frontend root continues streaming. |
| Subscriber bounds | Enforce actual queued bytes, frame count, and age, including connection catch-up. Bound reconnect keys without evicting active rate limits to admit new keys. | Mixed-size frames, slow writes, initial replay/state, and many distinct client keys. |
| Native coverage | Emit observations from actual lifecycle paths. Optional API availability alone is not proof of category coverage and must not suppress uncovered fallback observations. | Tests execute native lifecycle call sites; independent optional family and legacy-runtime behavior. |
| Provenance | Preserve producer, severity, source sequence/time, and typed identities through ingress, reduction, and safe public projection. Native facts share their established operation identity. | Native/Rust production adapters feed the real reducer and stream; gap and correlation checks. |

The implementation workers own engine lifecycle, stream recovery, and native
production separately. The parent reviews their designs and diffs, serializes
Rust validation, and checks their combined behavior before pushing.

The original six-outcome ingress plan omitted explicit state-capacity and
cancelled-reservation rejection. `RejectedCapacity` supplies the capacity
outcome: a new state key at capacity is rejected and counted, while already
accepted entries remain retained. `RejectedCancelled` keeps stale reservation
handles distinct from capacity loss and does not mark the reducer degraded.
These additive Rust contract changes require exhaustive consumers to handle
both outcomes. They do not change the native ABI or SSE version.
Degraded state remains explicit until authoritative reconciliation can restore
it; incrementing a rebuild generation alone is not evidence of recovery.

## Validation limits

This section states what the shipped measurements actually cover. Read it
before citing any performance number from this subsystem.

### What is measured

- **Deterministic correctness.** The affected Rust suites, warning-denying
  Clippy, native CPU replay and tests, public headers, generated contracts,
  and a composed product build. These establish the behavioral guarantees in
  the acceptance table above.
- **Ingress allocation, warmed, cold, and contended.**
  `crates/mesh-llm-host-runtime/tests/ingress_reservoir_no_alloc.rs` asserts a
  monotonic allocation-call counter stays flat across an accepted submission
  for each of the four delivery classes; across the *first* submission on a
  freshly constructed engine, per class, with no warm-up of any kind; and
  across 1,000 submissions per thread on eight threads running against a
  live drain. The counters are thread-local, so the consumer's own
  allocations cannot be mistaken for a producer's.
- **Producer-visible ingress p99, against the declared budget.**
  `crates/mesh-llm-host-runtime/tests/ingress_budget.rs` runs a mixed
  10,000-submission workload and asserts the measured `ingress_p99_us` is
  within `CALLBACK_INGRESS_P99_BUDGET`. Refused submissions are measured
  too, so the figure is not only the happy path. The benchmark comparator's
  copy of the same bound is cross-checked against the Rust constant by
  `scripts/tests/test_compare_event_benchmark_matrix.py`.
- **Throughput and time-to-first-token under the class bypass.** Production
  versus `event-disabled` on the same release binary, three independent seeds:
  `decode_tok_s` and `decode_only_tok_s` inside a 3% bound, `ttft_ms` at
  +0.34% with a 95% CI of [-1.34%, +2.30%] at n=80.
- **That a producer never waits on the consumer.**
  `runtime_events::engine::tests::nonblocking` parks a whole drain pass, for
  500 ms, inside everything it holds, and requires 32 producer threads
  across every delivery class -- plus `reserve_root` and `cancel` -- to each
  return within 50 ms. Separately, a lock-class audit makes a blocking
  acquisition in producer context panic in any debug build, and a
  source-shape test requires every production mutex in `runtime_events/` to
  declare a class so the audit cannot be sidestepped.

### What is not measured

- **Caller-side cost before `submit`.** `ingress_p99_us` times
  `RuntimeEventEngine::submit` end to end -- the metadata fill, the class
  decision, the reservation reads, the ring push, and the telemetry tail.
  That is what a producer thread pays to emit one event through the engine.
  It does not include a caller's own locking before it gets there: the
  skippy adapter's mutex and the native reporter's sink mutex are both
  outside this boundary and are not covered by this figure.
- **Total cost of the event system, as of this writing.** The
  measurement is now *possible*: `MESH_LLM_EVENT_SYSTEM_TRIAL_MODE=off`
  installs no engine, and therefore no driver, no presentation subscriber,
  no telemetry consumer, and no native reporter, so emitting an event is
  one `Option` check. `scripts/run-event-benchmark-matrix.py` accepts
  `--mode off`, and `production` versus `off` is comparison A0.
  What is not here is a *result*: no A0 run has been performed, so this
  document still carries no number for the event system's total cost.

  `event-disabled` remains what it always was -- a class bypass that still
  reserves slots, writes terminals and state transitions, and runs the
  reducer, the replay buffer, and subscriber fan-out. Comparison A bounds
  the cost of progress and diagnostic facts, and nothing more. Do not cite
  it as a total-cost figure.
- **Anything in CI.** There is no performance gate on any CI lane. All
  performance numbers above come from local runs on a single machine.

### Historical corrections

An earlier +28-40% decode figure measured against `v0.75.1` is retracted as
void: it compared a debug build to a release archive across a window
containing unrelated performance changes. Historical benchmark reports
predate deterministic bootstrap seeding and are not retroactively validated by
that fix.

TTFT needed three runs to settle. At 20 primary pairs it read +15.7%, then
+8.8% -- both underpowered. Rerun at 80 pairs on a fresh seed it collapsed to
+0.34% (absolute paired means 22.388 ms vs 22.368 ms). The point estimate
moved toward zero rather than tightening around +9-16%, which is the shape of
jitter averaging out. Minimum detectable difference fell from ~16.5-18% to
~2.59%.

### Declined: bare-relative-path redaction

`redact_local_path` collapses absolute, Windows, UNC, `./`, `../`, and `~`
paths to a basename. Bare relative paths with directory components are
deliberately left alone: `org/repo/file.gguf` is a legitimate canonical model
reference that this codebase itself produces, and it is structurally
indistinguishable from `models/secret/x.gguf`. Redacting it would break model
identity on the wire to buy nothing a caller can rely on. The contract is
pinned by a 14-case table test in
`crates/mesh-llm-host-runtime/src/runtime/model_lifecycle/events.rs`.

## Local validation before rebase

- Host runtime: 3,353 unit tests and 10 public architecture regressions passed;
  nine pre-existing ignored tests were not executed.
- UI validation: 1,718 tests passed with three skipped; lint, TypeScript, and
  production build passed.
- Static runtime/server Clippy also passed across all targets.
- Contracts: 17 unit tests and 14 inventory checks passed.
- Static runtime and server: 144 and 705 tests passed respectively; three
  server tests remain ignored.
- Native CPU queue: all 18 native tests passed, including actual lifecycle
  transitions with a generated model fixture. Public C/C++ headers and
  generated API documentation checks passed.
- The opted-in dynamic native test loaded a real model, observed two structured
  callbacks and two unload callbacks, then cleared the reporter successfully.
- Warning-denying Clippy passed for the application, host runtime, and event
  contracts across all targets. The host SDK emits its existing build-script
  linker warnings, which Rust excludes from `-D warnings`.

The shutdown deadline is cooperative. It is checked between bounded reducer
batches; waiting for an existing synchronous critical section or batch is not
preemptible. This does not establish a hard two-second wall-clock ceiling.

## Rebase validation

Rebased onto `main` at `2f8e609797aeec788766561df822c82dfcb3f068`.
The removed legacy CI batch planner stays removed, and the retained CI shim
check uses parsed YAML triggers. The composed CPU product build and its
version command passed. CI crate lists, publish dependencies, release targets,
Rust test coverage, console-print ratchet, actionlint, formatting, and generated
inventory checks passed on the rebased tree.

The full Python suite ran 972 tests: 964 passed, seven were skipped, and one
cache-installer test failed because the restricted macOS test `PATH` omitted
`sha256sum`. Both cache-installer tests then passed with GNU coreutils on
`PATH`. No production change was needed for that host prerequisite.

## CI follow-up

The first rewritten push exposed a static-feature test compilation error in
Linux SafeTensors smoke. The local model test called a dynamic-only loader
without feature gating. The fix gates runtime discovery while preserving the
static model test. The full static host suite then passed 3,321 tests with nine
ignored. Its shared-bus audit assertion now selects the model-open family so
unrelated startup observations cannot change the expected model event pair.

The smoke workflow now prints rendered Cargo compiler diagnostics to stderr
while retaining JSON artifact discovery and pipeline failures. All 16 related
workflow contract tests passed, and a direct fixture check verified separate
error output and executable discovery. The complete Python rerun also passed:
965 tests passed and seven skipped out of 972.

## Subsequent review corrections

Cancelled reservation generations now have a separate rejection outcome and
health counter. Intentional cancellation does not claim capacity loss or force
a state rebuild. Process sequence snapshots share their map until an actual
watermark change, and live replay and captured recovery use the same retention
classifier. A caught-up client remains in-window when all retained frames age
out. Retry delays remove their abort listener on either completion path.

Local-only startup owns its driver across fallible initialization. Every normal
error return stops and awaits that driver before clearing the installed engine;
identity-checked clearing protects a replacement runtime. Native unload events
are emitted after resource cleanup and before the model address is freed.

The macOS protocol-rejection test previously tried to write a request after
sending a forbidden stream discriminator. The peer could reject that stream
before the write, producing a valid STOP_SENDING that failed the test. The test
now waits for that specific rejection with a bounded timeout before checking
that no response frame is accepted. Successful transfer and resume checks are
unchanged.

The inventory generator uses Python 3.11's standard TOML parser. Benchmark
model discovery uses the request timeout, and CI artifact assertions inspect
the actual workflow configuration. These corrections were checked against the
current code rather than applied from review text alone.

The combined follow-up tree passed 3,364 default host unit tests, 3,332 static
host unit tests, ten public architecture regressions, and 31 contract tests.
Each host configuration retained nine ignored tests. UI validation passed
1,719 tests with three skipped, plus lint, TypeScript, and production build.
The Python suite passed 969 tests with seven skipped. Warning-denying Clippy
passed for the application, host, and contracts, including static host targets;
actionlint and the console-print ratchet passed. The updated native CPU patch
queue passed all 18 native tests.
The composed CPU product rebuild and version command passed. The opted-in
dynamic test loaded a real model through that rebuilt bundle, observed two
structured production callbacks and two unload callbacks, and cleared the
reporter successfully.

## Runtime ownership follow-through

The same early-error cleanup applies to mesh serving: the outer startup scope
retains the engine, driver, and presentation task until initialization succeeds.
It awaits retained tasks before clearing its engine on error, and clears the
engine after the established lifecycle shutdown on success. Invalid trial
selectors are resolved before replacing the global engine. Normal shutdown
also awaits the aborted presentation task.

Telemetry tasks hold weak references to their original engine. They sample
that engine rather than looking up a global replacement, and exit on a cadence
tick after it is released. This prevents an old exporter from continuing to
sample a later embedded runtime. Cancellation or panic requests task abortion
through the owning handles; the awaited cleanup guarantee applies to normal
returns and fallible initialization, not to cancellation of the outer future.

The ownership changes passed 3,369 default host tests and 3,337 static host
tests, with nine ignored in each configuration. The 15 telemetry tests include
original-engine lifetime and replacement isolation; the mesh startup tests
exercise a real invalid-plugin error before node setup and verify global
cleanup, plus retained-task release and selector replacement protection.
Default and static host Clippy passed with warnings denied. The composed CPU
product rebuilt successfully after these ownership changes.

## Final review accounting corrections

Drain reports and finite shutdown budgets count only facts successfully reduced
and published. A separate internal count tracks physically removed entries, so
rejected prefixes cannot stop shutdown before later eligible work. Regressions
failed against the previous implementation: three counts included rejected
facts, and a one-publication shutdown budget left 66 entries queued. All five
accounting and prefix regressions pass with the correction.

The native event prefix now has compile-time checks for its alignment, size,
and every shared field offset. UI fixtures resolve from their test module URL,
with successful runs from both repository and UI working directories.

The final accounting tree passed 3,374 default host tests, 3,342 static host
tests, ten public architecture regressions, and 13 native event tests. Each
host configuration retains nine ignored tests. Full UI validation again passed
1,719 tests with three skipped, including lint, type checking, and build.
Default and static warning-denying Clippy, formatting, and the composed CPU
product build also passed on this tree.

## Split serving lifecycle correction

The Linux KV-cache smoke exposed a startup regression: the host installed its
event-only observer in the exact generation-receipt slot. Skippy correctly
restricts exact receipts to local single-stage execution, so this unintentionally
rejected every split stage-0 backend. Two CI attempts failed to form a topology.
The exact PR CPU artifact reproduced the error in a Linux container; the exact
main-branch artifact passed the dense two-node KV-cache smoke there.

Lifecycle-only observation now has its own configuration and completion summary.
Local and split generation report starts, committed-token progress, and a terminal
summary without claiming a canonical distributed session position or exporting
state. Exact receipt consumers retain their local-only restriction and optional
state digest. Early returns abort the lifecycle observation, and normal completion
disarms that fallback. Cancellation after the first token preserves the completed
prefill while cancelling generation.

Local macOS split validation also exposed a pre-existing packaging defect:
versioned dylib aliases were copied as independent files and loaded more than
once. Removing absolute build RPATHs alone did not fix the crash. An isolated
runtime copy with canonical alias symlinks, corrected file hashes, and local
signatures loaded successfully and reproduced the same receipt restriction as
Linux. This diagnostic runtime does not certify the unchanged macOS packaging
path; that issue is separate from the event-system changes.

The corrected tree passed 3,378 default host tests, 3,346 static host tests,
708 server tests in each configuration, and 29 serving-plugin tests. The host
suites retain nine ignored tests and the server suites retain three. Default
and static Clippy passed with warnings denied; formatting and the composed CPU
product build passed. The rebuilt host with the diagnostic macOS runtime passed
both dense and recurrent two-node cache smokes, including `kv-recurrent` restore
payloads. The dense SSE capture contained six generation starts, six generation
completions, and six prefill completions; reported delivery failures and degraded
state remained zero. These observations do not establish benchmark certification.
