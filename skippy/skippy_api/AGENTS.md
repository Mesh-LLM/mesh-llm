# Porting a llama.cpp patch into `skippy_api`

This file is for whoever (human or agent) ports the next patch from
`skippy/llama_cpp/patches` into this library, and for anyone writing code here.
It records rules that were agreed the hard way; most of them exist because
something broke.

## How to work here (agents and humans)

1. **Edit through the compiler and the LSP**; navigate through the LSP. No perl/sed/awk/regex
   over source. (TYPE-RULES §1, "Source changes go through the compiler")
2. **Born red.** Show the failing test for a bug, on the gate's toolchain, before the fix.
   (TYPE-RULES §1)
3. **Rules that disagree: stop and ask; don't pick one.** (TYPE-RULES §8; `tests/rules_consistency_test.cpp`)
4. **A gate you didn't run didn't pass.** Report each gate's last line. (root AGENTS.md,
   "Pre-Commit Checklist"; trap "`ctest` can pass on a stale binary")
5. **Out-of-scope discoveries are recorded and reported, not fixed** — a trap, an `UPSTREAM.md`
   entry, or a note in your report. (Keeping this file current)
6. **Commit only through `just skippy-api-checkpoint`; push only with `--force-with-lease`**
   against a tag set before starting. (Tooling rules)

## What this library is

A C++23 façade over the patched llama.cpp staged runtime, plus Rust bindings in
`skippy/skippy-api-rs`. It owns *use* of the runtime, never the runtime itself.

```
include/            public headers, one per capability
  runtime.h         NativeRuntime / Model / Session handles, SessionConfig
  workload.h        workload classification, pooling
  embed.h           embed, embed_batch, EmbeddingView
  rerank.h          rerank, rerank_batch
  stage_plan.h      stage plans, profiles, frontier values, state effects
  stage_load.h      plan -> loader configuration
  stage_sink.h      compile-time sink concept (StageSink, DiscardSink, CopyFrameSink)
  execution_contract.h
  family.h          typed model families
  llama/            the only headers allowed to include llama/skippy headers
src/                implementations; src/llama/ owns everything native
tests/              GoogleTest, **no llama dependency allowed**
benchmarks/         A/B harness, case manifest, comparison script
tools/              probes (planner, stage pipeline, stage chain)
```

## Keeping this file current

This document is part of the deliverable, not commentary on it. Update it in the
same commit as the change that taught you something:

- **A new failure mode** goes in *Traps that have already bitten us*, with the
  symptom (what it looked like), the cause (why the compiler and the tests did
  not catch it) and the rule that prevents it. A trap without the symptom is not
  actionable; a trap without the cause gets "fixed" the wrong way next time.
- **A new rule or a changed rule** is written here before or with the code. If
  code and this file disagree, this file is wrong until you update it — do not
  leave the divergence for the next person.
- **A new boundary** (something that belongs in `skippy/llama_cpp/patches`, a
  missing ABI entry point, a model that cannot be staged) gets a manifest row
  with acceptance `blocked-upstream` and its reason, plus a line here if it
  changes what "ported" means.
- **Never delete a trap** because "that can't happen any more" unless the code
  that makes it impossible is in the same commit. Half of these entries exist
  because a guard was removed and the bug came back.

If you are an agent reading this: treat an undocumented discovery as unfinished
work — record it in this file as a trap or an `UPSTREAM.md` entry (and say so in
your report) rather than widening the task to fix it. The next session will not
have your context, and this file is the only thing that survives it.

## Code rules

### Types and interfaces

- **No out-parameters above the C ABI.** Functions return their result. The only
  out-params are inside `skippy_*` calls we do not own, and they stop at the
  wrapper.
- **`std::expected<T, E>` when a value comes back; `std::expected<void, E>`
  when only a failure can.** One failure spelling, and it chains: compose with
  `and_then` / `transform` / `or_else` / `transform_error`, never `if (!r)`
  ladders. `.value()` is banned (it throws); `optional<E>`-as-failure is
  retired because `optional::and_then` runs on the engaged value, i.e. on the
  failure. Ruling and migration plan: `TYPE-RULES.md` §4.
- **Never swallow a caught exception's message.** The rule lives in TYPE-RULES
  §4 "A native failure code cannot be constructed without its message" (pattern:
  `WorkloadFailure::native`, pattern: `LoadFailure`; anti-pattern: a member
  channel — see the trap below).
- **`bool` is only a yes/no predicate** (`contains`, `guards_match`, `ranges_join`).
  Never an error channel, never a placeholder, never "did it work".
- **`noexcept` is a terminate contract, not a style choice.** An exception
  escaping a `noexcept` function calls `std::terminate` — it is not an error
  path, and the compiler will not warn you that the body can throw. Every
  `noexcept` body must be nothrow all the way down: stored positions resolve by
  clamping pointer arithmetic (pattern: `StagePlan::tail`, pattern: `StagePlan::view`),
  never by `subspan`/`at`/`substr` over values that came from outside; and an
  allocation that must happen inside a failure handler gets its own
  `catch (...)` so the OOM reporter cannot itself OOM (`LlamaSession::run_pass`
  returning the message with its own inner `catch (...)` is the pattern;
  pattern: `LlamaSession::run_pass`). If
  a function can
  legitimately fail, it returns `expected`/`optional` and is not `noexcept`.
- **Strong types for identities.** `ProfileId`, `TensorId`, `ValueIdentity`,
  `LayerRange` — not bare strings and ints that can be swapped by accident.
- **Constrain at the type where you can.** A parser that takes `const char*` and
  returns an enum accepts every integer in the world; a function that takes a
  validated type accepts only the ones that mean something. Where a value has a
  finite domain (pooling kinds, workload kinds, families), parse into the
  bounded type and reject the rest at the boundary — `parse_pooling` rejecting
  out-of-range integers is the pattern (pattern: `parse_pooling`, pattern:
  `state_kind_of`). The native-planner converters
  (`state_kind_of` and siblings) reject unknown domain values the same way,
  because a defaulted kind would silently flip `summarize()`'s recurrent
  classification rather than merely mislabel. Do not smuggle an unconstrained
  `int` through the middle of the library.
- **No silent widening or narrowing.** Every `static_cast` between integer
  widths needs a reason: a bound was checked, an ABI demands exactly this width,
  or the value is provably in range. Cast *at the boundary* (where the check
  lives), not scattered through the call path — and when the check is "it fits
  because we checked three lines up", say so in a comment or name the helper.
- **`std::size_t` for counts and sizes.** Signed 32-bit is an ABI width, not a
  vocabulary type; use it only at the boundary and convert immediately.
- **Move-only by default** for anything owning a resource (models, sessions,
  plans, chains, runtimes). Delete copy, default move. No `shared_ptr` without
  a stated reason; no raw owning pointers.
- **`const` means what it says.** Mark a method `const` only when the object is
  genuinely unchanged. Mutating through a pointer from a `const` method is legal
  C++ and a lie to the reader — do not do it. `plan_stages`/`validate_chain` are
  const because their cache is not their state; `tokenize`/`embed` are not const
  because their buffers are.

### Style

- C++23. Concepts over templates-by-convention. Ranges over index loops, except
  where the ABI is index-based.
- Prefer compile-time policy (concepts) over runtime plugin registries.
- 4-space indent, 100 columns, attached braces, single-statement guards may stay
  on one line. `clang-format` is authoritative: `.clang-format` in this directory.
- Files under 1,000 lines; split by responsibility, never by symmetry. No
  `utils`/`helpers`/`common` buckets.
- Do not add a method that is a dozen lines of straight-line logic with three
  early returns — name the phases instead (clang-tidy enforces complexity).

### Performance

- **Wins where we can, parity everywhere else.** Every completed runnable row in
  `benchmarks/cases.txt` must read `win` or `parity`; unmeasured and upstream-blocked
  rows retain their documented `pending` or `blocked-upstream` state. A row that loses is a bug
  to fix or a design cost to justify in writing.
- **Never claim a win the measurement does not support.** If the gate cannot
  prove both sides did the same work, the row does not count.
- **A recorded number is only true of the tree it was measured on.** Before a
  push, re-run every manifest row and AGENTS.md figure whose measured path the
  change could touch, and every row the change itself records; a figure that
  predates the commits being pushed is wrong even if it was once true.
  `stage_materialize` sat at `parity` with AGENTS.md quoting 0.80x for two
  commits after the stored-positions and shared-mapping work had already
  flipped it to a ~1.2x win — nobody re-measured, so the record lied.
  Those timings were later withdrawn when the equivalence gate was corrected;
  historical timing claims are not current performance evidence.
- **Remove work, do not micro-optimise.** The façade owns tokenisation, batch
  fill, pooling and normalisation; everything else is llama. A change that
  shaves nanoseconds off our layer while the decode dominates is not a result.
- **Ask what the old ABI pays.** The retained-context pool, the tokenise scratch
  and the plan cache all came from asking that question rather than from
  profiling our own code.
- **Never allocate blind.** Nothing allocates in proportion to a file-supplied
  number until that number is bounded against the file and the domain caps —
  a GGUF count is bytes on disk, not a size class. Even then be timid: the
  model behind the header can be tens of gigabytes, so a buffer that looks
  reasonable per call OOMs the process when it is per layer or per shard.
  Stream through fixed-size blocks instead of materialising (`sha256_of`'s
  64-byte loop is the shape), and when a caller-supplied count drives a
  `reserve`, bound it against the model's real dimensions first
  (pattern: `split_layers`).
- **Zero-copy where the data is already on disk.** Read through the header
  parser's read-only `Mapping` — pages fault in from the page cache and
  nothing heap-sized is allocated — instead of materialising a second copy
  of bytes the process already owns.

### State

- **Avoid process-wide state.** A file-scope singleton is hidden global
  ownership in a library whose rule is that ownership is obvious: two `Runtime`s
  in one process silently share it, it cannot be reset in tests, and it makes a
  `const` method honest only by accident of where the cache lives. Own caches in
  the object that uses them (`LlamaModel::idle_contexts_` and `PlanStore` in
  `LlamaBackend` are the patterns).
- **Atomics and CAS before mutexes for shared caches.** The shape that works
  here is read-copy-update: the index is an immutable `shared_ptr`, readers take
  a snapshot with an atomic load and never block, and publishers build a
  copy-on-write index and install it with `compare_exchange`. Publish returns
  the winning index so racing publishers observe each other instead of planning
  twice. The handoff goes through `skippy::detail::atomic_shared_ptr`:
  `std::atomic<std::shared_ptr<T>>` where `__cpp_lib_atomic_shared_ptr` is
  defined, otherwise the free-function spelling that C++20 deprecated — the
  fallback lives only inside the shim with its suppression scoped to two
  bodies, because GCC 14 warns and this tree builds `-Werror`. Two things to
  say plainly: libc++ implements the shared_ptr atomics
  with a short internal spin rather than being truly lock-free (fine when reads
  dominate and publishes are once per model version — measure if a cache turns
  write-heavy), and anything borrowed out of the snapshot must be returned
  behind a lease that pins it (pattern: `PlanLease`; without it the views
  dangle the moment a later publish drops the old index).
- **No lock on shared state; the serialiser lives in the consumer.** The rule —
  the RCU/CAS design, the two allowed shapes (`std::once_flag` for process-wide
  init; the consumer-side serialiser) and the never-hold-a-lock-across-llama
  clause — lives in TYPE-RULES §3.2 (pattern: `PlanLease`, pattern:
  `Runtime.bridge`; anti-pattern: a mutex around the plan store or context
  pool). The check is `skippy-no-mutex` (TYPE-RULES §5).
- **llama returns before the backend finishes.** `llama_decode`/`llama_encode`
  may hand control back with work still in flight; the embeddings getters
  synchronise, `llama_memory_clear` and the retained-context pool do not.
  Clearing memory a pass is still writing is a race on every GPU backend —
  quiesce with `llama_synchronize` before every `llama_memory_clear` and
  before recycling a context into the pool (a no-op when idle). The pool
  labels a context by `min(llama_n_ctx, llama_n_batch, llama_n_ubatch) /
  llama_n_seq_max`, read from the context, never the requesting session's
  smaller geometry, or a warmed larger context hides capacity the pool then
  cannot serve.

### Rust

- **`rust::String(const std::string&)` throws on non-UTF-8.** Native failure
  text — a planner message built from a hostile GGUF's tensor names, say — is
  arbitrary bytes. Build error strings with `rust::String::lossy(...)`: it is
  `noexcept` (it crosses into Rust's `from_utf8_lossy`) so the message
  survives where the throwing conversion left a bare `ok=false` with no text
  after the guard caught it. The hex helpers are ASCII by construction and
  keep the plain conversion.

- Map to idiomatic Rust, not to the C++ shape. Handles wrap `UniquePtr<_>`; a
  session borrows its model (`StagedSession<'m>`), a model borrows its runtime.
- `&self` for read-only calls, `Pin<&mut _>` for mutating ones. Do not make
  anything `&mut` "just to be safe".
- Errors are typed (`WorkloadError`, `StageError`, `PlanError`) and carry the
  backend's message; do not flatten to strings at the boundary.
- Rust consumers link one static facade/llama/ggml closure. Build-script rpaths
  do not propagate to external executables; keep static linkage and its system
  dependencies complete. Never co-link this closure with the production dynamic
  llama runtime before a separately reviewed cutover.

## Never modify llama.cpp from here

`skippy/llama_cpp/patches` is the only place llama.cpp changes live. This
library **wraps** the runtime; it never edits it, and no PR here may touch a
patch file, regenerate it, or carry a private llama fork.

When a patch you are porting would work better if llama itself changed:

1. **Do not make the change here.** Flag it instead, with the exact evidence —
   the symbol that is missing, the function that re-derives state per call, the
   cost you measured.
2. **Record it as upstream-shaped work** in the manifest entry
   (`benchmarks/cases.txt`, acceptance `blocked-upstream`) and in the README's
   "not covered" section, so it is visible rather than lost. Append the
   finding — with its evidence — to `UPSTREAM.md`, the ledger of upstream
   opportunities; a discovery made while porting must survive the PR.
3. **Port what can be done outside llama**, and say plainly which half is not:
   for example the prepared-export work (`PreparedExport`) removes our per-hop
   allocation and descriptor rebuild, while llama still maps parts internally —
   that half needs an ABI entry point, an ABI version bump and a per-family
   implementation in the patch queue.
4. **Prefer an upstream-worthy patch to a local workaround.** If the change is
   generally useful (a missing accessor, a per-call re-derivation, a buffer that
   should be reusable), it belongs in `llama_cpp/patches` as its own patch with
   its own ABI note — not in a façade shim that hides the cost.

The test for "is this upstream-shaped?" is simple: could the change be expressed
as a C ABI function that any consumer of the staged runtime would want? If yes,
it is a patch, not a façade feature.

## Testing rules

- **The ASAN suite is one command** (`just skippy-api-asan`): the mocked suite
  rebuilt in `target/skippy-api-asan` under AddressSanitizer + UBSan in a
  separate tree, so sanitizer flags never leak into other targets. Any change
  that stores a borrow — span, `string_view`, pointer into a container — runs
  it. The borrow-stress tests exist to make reallocation-driven dangling
  detectable, and were proven to fire on the pre-fix `StagePlan::append`.
- **Mocked unit tests belong in C++** (`tests/`, no llama). They pin contracts:
  classification, buffer bounds, batch layout, rejection paths.
- **Anything that needs a real model belongs in Rust** (`skippy-api-rs/skippy/tests/end_to_end.rs`).
  The mock cannot see pooling selection, context geometry, or the retained-context
  pool — every bug found in those areas was found by the Rust suite.
- `SKIPPY_TEST_REQUIRE_MODELS=1` turns a missing GGUF into a failure; use it in CI.
- A change is not done until tests are green **and** the A/B row for its case has
  a verdict (see below).

## Porting a patch: the procedure

1. **Read the patch, not the docs.** `skippy/llama_cpp/patches/NNNN-*.patch`.
   Identify whether it adds (a) an ABI entry point, (b) graph/family behaviour,
   (c) metadata/validation, or (d) build wiring.
2. **Classify it** — this decides how it can ever be "proven":
   - **A — replaces an existing call.** The old C ABI does the same work.
     Proof: an A/B row, gated on equal outputs, then timed. Must be ≥ parity.
   - **B — new capability.** No old path exists (generation, staged execution,
     prepared tokens). Proof: a *budget* row against the cheapest correct
     reference implementation we can write (e.g. our chain vs a hand-driven
     native chain), gated on byte-identical results.
   - **C — metadata/validation.** Pure computation over a plan. Proof: a
     *ceiling* row — cost must stay a small fraction of what it serves
     (e.g. validate ≤ 1% of realize).
   If a patch cannot be gated, it is not ported yet.
3. **Port the pure half into C++** with no llama dependency, and unit-test it.
   Types and rules that do not need the runtime belong here.
4. **Wrap the native half** in `src/llama/` and expose it through the façade.
   One `expected`/`optional` at the boundary; convert native status + message
   into a typed failure and free the native error exactly once.
5. **Bind it in Rust** (`skippy-sys/cpp/bridge.{h,cpp}` + `skippy-api-rs/skippy/src/lib.rs`),
   then add an E2E test against a real GGUF.
6. **Add a manifest case** and the benchmark pair:
   - `benchmarks/cases.txt`: `patch <TAB> class <TAB> case <TAB> filter <TAB> acceptance`
   - benchmark pairs named `Family/FFI` and `Family/API` (the comparison tool
     pairs on that convention).
7. **Run it**: `just skippy-ab <case> <model.gguf>` and record the verdict.
8. **Commit** only through the checkpoint recipe — the Tooling rules' commit
   rule; nothing else commits on this tree.

### Manifest states

`benchmarks/cases.txt` is tab separated:
`patch  class  case  benchmark_filter  acceptance  note`.

| acceptance | meaning |
|---|---|
| `win` | measured faster than the old ABI (class A) or than the reference (class B) |
| `parity` | measured equal within the class's noise floor; the honest outcome when llama owns the work |
| `pending` | the row is ours to write and has not been written yet |
| `blocked-upstream` | the fix belongs in `skippy/llama_cpp/patches`; the note names the missing entry point or the owning patch |

`just skippy-ab <case>` refuses `pending` and `blocked-upstream` rows with their
note and exits 0 — an unwritten or upstream-blocked case is a known boundary,
not a test failure.

### Workflow commands

```bash
just skippy-api-asan                                 # mocked suite under ASan+UBSan
just skippy-api-verify [embed.gguf] [reranker.gguf]   # format, tidy, ctest, Rust
just skippy-api-checkpoint "<type(scope): subject>" [embed.gguf] [reranker.gguf]
just skippy-ab                                        # list A/B cases
just skippy-ab <case> <model.gguf>                    # run one case; prints acceptance= and the measured ratio
just skippy-llama-api-planner-probe <model.gguf>      # plan/probe a model
```

## Tooling rules

- **Prefer a lint rule to a review comment.** When you find a formatting or code
  issue — a shape that should never be written, a construct the house style
  forbids — first check whether `clang-tidy` already has a check for it
  (`clang-tidy --list-checks`, or the check catalog). If it does, enable it in
  `.clang-tidy` rather than fixing the single instance and leaving the next one
  to a human. If no check exists, say so in the commit message and propose the
  rule (what it should reject, and an example of code it should flag); a rule
  that is not written down is a rule that will be broken again.
- **Fix the class, not the instance.** When the same defect appears twice, the
  second occurrence is evidence that a guard is missing.

- **Do not write new Python tooling.** Benchmark report admission and comparison
  belong to `cargo xtool automation skippy-benchmark-compare`. New automation belongs in a `just` recipe or a typed `tools/`
  helper; the repository rule is in the root `AGENTS.md`.
- **Commit through the recipe**, never a bare `git commit` on this tree:
  `just skippy-api-checkpoint "<type(scope): subject>" [embed.gguf] [reranker.gguf]`.
  It runs format, tidy, the C++ suite and the Rust end-to-end suite first.
- **`cargo check` is not a build.** For a product build use `just build` /
  `just skippy`; for validation use the recipes above.
- Keep `benchmarks/README.md` current when a verdict changes; a stale table is
  worse than no table.

## Traps that have already bitten us

- **Cargo must track the selected native checkout.** At `aa627bb692`, adding an
  untracked native header after a fresh build still reported `Fresh skippy-sys`:
  only facade inputs had `rerun-if-changed` directives, so preparing a new native
  queue could reuse the old static closure. Watch the resolved native directory
  recursively, including its prepared pin/patch stamps and explicit source
  overrides; prove a native-input change reruns the build and an unchanged build
  stays fresh. Clearing Cargo's cache does not repair this missing dependency.
- **A selected sentinel is not an eligible token.** Empty, fully suppressed,
  or entirely non-finite logits have no selectable token. Preserve that failure
  through the selection boundary instead of accepting a negative token or
  allowing an infinity/NaN to stand in for a valid score. The exhaustion tests
  and the benchmark's exhausted-vocabulary gate cover the rejection path.
- **A copied token set can fail even when the model accessor cannot.** The
  suppress-token bridge returned a vector directly from a `noexcept` body;
  growing the result could terminate its Rust caller. Return a fallible envelope
  through `guarded_message`, keeping empty success distinct from allocation
  failure. cxx's infallible Rust reserve cannot be caught by the C++ barrier;
  admit the capacity with Rust `try_reserve` first. The capacity-failure and
  suppression-envelope tests pin both boundaries.
- **An asymptotically slower reference invents a facade speedup.** The
  suppression reference searched every suppress id for every candidate, while
  native greedy sampling builds one bitmap. Mirror the native algorithm and
  allocation shape before timing. The earlier speedup is withdrawn; the
  corrected class-A measurement applies only to its recorded synthetic fixture.
- **A successful result must pass the benchmark gate.** Stage validation changed
  to `expected<void, E>`, but the benchmark still rejected its engaged success
  as though it were an optional error. Admit both successful validators before
  timing, and compare materialization's exact identities and state-effect
  counts on both sides. Timing claims for corrected gates require remeasurement.
- **Every metadata reader validates its header.** The architecture shortcut
  skipped the GGUF magic and version checks used by the full inventory reader.
  `ArchitectureRejectsWrongMagicBeforeMetadata` and
  `ArchitectureRejectsUnsupportedVersionsBeforeMetadata` pin the shared
  boundary. Probe integers likewise use bounded complete-string parsing before
  geometry or allocation; malformed CLI input must not escape as an exception.
- **A `std::move` argument poisons every other argument in the same call.**
  Parameter construction order is unspecified, so
  `recycle(std::move(context), llama_n_ctx(context.get()))` moves the pointer
  out before — or after — the getter reads it; on the wrong order the getter
  dereferences null and the first witness is an e2e SEGFAULT. Hoist every use
  of a moved-from object into its own statement before the call
  (`~LlamaSession` is the fixed example).
- **The pool's label belongs to the context, never to the caller.** A
  `{300,1}` session dropped, then a `{400,1}` request, aborted in
  `llama_decode`: llama pads `n_ctx` to a multiple of 256 but leaves
  `n_batch`/`n_ubatch` at the requested value, so a label derived from
  `n_ctx` advertised a batch the context could not physically run. The pool
  must label from the context (`min(n_ctx, n_batch, n_ubatch)` per sequence);
  a caller cannot supply it. Pinned by
  `end_to_end.rs::pooled_context_never_undersizes_the_batch`.
- **A native getter with a non-zero default is not a presence test.**
  `llama_model_n_cls_out` defaults to 1 and is only overwritten when
  `classifier.output_labels` ships, so reading it ungated called every
  decoder-only chat GGUF a one-output reranker — with a "score" read from a
  head that does not exist. Neither llama flag separates the cases
  (`llama_model_has_encoder` is true only for T5-family pairs,
  `llama_model_has_decoder` for all but T5ENCODER), and the label test the
  upstream API suggests fails the fixture: `bge-reranker-v2-m3` ships no
  labels, and its default count of 1 *is* its real score head. Gate the count
  on the `cls.output.weight` tensor in the inventory, never on family; unknown
  architectures keep the conservative default. Pinned by
  `end_to_end.rs::a_causal_decoder_is_not_a_reranker`.
- **A throwing `std` call inside `noexcept` is `std::terminate` on the worst
  day, and nothing tests it.** The four `StagePlan` value accessors resolved a
  profile's recorded `{offset, count}` with `std::span::subspan`. A corrupt
  position violates its preconditions; C++23 does not specify an exception
  for this case, so behavior is undefined unless the implementation adds
  hardening. Every test passed because
  valid plans never hit the branch, and the compiler cannot see the throw.
  Same shape, softer landing: `run_pass`'s catch handler assigned
  `failure.what()` into a `std::string`, so the OOM reporter itself could
  `bad_alloc` out of `noexcept`. Rule: `noexcept` accessors over stored
  positions clamp via pointer arithmetic (`StagePlan::tail`, `view()`), and
  allocations inside failure handlers get their own `catch (...)`.
- **An FFI result shape `guarded<T>` cannot fold is an unguarded method.**
  The staged/plan bridge surface returns `ok`+`error` structs (and one
  `Vec<StagePlanInfo>`), so the `guarded<Result>` status machinery could not
  wrap them and the methods shipped bare: every `std::string`/`Vec`
  allocation inside those `noexcept` bodies was a latent terminate. They now
  catch in-band via `in_band_failed<T>` / `guarded_message` in-band alongside
  the `ok`/`error`/stages structs (`PlanStagesResult` is the shape). Rule: a
  new
  bridge method is either `guarded<T>`-wrapped or explicitly guarded in-band
  in the same change.
- **Do not trust the compiler to catch a polarity flip.** Converting
  `optional<E>`-as-failure → `std::expected<void, E>` flips every `!`:
  `optional`'s `operator bool` means "the failure is present", `expected`'s
  means "success", so `if (!valid) return valid;` turns from "propagate the
  failure" into "swallow it". Six tests passed while validation rejected
  nothing. Convert one file at a time, update the assertions in the same commit,
  and find every `operator bool` use of the converted type through the LSP
  before you build.
- **`return {}` becomes `nullopt` in some functions and an engaged `optional` in
  others.** Check every one.
- **`ctest` can pass on a stale binary.** If the build failed, the pass means
  nothing. Always read the compiler output before the test summary.
- **A benchmark row is only a comparison if both sides do the same work.** The
  materialise row read "2.8x slower" for two rounds because the reference loop
  skipped fields we were reading. Gate on outputs first.
- **A `$`-anchored benchmark filter matches nothing, silently.** Names
  registered with `->UseRealTime()` end in `/real_time` when the filter is
  applied, but the JSON report drops the suffix — so the manifest rows looked
  right against the report and still matched zero benchmarks. Symptom: google
  benchmark exits 0 having written a 0-byte `--benchmark_out` file (it writes
  no report at all when nothing matched), and the comparison script died on a
  Python traceback that hid the cause. Rule: manifest filters prefix-match and
  never `$`-anchor; `just skippy-ab` checks the report is non-empty and keeps
  the benchmark log before parsing; `--benchmark_list_tests` shows the
  registered names a filter really sees.
- **A lint gate that is not in the verify recipe does not exist.** The Rust
  facade compiled, tested and shipped with an unreachable match arm because
  `skippy-api-verify` ran format/tidy/ctest/cargo test but no clippy or
  `cargo fmt --check`. Rule: a new gate goes into the owning verify recipe in
  the same change, or it will not run. `clang-format`'s version is echoed by
  the format targets because two machines with different Homebrew LLVM
  disagreed on one wrap; the `.clang-format` beside the sources is the
  authority, and the check tells you which binary interpreted it.
- **A span into a growing vector is a dangling pointer waiting to happen.**
  The arena commit took per-profile spans into the flat value vectors while
  later profiles were still appending, so profile 0's spans dangled the moment
  profile 1's values reallocated the vector. One profile with values never
  reallocates — which is why Qwen (values on one profile) and every mocked
  test passed while BERT (values on every profile) segfaulted the bridge.
  Take positions, not pointers: profiles record `{offset, count}` ranges
  and the owning plan resolves them to spans on access
  (`activation_imports_of` and friends). The first fix (a two-phase
  materialise) missed `StagePlan::append`, which had the same bug on the
  incremental path — `clone()` dangled any earlier profile the moment a
  later one grew the vectors. Stored positions make the whole class of
  bug unrepresentable, at any call site, forever.
- **A refactor that moves a computation can leave the old one behind.**
  Moving stored_length into row_size_bytes left the original multiply loop
  in place, so every tensor's size was squared and the planner rejected the
  real model — while all 213 tests stayed green, because no mocked test
  compares stored_length against a real model. The probe is the only
  witness for file-derived values; run it after touching the parser.
- **A native sizing call is a process abort, not an error.** `ggml_row_size`
  trusts its caller: a file-supplied type past `GGML_TYPE_COUNT` reads
  `type_traits` out of bounds, the removed Q4_2/Q4_3 slots report a zero
  block size so the call divides by zero, a first dimension off the block
  grid trips `GGML_ASSERT`, and it multiplies before it divides so its
  intermediate can wrap even when the true byte count fits. `read_extents`
  only bounds the *element* product — the *byte* total can wrap past it.
  `row_size_bytes` therefore validates the whole domain (type in range,
  block > 0, first dimension non-negative and block-aligned, every product
  wrap-checked in `checked_tensor_bytes`) and returns nullopt; `read_tensors`
  turns that into a typed error. Feeding a native helper from file bytes is
  boundary parsing, and boundary parsing rejects instead of crashing.
- **A reverted harness silently un-fixes its own fairness.** The like-for-like
  loop above was lost when `benchmarks/` was reset to its committed state
  during a refactor; the row then read 0.7x for an evening while five wrong
  optimisation hypotheses were tested against it. When a benchmark file is
  restored or rebased, diff it against the last run that produced the number
  you are comparing to — the harness is code, and it regresses like code.
- **Medians of five lie on fixed-cost rows.** Use
  `--benchmark_repetitions=15 --benchmark_enable_random_interleaving=true` and
  read `_min`.
- **A digest that materialises the file materialises the model.** `sha256_of`
  read the whole shard into a `std::vector` and then appended the SHA padding
  to it, so the first plan of a file version — reachable from
  `StageChain::open` through `plan_model_stages` — allocated the entire model
  (30 GB for a 30 GB GGUF) just to feed a hash, and could OOM the process.
  Every fixture in the mocked suite is a few hundred KB, so no test could see
  it. Rule: nothing that touches a whole shard may allocate in proportion to
  it — hash and scan through the same read-only `Mapping` the header parser
  uses (pages fault in from the page cache), padding only the final partial
  block in a fixed 128-byte tail.
  `InventoryParser.Sha256MatchesReferenceDigests` pins digests against
  hashlib on both padding shapes. An invalid mapping is a typed failure, not
  an empty digest: `sha256_of` returns
  `std::expected<std::string, std::string>` and rejects `!mapping.valid()`,
  pinned by the directory-shape seed the fuzzer found
  (`fuzz/corpus/fuzz_inventory/seed_crash_directory`, replayed by
  `just skippy-api-asan`).
- **cxx cannot hold a non-copyable C++ object by value.** Use `UniquePtr<T>`,
  methods taking `Pin<&mut T>`, and declare `type T;` in the bridge block.
- **A failure's reason stored on the object "for later" is a reason lost.**
  `load_model` parked the inventory parser's message in
  `LlamaBackend::last_load_error_` (anti-pattern: `last_load_error_`); nothing read it, and a second concurrent
  load overwrote it before any reader could. Cause: the failure type
  (`RuntimeError`, a bare enum) had no room for a message, so the message had
  nowhere to travel but sideways into state. Rule: the message travels in the
  failure value — `LoadFailure{code, message}` — from the boundary that
  produced it to the caller that reports it; a member is never the channel.
  Twice now: P1-2's `LlamaSession::last_backend_error_` (e5600593;
  anti-pattern: `last_backend_error_`) recorded
  the backend's `what()`, was read by nobody, and the code rule above even
  cited it as *the pattern* — corrected in the same change that deleted the
  member. `WorkloadFailure::native` (a native code cannot be built without
  its message) is the rung-1 fix; two occurrences meets the §2 threshold.

- **The formatter reflows lambdas.** A `str.replace()` over formatted C++ can
  silently no-op — and so can any text substitution over source. Superseded by
  TYPE-RULES §1 "Source changes go through the compiler or the LSP, never text
  substitution" (anti-pattern: `str.replace` over source): edit through the
  compiler or the LSP, and the toolchain confirms the change landed.
- **Identities are opaque bytes, and each layer spells them differently.**
  Inside C++ a plan identity is a `string_view` slice of the plan's
  `identity_arena`, where neighbours sit back to back and nothing is
  NUL-terminated: handing `name.data()` to an API that wants `const char*`
  concatenates the rest of the arena into one garbage tensor name and the
  loader rejects it — copy to owning `std::string`s first (pattern: `own()`)
  and only then take `.c_str()`; NUL-terminated copies are what the native ABI
  needs until `skippy_bytes_ref` lands upstream. The Rust bridge hex-encodes
  the same bytes today (`StagePlanInfo`'s identity fields); `&[u8]` views over
  C++-owned storage (ownership design O2/O6) is the target, not the current
  state.
- **`ffi_vs_api_benchmark.cpp` warnings are errors.** A `nodiscard` result must
  be `(void)`-cast, and the benchmark will fail to compile (not run) if missed.
- **A green tidy gate that ran without its plugin.** `skippy-llama-api-tidy`
  passed `--load=target/skippy-tidy/SkippyTidyChecks.so` relative to the repo
  root, then `cd`'d into `skippy/skippy_api` before the first invocation;
  dlopen resolved the path against the new working directory, failed, and
  clang-tidy treats a failed `--load` as non-fatal — so every skippy-* check
  silently vanished and the gate passed on the built-ins alone. Symptom:
  `skippy-no-mutex` reported zero findings on a tree whose `backend.cpp`
  carried a `std::once_flag` the then-current regex forbade, while the same
  plugin loaded by absolute path fired exactly there; the recipe's own log
  carried `Error opening …SkippyTidyChecks.so` the whole time. Cause: a
  relative plugin path plus a `cd`, with nothing asserting the plugin was
  live. Rule: the gate loads the plugin by absolute path, refuses to run when
  it is missing, and proves liveness first — `-list-checks` must name one of
  the plugin's own checks before any file is checked.
- **A fixture that passes against stubs proves nothing about the gate.** Symptom: `skippy-no-mutex`
  forbade `std::once_flag`, `LlamaBackend::LlamaBackend` declared one with no NOLINT, and the gate
  reported zero findings. Cause: fixtures compile with `-I stubs`, where the name is literally
  `std::once_flag`; real TUs compile against libc++, where it is `std::__1::once_flag`, and a regex
  over `getQualifiedNameAsString()` anchored on `std::(…)$` never matches. Rule: TYPE-RULES §1
  "born red on the real artefact" — the first fixture is the tree's own violation, run on the gate's
  toolchain, and `std::` names are matched with `hasName`/`hasAnyName`.
- **A `const` local handed to `DoNotOptimize` selects the deprecated overload.**
  Bitten while adding the `SamplingEligible` row: declaring a benchmark local
  `const` (`const llama_token sampled = ...`) and passing it to
  `benchmark::DoNotOptimize` fails the build under `-Werror`, because google
  benchmark deprecates its `Tp const&` overload and a const lvalue can only
  bind to it — non-const lvalues bind to the plain `Tp&` overload and pointers
  to `Tp*`, neither deprecated. Nothing else in the file warns, so the error
  reads like a mystery three lines at a time. Rule: benchmark locals passed to
  `DoNotOptimize` are non-const lvalues (as every pre-existing row already
  spells), or are passed by address.

## Where things stand

See `README.md` for the patch coverage table and the current benchmark verdicts.
