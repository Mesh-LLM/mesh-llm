# Porting a llama.cpp patch into `skippy_api`

This file is for whoever (human or agent) ports the next patch from
`skippy/llama_cpp/patches` into this library, and for anyone writing code here.
It records rules that were agreed the hard way; most of them exist because
something broke.

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
work. The next session will not have your context, and this file is the only
thing that survives it.

## Code rules

### Types and interfaces

- **No out-parameters above the C ABI.** Functions return their result. The only
  out-params are inside `skippy_*` calls we do not own, and they stop at the
  wrapper.
- **`std::expected<T, E>` when a value comes back.** **`std::optional<E>` when
  only a failure can come back.** Never `std::expected<void, E>`.
  `nullopt` means "it worked"; an engaged optional is the reason it did not.
- **Never swallow a caught exception's message.** `catch (const std::exception&)`
  plus a generic error code makes OOM indistinguishable from a backend failure;
  the backend's `what()` must survive into the failure the caller sees
  (`LlamaSession::last_backend_error` is the pattern).
- **`bool` is only a yes/no predicate** (`contains`, `guards_match`, `ranges_join`).
  Never an error channel, never a placeholder, never "did it work".
- **`noexcept` is a terminate contract, not a style choice.** An exception
  escaping a `noexcept` function calls `std::terminate` — it is not an error
  path, and the compiler will not warn you that the body can throw. Every
  `noexcept` body must be nothrow all the way down: stored positions resolve by
  clamping pointer arithmetic (`StagePlan::tail` and `view()` are the pattern),
  never by `subspan`/`at`/`substr` over values that came from outside; and an
  allocation that must happen inside a failure handler gets its own
  `catch (...)` so the OOM reporter cannot itself OOM (`LlamaSession::run_pass`
  recording `last_backend_error_` is the pattern). If a function can
  legitimately fail, it returns `expected`/`optional` and is not `noexcept`.
- **Strong types for identities.** `ProfileId`, `TensorId`, `ValueIdentity`,
  `LayerRange` — not bare strings and ints that can be swapped by accident.
- **Constrain at the type where you can.** A parser that takes `const char*` and
  returns an enum accepts every integer in the world; a function that takes a
  validated type accepts only the ones that mean something. Where a value has a
  finite domain (pooling kinds, workload kinds, families), parse into the
  bounded type and reject the rest at the boundary — `parse_pooling` rejecting
  out-of-range integers is the pattern. The native-planner converters
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

- **Wins where we can, parity everywhere else.** Every row in
  `benchmarks/cases.txt` must read `win` or `parity`. A row that loses is a bug
  to fix or a design cost to justify in writing — `stage_materialize` at 0.80x
  is documented, not ignored.
- **Never claim a win the measurement does not support.** If the gate cannot
  prove both sides did the same work, the row does not count.
- **Remove work, do not micro-optimise.** The façade owns tokenisation, batch
  fill, pooling and normalisation; everything else is llama. A change that
  shaves nanoseconds off our layer while the decode dominates is not a result.
- **Ask what the old ABI pays.** The retained-context pool, the tokenise scratch
  and the plan cache all came from asking that question rather than from
  profiling our own code.

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
  behind a lease that pins it (`PlanLease` is the pattern; without it the views
  dangle the moment a later publish drops the old index).
- **CAS before mutex, scoped before global.** Reach for `compare_exchange` and
  atomics wherever a mutex is only protecting a cache; if a mutex is genuinely
  needed (complex invariants that cannot be expressed as a publish), hold it
  for the shortest scope possible — a `std::scoped_lock` acquired immediately
  before the shared state and released immediately after, never stored, never
  held across a call into llama or an allocation the caller controls. A mutex
  that is held across a call into the backend is a deadlock and a serialiser
  hiding as a lock; if you find yourself holding one that long, the state wants
  to be an RCU snapshot instead.

### Rust

- Map to idiomatic Rust, not to the C++ shape. Handles wrap `UniquePtr<_>`; a
  session borrows its model (`StagedSession<'m>`), a model borrows its runtime.
- `&self` for read-only calls, `Pin<&mut _>` for mutating ones. Do not make
  anything `&mut` "just to be safe".
- Errors are typed (`WorkloadError`, `StageError`, `PlanError`) and carry the
  backend's message; do not flatten to strings at the boundary.

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
- **Anything that needs a real model belongs in Rust** (`skippy/tests/end_to_end.rs`).
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
5. **Bind it in Rust** (`skippy-sys/cpp/bridge.{h,cpp}` + `skippy/src/lib.rs`),
   then add an E2E test against a real GGUF.
6. **Add a manifest case** and the benchmark pair:
   - `benchmarks/cases.txt`: `patch <TAB> class <TAB> case <TAB> filter <TAB> acceptance`
   - benchmark pairs named `Family/FFI` and `Family/API` (the comparison tool
     pairs on that convention).
7. **Run it**: `just skippy-ab <case> <model.gguf>` and record the verdict.
8. **Commit** with `just skippy-api-checkpoint "<conventional message>"`, which
   runs format, tidy, C++ tests and the Rust E2E suite first.

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
just skippy-ab <case> <model.gguf>                    # run one case
just skippy-ab <case> <model.gguf>                    # prints WIN/PARITY/LOSS verdict
just skippy-llama-api-planner-probe <model.gguf>      # plan/probe a model
just skippy-llama-api-stage-pipeline <model.gguf> N   # chain N stages, compare token
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
  second occurrence is evidence that a guard is missing."

- **Do not write new Python tooling.** `benchmarks/compare_benchmarks.py` is
  transitional. New automation belongs in a `just` recipe or a typed `tools/`
  helper; the repository rule is in the root `AGENTS.md`.
- **Commit through the recipe**, never a bare `git commit` on this tree:
  `just skippy-api-checkpoint "<type(scope): subject>" [embed.gguf] [reranker.gguf]`.
  It runs format, tidy, the C++ suite and the Rust end-to-end suite first.
- **`cargo check` is not a build.** For a product build use `just build` /
  `just skippy`; for validation use the recipes above.
- Keep `benchmarks/README.md` current when a verdict changes; a stale table is
  worse than no table.

## Traps that have already bitten us

- **A throwing `std` call inside `noexcept` is `std::terminate` on the worst
  day, and nothing tests it.** The four `StagePlan` value accessors resolved a
  profile's recorded `{offset, count}` with `std::span::subspan`, which throws
  `std::out_of_range` when the position is corrupt — out of a `noexcept`
  function that is a process crash, not an error. Every test passed because
  valid plans never hit the branch, and the compiler cannot see the throw.
  Same shape, softer landing: `run_pass`'s catch handler assigned
  `failure.what()` into a `std::string`, so the OOM reporter itself could
  `bad_alloc` out of `noexcept`. Rule: `noexcept` accessors over stored
  positions clamp via pointer arithmetic (`StagePlan::tail`, `view()`), and
  allocations inside failure handlers get their own `catch (...)`.
- **Do not trust the compiler to catch a polarity flip.** Converting
  `expected<void, E>` → `optional<E>` turns `if (!valid) return valid;` from
  "propagate the failure" into "swallow it". Six tests passed while validation
  rejected nothing. Convert one file at a time, update the assertions in the same
  commit, and grep for `!` before you build.
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
- **A reverted harness silently un-fixes its own fairness.** The like-for-like
  loop above was lost when `benchmarks/` was reset to its committed state
  during a refactor; the row then read 0.7x for an evening while five wrong
  optimisation hypotheses were tested against it. When a benchmark file is
  restored or rebased, diff it against the last run that produced the number
  you are comparing to — the harness is code, and it regresses like code.
- **Medians of five lie on fixed-cost rows.** Use
  `--benchmark_repetitions=15 --benchmark_enable_random_interleaving=true` and
  read `_min`.
- **`rust::String` requires UTF-8.** Planner identities are **binary**
  (`resident_tensor_at_v1` returns an identity, not a name). Hex-encode them.
- **cxx cannot hold a non-copyable C++ object by value.** Use `UniquePtr<T>`,
  methods taking `Pin<&mut T>`, and declare `type T;` in the bridge block.
- **The formatter reflows lambdas.** A `str.replace()` over formatted C++ can
  silently no-op; grep after editing, not after compiling.
- **`ffi_vs_api_benchmark.cpp` warnings are errors.** A `nodiscard` result must
  be `(void)`-cast, and the benchmark will fail to compile (not run) if missed.

## Where things stand

See `README.md` for the patch coverage table and the current benchmark verdicts.
