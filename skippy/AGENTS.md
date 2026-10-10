# Skippy Agent Notes

These instructions apply under `skippy/` in addition to the root `AGENTS.md`. Paths and `just` commands below are relative to the workspace root. Skippy owns standalone inference and native runtimes; it must not depend on Mesh crates or plugin hosts.

## Key docs

| Doc | What it covers |
|---|---|
| `skippy/docs/ARCHITECTURE.md` | Current product and API boundaries |
| `skippy/docs/SKIPPY.md` | Historical Mesh integration plan |
| `skippy/docs/SKIPPY_SPLITS.md` | Running large models with split serving |
| `skippy/docs/LAYER_PACKAGE_REPOS.md` | Layer-package publishing |
| `skippy/docs/design/LLAMA_STAGE_INTEGRATION_PLAN.md` | llama.cpp staged-runtime plan and patch-queue background |

## Building and native runtimes

- `just skippy` builds the standalone `skippy` CLI and one locally packaged native runtime; it does not build MeshLLM. The CLI discovers `target/debug/native-runtimes` beside the executable automatically. Use `--runtime-bundle` only to select a different bundle explicitly.
- `just skippy-cli-build` builds only the debug CLI. `just skippy-cli-release-build` builds only the release CLI. Those commands do not package a runtime.
- `just skippy-release` builds the release CLI and selected native runtime without MeshLLM. `just release-runtime-build <backend>` builds one packageable native runtime under `dist/native-runtimes/`.
- `just release-build` builds Skippy before MeshLLM. Release CI publishes one backend-neutral Skippy CLI archive per platform and native runtimes per backend. Static backend linkage is not a release or packaging lane.
- For branch-local Skippy ABI, llama.cpp patch, MAS hidden-state, or native tensor changes, use the local packaged runtime; do not test new ABI symbols against a downloaded release runtime.

### llama.cpp "generator does not match" error

If a native build fails with `CMake Error: ... generator : Ninja / Does not match the generator used previously: Unix Makefiles`, the build directory still holds a CMake cache from a build configured with a different generator. That happens when `ninja` appears on or leaves `PATH` between builds, because `scripts/build-llama.sh` picks the generator from `PATH` at configure time. The script detects the mismatch and clears the stale `.deps/llama-build/...` directory by itself before reconfiguring. On a checkout that predates the guard, remove the build directory named in the error and rebuild.

See `CONTRIBUTING.md` for full dev workflow.

## llama.cpp ABI Patch Queue

### No model-family special cases in shared Skippy machinery

- Do not branch on a model-family name or `LLM_ARCH_*` in shared Skippy stage
  extraction, planning, replay, proof, input ownership, or serving logic to
  make a family pass. This is a hard rule for new code and repairs.
- Express a difference through the graph, an explicit capability or input
  contract, runtime metadata, or a recipe supplied by the owning model builder.
  Put genuinely family-specific graph construction and annotations in that
  model's implementation and its owning `model_support/` patch.
- Before adding a shared-runtime exception, audit the same capability across
  other families and test the general contract. If the capability cannot yet
  be expressed generally, report the gap instead of adding an architecture
  allowlist or a family-named fallback.
- Existing family checks in shared Skippy paths are audit debt, not precedent.
  When touching one, replace it with a semantic contract or record the blocker
  and a migration plan; do not copy the pattern into another path.

mesh-llm embeds the stage runtime and links patched llama.cpp static ABI
libraries. The only durable llama.cpp patch queue is
`skippy/llama_cpp/patches`, pinned by `skippy/llama_cpp/upstream.txt`.

- `just build` packages the selected native runtime and builds the standalone
  Skippy CLI, then builds the UI and dynamic MeshLLM host. The hosts never link
  a backend library.
- Static llama.cpp compilation is the explicitly named native-runtime primitive
  (`just release-runtime-build` / `scripts/package-native-runtime.sh --build`), used
  when changing the Skippy ABI or patch queue. It is not a host build path.
- Do not reintroduce an external `llama-server` / `rpc-server` runtime lane.
- If you need to update upstream llama.cpp, use `scripts/prepare-llama.sh`,
  `scripts/build-llama.sh`, `scripts/update-llama-pin.sh`, and
  `scripts/summarize-llama-upstream.sh`.
- Keep the queue ordered by functional ownership, with unique contiguous patch
  numbers. A source-layout change must be folded into the patches that own the
  affected capabilities; do not append a terminal "split", "move", or
  "cleanup" patch that reorganizes code introduced by earlier patches.
- Before adding a patch, find the existing patch that introduces or owns the
  affected source, test, or fixture. Prefer updating that patch for fixes and
  extensions of its capability, including test-only changes. Rework any later
  patches that depend on it, update queue metadata, and validate a cold replay
  from the pinned upstream. Add a new patch only for a genuinely separate
  capability without an existing owner; explain that boundary in the change.
- Apply the queue in three lanes: numbered core patches directly under
  `patches/`, numbered family-enablement patches listed by
  `patches/model_support/series`, then generated graph-semantics shards listed
  by `patches/generated/series`. Numbering is contiguous within each lane.
- New model-family implementations and their family-specific conversion,
  template, multimodal, runtime, and tests belong in one focused
  `model_support/` patch. Keep reusable Skippy machinery in the core lane and
  mechanically generated graph annotations in the generated lane.
- When deliberately changing queue boundaries, recreate the affected series
  from the pinned upstream and prove that the rebuilt series produces the
  intended final tree.
  Once a capability has an owning module, every patch in the recreated series
  must edit that module directly rather than introducing code in an obsolete
  monolith and moving it later.
- Treat the public Skippy ABI surface and model lifecycle/loading as distinct
  patch boundaries. Public declarations may precede their implementation, but
  a patch should not combine ABI definition with independently reviewable model
  loading or package behavior.

### Skippy Native Source Layout

Treat the patched Skippy C ABI as a set of capability-owned modules, not as one
implementation file.

- Keep `include/skippy.h` as an umbrella header only. Public declarations
  belong in standalone C-compatible headers under `include/skippy/`, named for
  their capability: for example `sampling.h`, `speculative_decoding.h`,
  `state.h`, and `model_package.h`.
- Keep capability implementations in `src/skippy/<capability>.cpp`. Private
  C++ declarations belong beside them in narrowly named headers under
  `src/skippy/`; they are not part of the installed ABI.
- Put new behavior in its owning module. `src/skippy.cpp` is retired; do not
  recreate it. Runtime lifecycle, sessions, activation framing, execution,
  verification, sampling, state, tokenization, and model packaging each belong
  to their existing capability-owned source files.
- Use `snake_case` filenames and preserve the `skippy_` prefix for exported C
  symbols. Avoid generic `helpers`, `utils`, or expanded `common` buckets.
- Keep new implementation files below 1,000 lines. If a capability approaches
  that size, split it by a narrower responsibility before adding more code.
- Public headers must compile independently as C11 and C++17 headers. Declare
  each implementation source explicitly in CMake and install the complete
  `include/skippy/` header tree.
- Source include compatibility is not assumed. Do not add forwarding headers
  for retired paths unless a task explicitly requires them. Binary ABI changes
  still require the normal Skippy ABI version bump and synchronized Rust FFI
  constants.

### Skippy native API documentation

- Treat Doxygen-style comments in `include/skippy.h` and
  `include/skippy/*.h` as the source of truth for the public native API
  reference. Document every public header and exported `skippy_*` function
  beside its declaration with an `@brief` describing the capability it owns.
- After changing a public Skippy header or exported function, prepare the
  patched native checkout and regenerate the website page:

  ```bash
  scripts/prepare-llama.sh pinned
  python3 scripts/generate-skippy-api-doc.py
  python3 scripts/generate-skippy-api-doc.py --check
  ```

- Commit the regenerated `mesh/website/src/docs/pages/skippy-api.md` with the
  native queue change. Do not hand-edit the generated page or let a native API
  PR merge without updating the website reference.

## Workspace Crates

Product crates live under `skippy/crates/`. The most important crates:

- `skippy-cli/` — standalone `skippy` command.

Shared foundations:

- `skippy-guardrails/` — guardrail and compaction primitives for OpenAI-compatible paths.
- `skippy-hardware-profile/`, `skippy-native-runtime/`, `skippy-runtime-install/` — hardware profile detection, native runtime manifest/selection, runtime download/install/cache.

OpenAI-compatible API:

- `skippy-inference-api/` — OpenAI-compatible HTTP frontend (chat, completions, responses, models).

Models:

- `skippy-model-artifact/`, `skippy-model-hf/`, `skippy-model-package/`, `skippy-model-ref/`, `skippy-model-resolver/` — model catalog, HuggingFace download, packaging, reference resolution.

Embedded staged runtime (skippy):

- `skippy-ffi/` — Rust ABI bindings to the patched llama.cpp staged runtime.
- `skippy-runtime/` — Rust-side staged runtime, package materialization, model info.
- `skippy-serving/` — embedded staged-runtime serving (frontend, binary transport, runtime state, embedded HTTP).
- `skippy-protocol/`, `skippy-topology/`, `skippy-coordinator/`, `skippy-cache/`, `skippy-metrics/`, `skippy-bench/`, `skippy-correctness/`, `skippy-package-builder/` — supporting skippy infrastructure.

Tools and benchmarks:

- `metrics-server/` — standalone metrics collector binary.
- `skippy-gpu-bench/`, `llama-spec-bench/` — benchmarking binaries.

This list covers the crates you are most likely to touch; check `skippy/crates/` and each crate's `Cargo.toml` description for anything not listed.

Other product directories:

- `skippy/docs/design/` — Skippy architecture and native-runtime plans.
- `skippy/docs/` — family certification, configuration, benchmarks, parity, and split serving.
- `skippy/evals/` — Skippy benchmarking and evaluation scripts.
- `skippy/scripts/tools/`, `skippy/scripts/recipes/` — Clang rewriter and quantization recipes.
- `skippy/scripts/tests/` — product-owned script tests.
- `skippy/llama_cpp/patches/` — durable llama.cpp patch queue, pinned by `skippy/llama_cpp/upstream.txt`.

## Key Source Files

Embedded staged runtime (`skippy/crates/skippy-*`):

- `skippy-ffi/src/lib.rs` — Rust ABI mirror of the patched llama.cpp staged runtime; `ABI_VERSION_*` constants must stay in sync with `skippy/common.h` in the patch queue.
- `skippy-runtime/src/package.rs` — layer-package materialization, identity-bound cache.
- `skippy-runtime/src/devices.rs` — backend device enumeration.
- `skippy-serving/src/frontend.rs`, `skippy-serving/src/frontend/` — embedded chat/generation frontend.
- `skippy-serving/src/runtime_state.rs` — KV-slot, lane, session state machine.
- `skippy-serving/src/binary_transport.rs`, `binary_transport/` — binary transport to embedded server.

OpenAI-compatible HTTP frontend (`skippy/crates/skippy-inference-api/src/`):

- `router.rs`, `chat.rs`, `completions.rs`, `responses.rs`, `models.rs`, `sse.rs`, `backend.rs` — OpenAI surface.

## Skippy ABI Compatibility

The patched llama.cpp staged runtime has its own ABI version, tracked in `skippy/common.h` (inside the patch queue) and mirrored by `SKIPPY_ABI_VERSION_*` constants in `skippy/crates/skippy-ffi/src/lib.rs`.

- When changing the staged-runtime ABI in the patch queue, bump `SKIPPY_ABI_VERSION_PATCH` (or MINOR/MAJOR) in `skippy/common.h` AND keep the Rust constants in `skippy-ffi/src/lib.rs` in sync in the same change.
- `skippy-runtime` consumes the ABI version for package loading and feature probing; an out-of-sync mirror will silently advertise the wrong version.
- Treat the staged-runtime ABI the same as the mesh wire protocol: additive changes preferred, breaking changes need explicit acknowledgement.

## ABI validation

If `skippy-ffi` ABI constants change, run `cargo test -p skippy-ffi --lib` and `cargo test -p skippy-runtime --lib` in addition to the workspace's normal checks. Do not stop at a build-only validation.

## What NOT to add

Do not reintroduce an external `llama-server` or `rpc-server` runtime lane. The embedded staged runtime via patched llama.cpp is the supported path.
