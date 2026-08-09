---
name: llama-stage-patch-changes
description: Use this skill when changing mesh-llm's patched llama.cpp Skippy ABI, runtime hooks, model introspection, tensor filtering, activation-frame execution, GGUF writer surface, upstream pin, or patch queue.
metadata:
  short-description: Maintain the llama.cpp Skippy patch queue
---

# llama-stage-patch-changes

Use this skill when changing the Skippy staged-runtime ABI carried in
`third_party/llama.cpp/patches`.

## Boundaries

- Keep durable llama.cpp-side changes in `third_party/llama.cpp/patches/*.patch`.
- Keep the upstream pin in `third_party/llama.cpp/upstream.txt`.
- Do not edit `.deps/llama.cpp` as the final artifact; regenerate the
  patch queue from commits.
- Keep mesh orchestration, protocol compatibility, lifecycle, model management,
  and API status behavior in Rust.
- Keep one functional boundary per patch. Patch numbers must be unique and
  contiguous.
- Keep public ABI declarations separate from independently reviewable model
  lifecycle, loading, and package implementation changes.
- The Skippy native ABI is an internal lockstep boundary, not a stable
  cross-version compatibility contract. It may change whenever the feature
  requires it; update the Rust FFI mirror and all callers in the same change.
- Do not preserve old native ABI signatures for compatibility. Bump the ABI
  version when the boundary changes so mismatches are diagnosable, and make
  sure the shipped Rust side and native runtime are built from the same queue.
- Do not add a terminal source-reorganization patch. A deliberate layout or
  ownership change must be represented in the recreated patches that own the
  affected capabilities.

## Native Source Layout

- `include/skippy.h` is an umbrella only. Put public C ABI declarations in
  standalone `include/skippy/<capability>.h` headers.
- Put implementations in `src/skippy/<capability>.cpp` and private C++
  declarations in narrowly named `src/skippy/*.h` headers.
- Use `snake_case` capability names. Keep exported symbols prefixed with
  `skippy_` and avoid generic `helpers`, `utils`, or expanded `common` modules.
- `src/skippy.cpp` is retired. Extend the owning capability module and keep new
  implementation files below 1,000 lines.
- Make every public header independently compilable as both C11 and C++17.
  Update explicit CMake source lists and installation rules with new modules.
- Do not preserve retired source include paths unless the task explicitly asks
  for compatibility. Continue to version and mirror any binary ABI change.

## ABI PR documentation requirements

Every pull request that changes the Skippy ABI must include an explicit ABI
inventory in the PR description. Do not describe a changed function signature
as a newly added function.

The inventory must state, for each change:

- the exact symbol or declaration name and its complete signature or field
  change;
- whether it was added, changed, deprecated, deleted, or removed;
- the public header containing the declaration;
- the implementation source and Rust FFI mirror, when applicable;
- the ABI version before and after the change;
- why the change is required and what data or behavior it enables;
- that backward compatibility with older native runtimes is intentionally not
  required, and that the Rust FFI mirror and callers were updated in lockstep;
- the tests that exercise the native ABI boundary, including public-header
  compilation when a header changes.

Use this compact table in the PR description:

| Status | Symbol/declaration | Public header | Implementation / mirror | Reason | Lockstep update |
|---|---|---|---|---|---|
| Changed / Added / Removed | exact name and signature | `include/skippy/<capability>.h` | `src/skippy/<capability>.cpp`; Rust FFI path | behavior enabled | Rust mirror/callers updated; old ABI not supported |

For a changed function signature, call out that it is an ABI change even when
the symbol name is unchanged. List removed declarations explicitly as
“none” when no functions or fields were deleted; this prevents reviewers from
having to infer removals from a patch diff. Keep this inventory synchronized
with the ABI version constants in `include/skippy/common.h` and the mirrors in
`crates/skippy-ffi/src/lib.rs`. Do not add compatibility shims solely to
support an older native runtime; the acceptance criterion is a synchronized
Rust/native build and a clear version mismatch if the pieces are mixed.

## Local Flow

Prepare the pinned checkout and current patch queue:

```bash
scripts/prepare-llama.sh pinned
```

For llama-side editing, work in `.deps/llama.cpp` or another llama.cpp
checkout where commits can be named and inspected. Base the branch on the
pinned upstream, then carry the stage ABI patch commits on top.

For an ordinary capability change, emit one focused mail-format patch after
the current queue. Do not rewrite unrelated entries:

```bash
repo_root="$(pwd)"
llama_checkout="${LLAMA_CHECKOUT:-$repo_root/.deps/llama.cpp}"
last_patch="$(find third_party/llama.cpp/patches -maxdepth 1 -type f -name '*.patch' | sort | tail -n 1)"
last_number="${last_patch##*/}"
last_number="${last_number%%-*}"
next_number=$((10#$last_number + 1))
git -C "$llama_checkout" format-patch -1 \
  --start-number "$next_number" \
  --output-directory "$repo_root/third_party/llama.cpp/patches" HEAD
```

For a deliberate queue-boundary or source-layout change, rebuild the affected
series from the pinned upstream instead. Create capability-owned commits in
their intended order, place declarations and implementation in their final
modules from the first patch that introduces them, and format the complete
replacement series with contiguous numbering. Before replacing the durable
queue, verify both of these invariants:

```bash
# The reconstructed commit series has exactly the intended final tree.
git diff --exit-code <authoritative-final-commit> <reconstructed-series-head>

# No patch defers the structural change to the end of the series.
git log --reverse --oneline <pinned-upstream>..<reconstructed-series-head>
```

Move the old queue to an explicit temporary backup, generate the replacement
into a fresh `third_party/llama.cpp/patches` directory, and retain the backup
until clean application and native compilation pass. Never keep both series or
duplicate patch numbers in the durable directory.

## Validation

Validate patch application in a clean checkout:

```bash
tmp_root="$(mktemp -d /tmp/mesh-llama.XXXXXX)"
trap 'rm -rf -- "$tmp_root"' EXIT
LLAMA_WORKDIR="$tmp_root/llama.cpp" scripts/prepare-llama.sh pinned
LLAMA_WORKDIR="$tmp_root/llama.cpp" \
  MESH_LLM_LLAMA_BUILD_ROOT="$tmp_root/build" \
  LLAMA_STAGE_BACKEND=cpu \
  LLAMA_STAGE_LINK_MODE=static \
  scripts/build-llama.sh
```

Compile each new public header once as C11 and once as C++17 with warnings
treated as errors. For implementation moves, run the tests owned by the moved
capability in addition to the Rust fallout checks below.

For Rust fallout, run cargo commands serially:

```bash
cargo fmt --all --check
cargo check -p mesh-llm
cargo test -p skippy-runtime --lib
cargo test -p skippy-server --lib
cargo test -p mesh-llm --lib
```

Patch files are mail-format artifacts. Do not hand-normalize them in a way that
breaks `git am`.
