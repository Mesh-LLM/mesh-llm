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
- Prefer one ABI capability per patch.

## Native Source Layout

- `include/skippy.h` is an umbrella only. Put public C ABI declarations in
  standalone `include/skippy/<capability>.h` headers.
- Put implementations in `src/skippy/<capability>.cpp` and private C++
  declarations in narrowly named `src/skippy/*.h` headers.
- Use `snake_case` capability names. Keep exported symbols prefixed with
  `skippy_` and avoid generic `helpers`, `utils`, or expanded `common` modules.
- Do not add a separable responsibility to `src/skippy.cpp`. Extract or extend
  the owning capability module, and keep new implementation files below 1,000
  lines.
- Make every public header independently compilable as both C11 and C++17.
  Update explicit CMake source lists and installation rules with new modules.
- Do not preserve retired source include paths unless the task explicitly asks
  for compatibility. Continue to version and mirror any binary ABI change.

## Local Flow

Prepare the pinned checkout and current patch queue:

```bash
scripts/prepare-llama.sh pinned
```

For llama-side editing, work in `.deps/llama.cpp` or another llama.cpp
checkout where commits can be named and inspected. Base the branch on the
pinned upstream, then carry the stage ABI patch commits on top.

After editing and committing in that checkout, emit the new mail-format patch
after the current queue. Do not delete or rewrite unrelated queue entries:

```bash
repo_root="$(pwd)"
last_patch="$(find third_party/llama.cpp/patches -maxdepth 1 -type f -name '*.patch' | sort | tail -n 1)"
last_number="${last_patch##*/}"
last_number="${last_number%%-*}"
next_number=$((10#$last_number + 1))
git -C .deps/llama.cpp format-patch -1 \
  --start-number "$next_number" \
  --output-directory "$repo_root/third_party/llama.cpp/patches" HEAD
```

## Validation

Validate patch application in a clean checkout:

```bash
tmp_root="$(mktemp -d /tmp/mesh-llama.XXXXXX)"
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
