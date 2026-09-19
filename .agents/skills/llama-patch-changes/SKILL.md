---
name: llama-patch-changes
description: Use when changing mesh-llm's llama.cpp patch queue, upstream pin, prepare/build scripts, or carried RPC, MoE, and mesh-hook llama.cpp patches.
---

# llama-patch-changes

Use this skill when editing the llama.cpp patch queue, refreshing patches from a
llama.cpp checkout, updating the pinned upstream SHA, or changing build scripts
that prepare or consume patched llama.cpp.

## Boundaries

- Keep durable llama-side changes in the ordered queue under
  `third_party/llama.cpp/patches`: top-level core patches first,
  `model_support/series` second, and `generated/series` last.
- Keep the upstream pin in `third_party/llama.cpp/upstream.txt`.
- Do not add a submodule, vendor a llama checkout, or depend on the old
  Mesh-LLM llama.cpp fork.
- Do not treat edits in `.deps/llama.cpp` as durable until the patch queue has
  been regenerated and committed.
- Do not add llama-stage ABI/static in-process patches unless the task
  explicitly asks for that integration pass.
- Prefer small, reviewable llama commits with one functional boundary per
  patch. Keep patch numbers unique and contiguous within each queue lane.
- Put a new model family's implementation, conversion, templates, multimodal
  integration, runtime adaptations, and family tests in one focused patch in
  `model_support/`. Do not spread family-specific code through core patches.
- Keep generated graph-semantics edits in `generated/`; do not hand-maintain
  them in either the core or model-support lane.
- Do not append a terminal patch whose only purpose is to split, move, or clean
  up code introduced by earlier patches. Recreate the affected patches so they
  use the intended ownership boundaries from the outset.

## Local Flow

Prepare the pinned upstream checkout and current patch queue:

```bash
scripts/prepare-llama.sh pinned
```

For actual llama-side editing, prefer a normal llama.cpp checkout or branch
where commits can be named and inspected. Base the branch on upstream
`ggml-org/llama.cpp` `master`, then carry the Mesh-LLM patch commits on top.

For a deliberate queue rewrite, reconstruct capability-owned core commits from
the pinned upstream, add model-family support commits, then add the generated
family shards. Verify the reconstructed head is tree-identical to the
authoritative final checkout before regenerating each queue lane. Preserve the
`model_support/series` and `generated/series` manifests explicitly rather than
flattening their patches into the top-level queue.

```bash
repo_root="$(pwd)"
llama_checkout="${LLAMA_CHECKOUT:-$repo_root/.deps/llama.cpp}"
patch_backup="$(mktemp -d /tmp/mesh-llm-patches.XXXXXX)"
patch_root="$repo_root/third_party/llama.cpp/patches"
mkdir -p "$patch_backup/core"
mv "$patch_root"/*.patch "$patch_backup/core/"
mv "$patch_root/model_support" "$patch_backup/model_support"
mkdir -p "$patch_root/model_support"
git -C "$llama_checkout" format-patch \
  --start-number 1 \
  --output-directory "$patch_root" \
  "$(cat "$repo_root/third_party/llama.cpp/upstream.txt")..<core-head>"
git -C "$llama_checkout" format-patch \
  --start-number 1 \
  --output-directory "$patch_root/model_support" \
  "<core-head>..<model-support-head>"
```

Regenerate `model_support/series` from the sorted patch filenames. Leave the
generated lane in place unless its generator inputs changed; if they did,
regenerate it with the deterministic family-patch workflow rather than moving
or formatting those commits by hand.

Keep the temporary backup until clean patch application and the required native
build pass. Ordinary focused changes may append a patch without rebuilding
unrelated functional boundaries.

## Validation

Validate that patches apply in a clean checkout:

```bash
tmp_llama="$(mktemp -d /tmp/mesh-llm-llama.XXXXXX)"
trap 'rm -rf -- "$tmp_llama"' EXIT
LLAMA_WORKDIR="$tmp_llama" scripts/prepare-llama.sh pinned
```

For normal mesh-llm validation, use the repository build workflow:

```bash
just build
```

For Rust-only fallout from build-system or runtime call-site changes:

```bash
cargo fmt --all --check
cargo check -p mesh-llm
```

Run Cargo commands serially. This repo frequently hits Cargo lock conflicts
when multiple Cargo commands run at once.

### Model-load spot check (required for backend or model-switch changes)

A clean patch replay plus a green build does **not** prove the runtime works.
Metal shaders in `ggml-metal.metal` are JIT-compiled on-device at first model
open, so a broken shader builds green everywhere and only fails at load time.
Machine-reconciled patches can also silently drop arch cases from switches in
`src/llama-model.cpp` (for example `llama_model_rope_type`), which only fail
when a model of that arch creates a context.

After any queue change that touches backend sources (`.metal`, CUDA, Vulkan),
`ggml.c`/`ggml-*.h` kernel argument structs, or `src/llama-model.cpp` switch
statements, load a small real model on your local backend and confirm one
completion returns:

```bash
./target/debug/mesh-llm serve --model "Qwen/Qwen2.5-3B-Instruct-GGUF@main:q4_k_m" --log-format json
# wait for the model to appear, then:
curl -s http://127.0.0.1:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}'
```

On a Mac this exercises the Metal shader JIT path directly. Watch the JSON log
for `metal_library_init: error` and `model-open failure`. If the change adds or
touches support for a specific model family, spot-check a model of that family
as well.

When regenerating the queue against a new upstream pin, also diff the arch
case lists of reconciled switches against upstream and account for every
deletion:

```bash
# example: rope-type switch parity
git -C .deps/llama.cpp show <upstream-pin>:src/llama-model.cpp \
  | awk '/llama_rope_type llama_model_rope_type/,/^}$/' \
  | grep -oE "LLM_ARCH_[A-Z0-9_]+" | sort > /tmp/upstream-cases.txt
awk '/llama_rope_type llama_model_rope_type/,/^}$/' .deps/llama.cpp/src/llama-model.cpp \
  | grep -oE "LLM_ARCH_[A-Z0-9_]+" | sort > /tmp/patched-cases.txt
comm -23 /tmp/upstream-cases.txt /tmp/patched-cases.txt  # must be empty or explained
```

## Updating The Upstream Pin

Test the queue against current upstream without moving the pin:

```bash
scripts/prepare-llama.sh latest
just build
cargo test -p mesh-llm --lib
```

If the queue applies and validation passes, update the upstream pin:

```bash
cp third_party/llama.cpp/upstream.txt /tmp/old-llama-upstream.txt
git -C .deps/llama.cpp rev-parse "$(cat .deps/llama.cpp/.git/mesh-llm-upstream-sha)" > third_party/llama.cpp/upstream.txt
```

Commit the pin update with any patch refreshes.

## Gotchas

- `scripts/prepare-llama.sh` configures local git identity for `git am`; keep
  that responsibility there for fresh CI checkouts.
- Patch files are mail-format artifacts and may intentionally contain
  whitespace that `git diff --check` reports. Do not hand-normalize patches in
  a way that changes or breaks `git am`.
- Build outputs live under `.deps/llama.cpp/build`; the root `llama.cpp`
  symlink is compatibility-only.
- Important backend flags include `GGML_RPC=ON`, `BUILD_SHARED_LIBS=OFF`, and
  `LLAMA_OPENSSL=OFF`; preserve CPU, Metal, CUDA, Vulkan, and ROCm behavior
  when touching build scripts.
- See `mesh-llm/docs/LLAMA_CPP_FORK.md` for the full patch-queue maintenance
  notes and `mesh-llm/docs/LLAMA_STAGE_INTEGRATION_PLAN.md` for deferred
  llama-stage integration.
