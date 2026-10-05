---
name: skippy-model-package
description: Use this skill when inspecting GGUF models, planning layer ranges, generating or validating skippy package artifacts, fake packages for direct GGUFs, materialized stage cache behavior, or GGUF writer integration.
metadata:
  short-description: Inspect and package GGUF stages
---

# skippy-model-package

Use this skill for model inspection, package planning, stage materialization,
and cache behavior.

## Ownership

Rust owns package manifests, topology planning inputs, cache policy, and mesh
model-storage integration. The patched llama/skippy ABI owns GGUF tensor
inspection and GGUF artifact writing.

The native package ABI is declared in `include/skippy/model_package.h` and
implemented in `src/skippy/model_package.cpp`. Keep model inspection, tensor
filtering, metadata copying, and package writing in that module; do not grow
`src/skippy.cpp` with package behavior.

Direct GGUF loading in mesh should materialize as a fake package identity in
the skippy runtime so the split-serving path can use the same package-backed
stage machinery as Hugging Face packages.

## Commands

Check current package names before running commands:

```bash
just with-lld cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

Useful current checks in this repo:

```bash
just with-lld cargo test -p skippy-runtime --lib
just with-lld cargo test -p skippy-topology --lib
just with-lld cargo test -p mesh-llm-host-runtime --lib inference::skippy
```

For a published layer package, prefer package-local diagnostics before a live
split smoke. On macOS or Linux, `just release-runtime-build cpu` builds the
`skippy-package-builder` Cargo binary and packages it with its native libraries.
Replace `<runtime-id>` with the generated directory under `dist/native-runtimes`:

```bash
just with-lld cargo test -p skippy-package-builder --bin skippy-package-builder
just release-runtime-build cpu
package_builder="dist/native-runtimes/<runtime-id>/tools/skippy-package-builder"
"$package_builder" preflight <package-dir> --stages 2
```

## Cache Policy

Materialized stages are derived cache. Model storage commands may evict
materialized stage artifacts without deleting the source model/package. Preserve
pinned materialized artifacts unless the command explicitly asks for a stronger
cleanup.
