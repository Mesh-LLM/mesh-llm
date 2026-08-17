---
name: skippy-correctness
description: Use this skill when validating skippy staged execution against full-model execution, adding model families, changing split boundaries, testing activation wire dtypes, or diagnosing mismatch behavior.
metadata:
  short-description: Validate staged execution exactness
---

# skippy-correctness

Use this skill when staged execution must be proven equivalent to full-model
execution.

## What To Check

- Single-stage direct GGUF parity.
- Two-stage boundary parity for representative split points.
- Multi-stage chain parity for package-backed serving.
- Selected-device and pinned-device behavior.
- Activation wire dtype exactness (`f16` by default, `q8` only with evidence).
- Recurrent/hybrid family behavior and topology affinity.
- Multimodal projector handling once native media execution is wired.

When native code moves between Skippy modules, preserve the existing capability
boundary during validation:

- `execution` and `activation` changes require direct, two-stage, and
  multi-stage parity coverage.
- `state` changes require KV import/export, checkpoint, trim, and exact-prefix
  cache coverage.
- `sampling` and `speculative_decoding` changes require deterministic sampling,
  draft acceptance, rejection recovery, and checkpoint cleanup coverage.
- `model_package` changes require inspection, tensor filtering, and artifact
  writer coverage.

Do not put a test-only implementation hook back into `src/skippy.cpp`; keep the
test with the capability that owns the behavior.

## Commands

First check whether standalone correctness crates have been imported:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

Current mesh-level checks:

```bash
cargo test -p skippy-runtime --lib
cargo test -p skippy-server --lib
cargo test -p mesh-llm-host-runtime --lib inference::skippy
cargo test -p mesh-llm-host-runtime --lib
```

If `skippy-correctness` is imported later, prefer that harness for model-backed
exactness gates instead of adding one-off tests.
