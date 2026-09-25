# Scheduler change validation

New benchmark orchestration belongs in typed `tools/xtask` commands under the
rule in `.agents/skills/manage-ci/SKILL.md`; from the repository root,
`cargo xtool repo-consistency ci-crate-lists` is a working alias example. Keep
the Python-backed benchmark below as transitional validation until its Rust
replacement passes parity checks. Do not add new Python tooling.

Any change under `crates/skippy-scheduler/` must be checked for performance regressions before it is merged.

- Run the complete crate test suite: `cargo test -p skippy-scheduler`.
- Rerun the deterministic scheduler lab in release mode: `just with-lld cargo bench -p skippy-scheduler --features scheduler-lab --bench scheduler_lab`.
- When a change is reachable from serving, rerun the matching `evals/skippy-competitive-benchmark.py` workload. Admission, batching, cache-affinity, prefill/decode mixing, capacity, or preemption changes must include the Thoughtworks c64/c128/c256 cells.
- Compare against an artifact from the same model bytes, runtime configuration, hardware, and workload manifest. Record the exact commit SHA, commands, artifact path, scheduler-visible and outer queue depth, batch sizes, cached/new prompt tokens, evictions, throughput, and TTFT. Do not promote a scheduler change when the relevant benchmark regresses without an explicit reviewed rationale.
