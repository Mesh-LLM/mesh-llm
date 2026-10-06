# Scheduler change validation

New benchmark orchestration belongs in typed `tools/xtask` commands under the
rule in `.agents/skills/manage-ci/SKILL.md`; from the repository root,
`cargo xtool repo-consistency ci-crate-lists` is a working alias example. Use
the native competitive prepared-run/report commands documented in
`skippy/docs/COMPETITIVE_BENCHMARK.md`. Do not add new Python tooling.

Any change under `skippy/crates/skippy-scheduler/` must be checked for performance regressions before it is merged.

- Run the complete crate test suite: `cargo test -p skippy-scheduler`.
- Rerun the deterministic scheduler lab in release mode: `just with-lld cargo bench -p skippy-scheduler --features scheduler-lab --bench scheduler_lab`.
- When a change is reachable from serving, rerun the matching native `automation replay-matrix competitive-run --input /absolute/prepared-run.json` workload. Admission, batching, cache-affinity, prefill/decode mixing, capacity, or preemption changes must include the Thoughtworks c64/c128/c256 cells.
- Compare against an artifact from the same model bytes, runtime configuration, hardware, and workload manifest. Record the exact commit SHA, commands, artifact path, scheduler-visible and outer queue depth, batch sizes, cached/new prompt tokens, evictions, throughput, and TTFT. Do not promote a scheduler change when the relevant benchmark regresses without an explicit reviewed rationale.
