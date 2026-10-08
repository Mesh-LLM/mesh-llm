# Cache change validation

New benchmark orchestration belongs in typed `tools/xtask` commands under the
rule in `.agents/skills/manage-ci/SKILL.md`; from the repository root,
`cargo xtool repo-consistency ci-crate-lists` is a working alias example. Use
the native competitive prepared-run/report commands documented in
`skippy/docs/COMPETITIVE_BENCHMARK.md`. Do not add new Python tooling.

Any change under `skippy/crates/skippy-cache/` must be checked for performance regressions before it is merged.

- Run the complete crate test suite: `cargo test -p skippy-cache --lib`.
- Rerun the cache benchmark that covers the changed policy or data structure. For family cache behavior, use `skippy/evals/skippy-cache-family-bench.sh <artifact-dir>`; use `SKIPPY_CACHE_SKIP_BUILD=1` only after an exact release build of the commit under test.
- When a change can affect serving admission, eviction, prefix reuse, or resident capacity, rerun the matching native `automation replay-matrix competitive-run --input /absolute/prepared-run.json` Thoughtworks cells and include the high-load c64/c128/c256 comparison.
- Compare against an artifact from the same model bytes, runtime configuration, hardware, and workload manifest. Record the exact commit SHA, commands, artifact path, cached/new prompt tokens, evictions, throughput, and TTFT. Do not promote a cache change when the relevant benchmark regresses without an explicit reviewed rationale.
