# Cache change validation

Any change under `crates/skippy-cache/` must be checked for performance regressions before it is merged.

- Run the complete crate test suite: `cargo test -p skippy-cache --lib`.
- Rerun the cache benchmark that covers the changed policy or data structure. For family cache behavior, use `evals/skippy-cache-family-bench.sh <artifact-dir>`; use `SKIPPY_CACHE_SKIP_BUILD=1` only after an exact release build of the commit under test.
- When a change can affect serving admission, eviction, prefix reuse, or resident capacity, rerun the matching `evals/skippy-competitive-benchmark.py` Thoughtworks cells and include the high-load c64/c128/c256 comparison.
- Compare against an artifact from the same model bytes, runtime configuration, hardware, and workload manifest. Record the exact commit SHA, commands, artifact path, cached/new prompt tokens, evictions, throughput, and TTFT. Do not promote a cache change when the relevant benchmark regresses without an explicit reviewed rationale.
