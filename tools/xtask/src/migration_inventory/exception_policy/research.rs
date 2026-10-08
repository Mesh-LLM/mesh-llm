//! Exact existing optional references and task fixtures, distinct from actual SDK cadence.
use super::super::ledger::ExceptionEntry;
pub(in crate::migration_inventory) const PATHS: [&str; 16] = [
    "skippy/crates/skippy-cache/src/cachegen/fixtures/generate_lmcache_compat.py",
    "skippy/crates/skippy-quantize/scripts/compare-reference-quantization.py",
    "skippy/evals/latency-benchmarking/latency-proxy.py",
    "skippy/evals/latency-benchmarking/measure.py",
    "mesh/evals/moa-openrouter/analyze_ablation.py",
    "mesh/evals/moa-openrouter/lite_agent.py",
    "mesh/evals/moa-openrouter/make_fixture.py",
    "mesh/evals/moa-openrouter/orclient.py",
    "mesh/evals/moa-openrouter/probe_tools.py",
    "mesh/evals/moa-openrouter/record.py",
    "mesh/evals/moa-openrouter/record_agentic.py",
    "mesh/evals/scenarios/debug-session/buggy.py",
    "mesh/evals/scenarios/edit-file/server.py",
    "mesh/evals/scenarios/refactor/config.py",
    "mesh/evals/test_injection_framing.py",
    "mesh/evals/virtual_llm_eval.py",
];
pub(in crate::migration_inventory) fn is_upstream(path: &str) -> bool {
    path.starts_with("skippy/crates/skippy-cache/")
        || path.starts_with("skippy/crates/skippy-quantize/")
}
pub(in crate::migration_inventory) fn project(path: &str) -> &'static str {
    if path == "skippy/crates/skippy-quantize/scripts/compare-reference-quantization.py" {
        "evals/quantizer-reference/pyproject.toml"
    } else {
        "evals/research-python/pyproject.toml"
    }
}
pub(in crate::migration_inventory) fn admitted(entry: &ExceptionEntry) -> bool {
    PATHS.contains(&entry.path.as_str())
        && entry.status
            == if is_upstream(&entry.path) {
                "isolated_upstream_reference"
            } else {
                "isolated_research_evaluation"
            }
        && entry.cadence.as_deref()
            == Some(
                "optional explicit operator research/upstream regeneration only; never PR/main required setup",
            )
        && entry.local_dependency_files.as_ref().is_some_and(|files| {
            files.iter().map(String::as_str).eq([
                project(&entry.path),
                "evals/research-python/BOUNDARIES.json",
            ])
        })
}
