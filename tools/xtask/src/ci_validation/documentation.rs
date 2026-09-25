use crate::command::{DynResult, ensure_contains};

pub(super) fn check_documentation_invariants(
    readme: &str,
    contributing: &str,
    release: &str,
    release_package_source: &str,
    ci_docs: &str,
    depot_docs: &str,
) -> DynResult<()> {
    for (text, needle, context) in [
        (
            readme,
            "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
            "README ARM64 asset note",
        ),
        (
            readme,
            "mesh-llm-aarch64-unknown-linux-gnu-cuda.tar.gz",
            "README ARM64 CUDA asset note",
        ),
        (
            release,
            "Windows release artifacts use the `x86_64-pc-windows-msvc` target triple",
            "RELEASE Windows publish note",
        ),
        (
            release_package_source,
            "cargo run -p xtask -- repo-consistency release-targets",
            "Imported Just release consistency command",
        ),
        (
            contributing,
            "just check-release",
            "CONTRIBUTING release consistency command",
        ),
        (ci_docs, "CI · Manual Full", "CI manual-full workflow"),
        (ci_docs, "Main / Quality", "native main workflow results"),
        (ci_docs, "CI Required", "CI topology required summary"),
        (
            depot_docs,
            "Cache isolation",
            "Depot cache-isolation policy",
        ),
    ] {
        ensure_contains(text, needle, context)?;
    }
    Ok(())
}
