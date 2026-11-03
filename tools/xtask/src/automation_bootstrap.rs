//! xtask's bootstrap identity. The running `xtask` must be Cargo's own build
//! output and compile without any product, native-runtime, or UI crate (so no
//! llama.cpp or UI preparation); `ci_controller::bootstrap` owns the check.

use ci_controller::bootstrap::BootstrapTool;

/// Workspace-local packages xtask may compile. Every other workspace crate is
/// product, native-runtime, or UI code and would pull llama.cpp or UI
/// preparation into automation bootstrap.
pub(crate) const XTASK: BootstrapTool = BootstrapTool {
    package: "xtask",
    local_packages: &[
        "xtask",
        "ci-controller",
        "process-supervision",
        "mesh-llm-release-footer",
        "trajectory-reader",
    ],
    usage: "usage: cargo xtool automation bootstrap",
    recipe: "just automation-bootstrap",
};
