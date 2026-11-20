//! This executable's bootstrap identity; `ci_controller::bootstrap` owns the
//! check that it is Cargo's build output with no other workspace crate inside.

use ci_controller::bootstrap::BootstrapTool;

/// The cleanup executable may compile only itself, the controller library, and
/// process supervision.
pub const RUNNER_CLEANUP: BootstrapTool = BootstrapTool {
    package: "runner-cleanup",
    local_packages: &["runner-cleanup", "ci-controller", "process-supervision"],
    usage: "usage: runner-cleanup automation bootstrap",
    recipe: "just runner-cleanup-bootstrap",
};
