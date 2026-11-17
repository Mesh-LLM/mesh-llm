//! This executable's bootstrap identity; `ci_controller::bootstrap` owns the
//! check that it is Cargo's build output with no other workspace crate inside.

use ci_controller::bootstrap::BootstrapTool;

/// The authority executable may compile only itself and the controller library.
pub const AUTHORITY_AUTOMATION: BootstrapTool = BootstrapTool {
    package: "authority-automation",
    local_packages: &["authority-automation", "ci-controller"],
    usage: "usage: authority-automation automation bootstrap",
    recipe: "just authority-automation-bootstrap",
};
