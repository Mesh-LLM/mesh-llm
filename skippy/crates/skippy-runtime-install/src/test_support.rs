//! Keeps a product's bundle directory out of the runtime discovery tests.
//! Built for this crate's tests and, through the `test-support` feature, for
//! the crates whose tests discover runtimes; `tests/install_abi.rs` includes it
//! by path.
//!
//! A build that runs the workspace's tests with
//! `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` set, for the tests that load the
//! runtime, would also hand that bundle to every test that discovers
//! runtimes, and they would find runtimes they don't expect. Those tests start
//! with [`rerun_without_bundle_env!`]: when the variable is set, the test runs
//! again in a child process without it, the way
//! `explicit_sources_ignore_product_environment` already runs its child.

use std::process::Command;

use crate::NATIVE_RUNTIME_BUNDLE_DIR_ENV;

/// Re-runs `test` (a path from `module_path!()`, crate name included) in a
/// child process without the bundle variable, and returns whether it did, in
/// which case the caller should return. Without the variable it returns `false` at once.
pub fn rerun_without_bundle_env(test: &str) -> bool {
    if std::env::var_os(NATIVE_RUNTIME_BUNDLE_DIR_ENV).is_none() {
        return false;
    }
    // libtest names a test by its path inside the test binary's crate.
    let name = test.split_once("::").map_or(test, |(_, path)| path);
    let output = Command::new(std::env::current_exe().expect("locate the test binary"))
        .args(["--exact", name, "--nocapture"])
        .env_remove(NATIVE_RUNTIME_BUNDLE_DIR_ENV)
        .output()
        .expect("run the test without the bundle directory");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{name} failed without {NATIVE_RUNTIME_BUNDLE_DIR_ENV}:\n{stdout}{stderr}"
    );
    // A name that matches no test would pass with nothing run.
    assert!(
        stdout.contains("test result: ok. 1 passed"),
        "{name} did not run in the child:\n{stdout}"
    );
    true
}

/// Starts a runtime discovery test: when `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`
/// is set, runs the test again without it and returns.
#[macro_export]
macro_rules! rerun_without_bundle_env {
    ($test:ident) => {
        if $crate::test_support::rerun_without_bundle_env(concat!(
            module_path!(),
            "::",
            stringify!($test)
        )) {
            return;
        }
    };
}
