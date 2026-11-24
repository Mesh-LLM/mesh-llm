//! Keeps both bundle variables out of the tests that discover runtimes.
//!
//! Skippy's `rerun_without_bundle_env!` clears only
//! `SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR`. Mesh discovery also reads
//! `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`, so a test run with that one set would
//! find runtimes the tests don't expect.

use std::process::Command;

use super::{MESH_NATIVE_RUNTIME_BUNDLE_DIR_ENV, NATIVE_RUNTIME_BUNDLE_DIR_ENV};

/// Re-runs `test` (a path from `module_path!()`, crate name included) in a
/// child process without either bundle variable, and returns whether it did,
/// in which case the caller should return.
pub fn rerun_without_bundle_env(test: &str) -> bool {
    if std::env::var_os(MESH_NATIVE_RUNTIME_BUNDLE_DIR_ENV).is_none() {
        return skippy_runtime_install::test_support::rerun_without_bundle_env(test);
    }
    // libtest names a test by its path inside the test binary's crate.
    let name = test.split_once("::").map_or(test, |(_, path)| path);
    let output = Command::new(std::env::current_exe().expect("locate the test binary"))
        .args(["--exact", name, "--nocapture"])
        .env_remove(MESH_NATIVE_RUNTIME_BUNDLE_DIR_ENV)
        .env_remove(NATIVE_RUNTIME_BUNDLE_DIR_ENV)
        .output()
        .expect("run the test without the bundle directories");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{name} failed without the bundle directories:\n{stdout}{stderr}"
    );
    // A name that matches no test would pass with nothing run.
    assert!(
        stdout.contains("test result: ok. 1 passed"),
        "{name} did not run in the child:\n{stdout}"
    );
    true
}

/// Starts a runtime discovery test: when a bundle variable is set, runs the
/// test again without it and returns.
#[macro_export]
macro_rules! rerun_without_bundle_env {
    ($test:ident) => {
        if $crate::native_runtime_install::test_support::rerun_without_bundle_env(concat!(
            module_path!(),
            "::",
            stringify!($test)
        )) {
            return;
        }
    };
}
