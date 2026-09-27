use crate::support::{TestResult, Tool, check};

#[test]
fn migration_native_policy_windows_deps_collects_and_verifies() -> TestResult {
    check(Tool::WindowsDeps, "happy")
}

#[test]
fn migration_native_policy_windows_deps_rejects_policy_violations() -> TestResult {
    check(Tool::WindowsDeps, "rejected")
}

#[test]
fn migration_native_policy_windows_deps_malformed_input_fails() -> TestResult {
    check(Tool::WindowsDeps, "error")
}

#[test]
fn migration_native_policy_windows_deps_usage_errors_exit_two() -> TestResult {
    check(Tool::WindowsDeps, "usage")
}
