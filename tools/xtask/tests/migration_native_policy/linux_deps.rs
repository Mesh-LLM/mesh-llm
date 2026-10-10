use crate::support::{TestResult, Tool, check};

#[test]
fn migration_native_policy_linux_deps_collects_verifies_and_orders() -> TestResult {
    check(Tool::LinuxDeps, "happy")
}

#[test]
fn migration_native_policy_linux_deps_rejects_policy_violations() -> TestResult {
    check(Tool::LinuxDeps, "rejected")
}

#[test]
fn migration_native_policy_linux_deps_malformed_input_fails() -> TestResult {
    check(Tool::LinuxDeps, "error")
}

#[test]
fn migration_native_policy_linux_deps_usage_errors_exit_two() -> TestResult {
    check(Tool::LinuxDeps, "usage")
}
