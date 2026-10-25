use crate::support::{TestResult, Tool, check};

#[test]
fn migration_native_policy_host_dependencies_happy_paths() -> TestResult {
    check(Tool::HostDependencies, "happy")
}

#[test]
fn migration_native_policy_host_dependencies_rejects_backend_imports_and_glibc() -> TestResult {
    check(Tool::HostDependencies, "rejected")
}

#[test]
fn migration_native_policy_host_dependencies_handled_errors_exit_two() -> TestResult {
    check(Tool::HostDependencies, "error")
}

#[test]
fn migration_native_policy_host_dependencies_usage_errors_exit_two() -> TestResult {
    check(Tool::HostDependencies, "usage")
}
