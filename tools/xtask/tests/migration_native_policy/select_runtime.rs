use crate::support::{TestResult, Tool, check};

#[test]
fn migration_native_policy_select_runtime_happy_paths() -> TestResult {
    check(Tool::SelectRuntime, "happy")
}

#[test]
fn migration_native_policy_select_runtime_rejects_ambiguous_and_mismatched() -> TestResult {
    check(Tool::SelectRuntime, "failure")
}

#[test]
fn migration_native_policy_select_runtime_usage_errors_exit_two() -> TestResult {
    check(Tool::SelectRuntime, "usage")
}
