use crate::support::{TestResult, Tool, check};

#[test]
fn migration_native_policy_release_matrix_complete_targets() -> TestResult {
    check(Tool::ReleaseMatrix, "happy")
}

#[test]
fn migration_native_policy_release_matrix_missing_targets() -> TestResult {
    check(Tool::ReleaseMatrix, "rejected")
}

#[test]
fn migration_native_policy_release_matrix_bad_labels() -> TestResult {
    check(Tool::ReleaseMatrix, "error")
}
