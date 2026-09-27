use crate::support::{TestResult, Tool, check};

#[test]
fn migration_native_policy_runtime_package_portable() -> TestResult {
    check(Tool::RuntimePackage, "happy")
}

#[test]
fn migration_native_policy_runtime_package_rejects_unsafe() -> TestResult {
    check(Tool::RuntimePackage, "rejected")
}
