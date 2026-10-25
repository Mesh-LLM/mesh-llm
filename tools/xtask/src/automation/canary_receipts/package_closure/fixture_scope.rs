//! Isolate process-global signal fixtures from concurrent test harness owners.
pub(super) fn isolated(module: &str, name: &str) -> bool {
    if std::env::var_os("CANARY_PACKAGE_SECURITY_CHILD").is_some() {
        return false;
    }
    let module = module.split_once("::").map_or(module, |(_, path)| path);
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("{module}::{name}"),
            "--nocapture",
            "--test-threads=1",
        ])
        .env("CANARY_PACKAGE_SECURITY_CHILD", "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "fixture child failed: {output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
        "fixture did not execute: {output:?}"
    );
    true
}
