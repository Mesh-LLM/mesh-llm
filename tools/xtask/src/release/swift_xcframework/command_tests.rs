use super::fixtures::Fixture;
use std::process::Command;

#[test]
fn cli_verifies_host_when_inert_rust_lipo_is_explicit() {
    let fixture = Fixture::host();
    fixture.materialize();
    fixture.write(true);
    let directory =
        std::path::PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap()).join("debug");
    let executable = directory.join("xtask");
    let lipo = directory.join("examples/migration_xcframework_lipo_fixture");
    let before = super::layout_tests::snapshot(&fixture.root);
    let result = Command::new(executable)
        .args(["release", "swift-xcframework"])
        .arg(&fixture.root)
        .arg("host-only")
        .arg("--lipo")
        .arg(lipo)
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"verified 1 XCFramework slice(s) for host-only mode\n"
    );
    assert!(result.stderr.is_empty());
    assert_eq!(super::layout_tests::snapshot(&fixture.root), before);
    if let Some(destination) = std::env::var_os("XCFRAMEWORK_QA_FIXTURE") {
        std::fs::rename(fixture.directory.keep(), destination).unwrap();
    }
}
