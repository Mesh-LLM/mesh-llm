use std::process::Command;

#[test]
fn missing_library_and_arguments_reject_before_any_destination_write() {
    let dir = tempfile::tempdir().unwrap();
    let swift = dir.path().join("generated.swift");
    std::fs::write(&swift, "original bindings").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["prepared-input", "swift-api-checksum"])
        .arg(dir.path().join("absent.a"))
        .arg(&swift)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("unknown command"));
    assert_eq!(
        std::fs::read_to_string(&swift).unwrap(),
        "original bindings"
    );
    let help = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["prepared-input", "swift-api-checksum"])
        .output()
        .unwrap();
    assert!(!help.status.success());
    assert!(String::from_utf8_lossy(&help.stderr).contains("LIBRARY GENERATED_SWIFT"));
}
