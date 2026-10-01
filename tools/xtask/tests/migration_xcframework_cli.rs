use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["release", "swift-xcframework"])
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn usage_when_mode_is_invalid() {
    let result = run(&["missing.xcframework", "invalid"]);
    assert_eq!(result.status.code(), Some(2));
    assert!(result.stdout.is_empty());
}

#[test]
fn missing_input_precedes_native_when_lipo_is_nonexistent() {
    let directory = tempfile::tempdir().unwrap();
    let executable = directory.path().join(if cfg!(windows) {
        "missing.exe"
    } else {
        "missing-lipo"
    });
    let result = run(&[
        "missing.xcframework",
        "--lipo",
        executable.to_str().unwrap(),
    ]);
    assert_eq!(result.status.code(), Some(1));
    assert!(
        String::from_utf8(result.stderr)
            .unwrap()
            .contains("XCFramework or Info.plist is missing")
    );
}

#[test]
fn legacy_shell_sentinel_is_not_executed_when_path_is_literal() {
    let directory = tempfile::tempdir().unwrap();
    let sentinel = directory.path().join("sentinel");
    let path = format!("missing; touch {}", sentinel.display());
    let result = run(&[&path]);
    assert_eq!(result.status.code(), Some(1));
    assert!(!sentinel.exists());
}

#[test]
fn explicit_executable_when_lipo_override_is_relative() {
    let result = run(&["missing.xcframework", "--lipo", "relative-lipo"]);
    assert_eq!(result.status.code(), Some(2));
}
