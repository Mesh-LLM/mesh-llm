#![cfg(unix)]

#[path = "swift_privacy_adapter/lifecycle.rs"]
mod lifecycle;
#[path = "swift_privacy_adapter/support.rs"]
mod support;

use support::Fixture;

#[test]
fn interleaves_lint_when_all_embedded_bytes_match() {
    let fixture = Fixture::new("clean");
    let embedded = fixture.embed("first", true);
    let output = fixture.command().output().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(fixture.invocations(), [fixture.template.clone(), embedded]);
    assert_eq!(output.stdout, fixture.success_stdout(1));
    assert_eq!(
        output.stderr,
        b"safe lint diagnostic\nsafe lint diagnostic\n"
    );
}

#[test]
fn retains_template_success_when_framework_is_missing() {
    let fixture = Fixture::new("clean");
    std::fs::remove_dir(&fixture.framework).unwrap();
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert_eq!(output.stdout, fixture.template_stdout());
    assert_eq!(
        fixture.invocations(),
        std::slice::from_ref(&fixture.template)
    );
}

#[test]
fn retains_template_success_when_framework_has_no_embedding() {
    let fixture = Fixture::new("clean");
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert_eq!(output.stdout, fixture.template_stdout());
    assert_eq!(
        fixture.invocations(),
        std::slice::from_ref(&fixture.template)
    );
}

#[test]
fn stops_before_embedded_lint_when_bytes_differ() {
    let fixture = Fixture::new("clean");
    fixture.embed("first", false);
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert_eq!(output.stdout, fixture.template_stdout());
    assert_eq!(
        fixture.invocations(),
        std::slice::from_ref(&fixture.template)
    );
}

#[test]
fn retains_template_success_when_second_lint_fails() {
    let fixture = Fixture::new("fail");
    let embedded = fixture.embed("first", true);
    fixture.fail_at(&embedded);
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert_eq!(output.stdout, fixture.template_stdout());
    assert_eq!(fixture.invocations(), [fixture.template.clone(), embedded]);
    assert!(String::from_utf8_lossy(&output.stderr).contains("exit=Some(23)"));
}

#[test]
fn emits_only_template_success_when_framework_option_is_absent() {
    let fixture = Fixture::new("fail");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(fixture.root.path())
        .env_clear()
        .args(["release", "swift-privacy", "--template"])
        .arg(&fixture.template)
        .arg("--plutil")
        .arg(fixture.tool())
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, fixture.template_stdout());
    assert_eq!(
        fixture.invocations(),
        std::slice::from_ref(&fixture.template)
    );
}

#[test]
fn emits_no_success_when_template_native_lint_fails() {
    let fixture = Fixture::new("fail");
    fixture.fail_at(&fixture.template);
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(
        fixture.invocations(),
        std::slice::from_ref(&fixture.template)
    );
}

#[test]
fn rejects_policy_before_tool_when_template_and_tool_are_invalid() {
    let fixture = Fixture::new("clean");
    std::fs::write(&fixture.template, b"invalid plist").unwrap();
    let output = fixture
        .command_with_tool(std::path::Path::new("/missing-tool"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid privacy plist"));
    assert!(fixture.invocations().is_empty());
}

#[test]
fn forwards_raw_diagnostics_when_native_stderr_contains_credentials() {
    let fixture = Fixture::new("sensitive");
    fixture.embed("first", true);
    let output = fixture.command().output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        output.stderr,
        b"password=synthetic-never-print\nsafe diagnostic\npassword=synthetic-never-print\nsafe diagnostic\n"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("native stdout"));
}

#[test]
fn forwards_exact_raw_stderr_before_reporting_native_failure() {
    let fixture = Fixture::new("sensitive-fail");
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        output
            .stderr
            .starts_with(b"password=synthetic-raw\0\xff\r\nunterminatederror: ")
    );
    assert_eq!(
        fixture.invocations(),
        std::slice::from_ref(&fixture.template)
    );
}

#[test]
fn launches_no_child_when_template_policy_is_invalid() {
    let fixture = Fixture::new("clean");
    std::fs::write(&fixture.template, b"invalid plist").unwrap();
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(fixture.invocations().is_empty());
}

#[test]
fn preserves_first_lint_failure_before_later_byte_difference() {
    let fixture = Fixture::new("fail");
    fixture.embed("first", true);
    fixture.embed("second", true);
    let paths = fixture.discovery_order();
    fixture.fail_at(&paths[0]);
    std::fs::write(&paths[1], b"unequal").unwrap();
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert_eq!(output.stdout, fixture.template_stdout());
    assert_eq!(
        fixture.invocations(),
        [fixture.template.clone(), paths[0].clone()]
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("differs from template"));
}

#[test]
fn accepts_embedded_file_symlink_without_following_directory_symlinks() {
    let fixture = Fixture::new("fail");
    let embedded = fixture.framework.join("PrivacyInfo.xcprivacy");
    std::os::unix::fs::symlink(&fixture.template, &embedded).unwrap();
    let outside = fixture.root.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("PrivacyInfo.xcprivacy"), b"unequal").unwrap();
    std::os::unix::fs::symlink(&outside, fixture.framework.join("Resources")).unwrap();
    let output = fixture.command().output().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, fixture.success_stdout(1));
    assert_eq!(fixture.invocations(), [fixture.template.clone(), embedded]);
}
