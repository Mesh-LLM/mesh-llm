use super::support::{Fixture, OwnedChild};
use std::{
    fs,
    process::{Command, Stdio},
};

#[test]
fn fails_without_success_output_when_template_native_lint_times_out() {
    let fixture = Fixture::new("timeout");
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Deadline"));
    assert_eq!(
        fixture.invocations(),
        std::slice::from_ref(&fixture.template)
    );
}

#[test]
fn fails_without_success_output_when_raw_capture_exceeds_cap() {
    let fixture = Fixture::new("overflow");
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("RawCaptureOverflow"));
}

#[test]
fn cancels_owned_descendant_without_stopping_unrelated_sentinel() {
    assert_tree_cleanup(true);
}

#[test]
fn times_out_owned_descendant_without_stopping_unrelated_sentinel() {
    assert_tree_cleanup(false);
}

#[test]
fn reports_unavailable_raw_stderr_without_substituting_partial_diagnostics() {
    let fixture = Fixture::new("stderr-overflow");
    let output = fixture.command().output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RawCaptureOverflow"));
    assert!(stderr.contains("complete_raw_stderr=false"));
    assert!(!stderr.contains("partial diagnostic must not be substituted"));
}

fn assert_tree_cleanup(cancel: bool) {
    let fixture = Fixture::new("descendant");
    let mut sentinel = OwnedChild(
        Command::new(fixture.tool())
            .arg("--sentinel")
            .current_dir(fixture.root.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    fixture.await_file("sentinel.pid", &mut sentinel.0);
    let mut command = fixture.command();
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut adapter = OwnedChild(command.spawn().unwrap());
    fixture.await_file("leaf.pid", &mut adapter.0);
    let leaf = fs::read_to_string(fixture.root.path().join("leaf.pid")).unwrap();
    if cancel {
        let sent = Command::new("/bin/kill")
            .args(["-TERM", &adapter.0.id().to_string()])
            .status()
            .unwrap();
        assert!(sent.success());
    }
    let status = adapter.0.wait().unwrap();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    std::io::Read::read_to_end(&mut adapter.0.stdout.take().unwrap(), &mut stdout).unwrap();
    std::io::Read::read_to_end(&mut adapter.0.stderr.take().unwrap(), &mut stderr).unwrap();
    assert!(!status.success());
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains(if cancel {
        "Cancelled"
    } else {
        "Deadline"
    }));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    let alive = Command::new("/bin/kill")
        .args(["-0", &leaf])
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!alive.success(), "owned descendant survived cancellation");
}
