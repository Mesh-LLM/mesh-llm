use std::process::{Command, Output};

fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["repository", "cargo-packages"])
        .args(args)
        .output()
        .expect("candidate CLI")
}

#[test]
fn cli_help_succeeds_when_metadata_is_not_requested() {
    let output = invoke(&["--help"]);
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .expect("help")
            .contains("--generation")
    );
}

#[test]
fn cli_rejects_input_before_metadata_when_batch_is_out_of_plan() {
    let directory = tempfile::tempdir().expect("isolated cwd");
    let missing = directory.path().join("unavailable-cargo");
    let output = invoke(&[
        "--generation",
        "current",
        "--crates",
        "[\"b\"]",
        "--batches",
        "[{\"crates\":[\"a\"]}]",
        "--metadata",
        missing.to_str().expect("executable"),
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .expect("diagnostic")
            .contains("outside the protected plan")
    );
}

#[test]
fn cli_fails_closed_when_explicit_metadata_file_is_missing() {
    let directory = tempfile::tempdir().expect("isolated cwd");
    let missing = directory.path().join("unavailable-cargo");
    let output = invoke(&[
        "--generation",
        "current",
        "--crates",
        "[\"a\"]",
        "--metadata",
        missing.to_str().expect("executable"),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .expect("diagnostic")
            .starts_with("CI package resolution failed:")
    );
}
