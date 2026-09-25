use crate::support::{Case, Legacy, Scratch, TestResult, assert_output};

const SCRIPT: Legacy = Legacy::Script("scripts/verify-static-abi-build-stamp.py");
const STAMP: &str = "stamp-version=3\npatched-sha=abc\nbackend=cpu\nlink-mode=static\n\
                     toolchain-epoch=e1\ncmake-arg=-DGGML_NATIVE=OFF\ncmake-arg=-DX=a=b\n";

fn verify(stamp: &str, extra: &[&str]) -> Case {
    let mut args = vec![
        stamp,
        "--backend",
        "cpu",
        "--link-mode",
        "static",
        "--stamp-version",
        "3",
        "--toolchain-epoch",
        "e1",
    ];
    args.extend_from_slice(extra);
    Case::same(&["static-abi-stamp"], &args, SCRIPT)
}

fn run_with(
    contents: &str,
    extra: &[&str],
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let scratch = Scratch::new("abi-stamp")?;
    let stamp = scratch.write("stamp", contents.as_bytes())?;
    verify(stamp.to_str().ok_or("path")?, extra).run(scratch.path())
}

#[test]
fn migration_prepared_inputs_abi_stamp_accepts_matching_stamp() -> TestResult {
    let ok = "verified static ABI build stamp: backend=cpu cmake_arguments=2\n";
    assert_output(&run_with(STAMP, &[])?, 0, ok, "");
    assert_output(&run_with(STAMP, &["--patched-sha", "abc"])?, 0, ok, "");
    let crlf = STAMP.replace('\n', "\r\n");
    assert_output(&run_with(&crlf, &[])?, 0, ok, "");
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_stamp_rejects_changed_identity() -> TestResult {
    let output = run_with(STAMP, &["--patched-sha", "def"])?;
    let expected = "static ABI build stamp patched-sha mismatch: expected 'def', got 'abc'\n";
    assert_output(&output, 1, "", expected);
    let epoch = STAMP.replace("toolchain-epoch=e1", "toolchain-epoch=e2");
    let expected = "static ABI build stamp toolchain-epoch mismatch: expected 'e1', got 'e2'\n";
    assert_output(&run_with(&epoch, &[])?, 1, "", expected);
    let backend = STAMP.replace("backend=cpu", "backend=cuda");
    let expected = "static ABI build stamp backend mismatch: expected 'cpu', got 'cuda'\n";
    assert_output(&run_with(&backend, &[])?, 1, "", expected);
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_stamp_rejects_malformed_stamps() -> TestResult {
    let cases = [
        (
            STAMP.replace("backend=cpu", "Backend=cpu"),
            "static ABI build stamp line 3 is malformed\n",
        ),
        (
            format!("{STAMP}\n"),
            "static ABI build stamp line 8 is malformed\n",
        ),
        (
            format!("{STAMP}backend=cpu\n"),
            "static ABI build stamp repeats singleton field 'backend'\n",
        ),
        (
            STAMP
                .replace("patched-sha=abc\n", "")
                .replace("link-mode=static", "link-mode="),
            "static ABI build stamp is missing required fields: patched-sha, link-mode\n",
        ),
        (
            STAMP.replace("cmake-arg=-DGGML_NATIVE=OFF\ncmake-arg=-DX=a=b\n", ""),
            "static ABI build stamp must contain at least one cmake-arg\n",
        ),
    ];
    for (contents, expected) in cases {
        assert_output(&run_with(&contents, &[])?, 1, "", expected);
    }
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_stamp_rejects_unreadable_stamp() -> TestResult {
    let scratch = Scratch::new("abi-stamp-missing")?;
    let missing = scratch.join("missing");
    let output = verify(missing.to_str().ok_or("path")?, &[]).run(scratch.path())?;
    let expected = format!(
        "unable to read static ABI build stamp: [Errno 2] No such file or directory: '{}'\n",
        missing.display()
    );
    assert_output(&output, 1, "", &expected);
    let scratch = Scratch::new("abi-stamp-binary")?;
    let stamp = scratch.write("stamp", b"backend=\xff\n")?;
    let output = verify(stamp.to_str().ok_or("path")?, &[])
        .status_only()
        .run(scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_stamp_usage_errors_exit_two() -> TestResult {
    let scratch = Scratch::new("abi-stamp-usage")?;
    let output = Case::same(
        &["static-abi-stamp"],
        &["stamp", "--backend", "cpu"],
        SCRIPT,
    )
    .status_only()
    .run(scratch.path())?;
    assert_eq!(output.status.code(), Some(2));
    Ok(())
}
