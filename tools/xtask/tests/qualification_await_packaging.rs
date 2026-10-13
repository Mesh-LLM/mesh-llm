//! `release await-packaging` contract tests with a fake `gh` executable,
//! porting `test_await_packaging_release.py`.
#![cfg(feature = "automation-contracts")]

#[path = "qualification_await_packaging/support.rs"]
mod support;

use support::{
    CORRELATION, artifacts_page, await_packaging, readiness_receipt, runs_page, single_file_zip,
    stderr,
};

#[cfg(unix)]
fn happy() -> (tempfile::TempDir, std::path::PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let zip = single_file_zip(
        "packaging-readiness.json",
        serde_json::to_string(&readiness_receipt())
            .unwrap()
            .as_bytes(),
    );
    let gh = support::write_fake_gh(
        temp.path(),
        &runs_page(&[(
            &format!("Packaging · {CORRELATION}"),
            42,
            1,
            "completed",
            "success",
        )]),
        &artifacts_page(),
        &zip,
    );
    (temp, gh)
}

#[cfg(unix)]
#[test]
fn accepts_correlated_terminal_success() {
    let (temp, gh) = happy();
    let result = await_packaging(temp.path(), Some(&gh), &[]);
    assert!(result.status.success(), "{}", stderr(&result));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "Verified packaging readiness from run 42/1\n"
    );
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("readiness.json")).unwrap())
            .unwrap();
    assert_eq!(written["correlation_id"], CORRELATION);
}

#[cfg(unix)]
#[test]
fn rejects_foreign_manifest_and_wrong_attempt() {
    let (temp, gh) = happy();
    let mut receipt = readiness_receipt();
    receipt["upstream"]["manifest_sha256"] = serde_json::json!("b".repeat(64));
    let zip = single_file_zip(
        "packaging-readiness.json",
        serde_json::to_string(&receipt).unwrap().as_bytes(),
    );
    std::fs::write(temp.path().join("payload.zip"), zip).unwrap();
    let result = await_packaging(temp.path(), Some(&gh), &[]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("upstream release identity mismatch"),
        "{}",
        stderr(&result)
    );

    let (temp, gh) = happy();
    let mut receipt = readiness_receipt();
    receipt["packaging_run_attempt"] = serde_json::json!(2);
    let zip = single_file_zip(
        "packaging-readiness.json",
        serde_json::to_string(&receipt).unwrap().as_bytes(),
    );
    std::fs::write(temp.path().join("payload.zip"), zip).unwrap();
    let result = await_packaging(temp.path(), Some(&gh), &[]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("run identity mismatch"),
        "{}",
        stderr(&result)
    );
}

#[cfg(unix)]
#[test]
fn rejects_missing_publication_channel() {
    let (temp, gh) = happy();
    let mut receipt = readiness_receipt();
    receipt["requested"]["publish_npm"] = serde_json::json!("false");
    let zip = single_file_zip(
        "packaging-readiness.json",
        serde_json::to_string(&receipt).unwrap().as_bytes(),
    );
    std::fs::write(temp.path().join("payload.zip"), zip).unwrap();
    let result = await_packaging(temp.path(), Some(&gh), &[]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("required release channels"),
        "{}",
        stderr(&result)
    );
}

#[cfg(unix)]
#[test]
fn rejects_duplicate_correlated_runs() {
    let temp = tempfile::tempdir().unwrap();
    let title = format!("Packaging · {CORRELATION}");
    let gh = support::write_fake_gh(
        temp.path(),
        &runs_page(&[
            (&title, 42, 1, "completed", "success"),
            (&title, 43, 1, "completed", "success"),
        ]),
        &artifacts_page(),
        &single_file_zip("packaging-readiness.json", b"{}"),
    );
    let result = await_packaging(temp.path(), Some(&gh), &[]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("more than one packaging run"),
        "{}",
        stderr(&result)
    );
}

#[cfg(unix)]
#[test]
fn reads_only_the_exact_readiness_archive() {
    let (temp, gh) = happy();
    let zip = single_file_zip("other.json", b"{}");
    std::fs::write(temp.path().join("payload.zip"), zip).unwrap();
    let result = await_packaging(temp.path(), Some(&gh), &[]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("unexpected files"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn rejects_foreign_repositories_and_malformed_identities() {
    let temp = tempfile::tempdir().unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "release",
            "await-packaging",
            "--repository",
            "other/repo",
            "--correlation-id",
            CORRELATION,
            "--upstream-repository",
            "Mesh-LLM/mesh-llm",
            "--upstream-ref",
            "v1.2.3",
            "--upstream-sha",
            support::UPSTREAM_SHA,
            "--manifest-sha256",
            support::MANIFEST_SHA,
            "--out",
            temp.path().join("out.json").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(
        stderr(&result).contains("unexpected release repository"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn times_out_when_no_correlated_run_appears() {
    let temp = tempfile::tempdir().unwrap();
    // The zero timeout ends the poll loop before any gh call, so no fake gh
    // executable is needed.
    let result = await_packaging(temp.path(), None, &["--timeout-seconds", "0"]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("did not reach terminal readiness"),
        "{}",
        stderr(&result)
    );
}

#[test]
fn rejects_invalid_correlation_id_and_digests() {
    let temp = tempfile::tempdir().unwrap();
    let bad = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "release",
            "await-packaging",
            "--repository",
            "Mesh-LLM/mesh-packaging",
            "--correlation-id",
            "bad id!",
            "--upstream-repository",
            "Mesh-LLM/mesh-llm",
            "--upstream-ref",
            "v1.2.3",
            "--upstream-sha",
            support::UPSTREAM_SHA,
            "--manifest-sha256",
            support::MANIFEST_SHA,
            "--out",
            temp.path().join("out.json").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
    assert!(
        stderr(&bad).contains("invalid correlation ID"),
        "{}",
        stderr(&bad)
    );
}
