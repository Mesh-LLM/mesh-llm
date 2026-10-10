#[path = "migration_canary_receipts_parity/fixture_files.rs"]
#[expect(
    dead_code,
    reason = "CLI fixtures reuse package helpers, not parity capture helpers"
)]
mod fixture_files;
#[path = "migration_canary_aggregate_cli/support.rs"]
mod support;
use std::fs;
use support::{Fixture, TestResult};

#[test]
fn green_when_package_and_receipts_match_explicit_context() -> TestResult {
    let given = Fixture::new()?;
    fs::write(given.path("summary"), b"prior-summary\n")?;
    fs::write(given.path("output"), b"prior-output\n")?;

    let when = given.run(&[])?;

    assert_eq!(when.status.code(), Some(0));
    assert!(when.stderr.is_empty());
    assert_eq!(
        when.stdout,
        format!(
            "{}All 2 families passed for aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (repair-1)\n",
            given.report()
        )
        .as_bytes()
    );
    assert_eq!(
        fs::read(given.path("summary"))?,
        format!("prior-summary\n{}", given.report()).as_bytes()
    );
    assert_eq!(fs::read(given.path("output"))?, b"prior-output\ngreen=true\ncandidate=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nbranch=llama-canary/repair-123-2-aaaaaaaaaa\n");
    Ok(())
}

#[test]
fn newest_failure_withholds_green_when_old_success_exists() -> TestResult {
    let given = Fixture::new()?;
    fixture_files::write_worker(
        &given.path("evidence"),
        "new-dense",
        "3",
        "failure",
        include_bytes!("migration_canary_receipts/fixtures/dense.jsonl"),
        &given.identity,
    )?;
    fs::write(given.path("output"), b"retained\n")?;

    let when = given.run(&[])?;

    assert_eq!(when.status.code(), Some(1));
    assert_eq!(when.stdout, fs::read(given.path("summary"))?);
    assert!(String::from_utf8_lossy(&when.stdout).contains("1/2 family receipts passed"));
    assert!(
        String::from_utf8_lossy(&when.stderr)
            .starts_with("family aggregation failed:\ndense: failed or mismatched worker receipt")
    );
    assert_eq!(fs::read(given.path("output"))?, b"retained\n");
    Ok(())
}

#[test]
fn explicit_null_is_rejected_when_identity_digest_matches() -> TestResult {
    let mut given = Fixture::new()?;
    let path = given.path("package/identity.json");
    let mut identity: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    identity["mesh_source"] = serde_json::Value::Null;
    let bytes = serde_json::to_vec(&identity)?;
    given.identity = fixture_files::hash_bytes(&bytes);
    fs::write(path, bytes)?;

    let when = given.run(&[])?;

    assert_eq!(when.status.code(), Some(1));
    assert!(when.stdout.is_empty());
    assert!(String::from_utf8_lossy(&when.stderr).contains("selected source identity mismatch"));
    assert!(!given.path("summary").exists());
    assert!(!given.path("output").exists());
    Ok(())
}

#[test]
fn wrong_context_is_rejected_before_summary_or_output() -> TestResult {
    for (flag, value, message) in [
        ("--run-id", "foreign", "foreign workflow run or attempt"),
        ("--run-attempt", "1", "foreign workflow run or attempt"),
        (
            "--controller-revision",
            "wrong",
            "controller revision mismatch",
        ),
        (
            "--selected-source",
            "wrong",
            "selected source identity mismatch",
        ),
    ] {
        let given = Fixture::new()?;

        let when = given.run(&[flag, value])?;

        assert_eq!(when.status.code(), Some(1));
        assert!(when.stdout.is_empty());
        assert!(String::from_utf8_lossy(&when.stderr).contains(message));
        assert!(!given.path("summary").exists());
        assert!(!given.path("output").exists());
    }
    Ok(())
}

#[test]
fn summary_failure_preserves_stdout_and_withholds_green() -> TestResult {
    let given = Fixture::new()?;
    fs::create_dir(given.path("summary"))?;

    let when = given.run(&[])?;

    assert_eq!(when.status.code(), Some(1));
    assert_eq!(when.stdout, given.report().as_bytes());
    assert!(!when.stderr.is_empty());
    assert!(!given.path("output").exists());
    Ok(())
}

#[test]
fn output_failure_retains_report_and_summary_without_success_line() -> TestResult {
    let given = Fixture::new()?;
    fs::create_dir(given.path("output"))?;

    let when = given.run(&[])?;

    assert_eq!(when.status.code(), Some(1));
    assert_eq!(when.stdout, given.report().as_bytes());
    assert_eq!(fs::read(given.path("summary"))?, given.report().as_bytes());
    assert!(!when.stderr.is_empty());
    Ok(())
}

#[test]
fn unknown_option_is_usage_error_before_package_io() -> TestResult {
    let given = Fixture::new()?;
    fs::write(given.path("package/identity.json"), b"malformed")?;

    let when = given.run(&["--unknown"])?;

    assert_eq!(when.status.code(), Some(2));
    assert!(when.stdout.is_empty());
    assert!(String::from_utf8_lossy(&when.stderr).contains("unrecognized arguments: --unknown"));
    Ok(())
}

#[test]
fn tampered_artifact_is_rejected_before_receipt_aggregation() -> TestResult {
    let given = Fixture::new()?;
    fs::write(given.path("package/binaries.tar"), b"tampered")?;

    let when = given.run(&[])?;

    assert_eq!(when.status.code(), Some(1));
    assert!(when.stdout.is_empty());
    assert!(String::from_utf8_lossy(&when.stderr).contains("binaries.tar digest mismatch"));
    assert!(!given.path("summary").exists());
    assert!(!given.path("output").exists());
    Ok(())
}

#[test]
fn selected_source_accepts_matching_certify_only_package() -> TestResult {
    let mut given = Fixture::new()?;
    let path = given.path("package/identity.json");
    let mut identity: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    identity["mesh_source"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
    let bytes = serde_json::to_vec(&identity)?;
    given.identity = fixture_files::hash_bytes(&bytes);
    fs::write(path, bytes)?;
    for family in ["dense", "hybrid"] {
        let path = given.path(&format!("evidence/{family}/receipt.json"));
        let mut receipt: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        receipt["identity_sha256"] = given.identity.clone().into();
        fs::write(path, serde_json::to_vec(&receipt)?)?;
    }

    let when = given.run(&[
        "--selected-source",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ])?;

    assert_eq!(when.status.code(), Some(0));
    assert!(when.stderr.is_empty());
    assert!(fs::read(given.path("output"))?.starts_with(b"green=true\n"));
    Ok(())
}

#[test]
fn help_does_not_read_package_or_emit_github_files() -> TestResult {
    let given = Fixture::new()?;
    fs::write(given.path("package/identity.json"), b"malformed")?;

    let when = given.run(&["--help"])?;

    assert_eq!(when.status.code(), Some(0));
    assert!(when.stderr.is_empty());
    assert!(String::from_utf8_lossy(&when.stdout).starts_with("usage:"));
    assert!(!given.path("summary").exists());
    assert!(!given.path("output").exists());
    Ok(())
}

#[test]
fn certify_only_package_mutations_refuse_before_summary_green_or_receipt_work() -> TestResult {
    for (key, value) in [
        ("controller", serde_json::json!("c".repeat(40))),
        ("candidate", serde_json::json!("c".repeat(40))),
        ("base", serde_json::json!("c".repeat(40))),
        ("mesh_source", serde_json::json!("")),
        ("pass_id", serde_json::json!("verify-1")),
        ("bundle_sha256", serde_json::json!("d".repeat(64))),
    ] {
        let mut given = Fixture::new()?;
        let path = given.path("package/identity.json");
        let mut identity: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        identity["mesh_source"] = serde_json::json!("a".repeat(40));
        identity[key] = value;
        let bytes = serde_json::to_vec(&identity)?;
        given.identity = fixture_files::hash_bytes(&bytes);
        fs::write(&path, &bytes)?;
        // Rebind expected identity to changed bytes: refusal must concern logical
        // selected-source admission, not merely a stale outer digest.
        let when = given.run(&["--selected-source", &"a".repeat(40)])?;
        assert!(!when.status.success(), "{key}: {when:?}");
        assert!(when.stdout.is_empty(), "{key}");
        assert!(!given.path("summary").exists(), "{key}");
        assert!(!given.path("output").exists(), "{key}");
        assert_eq!(fs::read(&path)?, bytes);
    }
    Ok(())
}
