//! Protected catalog, every profile and every frozen case: Rust against the
//! frozen goldens always, and against the legacy planner when opted in.

use crate::support::{INTERPRETER_ENV, Scratch, TestResult, parity, show, source_commit, text};
use serde_json::Value;
use std::fs;

fn count(summary: &Value, group: &str, outcome: &str) -> u64 {
    summary["counts"][group][outcome].as_u64().unwrap_or(0)
}

#[test]
fn migration_ci_shadow_rust_matches_every_golden_profile_and_projection() -> TestResult {
    // Given: the frozen fixtures, the protected catalogs committed as an
    // immutable source revision, and the legacy side only when opted in.
    let scratch = Scratch::new("happy")?;
    let sha = source_commit(&scratch.path().join("source"), |_| Ok(()))?;
    let source = scratch.path().join("source");
    let evidence = scratch.path().join("evidence");
    let source_arg = source.to_str().ok_or("non-UTF8 scratch path")?;
    let evidence_arg = evidence.to_str().ok_or("non-UTF8 scratch path")?;
    let legacy = std::env::var_os(INTERPRETER_ENV).is_some();
    let mut args = vec![
        "--suite",
        "ci",
        "--evidence",
        evidence_arg,
        "--source-repo",
        source_arg,
        "--source-sha",
        &sha,
    ];
    if !legacy {
        args.push("--rust-only");
    }
    // When: the parity suite runs.
    let output = parity(&args, legacy)?;
    show("happy", &output);
    // Then: zero unexplained differences over the full coverage set.
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let summary: Value = serde_json::from_slice(&fs::read(evidence.join("summary.json"))?)?;
    assert_eq!(summary["unexplained"], 0);
    assert_eq!(summary["legacy"]["enabled"], legacy);
    for group in [
        "frozen-cases",
        "action-outputs",
        "manifest-root",
        "protected-profiles",
        "runner-image-identity",
        "action-source",
    ] {
        assert_eq!(count(&summary, group, "different"), 0, "{group}");
    }
    let frozen = count(&summary, "frozen-cases", "identical");
    assert_eq!(frozen, if legacy { 126 } else { 63 }, "frozen comparisons");
    let outputs = count(&summary, "action-outputs", "identical");
    assert!(
        outputs >= if legacy { 38 + 38 + 8 } else { 38 },
        "{outputs}"
    );
    assert!(count(&summary, "manifest-root", "identical") >= 5);
    let profiles = if legacy { "identical" } else { "rust_only" };
    assert!(count(&summary, "protected-profiles", profiles) >= 8);
    assert!(count(&summary, "runner-image-identity", "identical") >= 1);
    assert_eq!(count(&summary, "action-source", "identical"), 1);
    assert_eq!(
        summary["manifest_root_entries"],
        serde_json::json!(["ci/ownership.yml", "ci/slices.yml"])
    );
    Ok(())
}

#[test]
fn migration_ci_shadow_requires_an_interpreter_unless_rust_only() -> TestResult {
    // Given: no interpreter flag and no opt-in environment.
    // When: a legacy comparison is requested.
    let output = parity(&["--suite", "ci"], false)?;
    show("missing interpreter", &output);
    // Then: it fails with a message naming every way to proceed.
    assert_ne!(output.status.code(), Some(0));
    let stderr = text(&output.stderr);
    assert!(stderr.contains(INTERPRETER_ENV), "{stderr}");
    assert!(stderr.contains("--interpreter"), "{stderr}");
    assert!(stderr.contains("--rust-only"), "{stderr}");
    Ok(())
}

#[test]
fn migration_ci_shadow_rejects_an_unknown_suite() -> TestResult {
    // Given/When: a suite other than `ci`.
    let output = parity(&["--suite", "release", "--rust-only"], false)?;
    // Then: usage fails without running anything.
    assert_ne!(output.status.code(), Some(0));
    assert!(text(&output.stderr).contains("--suite ci"));
    Ok(())
}
