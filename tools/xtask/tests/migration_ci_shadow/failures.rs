//! Failure contracts. None of these needs an interpreter: each is rejected
//! before any legacy process could start, or runs in `--rust-only` mode.

use crate::support::{
    Scratch, TestResult, copy_tree, fixture_root, parity, repository_root, show, source_commit,
    text,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

fn fails_with(args: &[&str], needle: &str, label: &str) -> TestResult {
    let output = parity(args, false)?;
    show(label, &output);
    assert_ne!(output.status.code(), Some(0), "{label}: must fail");
    let stderr = text(&output.stderr);
    assert!(stderr.contains(needle), "{label}: {stderr}");
    Ok(())
}

fn source_args<'a>(repo: &'a str, sha: &'a str) -> Vec<&'a str> {
    vec![
        "--suite",
        "ci",
        "--rust-only",
        "--source-repo",
        repo,
        "--source-sha",
        sha,
    ]
}

#[test]
fn migration_ci_shadow_rejects_a_pr_controlled_planner_path() -> TestResult {
    // Given: a source (PR-controlled) tree carrying its own planner copy.
    let scratch = Scratch::new("planner-path")?;
    let planner = scratch.path().join("scripts/plan-ci.py");
    fs::create_dir_all(planner.parent().ok_or("parent")?)?;
    fs::copy(repository_root().join("scripts/plan-ci.py"), &planner)?;
    let planner_arg = planner.to_str().ok_or("non-UTF8 path")?;
    // When: the comparison is pointed at that planner.
    // Then: only the protected checkout's planner is accepted; nothing runs.
    fails_with(
        &[
            "--suite",
            "ci",
            "--interpreter",
            "/usr/bin/false",
            "--planner",
            planner_arg,
        ],
        "protected checkout",
        "pr-controlled planner",
    )
}

#[test]
fn migration_ci_shadow_rejects_a_symlinked_catalog() -> TestResult {
    // Given: a source revision whose slice catalog is a symlink (mode 120000).
    let scratch = Scratch::new("symlink")?;
    let repo = scratch.path().join("source");
    let sha = source_commit(&repo, |repo: &Path| {
        fs::rename(repo.join("ci/slices.yml"), repo.join("ci/real-slices.yml"))?;
        std::os::unix::fs::symlink("real-slices.yml", repo.join("ci/slices.yml"))?;
        Ok(())
    })?;
    let repo_arg = repo.to_str().ok_or("non-UTF8 path")?;
    // When/Then: materializing the inert manifest root fails closed.
    fails_with(
        &source_args(repo_arg, &sha),
        "source manifest is missing or is not a regular file: ci/slices.yml",
        "symlink catalog",
    )
}

#[test]
fn migration_ci_shadow_rejects_an_executable_catalog() -> TestResult {
    // Given: a source revision whose ownership catalog has mode 100755.
    let scratch = Scratch::new("mode")?;
    let repo = scratch.path().join("source");
    let sha = source_commit(&repo, |repo: &Path| {
        use std::os::unix::fs::PermissionsExt;
        let path = repo.join("ci/ownership.yml");
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
        Ok(())
    })?;
    let repo_arg = repo.to_str().ok_or("non-UTF8 path")?;
    // When/Then: only mode 100644 blobs are accepted.
    fails_with(
        &source_args(repo_arg, &sha),
        "source manifest is missing or is not a regular file: ci/ownership.yml",
        "executable catalog",
    )
}

#[test]
fn migration_ci_shadow_detects_catalog_byte_drift() -> TestResult {
    // Given: a source revision whose slice catalog differs by one byte.
    let scratch = Scratch::new("drift")?;
    let repo = scratch.path().join("source");
    let sha = source_commit(&repo, |repo: &Path| {
        let path = repo.join("ci/slices.yml");
        let mut bytes = fs::read(&path)?;
        bytes.push(b'\n');
        fs::write(path, bytes)?;
        Ok(())
    })?;
    let repo_arg = repo.to_str().ok_or("non-UTF8 path")?;
    // When/Then: the protected-catalog equality check rejects it.
    fails_with(
        &source_args(repo_arg, &sha),
        "source slice catalog differs from the protected planner catalog",
        "catalog drift",
    )
}

#[test]
fn migration_ci_shadow_rejects_a_malformed_source_sha() -> TestResult {
    // Given: an uppercase and a short source revision.
    let repo = repository_root();
    let repo_arg = repo.to_str().ok_or("non-UTF8 path")?;
    for sha in ["A".repeat(40), "abc123".to_owned()] {
        // When/Then: only 40 lowercase hex characters are accepted.
        fails_with(
            &source_args(repo_arg, &sha),
            "pull request source SHA is malformed",
            "source sha",
        )?;
    }
    Ok(())
}

/// Runs the Rust-only suite against a mutated fixture copy and returns the
/// summary it wrote.
fn run_mutated(
    label: &str,
    mutate: impl FnOnce(&Path) -> TestResult,
) -> Result<Value, Box<dyn std::error::Error>> {
    let scratch = Scratch::new(label)?;
    let fixtures = scratch.path().join("fixtures");
    copy_tree(&fixture_root(), &fixtures)?;
    mutate(&fixtures)?;
    let evidence = scratch.path().join("evidence");
    let fixtures_arg = fixtures.to_str().ok_or("non-UTF8 path")?;
    let evidence_arg = evidence.to_str().ok_or("non-UTF8 path")?;
    let output = parity(
        &[
            "--suite",
            "ci",
            "--rust-only",
            "--fixtures",
            fixtures_arg,
            "--evidence",
            evidence_arg,
        ],
        false,
    )?;
    show(label, &output);
    assert_eq!(output.status.code(), Some(1), "{label}: a difference fails");
    Ok(serde_json::from_slice(&fs::read(
        evidence.join("summary.json"),
    )?)?)
}

fn different(summary: &Value, group: &str) -> Vec<Value> {
    summary["comparisons"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| row["group"] == group && row["outcome"] == "different")
        .cloned()
        .collect()
}

#[test]
fn migration_ci_shadow_reports_a_stale_lane_projection() -> TestResult {
    // Given: a recorded Linux lane projection that no longer matches.
    let summary = run_mutated("stale-projection", |fixtures| {
        let path = fixtures.join("expected/main.outputs.txt");
        let stale = fs::read_to_string(&path)?.replace(
            "linux_lane_plan={\"lane\":\"linux\"",
            "linux_lane_plan={\"lane\":\"linux-stale\"",
        );
        fs::write(path, stale)?;
        Ok(())
    })?;
    // When/Then: exactly that projection is reported, naming the output.
    let rows = different(&summary, "action-outputs");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["label"], "main");
    assert!(
        rows[0]["detail"]
            .as_str()
            .unwrap_or("")
            .contains("linux_lane_plan")
    );
    assert_eq!(summary["unexplained"], 1);
    Ok(())
}

#[test]
fn migration_ci_shadow_reports_a_single_byte_serialization_difference() -> TestResult {
    // Given: a frozen plan golden differing from the planner by one byte.
    let summary = run_mutated("single-byte", |fixtures| {
        let path = fixtures.join("expected/docs-only.plan.json");
        let golden = fs::read_to_string(&path)?.replacen("\"idx\":", "\"idx\" :", 1);
        let golden = if golden.contains("\"idx\" :") {
            golden
        } else {
            fs::read_to_string(&path)?.replacen("\"profile\":", "\"profile\" :", 1)
        };
        fs::write(path, golden)?;
        Ok(())
    })?;
    // When/Then: the frozen comparison fails and names the first byte.
    let rows = different(&summary, "frozen-cases");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["label"], "docs-only");
    assert!(
        rows[0]["detail"]
            .as_str()
            .unwrap_or("")
            .contains("first differing byte")
    );
    assert_eq!(summary["unexplained"], 1);
    Ok(())
}
