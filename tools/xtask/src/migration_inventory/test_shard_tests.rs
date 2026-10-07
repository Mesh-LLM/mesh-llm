use super::scan;
use super::shards::check_test_source_shard;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

fn fixture() -> DynResult<(PathBuf, Vec<scan::Candidate>, serde_json::Value)> {
    let root = crate::command::unique_temp_dir("migration-test-shard");
    let path = "scripts/tests/test_probe.py";
    fs::create_dir_all(root.join("scripts/tests"))?;
    let source = "subprocess.run(['python3', 'runner.py'])\n";
    fs::write(root.join(path), source)?;
    let observed = scan::scan_source(path, source);
    let mut digest = Sha256::new();
    digest.update(path.as_bytes());
    digest.update([0]);
    digest.update(source.as_bytes());
    digest.update([0]);
    let ledger = serde_json::json!({
        "schema_version": 1,
        "observed_source_sha256": hex::encode(digest.finalize()),
        "groups": [],
        "additional_groups": [[path, [[
            "launch", "runner.py", "test calls the fixed runner", [1]
        ]]]]
    });
    Ok((root, observed, ledger))
}

#[test]
fn test_shard_accepts_additional_source_member() -> DynResult<()> {
    // Given a candidate bound by an additional source group.
    let (root, observed, ledger) = fixture()?;
    // When checked, then its source identity is owned.
    let result = check_test_source_shard(&root, &ledger.to_string(), &observed);
    fs::remove_dir_all(root)?;
    result
}

#[test]
fn test_shard_rejects_omitted_additional_member() -> DynResult<()> {
    // Given a candidate omitted from its additional group.
    let (root, observed, mut ledger) = fixture()?;
    ledger["additional_groups"] = serde_json::json!([]);
    // When checked, then the source remains unowned.
    let error = check_test_source_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn test_shard_rejects_stale_additional_line() -> DynResult<()> {
    // Given an additional member pointing at a different physical line.
    let (root, observed, mut ledger) = fixture()?;
    ledger["additional_groups"][0][1][0][3] = serde_json::json!([2]);
    // When checked, then the mismatched location is stale.
    let error = check_test_source_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("stale"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn test_shard_rejects_duplicate_additional_identity() -> DynResult<()> {
    // Given two records for the same candidate source line.
    let (root, observed, mut ledger) = fixture()?;
    ledger["additional_groups"][0][1][0][3] = serde_json::json!([1, 1]);
    // When checked, then the repeated identity is rejected.
    let error = check_test_source_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("duplicate"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn test_shard_rejects_unresolved_additional_target() -> DynResult<()> {
    // Given an executable call whose target is not resolved.
    let (root, observed, mut ledger) = fixture()?;
    ledger["additional_groups"][0][1][0][1] = "unresolved: runner".into();
    // When checked, then a label cannot resolve the executable.
    let error = check_test_source_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("unresolved"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn test_shard_rejects_changed_target_outside_candidate_line() -> DynResult<()> {
    // Given a source file with a target declaration outside the scanned call.
    let (root, observed, ledger) = fixture()?;
    fs::write(
        root.join("scripts/tests/test_probe.py"),
        "RUNNER = 'other.py'\nsubprocess.run(['python3', 'runner.py'])\n",
    )?;
    // When checked, then a whole-file source mismatch invalidates the review.
    let error = check_test_source_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("changed test source tree"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn test_shard_rejects_duplicate_across_group_formats() -> DynResult<()> {
    // Given one identity repeated in the original and additional groups.
    let (root, observed, mut ledger) = fixture()?;
    ledger["groups"] = serde_json::json!([{
        "file": "scripts/tests/test_probe.py",
        "members": [[1, observed[0].id.split(':').nth(1).unwrap(), 1,
            "subprocess-or-interpreter", "launch", "runner.py", "fixed runner"]]
    }]);
    // When checked, then the identity cannot be owned twice.
    let error = check_test_source_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("duplicate"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn test_shard_checks_all_tracked_test_candidates() -> DynResult<()> {
    // Given the checked-in test shard and the tracked test sources.
    let root = crate::repo_consistency::repo_root()?;
    let paths = super::ledger::tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?
        .into_iter()
        .filter(|row| scan::is_script_test(&row.path))
        .collect::<Vec<_>>();
    let text = fs::read_to_string(root.join("ci/automation-migration/test-edges.json"))?;
    let ledger: super::shard_rows::TestShard = serde_json::from_str(&text)?;
    let additional = ledger
        .additional_groups
        .iter()
        .flat_map(|(_, cases)| cases.iter().map(|(_, _, _, lines)| lines.len()))
        .sum::<usize>();
    let grouped = ledger
        .groups
        .iter()
        .map(|group| group.members.len())
        .sum::<usize>();
    assert_eq!(observed.len(), grouped + additional);
    check_test_source_shard(&root, &text, &observed)
}

#[test]
fn test_shard_accepts_exact_product_test_roots_without_unclassified_exemptions() -> DynResult<()> {
    for prefix in ["mesh/", "skippy/"] {
        let (root, _, mut ledger) = fixture()?;
        let old = "scripts/tests/fixture.py";
        let path = format!("{prefix}{old}");
        let source = "subprocess.run(['python3', 'runner.py'])\n";
        fs::create_dir_all(root.join(&path).parent().unwrap())?;
        fs::write(root.join(&path), source)?;
        let observed = scan::scan_source(&path, source);
        let mut digest = Sha256::new();
        digest.update(path.as_bytes());
        digest.update([0]);
        digest.update(source.as_bytes());
        digest.update([0]);
        ledger["observed_source_sha256"] = serde_json::json!(hex::encode(digest.finalize()));
        ledger["additional_groups"][0][0] = serde_json::json!(path);
        check_test_source_shard(&root, &ledger.to_string(), &observed)?;
        ledger["additional_groups"][0][0] =
            serde_json::json!(format!("{prefix}scripts/tests/../fixture.py"));
        assert!(check_test_source_shard(&root, &ledger.to_string(), &observed).is_err());
        fs::remove_dir_all(root)?;
    }
    Ok(())
}
