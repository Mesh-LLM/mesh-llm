use super::scan;
use super::shards::check_other_shard;
use crate::command::DynResult;
use std::fs;

#[path = "manual_tsv_tests.rs"]
mod manual_tsv_tests;

pub(super) fn fixture() -> DynResult<(std::path::PathBuf, Vec<scan::Candidate>, serde_json::Value)>
{
    let root = crate::command::unique_temp_dir("migration-other-shard");
    fs::create_dir_all(root.join("evals"))?;
    let source = "python3 evals/run.py\n";
    fs::write(root.join("evals/run.sh"), source)?;
    fs::write(
        root.join("evals/README.md"),
        "```bash\npython3 evals/run.py\n```\n",
    )?;
    let observed = scan::scan_source("evals/run.sh", source);
    let hash = observed[0].id.split(':').nth(1).ok_or("missing hash")?;
    let ledger = serde_json::json!({
        "schema_version": 1,
        "groups": [{"file": "evals/run.sh", "members": [
            [1, hash, 1, "execution", "evals/run.py", "optional invocation"]
        ]}],
        "python_implementation_edges": [],
        "outside_scanner_source_calls": [{
            "file": "evals/README.md", "line": 2,
            "source_block": "python3 evals/run.py", "target": "evals/run.py",
            "boundary": "optional command", "kind": "instruction"
        }]
    });
    Ok((root, observed, ledger))
}

#[test]
fn other_shard_accepts_exact_source_and_instruction() -> DynResult<()> {
    let (root, observed, ledger) = fixture()?;
    let result = check_other_shard(&root, &ledger.to_string(), &observed);
    fs::remove_dir_all(root)?;
    result
}

#[test]
fn other_shard_rejects_new_scanner_candidate() -> DynResult<()> {
    let (root, mut observed, ledger) = fixture()?;
    observed.extend(scan::scan_source("evals/new.sh", "python3 -c 'print(1)'"));
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_changed_source_line() -> DynResult<()> {
    let (root, observed, ledger) = fixture()?;
    fs::write(root.join("evals/run.sh"), "python3 evals/changed.py\n")?;
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("stale"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_omitted_row_even_with_acceptance_flag() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture()?;
    ledger["acceptance"] = true.into();
    ledger["groups"][0]["members"] = serde_json::json!([]);
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_unresolved_execution_target() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture()?;
    ledger["groups"][0]["members"][0][4] = "unresolved: unknown".into();
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("target"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_new_manual_instruction() -> DynResult<()> {
    let (root, observed, ledger) = fixture()?;
    fs::write(
        root.join("evals/README.md"),
        "```bash\npython3 evals/run.py\npython3 evals/new.py\n```\n",
    )?;
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("instruction"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_omitted_manual_instruction() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture()?;
    ledger["outside_scanner_source_calls"] = serde_json::json!([]);
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("instruction"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_changed_manual_instruction() -> DynResult<()> {
    let (root, observed, ledger) = fixture()?;
    fs::write(
        root.join("evals/README.md"),
        "```bash\npython3 evals/changed.py\n```\n",
    )?;
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("stale instruction"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_omitted_python_implementation_row() -> DynResult<()> {
    let (root, mut observed, ledger) = fixture()?;
    fs::write(root.join("evals/loader.py"), "subprocess.run(command)\n")?;
    observed.extend(scan::scan_source(
        "evals/loader.py",
        "subprocess.run(command)",
    ));
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing source"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_unknown_conditional_target_without_boundary() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture()?;
    ledger["groups"][0]["members"][0][3] = "conditional".into();
    ledger["groups"][0]["members"][0][4] = "unknown".into();
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("boundary"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_rejects_executable_as_data() -> DynResult<()> {
    let (root, observed, mut ledger) = fixture()?;
    ledger["groups"][0]["members"][0][3] = "data".into();
    let error = check_other_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("disposition"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn other_shard_checks_live_source_without_claiming_completion() -> DynResult<()> {
    let root = crate::repo_consistency::repo_root()?;
    let text = fs::read_to_string(root.join("ci/automation-migration/other-edges.json"))?;
    let paths = super::ledger::tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?;
    let other = observed
        .iter()
        .filter(|row| {
            !row.path.starts_with(".github/")
                && row.path != "Justfile"
                && !row.path.starts_with("just/")
                && !row.path.starts_with("scripts/")
                && row.path != "tools/skippy-stage-rewriter/CMakeLists.txt"
        })
        .count();
    assert_eq!(other, 159);
    let ledger: serde_json::Value = serde_json::from_str(&text)?;
    assert_eq!(
        ledger["outside_scanner_source_calls"]
            .as_array()
            .map(Vec::len),
        Some(11)
    );
    // Given the checked-in shard and actual sources.
    // When reconciling the fourth shard, then every runnable TSV row is owned.
    check_other_shard(&root, &text, &observed)
}
