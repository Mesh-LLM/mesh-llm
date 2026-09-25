use super::scan;
use super::shards::{check_script_source_shard, check_shard};
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::fs;

fn github_source() -> Vec<scan::Candidate> {
    scan::scan_source(
        ".github/workflows/check.yml",
        "run: |\n  python3 -c 'print(1)'\n  command -v python3\n",
    )
}

fn github_ledger(observed: &[scan::Candidate]) -> serde_json::Value {
    serde_json::json!({"schema_version": 1, "groups": [{"file": ".github/workflows/check.yml", "members": [
        [2, observed[0].id.split(':').nth(1).unwrap(), 1, "inline", "call"],
        [3, observed[1].id.split(':').nth(1).unwrap(), 1, "probe", "availability"]
    ]}]})
}

#[test]
fn shard_rejects_missing_executable_even_if_self_asserted_complete() -> DynResult<()> {
    // Given a shard asserting completion but omitting its executable row.
    let observed = github_source();
    let mut ledger = github_ledger(&observed);
    ledger["complete"] = true.into();
    ledger["acceptance"] = true.into();
    ledger["groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    // When the source and ledger are reconciled, then execution is unowned.
    let error = check_shard("github-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    Ok(())
}

#[test]
fn shard_rejects_missing_nonexecution() -> DynResult<()> {
    // Given an omitted interpreter availability probe.
    let observed = github_source();
    let mut ledger = github_ledger(&observed);
    ledger["groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .remove(1);
    // When checked, then the distinct nonexecution row is missing.
    let error = check_shard("github-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    Ok(())
}

#[test]
fn shard_rejects_stale_source_identity() -> DynResult<()> {
    // Given a hash from an old version of the executable source line.
    let observed = github_source();
    let mut ledger = github_ledger(&observed);
    ledger["groups"][0]["members"][0][1] = "0000000000000000".into();
    // When checked, then the stale identity cannot approve the new source.
    let error = check_shard("github-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("stale"), "{error}");
    Ok(())
}

#[test]
fn shard_rejects_fabricated_exception() -> DynResult<()> {
    // Given a ledger row claiming a new Python call is an SDK exception.
    let observed = github_source();
    let mut ledger = github_ledger(&observed);
    ledger["groups"][0]["members"][0][3] = "exception".into();
    // When checked, then the fabricated disposition cannot authorize it.
    let error = check_shard("github-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("exception"), "{error}");
    Ok(())
}

#[test]
fn shard_rejects_duplicate_member() -> DynResult<()> {
    // Given two ledger entries for the same source candidate.
    let observed = github_source();
    let mut ledger = github_ledger(&observed);
    let extra = ledger["groups"][0]["members"][0].clone();
    ledger["groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .push(extra);
    // When checked, then the duplicate does not hide an extra row.
    let error = check_shard("github-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("duplicate"), "{error}");
    Ok(())
}

#[test]
fn shard_rejects_misclassified_nonexecution() -> DynResult<()> {
    // Given a probe row falsely labelled as a Python execution.
    let observed = github_source();
    let mut ledger = github_ledger(&observed);
    ledger["groups"][0]["members"][1][3] = "inline".into();
    // When checked, then the scanner and classification must agree.
    let error = check_shard("github-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("disposition"), "{error}");
    Ok(())
}

#[test]
fn script_shard_rejects_unowned_versioned_interpreter() -> DynResult<()> {
    // Given a partial script shard with one new versioned interpreter call.
    let observed = scan::scan_source("scripts/check.sh", "python3.12 -c 'print(1)'");
    let ledger = serde_json::json!({"schema_version": 1, "groups": []});
    // When checked, then a partial ledger does not approve that executable.
    let error = check_shard("script-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    Ok(())
}

#[test]
fn script_shard_rejects_missing_nonexecution_selection() -> DynResult<()> {
    // Given a script candidate that selects, but does not launch, Python.
    let observed = scan::scan_source("scripts/check.sh", "python_bin=python3");
    assert_eq!(observed.len(), 1);
    assert!(!observed[0].executable);
    let ledger = serde_json::json!({"schema_version": 1, "groups": []});
    // When its source-bound shard omits the selection.
    let error = check_shard("script-edges.json", &ledger.to_string(), &observed).unwrap_err();
    // Then a nonexecution row cannot silently disappear from the census.
    assert!(error.to_string().contains("missing"), "{error}");
    Ok(())
}

#[test]
fn test_shard_rejects_unowned_dynamic_import() -> DynResult<()> {
    // Given a test source using dynamic import without a reviewed row.
    let observed = scan::scan_source(
        "scripts/tests/test_loader.py",
        "importlib.import_module(name)",
    );
    let ledger = serde_json::json!({"schema_version": 1, "groups": []});
    // When checked, then an incomplete test shard remains red.
    let error = check_shard("test-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    Ok(())
}

fn python_source() -> Vec<scan::Candidate> {
    scan::scan_source(
        "scripts/loader.py",
        "importlib.import_module(module)\nsubprocess.run(command)\n",
    )
}

fn python_ledger(observed: &[scan::Candidate]) -> serde_json::Value {
    serde_json::json!({"schema_version": 1, "groups": [],
        "python_implementation_groups": [{"file": "scripts/loader.py", "root": "selected caller",
            "boundary": "target depends on caller input", "members": observed.iter().enumerate().map(|(i, _)| {
                serde_json::json!([i + 1, observed[i].id.split(':').nth(1).unwrap(), 1,
                    if i == 0 { "dynamic-import" } else { "subprocess-or-interpreter" },
                    "execution", "conditional: caller-provided module or command", "caller runs this source"])
            }).collect::<Vec<_>>() }],
        "python_implementation_edges": []})
}

#[test]
fn script_shard_rejects_omitted_python_source_member() -> DynResult<()> {
    // Given a source-bound ledger that omits one observed Python subprocess.
    let observed = python_source();
    let mut ledger = python_ledger(&observed);
    ledger["python_implementation_groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .remove(1);
    // When reconciled, then the missing source member is rejected.
    let error = check_shard("script-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("missing source members"),
        "{error}"
    );
    Ok(())
}

#[test]
fn script_shard_rejects_changed_python_member_identity() -> DynResult<()> {
    // Given a ledger whose Python member hash no longer matches the source.
    let observed = python_source();
    let mut ledger = python_ledger(&observed);
    ledger["python_implementation_groups"][0]["members"][0][1] = "0000000000000000".into();
    // When reconciled, then the changed source identity is rejected as stale.
    let error = check_shard("script-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("stale source identity"),
        "{error}"
    );
    Ok(())
}

#[test]
fn script_shard_rejects_duplicate_python_member() -> DynResult<()> {
    // Given a duplicated source-bound Python row.
    let observed = python_source();
    let mut ledger = python_ledger(&observed);
    let duplicate = ledger["python_implementation_groups"][0]["members"][0].clone();
    ledger["python_implementation_groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    // When reconciled, then one source identity cannot count twice.
    let error = check_shard("script-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("duplicate Python member"),
        "{error}"
    );
    Ok(())
}

#[test]
fn script_shard_rejects_unresolved_python_execution() -> DynResult<()> {
    // Given an execution member with no identified target.
    let observed = python_source();
    let mut ledger = python_ledger(&observed);
    ledger["python_implementation_groups"][0]["members"][1][5] = "unresolved: command".into();
    // When reconciled, then self-declared ownership does not resolve a launch.
    let error = check_shard("script-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("unresolved Python target"),
        "{error}"
    );
    Ok(())
}

#[test]
fn script_shard_rejects_unowned_python_candidate_even_if_nonexecution() -> DynResult<()> {
    // Given a source scan containing an unowned candidate classified as nonexecution.
    let mut observed = python_source();
    observed.push(scan::scan_source("scripts/extra.py", "sys.executable")[0].clone());
    observed.last_mut().unwrap().executable = false;
    let ledger = python_ledger(&observed[..2]);
    // When reconciled, then even a selection needs source ownership.
    let error = check_shard("script-edges.json", &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("missing source members"),
        "{error}"
    );
    Ok(())
}

#[test]
fn script_shard_checks_python_roster_against_full_source() -> DynResult<()> {
    // Given a source-bound Python group and an exact digest of its file.
    let root = crate::command::unique_temp_dir("migration-script-python-shard");
    fs::create_dir_all(root.join("scripts"))?;
    let source = "importlib.import_module(module)\nsubprocess.run(command)\n";
    fs::write(root.join("scripts/loader.py"), source)?;
    let observed = python_source();
    let mut ledger = python_ledger(&observed);
    ledger["python_implementation_sources"] = serde_json::json!([[
        "scripts/loader.py",
        hex::encode(Sha256::digest(source.as_bytes())),
        3
    ]]);
    // When source is unchanged, then all recorded members and the roster pass.
    check_script_source_shard(&root, &ledger.to_string(), &observed)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn script_shard_rejects_changed_python_file_outside_member_line() -> DynResult<()> {
    // Given a full-file roster for a Python module with two source members.
    let root = crate::command::unique_temp_dir("migration-script-python-shard");
    fs::create_dir_all(root.join("scripts"))?;
    let source = "importlib.import_module(module)\nsubprocess.run(command)\n";
    fs::write(root.join("scripts/loader.py"), source)?;
    let observed = python_source();
    let mut ledger = python_ledger(&observed);
    ledger["python_implementation_sources"] = serde_json::json!([[
        "scripts/loader.py",
        hex::encode(Sha256::digest(source.as_bytes())),
        3
    ]]);
    fs::write(
        root.join("scripts/loader.py"),
        format!("{source}# changed\n"),
    )?;
    // When an unrelated line changes, then the exact source digest rejects it.
    let error = check_script_source_shard(&root, &ledger.to_string(), &observed).unwrap_err();
    assert!(
        error.to_string().contains("changed Python source"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
