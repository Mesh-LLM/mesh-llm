use super::ledger::MigrationLedgers;
use super::required_closure::check_required_closure;
use super::scan;
use super::selected_process::check_selected_processes;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;

#[test]
fn required_closure_rejects_unowned_selected_version_probe() -> DynResult<()> {
    // Given a required script that selects and executes Python on a continuation line.
    let root = crate::command::unique_temp_dir("closure-selected-probe");
    let script = root.join("scripts/package-native-runtime.sh");
    fs::create_dir_all(script.parent().ok_or("missing parent")?)?;
    fs::create_dir_all(root.join("just"))?;
    fs::write(
        root.join("just/ci.just"),
        "ci-validate:\n    scripts/package-native-runtime.sh\n",
    )?;
    fs::write(
        &script,
        "#!/usr/bin/env bash\npython_bin() {\n  for candidate in python3 python; do\n    if command -v \"$candidate\" >/dev/null 2>&1 &&\n      \"$candidate\" -c 'import sys; raise SystemExit(0)' >/dev/null 2>&1; then\n      return 0\n    fi\n  done\n}\npython_bin\n",
    )?;
    let paths = vec![
        "just/ci.just".to_owned(),
        "scripts/package-native-runtime.sh".to_owned(),
    ];
    let observed = scan::scan_paths(&root, &paths)?;
    let owned = observed
        .iter()
        .map(|row| row.id.clone())
        .collect::<BTreeSet<_>>();
    check_required_closure(&root, &paths, &observed, &owned, &["just/ci.just"])?;
    // When all token rows are owned, the separately omitted process must still fail.
    let error = check_selected_processes(&root, &[]).unwrap_err();
    assert!(
        error.to_string().contains("selected interpreter"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn selected_process_rejects_omitted_required_probe_and_planner() -> DynResult<()> {
    // Given source-reviewed process calls separate from the line-token shard.
    let root = crate::repo_consistency::repo_root()?;
    let ledgers = MigrationLedgers::load(&root)?;
    check_selected_processes(&root, &ledgers.invocations.selected_process_calls)?;
    for (path, line) in [
        ("scripts/package-native-runtime.sh", 192),
        ("scripts/skippy-family-battery.sh", 190),
    ] {
        let mut records = ledgers.invocations.selected_process_calls.clone();
        records.retain(|record| record.caller != path || record.line != line);
        // When one launch is omitted, then the gate rejects its physical source.
        let error = check_selected_processes(&root, &records).unwrap_err();
        assert!(
            error.to_string().contains(&format!("{path}:{line}")),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn selected_process_rejects_new_tokenless_call() -> DynResult<()> {
    // Given a new selected command after the source-reviewed family planner calls.
    let repo = crate::repo_consistency::repo_root()?;
    let root = crate::command::unique_temp_dir("selected-process-added");
    let path = "scripts/skippy-family-battery.sh";
    let source = fs::read_to_string(repo.join(path))?;
    let file = root.join(path);
    fs::create_dir_all(file.parent().ok_or("missing script directory")?)?;
    fs::write(
        &file,
        format!("{source}\n\"$PLANNER\" --verify-plan extra.json\n"),
    )?;
    fs::copy(
        repo.join("scripts/plan-family-battery.py"),
        root.join("scripts/plan-family-battery.py"),
    )?;
    let mut records = MigrationLedgers::load(&repo)?
        .invocations
        .selected_process_calls;
    records.retain(|record| record.caller == path);
    let error = check_selected_processes(&root, &records).unwrap_err();
    assert!(
        error.to_string().contains("unowned required execution"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn selected_process_rejects_changed_planner_target() -> DynResult<()> {
    // Given a family battery whose selected planner now points at a different script.
    let repo = crate::repo_consistency::repo_root()?;
    let root = crate::command::unique_temp_dir("selected-planner-target");
    let path = "scripts/skippy-family-battery.sh";
    let text = fs::read_to_string(repo.join(path))?;
    let file = root.join(path);
    fs::create_dir_all(file.parent().ok_or("missing script directory")?)?;
    fs::write(
        &file,
        text.replace(
            "PLANNER=\"$ROOT/scripts/plan-family-battery.py\"",
            "PLANNER=\"$ROOT/scripts/other.py\"",
        ),
    )?;
    fs::copy(
        repo.join("scripts/plan-family-battery.py"),
        root.join("scripts/plan-family-battery.py"),
    )?;
    let mut records = MigrationLedgers::load(&repo)?
        .invocations
        .selected_process_calls;
    records.retain(|record| record.caller == path);
    // When the selected target changes, then a matching launch line is insufficient.
    let error = check_selected_processes(&root, &records).unwrap_err();
    assert!(
        error.to_string().contains("changed family planner binding"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
