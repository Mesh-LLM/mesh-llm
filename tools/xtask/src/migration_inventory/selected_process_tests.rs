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
    for record in &ledgers.invocations.selected_process_calls {
        let path = record.caller.as_str();
        let line = record.line;
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
        format!(
            "{source}\nPLANNER=\"$ROOT/scripts/plan-family-battery.py\"\n\"$PLANNER\" --verify-plan extra.json\n"
        ),
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
        format!(
            "{text}\nPLANNER=\"$ROOT/scripts/other.py\"\n\"$PLANNER\" --verify-plan extra.json\n"
        ),
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

fn typed_records_fixture()
-> DynResult<(std::path::PathBuf, Vec<super::ledger::SelectedProcessCall>)> {
    let root = crate::command::unique_temp_dir("selected-typed-processes");
    let cases = [
        ("scripts/skippy-ci-smoke.sh", "automation"),
        ("scripts/skippy-workload-certify.sh", "workload_automation"),
        (
            "scripts/llama-canary-agent-repair.sh",
            "repair_workload_automation",
        ),
    ];
    fs::create_dir_all(root.join("scripts"))?;
    let mut records = Vec::new();
    for (caller, name) in cases {
        let expansion = format!("\"${{{name}[@]}}\"");
        let declaration = format!("command=({expansion} automation local-ports 2)");
        let direct = format!("{expansion} automation local-ports \\");
        let condition = format!("if {expansion} automation local-ports 1; then");
        let substitution = format!("PORT=\"$({expansion} automation local-ports 1)\"");
        let source = format!(
            "# {direct}\necho '{direct}'\n{declaration}\n{direct}\n  2\n{condition}\n true\nfi\n{substitution}\n"
        );
        fs::write(root.join(caller), source)?;
        for (line, source_block, argv) in [
            (4, direct.clone(), format!("{direct}\n2")),
            (6, condition.clone(), condition),
            (9, substitution.clone(), substitution),
        ] {
            records.push(super::ledger::SelectedProcessCall {
                caller: caller.to_owned(),
                line,
                source_block,
                argv,
                child: "tools/xtask".to_owned(),
                child_source_known: true,
                replacement_owner: "tools/xtask/src/automation/local_ports.rs".to_owned(),
                status_streams_effects: "Typed finite fixture output and failure contract"
                    .to_owned(),
            });
        }
    }
    Ok((root, records))
}

#[test]
fn typed_selected_calls_admit_direct_condition_substitution_and_continuation_only() -> DynResult<()>
{
    let (root, records) = typed_records_fixture()?;
    check_selected_processes(&root, &records)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn typed_selected_calls_reject_omission_duplicate_and_wrong_literal_owner() -> DynResult<()> {
    let (root, records) = typed_records_fixture()?;
    let mut omitted = records.clone();
    omitted.remove(0);
    assert!(
        check_selected_processes(&root, &omitted)
            .unwrap_err()
            .to_string()
            .contains("unowned required execution")
    );
    let mut duplicate = records.clone();
    duplicate.push(records[0].clone());
    assert!(
        check_selected_processes(&root, &duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    let mut wrong = records.clone();
    wrong[0].replacement_owner = "tools/xtask/src/automation/canary_timeout.rs".to_owned();
    assert!(
        check_selected_processes(&root, &wrong)
            .unwrap_err()
            .to_string()
            .contains("changed typed source or owner")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn typed_selected_calls_reject_changed_continued_flags_even_when_launch_line_matches()
-> DynResult<()> {
    let (root, records) = typed_records_fixture()?;
    let path = root.join("scripts/skippy-ci-smoke.sh");
    let source = fs::read_to_string(&path)?;
    fs::write(path, source.replace("\n  2\n", "\n  3\n"))?;
    assert!(
        check_selected_processes(&root, &records)
            .unwrap_err()
            .to_string()
            .contains("changed typed source or owner")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn typed_evidence_array_is_owned_at_launch_and_binds_its_declaration() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("selected-evidence-array");
    fs::create_dir_all(root.join("scripts"))?;
    let caller = "scripts/skippy-workload-certify.sh";
    let declaration = "evidence_command=(\"${workload_automation[@]}\" automation workload-oracle-evidence write\n--output \"$OUTPUT\")";
    let launch = "\"${evidence_command[@]}\"";
    fs::write(root.join(caller), format!("{declaration}\n{launch}\n"))?;
    let record = super::ledger::SelectedProcessCall {
        caller: caller.to_owned(),
        line: 3,
        source_block: launch.to_owned(),
        child: "tools/xtask".to_owned(),
        child_source_known: true,
        argv: declaration.to_owned(),
        replacement_owner: "tools/xtask/src/automation/workload_oracle_evidence/mod.rs".to_owned(),
        status_streams_effects: "Actual evidence command launch".to_owned(),
    };
    check_selected_processes(&root, std::slice::from_ref(&record))?;
    fs::write(
        root.join(caller),
        format!("{}\n{launch}\n", declaration.replace("$OUTPUT", "$OTHER")),
    )?;
    assert!(
        check_selected_processes(&root, &[record])
            .unwrap_err()
            .to_string()
            .contains("changed typed source or owner")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn current_battery_plan_record_preserves_original_scope_with_other_typed_calls_present()
-> DynResult<()> {
    let repo = crate::repo_consistency::repo_root()?;
    let root = crate::command::unique_temp_dir("selected-current-battery-scope");
    let caller = "scripts/skippy-family-battery.sh";
    fs::create_dir_all(root.join("scripts"))?;
    fs::copy(repo.join(caller), root.join(caller))?;
    let records = MigrationLedgers::load(&repo)?
        .invocations
        .selected_process_calls
        .into_iter()
        .filter(|record| record.caller == caller)
        .collect::<Vec<_>>();
    assert!(!records.is_empty());
    assert!(
        records
            .iter()
            .any(|record| record.source_block == "\"${plan_args[@]}\"")
    );
    check_selected_processes(&root, &records)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn system_one_actual_mixed_launch_requires_both_default_and_explicit_override_binding()
-> DynResult<()> {
    let root = crate::command::unique_temp_dir("selected-system-one-mixed");
    fs::create_dir_all(root.join("scripts"))?;
    let caller = "scripts/skippy-system-one-smoke.sh";
    let source = concat!(
        "local case_command=(\"${automation[@]}\" automation system-one-cases)\n",
        "if [[ \"${SYSTEMONE_SMOKE_DRIVER+set}\" == set ]]; then\n",
        "case_command=(python3 \"$CASES_DRIVER\")\n",
        "fi\n",
        "\"${case_command[@]}\" \\\n",
        "--mode \"$mode\" --json-out \"$REPORT\" || rc=$?\n"
    );
    fs::write(root.join(caller), source)?;
    let record = super::ledger::SelectedProcessCall {
        caller: caller.to_owned(),
        line: 5,
        source_block: source.lines().nth(4).unwrap().trim().to_owned(),
        child: "tools/xtask default; explicit Python $CASES_DRIVER override".to_owned(),
        child_source_known: false,
        argv: source.trim_end().to_owned(),
        replacement_owner: "tools/xtask/src/automation/system_one_cases/mod.rs; retained explicit SYSTEMONE_SMOKE_DRIVER Python override".to_owned(),
        status_streams_effects: "Typed default and explicitly retained Python override; same report/status consumer".to_owned(),
    };
    check_selected_processes(&root, std::slice::from_ref(&record))?;
    assert!(
        check_selected_processes(&root, &[])
            .unwrap_err()
            .to_string()
            .contains("unowned required execution")
    );
    let mut pure_typed = record.clone();
    pure_typed.replacement_owner = "tools/xtask/src/automation/system_one_cases/mod.rs".to_owned();
    assert!(
        check_selected_processes(&root, &[pure_typed])
            .unwrap_err()
            .to_string()
            .contains("mixed launch binding")
    );
    for (old, new) in [
        ("automation system-one-cases", "automation workload-smoke"),
        ("${SYSTEMONE_SMOKE_DRIVER+set}", "${OTHER_DRIVER+set}"),
        ("$CASES_DRIVER", "$OTHER_DRIVER"),
        ("--json-out \"$REPORT\"", "--json-out \"$OTHER_REPORT\""),
    ] {
        fs::write(root.join(caller), source.replace(old, new))?;
        assert!(
            check_selected_processes(&root, std::slice::from_ref(&record)).is_err(),
            "unbound mutation: {old}"
        );
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
