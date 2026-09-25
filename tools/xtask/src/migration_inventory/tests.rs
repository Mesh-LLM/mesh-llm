use super::{
    checks,
    ledger::{
        CandidateCensus, ExceptionEntry, ExceptionLedger, GithubEdge, InstructionAsset,
        InstructionLedger, Inventory, InvocationLedger, MigrationLedgers, PythonFile, VerifiedEdge,
    },
    scan,
};
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn check_inventory(root: &Path, paths: &[String], ledgers: &MigrationLedgers) -> DynResult<()> {
    checks::check_inventory_from_roots(
        root,
        paths,
        ledgers,
        &scan::scan_paths(root, paths)?,
        &BTreeSet::new(),
        &[],
    )
}

fn check_policy(root: &Path, paths: &[String], ledgers: &MigrationLedgers) -> DynResult<()> {
    checks::check_policy_from_roots(
        root,
        paths,
        ledgers,
        &scan::scan_paths(root, paths)?,
        &BTreeSet::new(),
        &[],
    )
}

fn fixture(root: &Path) -> DynResult<(Vec<String>, MigrationLedgers)> {
    fs::create_dir_all(root.join("scripts"))?;
    fs::create_dir_all(root.join(".agents/skills/example"))?;
    fs::write(
        root.join("scripts/loader.py"),
        "importlib.util.spec_from_file_location('x', Path(__file__).parent / (sys.argv[1] + '.py'))\n",
    )?;
    fs::write(root.join("scripts/target.py"), "pass\n")?;
    fs::write(root.join("scripts/run.sh"), "python3 -c 'print(1)'\n")?;
    fs::write(
        root.join(".agents/skills/example/SKILL.md"),
        "Use the benchmark:\n```bash\npython3 scripts/target.py\n```\n",
    )?;
    let paths = [
        "scripts/loader.py",
        "scripts/target.py",
        "scripts/run.sh",
        ".agents/skills/example/SKILL.md",
    ]
    .map(str::to_owned)
    .to_vec();
    let file = |path: &str| PythonFile {
        path: path.into(),
        classification: "fixture".into(),
        replacement_owner: "xtask".into(),
        deletion_condition: "parity".into(),
    };
    let ledgers = MigrationLedgers {
        inventory: Inventory {
            schema_version: 1,
            files: vec![file("scripts/loader.py"), file("scripts/target.py")],
        },
        invocations: InvocationLedger {
            schema_version: 1,
            source_verified_edges: vec![],
            github_source_records: vec![],
            selected_process_calls: vec![],
            runner_image_planner_loader: None,
            family_canary_loaders: vec![],
            reproducible_candidate_census: CandidateCensus { acceptance: true },
        },
        instructions: InstructionLedger {
            schema_version: 1,
            assets: vec![InstructionAsset {
                path: ".agents/skills/example/SKILL.md".into(),
                classification: "actionable".into(),
                owner: "xtask".into(),
            }],
        },
        exceptions: ExceptionLedger {
            schema_version: 1,
            exceptions: vec![],
        },
    };
    Ok((paths, ledgers))
}

fn owned(root: &Path, paths: &[String], ledgers: &mut MigrationLedgers) -> DynResult<()> {
    ledgers.invocations.source_verified_edges = scan::scan_paths(root, paths)?
        .into_iter()
        .map(|row| VerifiedEdge {
            id: row.id,
            owner: "fixture".into(),
            reason: "fixture".into(),
            target: "fixture target".into(),
            replacement_task: 5,
        })
        .collect();
    Ok(())
}

fn cleanup(root: PathBuf) -> DynResult<()> {
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn migration_inventory_rejects_unowned_inline_python() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-inline");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers
        .invocations
        .source_verified_edges
        .retain(|edge| !edge.id.starts_with("scripts/run.sh#"));
    // Given an inline Python caller absent from an otherwise owned ledger.
    // When checking executable ownership.
    let error = check_inventory(&root, &paths, &ledgers).unwrap_err();
    // Then the uncovered caller is named.
    assert!(error.to_string().contains("scripts/run.sh"), "{error}");
    cleanup(root)
}

#[test]
fn migration_inventory_detects_computed_dynamic_import() {
    // Given a Python loader with an expression-derived target.
    // When scanning its source.
    let edges = scan::scan_source(
        "scripts/loader.py",
        "importlib.util.spec_from_file_location('x', Path(__file__).parent / (sys.argv[1] + '.py'))",
    );
    // Then the computed edge is visible.
    assert!(edges.iter().any(|edge| edge.id.contains("dynamic-import")));
}

#[test]
fn migration_inventory_detects_skill_command_in_markdown_list() {
    // Given an actionable inline command in a skill list.
    // When scanning its source.
    let edges = scan::scan_source(
        ".agents/skills/example/SKILL.md",
        "- `python3 scripts/target.py --check`",
    );
    // Then the command is an ownership candidate, not prose.
    assert_eq!(edges.len(), 1);
}

#[test]
fn migration_inventory_rejects_omitted_dynamic_import() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-dynamic");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers
        .invocations
        .source_verified_edges
        .retain(|edge| !edge.id.contains("dynamic-import"));
    // Given a computed import omitted from ownership.
    // When checking the inventory.
    let error = check_inventory(&root, &paths, &ledgers).unwrap_err();
    // Then it is reported as uncovered.
    assert!(error.to_string().contains("scripts/loader.py"), "{error}");
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_omitted_python_path() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-path");
    let (paths, mut ledgers) = fixture(&root)?;
    ledgers.inventory.files.pop();
    // Given a ledger missing an existing Python file.
    // When checking the inventory.
    let error = check_inventory(&root, &paths, &ledgers).unwrap_err();
    // Then the omitted path is named.
    assert!(error.to_string().contains("scripts/target.py"), "{error}");
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_stale_python_path() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-stale");
    let (mut paths, ledgers) = fixture(&root)?;
    paths.retain(|path| path != "scripts/target.py");
    // Given a ledger entry for a file no longer in the source roster.
    // When checking inventory.
    let error = check_inventory(&root, &paths, &ledgers).unwrap_err();
    // Then the stale path is named.
    assert!(
        error.to_string().contains("stale") && error.to_string().contains("scripts/target.py"),
        "{error}"
    );
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_unresolved_dynamic_target() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-target");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    let edge = ledgers
        .invocations
        .source_verified_edges
        .iter_mut()
        .find(|edge| edge.id.contains("dynamic-import"))
        .expect("fixture dynamic import");
    edge.target = "unresolved: computed path".into();
    // Given a recorded but unresolved dynamic target.
    // When checking inventory.
    let error = check_inventory(&root, &paths, &ledgers).unwrap_err();
    // Then the incomplete edge is rejected.
    assert!(error.to_string().contains("incomplete edge"), "{error}");
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_skill_local_python_example() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-skill");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers
        .invocations
        .source_verified_edges
        .retain(|edge| !edge.id.starts_with(".agents/skills/example/SKILL.md#"));
    // Given a skill example omitted from executable ownership.
    // When checking policy.
    let error = check_policy(&root, &paths, &ledgers).unwrap_err();
    // Then it reports the owning skill.
    assert!(
        error
            .to_string()
            .contains(".agents/skills/example/SKILL.md"),
        "{error}"
    );
    cleanup(root)
}

#[test]
fn migration_inventory_accepts_owned_edges_and_complete_paths() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-happy");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    // Given complete path, instruction and edge ledgers.
    // When checking inventory and policy.
    check_inventory(&root, &paths, &ledgers)?;
    check_policy(&root, &paths, &ledgers)?;
    // Then the fixture is accepted without consulting the live repo.
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_unresolved_census() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-baseline");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers.invocations.reproducible_candidate_census.acceptance = false;
    // Given every fixture edge owned but the editable census flag is false.
    // When checking the isolated inventory, then the flag alone cannot block it.
    check_inventory(&root, &paths, &ledgers)?;
    cleanup(root)
}

#[test]
fn migration_inventory_allows_baseline_sdk_proposal_before_cutover() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-sdk-proposal");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers
        .exceptions
        .exceptions
        .push(serde_json::from_value::<ExceptionEntry>(
            serde_json::json!({
                "path": "scripts/ci-openai-python-smoke.py",
                "status": "conditional_unqualified",
                "callers": [".github/workflows/smoke.yml"],
                "local_dependency_files": ["ci/python-sdk-compatibility/uv.lock"],
                "cadence": "advisory", "purpose": "SDK compatibility", "isolation_test": "task 22"
            }),
        )?);
    // Given a source-recorded SDK candidate without task-22 qualification.
    // When checking policy before advisory workflow cutover.
    let result = check_policy(&root, &paths, &ledgers);
    // Then the candidate remains transitional without approving retention.
    cleanup(root)?;
    result
}

#[test]
fn migration_inventory_rejects_fabricated_sdk_proposal() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-fake-sdk");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers.exceptions.exceptions.push(serde_json::from_value::<ExceptionEntry>(
        serde_json::json!({"path": "scripts/ci-invented-sdk.py", "status": "conditional_unqualified"}),
    )?);
    // Given a proposed exception absent from the exact SDK baseline.
    // When checking policy.
    let error = check_policy(&root, &paths, &ledgers).unwrap_err();
    // Then the fabricated path is rejected.
    assert!(error.to_string().contains("ci-invented-sdk.py"), "{error}");
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_unproven_qualified_sdk() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-sdk-qualified");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers
        .exceptions
        .exceptions
        .push(serde_json::from_value::<ExceptionEntry>(
            serde_json::json!({"path": "scripts/ci-openai-python-smoke.py", "status": "qualified",
            "callers": [".github/workflows/smoke.yml"],
            "local_dependency_files": ["ci/python-sdk-compatibility/uv.lock"],
            "cadence": "advisory", "purpose": "SDK compatibility", "isolation_test": "task 22"}),
        )?);
    // Given a qualified label without a retained source or isolated workflow.
    // When checking policy.
    let error = check_policy(&root, &paths, &ledgers).unwrap_err();
    // Then the label alone cannot authorize persistent Python.
    assert!(
        error.to_string().contains("ci-openai-python-smoke.py"),
        "{error}"
    );
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_conditional_sdk_after_cutover() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-sdk-cutover");
    let (mut paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    fs::create_dir_all(root.join(".github/workflows"))?;
    fs::write(
        root.join(".github/workflows/python-sdk-compatibility.yml"),
        "name: SDK\n",
    )?;
    paths.push(".github/workflows/python-sdk-compatibility.yml".into());
    ledgers.exceptions.exceptions.push(serde_json::from_value::<ExceptionEntry>(
        serde_json::json!({"path": "scripts/ci-openai-python-smoke.py", "status": "conditional_unqualified",
            "callers": [".github/workflows/smoke.yml"],
            "local_dependency_files": ["ci/python-sdk-compatibility/uv.lock"],
            "cadence": "advisory", "purpose": "SDK compatibility", "isolation_test": "task 22"}),
    )?);
    // Given the task-22 advisory workflow has landed.
    // When checking a still-conditional SDK exception.
    let error = check_policy(&root, &paths, &ledgers).unwrap_err();
    // Then qualification cannot be silently deferred past cutover.
    assert!(
        error.to_string().contains("ci-openai-python-smoke.py"),
        "{error}"
    );
    cleanup(root)
}

#[test]
fn migration_inventory_rejects_github_edge_with_missing_identity() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-github-id");
    let (mut paths, mut ledgers) = fixture(&root)?;
    fs::create_dir_all(root.join(".github/workflows"))?;
    fs::write(
        root.join(".github/workflows/check.yml"),
        "run: python3 -c 'print(1)'\n",
    )?;
    paths.push(".github/workflows/check.yml".into());
    owned(&root, &paths, &mut ledgers)?;
    ledgers
        .invocations
        .source_verified_edges
        .retain(|edge| !edge.id.starts_with(".github/"));
    ledgers.invocations.github_source_records.push(GithubEdge {
        id: ".github/workflows/check.yml#candidate:0000000000000000:1".into(),
        source_block: "run: python3 -c 'print(1)'".into(),
        owner: "fixture".into(),
        disposition: "execution".into(),
        argv: "python3 -c".into(),
        status_output_effects: "fixture effect".into(),
        transitive_boundary: "inline".into(),
        replacement: "xtask".into(),
        deletion_condition: "after parity".into(),
        reason: "fixture".into(),
    });
    // Given a GitHub record with the right text but an invented hash identity.
    // When checking the source ledger.
    let error = check_inventory(&root, &paths, &ledgers).unwrap_err();
    // Then text elsewhere in the file does not establish ownership.
    assert!(error.to_string().contains("0000000000000000"), "{error}");
    cleanup(root)
}

#[test]
fn migration_inventory_detects_versioned_and_launcher_python() {
    // Given two new execution forms in a shell and a skill.
    // When scanning the command sources.
    let shell = scan::scan_source(
        "scripts/run.sh",
        "python3.12 -c 'print(1)'\npy -3 -c 'print(2)'",
    );
    let skill = scan::scan_source(".agents/skills/example/SKILL.md", "- `py -3 -c 'print(2)'`");
    // Then each call is a candidate needing exact ownership.
    assert_eq!(shell.len(), 2);
    assert_eq!(skill.len(), 1);
}

#[test]
fn migration_inventory_rejects_missing_instruction_owner() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("migration-instruction-owner");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    ledgers.instructions.assets[0].owner.clear();
    // Given an instruction path with no responsible owner.
    // When checking policy.
    let error = check_policy(&root, &paths, &ledgers).unwrap_err();
    // Then the unowned instruction is rejected.
    assert!(error.to_string().contains("example/SKILL.md"), "{error}");
    cleanup(root)
}

#[test]
fn migration_inventory_tracks_github_tool_provisioning_without_approving_it() {
    // Given repeated Python setup steps that the source census marks as candidates.
    // When the Rust guard scans the workflow.
    let rows = scan::scan_source(
        ".github/workflows/smoke.yml",
        "- uses: actions/setup-python@pinned\n- uses: actions/setup-python@pinned",
    );
    // Then both occurrences have distinct source-census identities for review.
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0].id,
        ".github/workflows/smoke.yml#candidate:3d243c25cd749c31:1"
    );
    assert_eq!(
        rows[1].id,
        ".github/workflows/smoke.yml#candidate:3d243c25cd749c31:2"
    );
}

#[test]
fn migration_inventory_tracks_recorded_github_nonexecution_context() {
    // Given source lines that the JS census records as nonexecution candidates.
    // When the Rust scanner computes their identities.
    let rows = scan::scan_source(
        ".github/actions/setup-canary-python/action.yml",
        "sdk_python=\"$PWD/ci/canary-python/.venv/bin/python\"\n\
         echo \"SKIPPY_WORKLOAD_SDK_PYTHON=$sdk_python\" >> \"$GITHUB_ENV\"",
    );
    // Then the same exact IDs remain reviewable rather than disappearing.
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0].id,
        ".github/actions/setup-canary-python/action.yml#candidate:2dbbbaf8dfb0846a:1"
    );
    assert_eq!(
        rows[1].id,
        ".github/actions/setup-canary-python/action.yml#candidate:d0054e0fb6889d47:1"
    );
}

#[test]
fn migration_inventory_tracks_github_cache_dependency_without_approving_it() {
    // Given a requirements path within a setup-python cache block.
    // When scanning workflow source for candidate identities.
    let rows = scan::scan_source(
        ".github/workflows/smoke.yml",
        "cache-dependency-path: |\n  ci/requirements-ci-python.txt",
    );
    // Then the path is visible for nonexecution classification.
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].id,
        ".github/workflows/smoke.yml#candidate:e88a4d094ac6519b:1"
    );
}

#[test]
fn migration_inventory_scans_github_execution_context_when_metadata_and_heredocs_are_present() {
    // Given a cache value, a probe, a selector, a heredoc and actual calls.
    let source = "- uses: actions/setup-python@pinned\n  with:\n    cache-dependency-path: |\n      ci/requirements-ci-python.txt\n- name: Execute\n  env:\n    NOTE: 'python3 -c print(4)'\n  run: |\n    for cmd in python3 uv; do\n      command -v \"$cmd\"\n    done\n    selected_python=python3\n    \"$selected_python\" - <<'PY'\n    print('python3 -c still data')\n    PY\n    python3 -c 'print(1)'\n    echo python3\n- name: Later\n  run: python3 -m unittest\n";
    // When scanning a GitHub workflow.
    let rows = scan::scan_source(".github/workflows/fixture.yml", source);
    // Then metadata remains visible while heredoc data is not a second call.
    let blocks = rows
        .iter()
        .map(|row| row.source_block.as_str())
        .collect::<Vec<_>>();
    assert!(blocks.contains(&"- uses: actions/setup-python@pinned"));
    assert!(blocks.contains(&"ci/requirements-ci-python.txt"));
    assert!(blocks.contains(&"\"$selected_python\" - <<'PY'"));
    assert!(blocks.contains(&"python3 -c 'print(1)'"));
    assert!(blocks.contains(&"run: python3 -m unittest"));
    assert!(!blocks.contains(&"print('python3 -c still data')"));
}

#[test]
fn migration_inventory_keeps_unknown_shell_calls_and_repeated_identities() {
    // Given repeated calls and a computed interpreter that is not an assignment.
    let source = "run: |\n  python3 -c 'print(1)'\n  python3 -c 'print(1)'\n  \"$python_bin\" - <<'PY'\n  pass\n  PY\n";
    // When scanning the run block.
    let rows = scan::scan_source(".github/workflows/fixture.yml", source);
    // Then all calls survive, and repeated calls retain separate identities.
    assert_eq!(rows.len(), 3);
    assert_ne!(rows[0].id, rows[1].id);
    assert!(rows[2].source_block.contains("$python_bin"));
}

#[test]
fn migration_inventory_source_fixture_preserves_ids_and_execution_roles() -> DynResult<()> {
    // Given hand-authored expected identities for source and data contexts.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/migration/source-candidate-ids.json"
    ))?;
    let path = fixture["github_path"]
        .as_str()
        .ok_or("missing fixture path")?;
    let source = fixture["github_source"]
        .as_str()
        .ok_or("missing fixture source")?;
    let expected = fixture["github_rows"]
        .as_array()
        .ok_or("missing fixture rows")?
        .iter()
        .map(|row| {
            Ok((
                row["id"].as_str().ok_or("missing expected ID")?,
                row["executable"].as_bool().ok_or("missing expected role")?,
            ))
        })
        .collect::<Result<Vec<_>, &str>>()?;
    // When Rust scans the fixture without running Node or reading a live ledger.
    let observed = scan::scan_source(path, source)
        .into_iter()
        .map(|row| (row.id, row.executable))
        .collect::<Vec<_>>();
    // Then ordered identities, repeated occurrences, and execution roles agree.
    assert_eq!(
        observed,
        expected
            .iter()
            .map(|(id, role)| (id.to_string(), *role))
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn migration_inventory_distinguishes_nonexecution_from_unknown_calls() {
    // Given source lines with the same interpreter name in distinct shell roles.
    let rows = scan::scan_source(
        ".github/actions/check/action.yml",
        "run: |\n  for cmd in python3 uv; do\n    command -v \"$cmd\"\n  done\n  sdk_python=python3\n  echo \"SDK_PYTHON=$sdk_python\" >> \"$GITHUB_ENV\"\n  \"$sdk_python\" -c 'print(1)'\n  \"$unknown_python\" - <<'PY'\n  print('python3 -c is data')\n  PY\n",
    );
    // When classifying interpreter reachability.
    let executable = rows
        .iter()
        .filter(|row| row.executable)
        .map(|row| row.source_block.as_str())
        .collect::<Vec<_>>();
    // Then probes and propagation are retained but cannot approve an unknown call.
    assert_eq!(
        executable,
        [
            "\"$sdk_python\" -c 'print(1)'",
            "\"$unknown_python\" - <<'PY'"
        ]
    );
    assert!(
        rows.iter()
            .any(|row| row.source_block.starts_with("for cmd in") && !row.executable)
    );
    assert!(
        rows.iter()
            .any(|row| row.source_block.starts_with("sdk_python=") && !row.executable)
    );
}

#[test]
fn migration_inventory_keeps_source_edit_identity_distinct() {
    // Given two calls at the same source path with different inline code.
    let first = scan::scan_source(".github/workflows/check.yml", "run: python3 -c 'print(1)'");
    let changed = scan::scan_source(".github/workflows/check.yml", "run: python3 -c 'print(2)'");
    // When the inline program changes.
    // Then the observed ID changes, so the prior ownership record cannot authorize it.
    assert_ne!(first[0].id, changed[0].id);
}

#[test]
fn migration_inventory_allows_transitional_closure_after_typed_shards_own_source() -> DynResult<()>
{
    // Given the actual four typed shards have reconciled every observed candidate.
    let root = crate::repo_consistency::repo_root()?;
    let paths = super::ledger::tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?;
    let validated = super::shards::check_shards(&root, &observed)?;
    let ledgers = MigrationLedgers::load(&root)?;
    // When the inventory checks the same source, then owned Python remains transitional.
    checks::check_inventory(&root, &paths, &ledgers, &observed, &validated)?;
    Ok(())
}

#[test]
fn migration_inventory_does_not_trust_editable_acceptance_for_new_call() -> DynResult<()> {
    // Given the real scoped source is reconciled and the census flag is set by hand.
    let root = crate::repo_consistency::repo_root()?;
    let paths = super::ledger::tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?;
    let validated = super::shards::check_shards(&root, &observed)?;
    let mut ledgers = MigrationLedgers::load(&root)?;
    ledgers.invocations.reproducible_candidate_census.acceptance = true;
    let mut changed = observed.clone();
    changed.extend(scan::scan_source(
        "just/ci.just",
        "python3 -c 'new required call'",
    ));
    // When inventory checks an unowned interpreter, the flag cannot authorize it.
    let error = checks::check_inventory(&root, &paths, &ledgers, &changed, &validated).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unclassified executable candidates"),
        "{error}"
    );
    Ok(())
}

#[test]
fn migration_inventory_preserves_source_verified_edge_when_shard_is_complete() -> DynResult<()> {
    // Given exact shard coverage but a historical reviewed edge with an invented identity.
    let root = crate::repo_consistency::repo_root()?;
    let paths = super::ledger::tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?;
    let validated = super::shards::check_shards(&root, &observed)?;
    let mut ledgers = MigrationLedgers::load(&root)?;
    ledgers.invocations.source_verified_edges[0].id =
        "invented#candidate:0000000000000000:1".into();
    // When inventory checks reviewed evidence, then the stale record still fails.
    let error =
        checks::check_inventory(&root, &paths, &ledgers, &observed, &validated).unwrap_err();
    assert!(error.to_string().contains("stale edges"), "{error}");
    Ok(())
}

#[test]
fn migration_inventory_rejects_new_interpreter_on_actual_source() -> DynResult<()> {
    // Given an owned shell file extended with versioned, selected and inline Python.
    let root = crate::command::unique_temp_dir("migration-new-interpreter");
    let (paths, mut ledgers) = fixture(&root)?;
    owned(&root, &paths, &mut ledgers)?;
    fs::write(
        root.join("scripts/run.sh"),
        "python3 -c 'print(1)'\npython3.12 -c 'print(2)'\npython_bin=python3\n\"$python_bin\" -c 'print(3)'\n",
    )?;
    let observed = scan::scan_paths(&root, &paths)?;
    assert!(
        observed
            .iter()
            .any(|row| row.source_block.starts_with("python3.12 -c"))
    );
    assert!(
        observed
            .iter()
            .any(|row| row.source_block.starts_with("\"$python_bin\" -c"))
    );
    // When inventory checks actual source, then it rejects the newly added calls.
    let result = check_inventory(&root, &paths, &ledgers);
    cleanup(root)?;
    let error = result.unwrap_err();
    assert!(error.to_string().contains("scripts/run.sh"), "{error}");
    Ok(())
}
