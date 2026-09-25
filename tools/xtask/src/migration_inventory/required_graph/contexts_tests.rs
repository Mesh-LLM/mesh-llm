use super::super::required_graph::report;
use super::super::scan;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;

#[test]
fn prose_path_does_not_reach_python_but_selected_call_does() -> DynResult<()> {
    // Given a shell script that prints a path and invokes another Python file.
    let root = crate::command::unique_temp_dir("graph-context");
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(
        root.join("scripts/start.sh"),
        "printf 'scripts/note.py\\n'\npython3 scripts/run.py --check\n",
    )?;
    fs::write(root.join("scripts/note.py"), "pass\n")?;
    fs::write(root.join("scripts/run.py"), "pass\n")?;
    let files = ["scripts/start.sh", "scripts/note.py", "scripts/run.py"].map(str::to_owned);
    let graph = report(
        &root,
        &files,
        &scan::scan_paths(&root, &files)?,
        &BTreeSet::new(),
        &["scripts/start.sh"],
    )?;
    // When walking executable contexts, then a printed token is not traversed.
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/note.py"))
    );
    assert!(graph.edges.iter().any(|edge| {
        edge.child.as_deref() == Some("scripts/run.py")
            && edge.status == "reached_execution"
            && edge
                .unresolved_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("missing"))
    }));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn verified_invocation_joins_only_matching_source_call() -> DynResult<()> {
    // Given one source-backed record with argv and observed streams/effects.
    let root = crate::command::unique_temp_dir("graph-contract");
    fs::create_dir_all(root.join("scripts"))?;
    fs::create_dir_all(root.join("ci/automation-migration"))?;
    fs::write(
        root.join("scripts/start.sh"),
        "python3 scripts/run.py --check\n",
    )?;
    fs::write(root.join("scripts/run.py"), "pass\n")?;
    let files = ["scripts/start.sh", "scripts/run.py"].map(str::to_owned);
    let observed = scan::scan_paths(&root, &files)?;
    let id = &observed[0].id;
    fs::write(root.join("ci/automation-migration/invocations.json"), serde_json::json!({"github_source_records": [{"id": id, "source_block": "python3 scripts/run.py --check", "disposition": "execution", "argv": "python3 scripts/run.py --check", "status_output_effects": "exit 0, stdout ok, stderr empty, reads input", "transitive_boundary": "scripts/run.py", "replacement": "xtask runner", "deletion_condition": "after parity", "reason": "script invocation"}]}).to_string())?;
    // When joining this call, then its exact contract is attached.
    let graph = report(
        &root,
        &files,
        &observed,
        &BTreeSet::from([id.clone()]),
        &["scripts/start.sh"],
    )?;
    let edge = graph
        .edges
        .iter()
        .find(|edge| edge.child.as_deref() == Some("scripts/run.py"))
        .ok_or("missing call")?;
    assert_eq!(edge.status, "reached_execution");
    assert_eq!(edge.unresolved_reason, None);
    assert_eq!(edge.argv.as_deref(), Some("python3 scripts/run.py --check"));
    assert_eq!(
        edge.status_streams_effects.as_deref(),
        Some("exit 0, stdout ok, stderr empty, reads input")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn provision_optional_and_unknown_do_not_expand_children() -> DynResult<()> {
    // Given a selection probe, a conditional call, and unsupported selection syntax.
    let root = crate::command::unique_temp_dir("graph-branches");
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(
        root.join("scripts/start.sh"),
        "command -v scripts/probe.py\nif python3 scripts/optional.py --check; then\n  :\nfi\nchosen=(scripts/unknown.py)\n",
    )?;
    for name in ["probe", "optional", "unknown"] {
        fs::write(root.join(format!("scripts/{name}.py")), "pass\n")?;
    }
    let files = [
        "scripts/start.sh",
        "scripts/probe.py",
        "scripts/optional.py",
        "scripts/unknown.py",
    ]
    .map(str::to_owned);
    // When classified, then none of these is unconditional execution.
    let graph = report(
        &root,
        &files,
        &scan::scan_paths(&root, &files)?,
        &BTreeSet::new(),
        &["scripts/start.sh"],
    )?;
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/optional.py")
                && edge.status == "optional_branch")
    );
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/probe.py")
                && edge.status == "reached_execution")
    );
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/unknown.py")
                && edge.status == "reached_execution")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn setup_action_provisions_without_launching_python() -> DynResult<()> {
    // Given an external setup action and a later local run step.
    let root = crate::command::unique_temp_dir("graph-provision");
    fs::create_dir_all(root.join(".github/workflows"))?;
    fs::write(
        root.join(".github/workflows/main_linux.yml"),
        "steps:\n  - uses: actions/setup-python@1234\n    with:\n      python-version: '3.12'\n",
    )?;
    let files = [".github/workflows/main_linux.yml".to_owned()];
    // When reading YAML, then setup is provisioning and has no executable child.
    let graph = report(&root, &files, &[], &BTreeSet::new(), &[&files[0]])?;
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.status == "provisioning_selection" && edge.child.is_none())
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn python_child_does_not_recursively_launch_prose_path() -> DynResult<()> {
    // Given Python source mentioning a second script only in a string constant.
    let root = crate::command::unique_temp_dir("graph-python-prose");
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(root.join("scripts/start.sh"), "python3 scripts/entry.py\n")?;
    fs::write(root.join("scripts/entry.py"), "NAME = 'scripts/other.py'\n")?;
    fs::write(root.join("scripts/other.py"), "pass\n")?;
    let files = ["scripts/start.sh", "scripts/entry.py", "scripts/other.py"].map(str::to_owned);
    // When Python source is visited, then its data string does not launch another file.
    let graph = report(
        &root,
        &files,
        &scan::scan_paths(&root, &files)?,
        &BTreeSet::new(),
        &["scripts/start.sh"],
    )?;
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/other.py"))
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn selected_interpreter_continuation_reaches_child_but_assignment_does_not() -> DynResult<()> {
    // Given a selected interpreter, a continued executable call and inert shell data.
    let root = crate::command::unique_temp_dir("graph-shell-continuation");
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(
        root.join("scripts/start.sh"),
        concat!(
            "python_bin=python3\nhelper=scripts/data.py\n",
            "\"$python_bin\" scripts/child.py \\\n",
            "    --check\n"
        ),
    )?;
    fs::write(root.join("scripts/child.py"), "pass\n")?;
    fs::write(root.join("scripts/data.py"), "pass\n")?;
    let files = ["scripts/start.sh", "scripts/child.py", "scripts/data.py"].map(str::to_owned);
    // When traversing the root, then the selected call is reached, not the assignment.
    let observed = scan::scan_paths(&root, &files)?;
    let validated = observed.iter().map(|row| row.id.clone()).collect();
    let graph = report(&root, &files, &observed, &validated, &["scripts/start.sh"])?;
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/child.py")
                && edge.status == "unknown_selection"
                && edge.unresolved_reason.is_some())
    );
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/data.py")
                && edge.status == "reached_execution")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn unbound_selected_interpreter_remains_unresolved() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("graph-unbound-interpreter");
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(
        root.join("scripts/start.sh"),
        "\"$python_bin\" scripts/child.py\n",
    )?;
    fs::write(root.join("scripts/child.py"), "pass\n")?;
    let paths = ["scripts/start.sh", "scripts/child.py"].map(str::to_owned);
    let observed = scan::scan_paths(&root, &paths)?;
    let graph = report(&root, &paths, &observed, &BTreeSet::new(), &[&paths[0]])?;
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/child.py")
                && edge.status == "unknown_selection"
                && edge.unresolved_reason.is_some())
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
