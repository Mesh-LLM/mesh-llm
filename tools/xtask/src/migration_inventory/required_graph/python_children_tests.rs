use super::report;
use crate::command::DynResult;
use crate::migration_inventory::required_graph_tests::source;
use crate::migration_inventory::{ledger, scan};
use std::collections::BTreeSet;
use std::fs;

#[test]
fn reached_python_calls_keep_finite_descendants_and_runtime_boundaries() -> DynResult<()> {
    // Given a reached Python program with a subprocess, computed import and runtime command.
    let root = crate::command::unique_temp_dir("graph-python-children");
    source(&root, "scripts/entry.sh", "python3 scripts/entry.py\n")?;
    source(
        &root,
        "scripts/entry.py",
        concat!(
            "import importlib.util\nimport subprocess\n",
            "literal = 'subprocess.run([\"bash\", \"scripts/inert.sh\"])'\n",
            "subprocess.run([\"bash\", str(Path(__file__).resolve().parent / \"next.sh\")], check=True)\n",
            "spec = importlib.util.spec_from_file_location(\"helper\", Path(__file__).resolve().parent / \"helper.py\")\n",
            "spec.loader.exec_module(module)\n",
            "subprocess.run(command, check=True)\n"
        ),
    )?;
    source(&root, "scripts/next.sh", "true\n")?;
    source(
        &root,
        "scripts/helper.py",
        "import subprocess\nsubprocess.run(['bash', 'scripts/next.sh'], check=True)\n",
    )?;
    let paths = [
        "scripts/entry.sh",
        "scripts/entry.py",
        "scripts/next.sh",
        "scripts/helper.py",
    ]
    .map(str::to_owned);
    // When the graph walks the shell's Python child.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &[&paths[0]])?;
    // Then both finite descendants are attributed to their source and unknown bytes stay unknown.
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/entry.py"
                && edge.line == 4
                && edge.child.as_deref() == Some("scripts/next.sh")),
        "{:?}",
        graph
            .edges
            .iter()
            .filter(|edge| edge.parent == "scripts/entry.py")
            .map(|edge| (&edge.line, &edge.child))
            .collect::<Vec<_>>()
    );
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/entry.py"
                && edge.line == 5
                && edge.child.as_deref() == Some("scripts/helper.py"))
    );
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/helper.py"
                && edge.line == 2
                && edge.child.as_deref() == Some("scripts/next.sh"))
    );
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/entry.py"
                && edge.line == 7
                && edge.child.is_none()
                && edge.unresolved_reason.is_some())
    );
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/inert.sh"))
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn checked_in_planner_exposes_selected_shell_child_without_certifying_root() -> DynResult<()> {
    // Given the real planner reached through its own source path.
    let root = crate::repository::RepositoryRoot::resolve(None)?;
    let paths = ledger::tracked_paths(root.as_path())?;
    let observed = scan::scan_paths(root.as_path(), &paths)?;
    let validated = observed
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect();
    // When the graph reads the planner's multiline subprocess call.
    let graph = report(
        root.as_path(),
        &paths,
        &observed,
        &validated,
        &["scripts/plan-ci.py"],
    )?;
    // Then source proves the relative script, but caller-selected root bytes stay conditional.
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/plan-ci.py"
                && edge.line == 375
                && edge.child.as_deref() == Some("scripts/affected-crates.sh")
                && edge.status == "unknown_selection"
                && edge.argv.is_none())
    );
    Ok(())
}

#[test]
fn computed_import_without_execution_does_not_reach_child() -> DynResult<()> {
    // Given a loader specification never executed and a selected runtime command.
    let root = crate::command::unique_temp_dir("graph-python-selection");
    source(
        &root,
        "scripts/entry.py",
        concat!(
            "spec = importlib.util.spec_from_file_location(\"helper\", Path(__file__).resolve().parent / \"helper.py\")\n",
            "subprocess.run(command)\n"
        ),
    )?;
    source(
        &root,
        "scripts/helper.py",
        "subprocess.run([\"bash\", \"scripts/missing.sh\"])\n",
    )?;
    let paths = ["scripts/entry.py", "scripts/helper.py"].map(str::to_owned);
    // When only the entry is reached.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &[&paths[0]])?;
    // Then loader construction alone does not descend into the helper or guess the command.
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/helper.py")
    );
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/entry.py"
                && edge.child.as_deref() == Some("scripts/helper.py")
                && edge.unresolved_reason.is_some())
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn source_selected_python_subprocess_stays_conditional_without_root_bytes() -> DynResult<()> {
    // Given a reached Python child with a named script relative to a runtime root.
    let root = crate::command::unique_temp_dir("graph-python-root");
    source(
        &root,
        "scripts/entry.py",
        concat!(
            "script = root / \"scripts\" / \"affected-crates.sh\"\n",
            "result = subprocess.run(\n    [\"bash\", str(script), \"--stdin\"],\n    check=False,\n)\n"
        ),
    )?;
    source(&root, "scripts/affected-crates.sh", "true\n")?;
    let paths = ["scripts/entry.py", "scripts/affected-crates.sh"].map(str::to_owned);
    // When the graph reads the whole Python call.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &[&paths[0]])?;
    // Then the finite relative name is visible, but runtime root bytes are not certified.
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/entry.py"
                && edge.line == 2
                && edge.child.as_deref() == Some("scripts/affected-crates.sh")
                && edge.status == "unknown_selection")
    );
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/affected-crates.sh")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn reached_python_import_descends_only_after_execution_and_rejects_missing_child() -> DynResult<()>
{
    // Given a multiline computed import executed through its spec loader.
    let root = crate::command::unique_temp_dir("graph-python-missing-import");
    source(
        &root,
        "scripts/entry.py",
        concat!(
            "spec = importlib.util.spec_from_file_location(\n",
            "    \"helper\", Path(__file__).resolve().parent / \"lib/helper.py\")\n",
            "spec.loader.exec_module(module)\n"
        ),
    )?;
    let paths = ["scripts/entry.py".to_owned()];
    // When the path roster omits the executed import's local child.
    let error = report(&root, &paths, &[], &BTreeSet::new(), &[&paths[0]]).unwrap_err();
    // Then the graph rejects that missing source rather than silently closing the edge.
    assert!(
        error.to_string().contains("scripts/lib/helper.py"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
