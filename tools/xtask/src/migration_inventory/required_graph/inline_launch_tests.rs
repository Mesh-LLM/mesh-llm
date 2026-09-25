use super::report;
use crate::command::DynResult;
use crate::migration_inventory::{ledger, required_closure, scan, shards};
use std::collections::BTreeSet;
use std::fs;

const GENERIC: &str = "inline interpreter/installer or Python dynamic process candidate requires source-backed caller contract";
const SELECTED: &str = "inline Python program launched through a runtime-selected interpreter binding; selector output and launch are not source-joined";
const VARIABLE: &str = "Python script target is a shell variable; its bound path and branch are not joined as a child edge";

fn reason_at(graph: &super::Graph, line: usize) -> Option<&str> {
    graph
        .edges
        .iter()
        .find(|edge| edge.line == line)
        .and_then(|edge| edge.unresolved_reason.as_deref())
}

#[test]
fn selected_interpreter_inline_program_names_selection_boundary() -> DynResult<()> {
    // Given reached shell launches whose interpreter comes from a selector.
    let root = crate::command::unique_temp_dir("graph-inline-selected");
    let path = "scripts/selected.sh";
    super::super::required_graph_tests::source(
        &root,
        path,
        concat!(
            "\"$(python_bin)\" - \"$manifest\" <<'PY'\n",
            "print(1)\n",
            "PY\n",
            "done < <(\"$py\" - \"$dir\" <<'PY'\n",
            "print(2)\n",
            "PY\n",
            "\"$python\" -c 'print(3)' \"$1\"\n",
            "\"$python_bin\" - \\\n",
        ),
    )?;
    let files = [path.to_owned()];
    let observed = scan::scan_paths(&root, &files)?;

    // When the graph walks the source without contracts.
    let graph = report(&root, &files, &observed, &BTreeSet::new(), &[path])?;

    // Then each launch is unresolved as a selected interpreter, never generic.
    for line in [1, 4, 7, 8] {
        assert_eq!(reason_at(&graph, line), Some(SELECTED), "{path}:{line}");
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn variable_script_target_is_named_and_fixed_inline_stays_generic() -> DynResult<()> {
    // Given a variable-selected script target and a new fixed inline launch.
    let root = crate::command::unique_temp_dir("graph-inline-variable");
    let path = "scripts/variable.sh";
    super::super::required_graph_tests::source(
        &root,
        path,
        concat!(
            "python3 \"$WINDOWS_PROCESS_HELPER\" force-stop --pid \"$pid\" || true\n",
            "python3 -c 'import sys; print(sys.argv)'\n",
            "if python3 - \"$LOG\" <<'PY'\n",
            "raise SystemExit(0)\n",
            "PY\n",
        ),
    )?;
    let files = [path.to_owned()];
    let observed = scan::scan_paths(&root, &files)?;

    // When the graph walks the source without contracts.
    let graph = report(&root, &files, &observed, &BTreeSet::new(), &[path])?;

    // Then the variable target is named while an unreviewed fixed python3
    // launch remains the generic unresolved execution candidate.
    assert_eq!(reason_at(&graph, 1), Some(VARIABLE));
    assert_eq!(reason_at(&graph, 2), Some(GENERIC));
    assert_eq!(reason_at(&graph, 3), Some(GENERIC));
    assert!(
        graph
            .edges
            .iter()
            .all(|edge| edge.status == "unknown_selection")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn checked_in_required_graph_has_no_generic_inline_reason() -> DynResult<()> {
    // Given the checked-in required roots, ledgers and validated candidates.
    let root = crate::repository::RepositoryRoot::resolve(None)?;
    let paths = ledger::tracked_paths(root.as_path())?;
    let observed = scan::scan_paths(root.as_path(), &paths)?;
    let validated = shards::check_shards(root.as_path(), &observed)?;
    let roots = required_closure::required_roots(root.as_path(), &paths)?;
    let refs = roots.iter().map(String::as_str).collect::<Vec<_>>();

    // When the full graph is built.
    let graph = report(root.as_path(), &paths, &observed, &validated, &refs)?;
    let count = |reason: &str| {
        graph
            .edges
            .iter()
            .filter(|edge| edge.unresolved_reason.as_deref() == Some(reason))
            .count()
    };

    // Then every former generic entry has a joined contract or a named boundary.
    assert_eq!(count(GENERIC), 0);
    assert_eq!(count(SELECTED), 17);
    // The Windows helper calls now bind through the proven SCRIPT_DIR root instead.
    assert_eq!(count(VARIABLE), 0);
    let joined = graph
        .edges
        .iter()
        .filter(|edge| {
            edge.contract_source.as_deref().is_some_and(|source| {
                source
                    .starts_with("ci/automation-migration/invocations.json#inline_source_records:")
            })
        })
        .count();
    assert_eq!(joined, 5 + 28);
    Ok(())
}
