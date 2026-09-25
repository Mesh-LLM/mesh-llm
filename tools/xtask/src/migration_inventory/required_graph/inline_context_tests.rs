use super::report;
use crate::command::DynResult;
use crate::migration_inventory::scan;
use std::collections::BTreeSet;
use std::fs;

#[test]
fn inline_launch_is_distinct_from_shell_data_and_unbound_selection() -> DynResult<()> {
    // Given a reached shell source with an inline call, inert declarations/data,
    // and an interpreter selected only at runtime.
    let root = crate::command::unique_temp_dir("graph-inline-context");
    super::super::required_graph_tests::source(
        &root,
        "scripts/start.sh",
        concat!(
            "python_bin() {\n",
            "  local python\n",
            "  python=\"$(command -v python3)\"\n",
            "  printf '%s\\n' 'python3 -c quoted-data'\n",
            "}\n",
            "python=\"$(python_bin)\" || {\n",
            "python3 -c 'print(1)'\n",
            "\"$python\" -c 'print(2)'\n",
            "value=\"$(python3 -c 'print(3)')\"\n",
            "python3 \"$dynamic_child\" --check\n",
        ),
    )?;
    let files = ["scripts/start.sh".to_owned()];
    let observed = scan::scan_paths(&root, &files)?;

    // When walking this executable root, then only actual call syntax gets
    // an unresolved execution edge; a selected target remains unresolved.
    let graph = report(&root, &files, &observed, &BTreeSet::new(), &[&files[0]])?;
    let inline: Vec<_> = graph
        .edges
        .iter()
        .filter(|edge| edge.unresolved_reason.as_deref() == Some(
            "inline interpreter/installer or Python dynamic process candidate requires source-backed caller contract",
        ))
        .collect();
    assert_eq!(inline.len(), 2);
    assert!(
        inline
            .iter()
            .any(|edge| edge.source_block == "python3 -c 'print(1)'")
    );
    assert!(
        inline
            .iter()
            .any(|edge| edge.source_block == "value=\"$(python3 -c 'print(3)')\"")
    );
    assert!(inline.iter().all(|edge| edge.status == "unknown_selection"));
    // The selected interpreter and variable script target are still
    // unresolved launches, but under their specific boundary reasons.
    for (block, prefix) in [
        (
            "\"$python\" -c 'print(2)'",
            "inline Python program launched through a runtime-selected",
        ),
        (
            "python3 \"$dynamic_child\" --check",
            "Python script target is a shell variable",
        ),
    ] {
        assert!(graph.edges.iter().any(|edge| {
            edge.source_block == block
                && edge.status == "unknown_selection"
                && edge.child.is_none()
                && edge
                    .unresolved_reason
                    .as_deref()
                    .is_some_and(|reason| reason.starts_with(prefix))
        }));
    }
    assert!(graph.edges.iter().any(|edge| {
        edge.source_block == "python=\"$(python_bin)\" || {"
            && edge.status == "unknown_selection"
            && edge
                .unresolved_reason
                .as_deref()
                .is_some_and(|reason| reason.starts_with("shell assignment selects"))
    }));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn verified_inline_occurrence_does_not_approve_heredoc_data_or_dynamic_target() -> DynResult<()> {
    // Given one reviewed execution, a data heredoc, and a runtime-selected child.
    let root = crate::command::unique_temp_dir("graph-inline-contract");
    let path = "scripts/check.sh";
    let call = "python3 -c 'print(1)'";
    super::super::required_graph_tests::source(
        &root,
        path,
        &format!(
            "{call}\npayload=$(cat <<'DATA'\npython3 -c 'print(9)'\nDATA\n)\npython3 \"$dynamic_child\" --check\n"
        ),
    )?;
    let files = [path.to_owned()];
    let observed = scan::scan_paths(&root, &files)?;
    let id = &observed
        .iter()
        .find(|row| row.source_block == call)
        .ok_or("missing call")?
        .id;
    let ledger = root.join("ci/automation-migration/invocations.json");
    fs::create_dir_all(ledger.parent().ok_or("missing ledger directory")?)?;
    fs::write(
        &ledger,
        serde_json::to_vec(&serde_json::json!({
            "github_source_records": [],
            "inline_source_records": [{
                "id": id, "source_block": call, "disposition": "execution",
                "argv": call, "status_output_effects": "stdout inherited; nonzero fails; no writes",
                "transitive_boundary": "inline stdlib only", "replacement": "xtask check",
                "deletion_condition": "after parity", "reason": "shell launches python3"
            }]
        }))?,
    )?;
    let validated = observed.iter().map(|row| row.id.clone()).collect();

    // When the source is walked, then only the exact reviewed call is reached.
    let graph = report(&root, &files, &observed, &validated, &[path])?;
    assert!(graph.edges.iter().any(|edge| {
        edge.line == 1 && edge.status == "reached_execution" && edge.contract_source.is_some()
    }));
    assert!(!graph.edges.iter().any(|edge| edge.line == 3));
    assert!(graph.edges.iter().any(|edge| {
        edge.line == 6 && edge.status == "unknown_selection" && edge.contract_source.is_none()
    }));
    fs::write(
        &ledger,
        serde_json::to_vec(&serde_json::json!({
            "github_source_records": [],
            "inline_source_records": [{
                "id": "scripts/check.sh#candidate:stale:1", "source_block": call,
                "disposition": "execution", "argv": call,
                "status_output_effects": "stdout inherited; nonzero fails; no writes",
                "transitive_boundary": "inline stdlib only", "replacement": "xtask check",
                "deletion_condition": "after parity", "reason": "shell launches python3"
            }]
        }))?,
    )?;
    let stale = report(&root, &files, &observed, &validated, &[path])?;
    assert!(stale.edges.iter().any(|edge| {
        edge.line == 1 && edge.status == "unknown_selection" && edge.contract_source.is_none()
    }));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn smoke_inline_contracts_bind_each_physical_occurrence() -> DynResult<()> {
    let root = crate::repository::RepositoryRoot::resolve(None)?;
    let path = "scripts/ci-smoke-test.sh";
    let files = [path.to_owned()];
    let observed = scan::scan_paths(root.as_path(), &files)?;
    let validated = observed.iter().map(|row| row.id.clone()).collect();

    let graph = report(root.as_path(), &files, &observed, &validated, &[path])?;
    for line in [37, 46, 123, 150, 280] {
        let edge = graph
            .edges
            .iter()
            .find(|edge| edge.parent == path && edge.line == line)
            .ok_or("missing smoke inline call")?;
        assert_eq!(edge.status, "reached_execution", "{path}:{line}");
        assert!(edge.contract_source.as_deref().is_some_and(|source| {
            source.starts_with("ci/automation-migration/invocations.json#inline_source_records:")
        }));
        assert!(edge.unresolved_reason.is_none());
    }
    assert_ne!(
        graph
            .edges
            .iter()
            .find(|edge| edge.line == 46)
            .and_then(|edge| edge.contract_source.as_ref()),
        graph
            .edges
            .iter()
            .find(|edge| edge.line == 280)
            .and_then(|edge| edge.contract_source.as_ref())
    );
    Ok(())
}
