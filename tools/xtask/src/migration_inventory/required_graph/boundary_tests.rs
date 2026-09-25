use super::super::required_graph::{Graph, report, require_census};
use super::super::required_graph_tests::source;
use super::boundaries::EdgeDisposition;
use crate::command::DynResult;
use crate::migration_inventory::scan;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const SELECTED: &str = "python_bin() {\n    for candidate in python3 python; do\n        command -v \"$candidate\" && printf '%s\\n' \"$candidate\" && return 0\n    done\n}\n\"$(python_bin)\" - \"$1\" <<'PY'\nprint(1)\nPY\n";

fn ledger(root: &Path, records: &[Value]) -> DynResult<()> {
    source(
        root,
        "ci/automation-migration/invocations.json",
        &json!({"github_source_records": [], "boundary_records": records}).to_string(),
    )
}

fn record(caller: &str, line: usize, block: &str, disposition: &str) -> Value {
    json!({
        "id": format!("boundary:{caller}:{line}"), "disposition": disposition,
        "caller": caller, "line": line, "source_block": block, "occurrence": 1,
        "candidate_id": null, "child": null, "selector": "fixture selector",
        "reachable_bytes": "fixture bytes", "replacement_owner": "fixture owner",
        "deletion_condition": "fixture deletion", "rationale": "fixture rationale",
        "evidence": [{"path": caller, "line": line, "text": block}],
    })
}

fn graph(root: &Path, paths: &[&str], roots: &[&str]) -> DynResult<Graph> {
    let files = paths
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<Vec<_>>();
    let observed = scan::scan_paths(root, &files)?;
    let validated = observed
        .iter()
        .map(|row| row.id.clone())
        .collect::<BTreeSet<_>>();
    report(root, &files, &observed, &validated, roots)
}

fn selected_candidate(root: &Path) -> DynResult<scan::Candidate> {
    let files = vec!["scripts/run.sh".to_owned()];
    Ok(scan::scan_paths(root, &files)?
        .into_iter()
        .find(|row| row.source_block.starts_with("\"$(python_bin)\""))
        .ok_or("missing candidate")?)
}

#[test]
fn boundary_unrecorded_selected_interpreter_program_stays_unresolved() -> DynResult<()> {
    // Given a new inline program launched through a runtime-selected interpreter.
    let root = crate::command::unique_temp_dir("boundary-selected");
    source(&root, "scripts/run.sh", SELECTED)?;
    ledger(&root, &[])?;
    // When no boundary record owns it, then it is class-c unresolved and census stays open.
    let open = graph(&root, &["scripts/run.sh"], &["scripts/run.sh"])?;
    assert_eq!(open.unresolved, 1, "{open:?}");
    assert!(!open.complete_census);
    // When a complete record binds block, occurrence and candidate, then it is a bounded selector.
    let candidate = selected_candidate(&root)?;
    let mut owned = record(
        "scripts/run.sh",
        6,
        &candidate.source_block,
        "bounded_selector",
    );
    owned["candidate_id"] = json!(candidate.id);
    ledger(&root, &[owned])?;
    let closed = graph(&root, &["scripts/run.sh"], &["scripts/run.sh"])?;
    assert_eq!(closed.unresolved, 0);
    assert!(closed.complete_census);
    assert!(
        closed
            .edges
            .iter()
            .any(|edge| edge.disposition == EdgeDisposition::BoundedSelector)
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn boundary_record_with_stale_source_line_fails() -> DynResult<()> {
    // Given the selected call moved from line 6 to line 7 after a record was written.
    let root = crate::command::unique_temp_dir("boundary-stale");
    source(&root, "scripts/run.sh", &format!("\n{SELECTED}"))?;
    let candidate = selected_candidate(&root)?;
    let mut stale = record(
        "scripts/run.sh",
        6,
        &candidate.source_block,
        "bounded_selector",
    );
    stale["candidate_id"] = json!(candidate.id);
    stale["evidence"] =
        json!([{"path": "scripts/run.sh", "line": 7, "text": candidate.source_block}]);
    ledger(&root, &[stale])?;
    // When the graph is built, then the unbound record fails.
    let moved = graph(&root, &["scripts/run.sh"], &["scripts/run.sh"])?;
    assert_eq!(moved.unbound_boundary_records.len(), 1, "{moved:?}");
    let error = require_census(&moved).unwrap_err();
    assert!(error.to_string().contains("stale or unbound"), "{error}");
    // When only an evidence line drifts, then the record is rejected as stale evidence.
    let mut drifted = record(
        "scripts/run.sh",
        7,
        &candidate.source_block,
        "bounded_selector",
    );
    drifted["candidate_id"] = json!(candidate.id);
    drifted["evidence"] = json!([{"path": "scripts/run.sh", "line": 1, "text": "python_bin() {"}]);
    ledger(&root, &[drifted])?;
    let error = graph(&root, &["scripts/run.sh"], &["scripts/run.sh"]).unwrap_err();
    assert!(error.to_string().contains("stale evidence"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn boundary_gated_body_that_adds_python_is_still_reported() -> DynResult<()> {
    // Given a macOS-gated recipe whose launched child is owned as a platform conditional.
    let root = crate::command::unique_temp_dir("boundary-gated");
    source(
        &root,
        "Justfile",
        "default: build\n[macos]\nbuild:\n    python3 scripts/child.py\n",
    )?;
    source(&root, "scripts/child.py", "pass\n")?;
    let mut gated = record(
        "just:build",
        0,
        "python3 scripts/child.py",
        "platform_conditional",
    );
    gated["child"] = json!("scripts/child.py");
    gated["evidence"] = json!([{"path": "Justfile", "line": 2, "text": "[macos]"}]);
    ledger(&root, &[gated])?;
    let owned = graph(&root, &["Justfile", "scripts/child.py"], &["Justfile"])?;
    assert_eq!(owned.unresolved, 0, "{owned:?}");
    // When the gated child newly launches Python, then the walked body reports it.
    source(
        &root,
        "scripts/child.py",
        "import subprocess\nsubprocess.run(['python3', 'scripts/other.py'], check=True)\n",
    )?;
    source(&root, "scripts/other.py", "pass\n")?;
    let paths = ["Justfile", "scripts/child.py", "scripts/other.py"];
    let reported = graph(&root, &paths, &["Justfile"])?;
    assert!(
        reported
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/child.py"
                && edge.child.as_deref() == Some("scripts/other.py")
                && edge.disposition == EdgeDisposition::Unresolved),
        "{reported:?}"
    );
    assert!(!reported.complete_census);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn boundary_removed_or_renamed_main_call_invalidates_record() -> DynResult<()> {
    // Given a PR entrypoint calling a protected @main lane with an external trust record.
    let root = crate::command::unique_temp_dir("boundary-main");
    let entry = ".github/workflows/pr_linux.yml";
    let call = "uses: Mesh-LLM/mesh-llm/.github/workflows/ci-linux-lane.yml@main";
    source(&root, entry, &format!("jobs:\n  lane:\n    {call}\n"))?;
    ledger(&root, &[record(entry, 3, call, "external_trust_boundary")])?;
    let owned = graph(&root, &[entry], &[entry])?;
    assert_eq!(owned.unresolved, 0, "{owned:?}");
    assert!(
        owned
            .edges
            .iter()
            .any(|edge| edge.disposition == EdgeDisposition::ExternalTrustBoundary)
    );
    // When the call is renamed, then the record no longer binds and the graph fails.
    let renamed_call = "uses: Mesh-LLM/mesh-llm/.github/workflows/ci-linux-lane2.yml@main";
    source(
        &root,
        entry,
        &format!("jobs:\n  lane:\n    {renamed_call}\n"),
    )?;
    let error = graph(&root, &[entry], &[entry])
        .and_then(|renamed| require_census(&renamed))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("boundary:.github/workflows/pr_linux.yml:3"),
        "{error}"
    );
    // When the call is removed, then the record is invalid as well.
    source(&root, entry, "jobs: {}\n")?;
    let removed = graph(&root, &[entry], &[entry]).and_then(|graph| require_census(&graph));
    assert!(removed.is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn boundary_record_without_owner_or_rationale_fails() -> DynResult<()> {
    // Given an otherwise matching record with an empty rationale.
    let root = crate::command::unique_temp_dir("boundary-incomplete");
    let entry = ".github/workflows/pr_linux.yml";
    let call = "uses: Mesh-LLM/mesh-llm/.github/workflows/ci-linux-lane.yml@main";
    source(&root, entry, &format!("jobs:\n  lane:\n    {call}\n"))?;
    let mut incomplete = record(entry, 3, call, "external_trust_boundary");
    incomplete["rationale"] = json!("");
    ledger(&root, &[incomplete])?;
    // When the graph loads it, then the missing rationale fails.
    let error = graph(&root, &[entry], &[entry]).unwrap_err();
    assert!(
        error.to_string().contains("missing owner or rationale"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
