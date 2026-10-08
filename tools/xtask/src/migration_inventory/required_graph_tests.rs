use super::required_graph::report;
use super::scan;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub(super) fn source(root: &Path, path: &str, text: &str) -> DynResult<()> {
    let file = root.join(path);
    fs::create_dir_all(file.parent().ok_or("missing parent")?)?;
    fs::write(file, text)?;
    Ok(())
}

#[test]
fn graph_reaches_script_through_action_and_rejects_omitted_edge() -> DynResult<()> {
    // Given a main entry calling a reusable lane, action, and script.
    let root = crate::command::unique_temp_dir("graph-required-edge");
    source(
        &root,
        ".github/workflows/main_linux.yml",
        "jobs:\n  lane:\n    uses: ./.github/workflows/ci-linux-lane.yml\n",
    )?;
    source(
        &root,
        ".github/workflows/ci-linux-lane.yml",
        "steps:\n  - uses: ./.github/actions/prepare\n",
    )?;
    source(
        &root,
        ".github/actions/prepare/action.yml",
        "run: bash scripts/build.sh\n",
    )?;
    source(&root, "scripts/build.sh", "python3 scripts/build.py\n")?;
    source(&root, "scripts/build.py", "pass\n")?;
    let paths = [
        ".github/workflows/main_linux.yml",
        ".github/workflows/ci-linux-lane.yml",
        ".github/actions/prepare/action.yml",
        "scripts/build.sh",
        "scripts/build.py",
    ];
    let files = paths.map(str::to_owned);
    let observed = scan::scan_paths(&root, &files)?;
    let graph = report(&root, &files, &observed, &BTreeSet::new(), &[paths[0]])?;
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "scripts/build.sh"
                && edge.child.as_deref() == Some("scripts/build.py"))
    );
    // When the direct script edge is omitted from the source, then reconciliation fails.
    source(&root, "scripts/build.sh", "python3 scripts/absent.py\n")?;
    let error = report(&root, &files, &observed, &BTreeSet::new(), &[paths[0]]).unwrap_err();
    assert!(error.to_string().contains("absent.py"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn graph_distinguishes_metadata_from_protected_remote_execution() -> DynResult<()> {
    // Given a PR protected lane and a cache-only Python path.
    let root = crate::command::unique_temp_dir("graph-protected");
    let entry = ".github/workflows/pr_linux.yml";
    source(
        &root,
        entry,
        "jobs:\n  lane:\n    uses: Mesh-LLM/mesh-llm/.github/workflows/ci-linux-lane.yml@main\n  cache:\n    cache-dependency-path: scripts/requirements.py\n",
    )?;
    let paths = vec![entry.to_owned()];
    let observed = scan::scan_paths(&root, &paths)?;
    let graph = report(&root, &paths, &observed, &BTreeSet::new(), &[entry])?;
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.trust_revision == "protected_main_external"
                && edge.unresolved_reason.is_some())
    );
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.child.as_deref() == Some("scripts/requirements.py"))
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn graph_keeps_interpolated_local_child_unresolved() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("graph-interpolated");
    let entry = ".github/workflows/main_linux.yml";
    source(
        &root,
        entry,
        "jobs:\n  lane:\n    uses: ./.github/workflows/${{ inputs.lane }}.yml\n",
    )?;
    let paths = vec![entry.to_owned()];
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &[entry])?;
    assert!(graph.edges.iter().any(|edge| {
        edge.status == "unknown_selection"
            && edge.child.is_none()
            && edge
                .unresolved_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("unresolved workflow/action target"))
    }));
    fs::remove_dir_all(root)?;
    Ok(())
}
