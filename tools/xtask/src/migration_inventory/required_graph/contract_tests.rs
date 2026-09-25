use super::report;
use crate::command::DynResult;
use crate::migration_inventory::{ledger, scan};
use std::collections::BTreeSet;
use std::fs;

#[test]
fn actionlint_extractor_joins_checked_in_invocation_contract() -> DynResult<()> {
    let root = crate::repository::RepositoryRoot::resolve(None)?;
    let paths = [
        ".github/actions/install-actionlint/action.yml".to_owned(),
        "scripts/safe-extract-tar.py".to_owned(),
    ];
    let observed = scan::scan_paths(root.as_path(), &paths)?;
    let validated = observed.iter().map(|row| row.id.clone()).collect();
    let graph = report(root.as_path(), &paths, &observed, &validated, &[&paths[0]])?;
    let edge = graph
        .edges
        .iter()
        .find(|edge| edge.child.as_deref() == Some("scripts/safe-extract-tar.py"))
        .ok_or("missing extractor call")?;
    assert_eq!(edge.unresolved_reason, None);
    assert!(
        edge.argv
            .as_deref()
            .is_some_and(|argv| argv.contains("$archive $install_dir"))
    );
    assert!(
        edge.status_streams_effects
            .as_deref()
            .is_some_and(|effects| effects.contains("nonzero fails"))
    );
    Ok(())
}

#[test]
fn composer_selected_child_joins_script_identity_and_owner() -> DynResult<()> {
    let root = crate::repository::RepositoryRoot::resolve(None)?;
    let paths = ledger::tracked_paths(root.as_path())?;
    let observed = scan::scan_paths(root.as_path(), &paths)?;
    let validated = observed.iter().map(|row| row.id.clone()).collect();
    let graph = report(
        root.as_path(),
        &paths,
        &observed,
        &validated,
        &["scripts/ci-compose-product-input.sh"],
    )?;
    let edge = graph
        .edges
        .iter()
        .find(|edge| {
            edge.parent == "scripts/ci-compose-product-input.sh"
                && edge.line == 192
                && edge.child.as_deref() == Some("scripts/safe-extract-tar.py")
        })
        .ok_or("missing composer extraction")?;
    assert_eq!(edge.unresolved_reason, None);
    assert!(
        edge.status_streams_effects
            .as_deref()
            .is_some_and(|effects| effects.contains("replacement owner"))
    );
    Ok(())
}

#[test]
fn reached_source_children_have_physical_contracts() -> DynResult<()> {
    // Given the checked-in required graph, when each reached same-commit Python child
    // is inspected, then none of the 22 source-backed calls lacks a contract.
    let root = crate::repository::RepositoryRoot::resolve(None)?;
    let paths = ledger::tracked_paths(root.as_path())?;
    let observed = scan::scan_paths(root.as_path(), &paths)?;
    let validated = observed.iter().map(|row| row.id.clone()).collect();
    let graph = report(
        root.as_path(),
        &paths,
        &observed,
        &validated,
        &[
            ".github/workflows/main_quality.yml",
            ".github/workflows/main_website.yml",
            ".github/workflows/main_linux.yml",
            ".github/workflows/main_macos.yml",
            ".github/workflows/main_windows.yml",
            "Justfile",
            "just/ci.just",
        ],
    )?;
    let missing: Vec<_> = graph
        .edges
        .iter()
        .filter(|edge| {
            edge.unresolved_reason.as_deref().is_some_and(|reason| {
                reason.starts_with("missing source-backed interpreter contract")
            })
        })
        .map(|edge| format!("{}:{} -> {:?}", edge.parent, edge.line, edge.child))
        .collect();
    assert!(missing.is_empty(), "{missing:?}");
    Ok(())
}

#[test]
fn child_contract_rejects_stale_wrong_and_unowned_physical_calls() -> DynResult<()> {
    use super::contracts::Contracts;
    let root = crate::command::unique_temp_dir("graph-child-contract");
    let path = ".github/actions/prepare/action.yml";
    let child = "scripts/plan-ci.py";
    let block = "python3 scripts/plan-ci.py --manifest-root root";
    super::super::required_graph_tests::source(
        &root,
        path,
        &format!("run: |\n  {block}\n  {block}\n"),
    )?;
    let record = |identity: String, source: &str, target: &str| {
        serde_json::json!({
            "id": identity, "source_block": source, "disposition": "execution",
            "argv": block, "status_output_effects": "inherited streams; nonzero propagates",
            "transitive_boundary": target, "replacement": "tools/xtask planner",
            "deletion_condition": "after planner parity", "reason": "actual child execution"
        })
    };
    let ledger = root.join("ci/automation-migration/invocations.json");
    fs::create_dir_all(ledger.parent().ok_or("missing ledger parent")?)?;
    let observed = scan::scan_paths(&root, &[path.to_owned()])?;
    let ids: Vec<_> = observed
        .iter()
        .filter(|row| row.source_block == block)
        .map(|row| row.id.clone())
        .collect();
    assert_eq!(ids.len(), 2);
    let validated: BTreeSet<_> = observed.iter().map(|row| row.id.clone()).collect();
    for (first, second, expected) in [
        (
            record(ids[0].clone(), "stale", child),
            record(ids[1].clone(), block, child),
            false,
        ),
        (
            record(ids[0].clone(), block, "scripts/plan-ci.py-backup"),
            record(ids[1].clone(), block, child),
            false,
        ),
        (
            record(ids[0].clone(), block, child),
            record(ids[1].clone(), block, child),
            true,
        ),
    ] {
        fs::write(
            &ledger,
            serde_json::to_vec(&serde_json::json!({"github_source_records": [first, second]}))?,
        )?;
        let contracts = Contracts::load(&root)?;
        assert_eq!(
            contracts
                .for_call(path, 2, block, child, &observed, &validated)
                .is_some(),
            expected
        );
    }
    fs::write(
        &ledger,
        serde_json::to_vec(
            &serde_json::json!({"github_source_records": [record(ids[0].clone(), block, child)]}),
        )?,
    )?;
    let contracts = Contracts::load(&root)?;
    assert!(
        contracts
            .for_call(path, 3, block, child, &observed, &validated)
            .is_none()
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
