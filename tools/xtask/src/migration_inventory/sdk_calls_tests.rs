use super::*;
use crate::migration_inventory::{required_closure, required_graph, scan};
use std::collections::BTreeSet;
fn write(root: &Path, path: &str, bytes: impl AsRef<[u8]>) {
    let path = root.join(path);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}
fn fixture(
    client: &str,
) -> (
    tempfile::TempDir,
    Vec<String>,
    BTreeSet<String>,
    &'static str,
) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let (caller, child) = match client {
        "openai" => (
            "scripts/ci-compat-smoke.sh",
            "scripts/ci-openai-python-smoke.py",
        ),
        "litellm" => ("scripts/ci-compat-smoke.sh", "scripts/ci-litellm-smoke.py"),
        "langchain" => (
            "scripts/ci-compat-smoke.sh",
            "scripts/ci-langchain-openai-smoke.py",
        ),
        _ => (
            "scripts/skippy-workload-certify.sh",
            "scripts/ci-openai-embeddings-smoke.py",
        ),
    };
    let text = if client == "embeddings" {
        format!(
            "\"${{workload_automation[@]}}\" automation smoke-observation sdk-client \\\n  --client {client} --python \"$SDK_PYTHON\" \\\n  --base-url \"$BASE_URL\"\n"
        )
    } else {
        format!(
            "\"${{automation[@]}}\" automation smoke-observation sdk-client --client {client} --python \"$SDK_PYTHON\" --base-url \"$BASE_URL\"\n"
        )
    };
    write(root, caller, &text);
    write(root, child, "pass\n");
    let repo = crate::repo_consistency::repo_root().unwrap();
    write(root, OWNER, fs::read(repo.join(OWNER)).unwrap());
    let paths = vec![caller.into(), child.into(), OWNER.into()];
    let observed = scan::scan_paths(root, &paths).unwrap();
    let owned = observed
        .iter()
        .map(|r| r.id.clone())
        .collect::<BTreeSet<_>>();
    let candidate = observed
        .iter()
        .find(|r| r.source_block.contains("--client "))
        .unwrap();
    let line = text
        .lines()
        .position(|l| l.trim() == candidate.source_block)
        .unwrap()
        + 1;
    let id = candidate.id.split(':').collect::<Vec<_>>();
    write(root,"ci/automation-migration/script-edges.json",serde_json::to_vec(&serde_json::json!({"groups":[{"file":caller,"members":[[line,id[1],1,"script",child,"Actual closed SDK -I child through native owner"]]}]})).unwrap());
    write(root,"ci/automation-migration/python-inventory.json",serde_json::to_vec(&serde_json::json!({"files":[{"path":child,"replacement_owner":"retained SDK compatibility","deletion_condition":"Retain actual SDK behavior, qualification pending"}]})).unwrap());
    (temp, paths, owned, caller)
}
#[test]
fn sdk_closed_adapter_graph_and_closure_reach_each_actual_client() {
    for client in ["openai", "litellm", "langchain", "embeddings"] {
        let (temp, paths, owned, caller) = fixture(client);
        let observed = scan::scan_paths(temp.path(), &paths).unwrap();
        let graph =
            required_graph::report(temp.path(), &paths, &observed, &owned, &[caller]).unwrap();
        assert!(graph.edges.iter().any(|e| e.parent == caller
            && e.child.as_deref() == Some(paths[1].as_str())
            && e.status == "reached_execution"
            && e.contract_source.is_some()));
        required_closure::check_required_closure(temp.path(), &paths, &observed, &owned, &[caller])
            .unwrap();
        let missing = vec![paths[0].clone(), paths[2].clone()];
        assert!(
            required_graph::report(temp.path(), &missing, &observed, &owned, &[caller]).is_err()
        );
        assert!(
            required_closure::check_required_closure(
                temp.path(),
                &missing,
                &observed,
                &owned,
                &[caller]
            )
            .is_err()
        );
        temp.close().unwrap();
    }
}
#[test]
fn sdk_closed_adapter_refuses_unknown_client_and_changed_native_roster() {
    let (temp, paths, owned, caller) = fixture("litellm");
    let original = fs::read_to_string(temp.path().join(caller)).unwrap();
    for changed in [
        original.replace("--client litellm", "--client foreign"),
        original.replace("--python \"$SDK_PYTHON\"", "--python foreign"),
    ] {
        write(temp.path(), caller, &changed);
        let observed = scan::scan_paths(temp.path(), &paths).unwrap();
        assert!(required_graph::report(temp.path(), &paths, &observed, &owned, &[caller]).is_err());
        assert!(
            required_closure::check_required_closure(
                temp.path(),
                &paths,
                &observed,
                &owned,
                &[caller]
            )
            .is_err()
        );
    }
    write(temp.path(), caller, original);
    let source = fs::read_to_string(temp.path().join(OWNER)).unwrap();
    write(
        temp.path(),
        OWNER,
        source.replace("ci-litellm-smoke.py", "different-client.py"),
    );
    let observed = scan::scan_paths(temp.path(), &paths).unwrap();
    assert!(required_graph::report(temp.path(), &paths, &observed, &owned, &[caller]).is_err());
    assert!(
        required_closure::check_required_closure(temp.path(), &paths, &observed, &owned, &[caller])
            .is_err()
    );
    temp.close().unwrap();
}
#[test]
fn sdk_closed_adapter_unowned_contract_does_not_hide_required_python() {
    let (temp, paths, _, caller) = fixture("openai");
    let observed = scan::scan_paths(temp.path(), &paths).unwrap();
    let graph = required_graph::report(temp.path(), &paths, &observed, &BTreeSet::new(), &[caller])
        .unwrap();
    assert!(
        graph
            .edges
            .iter()
            .any(|e| e.child.as_deref() == Some(paths[1].as_str())
                && e.status == "unknown_selection"
                && e.unresolved_reason.is_some())
    );
    assert!(
        required_closure::check_required_closure(
            temp.path(),
            &paths,
            &observed,
            &BTreeSet::new(),
            &[caller]
        )
        .is_err()
    );
    temp.close().unwrap();
}

#[test]
fn checked_in_sdk_callers_have_exact_required_children_and_process_contract()
-> crate::command::DynResult<()> {
    use crate::migration_inventory::{ledger, selected_process, shards};
    let root = crate::repo_consistency::repo_root()?;
    let paths = ledger::tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?;
    let validated = shards::check_shards(&root, &observed)?;
    let callers = [
        "scripts/ci-compat-smoke.sh",
        "scripts/skippy-workload-certify.sh",
    ];
    let graph = required_graph::report(&root, &paths, &observed, &validated, &callers)?;
    let children = [
        "scripts/ci-openai-python-smoke.py",
        "scripts/ci-litellm-smoke.py",
        "scripts/ci-langchain-openai-smoke.py",
        "scripts/ci-openai-embeddings-smoke.py",
    ];
    for child in children {
        let edges = graph
            .edges
            .iter()
            .filter(|edge| {
                callers.contains(&edge.parent.as_str()) && edge.child.as_deref() == Some(child)
            })
            .collect::<Vec<_>>();
        assert_eq!(edges.len(), 1, "{child}");
        assert_eq!(edges[0].status, "reached_execution", "{child}");
        assert!(edges[0].contract_source.is_some(), "{child}");
    }
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.source_block.contains("sdk-client")
                && edge.status == "unknown_selection")
    );
    required_closure::check_required_closure(&root, &paths, &observed, &validated, &callers)?;
    let records = ledger::MigrationLedgers::load(&root)?
        .invocations
        .selected_process_calls;
    selected_process::check_selected_processes(&root, &records)?;
    let sdk = records
        .iter()
        .find(|record| record.caller == callers[1] && record.source_block.contains("sdk-client"))
        .ok_or("SDK process record missing")?;
    for changed in ["scripts/ci-litellm-smoke.py", "scripts/foreign-sdk.py"] {
        let mut changed_records = records.clone();
        let row = changed_records
            .iter_mut()
            .find(|record| record.caller == sdk.caller && record.line == sdk.line)
            .ok_or("SDK record missing")?;
        row.child = changed.into();
        assert!(selected_process::check_selected_processes(&root, &changed_records).is_err());
    }
    let mut unowned = validated.clone();
    for row in &observed {
        if callers.contains(&row.path.as_str()) && row.source_block.contains("--client ") {
            unowned.remove(&row.id);
        }
    }
    let refused = required_graph::report(&root, &paths, &observed, &unowned, &callers)?;
    for child in children {
        assert!(
            refused
                .edges
                .iter()
                .any(|edge| edge.child.as_deref() == Some(child)
                    && edge.status == "unknown_selection"),
            "{child}"
        );
    }
    assert!(
        required_closure::check_required_closure(&root, &paths, &observed, &unowned, &callers)
            .is_err()
    );
    Ok(())
}
