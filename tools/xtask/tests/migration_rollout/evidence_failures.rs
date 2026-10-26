use super::fixture::Fixture;
use super::rollout::Issue;
use serde_json::json;
use std::fs;

#[test]
fn migration_rollout_missing_predecessor_reference_is_rejected() {
    let fixture = Fixture::new();
    fixture.change(|input| {
        input["predecessors"].as_array_mut().unwrap().pop();
    });
    fixture.rejected(Issue::Predecessor);
}

#[test]
fn migration_rollout_duplicate_predecessor_does_not_cover_missing_task() {
    let fixture = Fixture::new();
    fixture.change(|input| input["predecessors"][14]["task"] = json!(11));
    fixture.rejected(Issue::Predecessor);
}

#[test]
fn migration_rollout_missing_predecessor_file_is_rejected() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.root.join("task-21.txt")).unwrap();
    fixture.rejected(Issue::Evidence);
}

#[test]
fn migration_rollout_mismatched_predecessor_bytes_are_rejected() {
    let fixture = Fixture::new();
    fs::write(
        fixture.root.join("task-19.txt"),
        b"different platform evidence",
    )
    .unwrap();
    fixture.rejected(Issue::Evidence);
}

#[test]
fn migration_rollout_empty_predecessor_is_not_available_evidence() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("task-20.txt"), b"").unwrap();
    fixture.change(|input| input["predecessors"][9]["evidence"] = fixture.binding("task-20.txt"));
    fixture.rejected(Issue::Predecessor);
}

#[test]
fn migration_rollout_stale_binary_bytes_are_rejected_before_execution() {
    let fixture = Fixture::new();
    fixture.change(|input| input["bootstrap"]["binary"]["sha256"] = json!("0".repeat(64)));
    fixture.rejected(Issue::Evidence);
}

#[test]
fn migration_rollout_bootstrap_report_cannot_select_another_binary() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("another-binary"), b"another inert binary").unwrap();
    fixture.change(|input| input["bootstrap"]["binary"] = fixture.binding("another-binary"));
    fixture.rejected(Issue::Bootstrap);
}

#[test]
fn migration_rollout_duplicate_bootstrap_output_is_rejected() {
    let fixture = Fixture::new();
    let path = fixture.root.join("bootstrap.txt");
    let mut bytes = fs::read(&path).unwrap();
    bytes.extend_from_slice(b"host=other-host\n");
    fs::write(path, bytes).unwrap();
    fixture.change(|input| input["bootstrap"]["report"] = fixture.binding("bootstrap.txt"));
    fixture.rejected(Issue::Bootstrap);
}

#[test]
fn migration_rollout_local_observations_cannot_supply_authorization() {
    let fixture = Fixture::new();
    fixture.change(|input| input["authorized"] = json!(true));
    fixture.rejected(Issue::Input);
}
