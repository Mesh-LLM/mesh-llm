use super::fixture::Fixture;
use super::rollout::{self, Issue};
use serde_json::json;
use std::fs;

#[test]
fn migration_rollout_unmerged_catalog_is_not_a_protected_predecessor() {
    let fixture = Fixture::new();
    fixture.write_repo("unmerged.txt", b"candidate-only prerequisite");
    let unmerged = fixture.commit("unmerged catalog claim");
    fixture.change(|input| input["catalog_sha"] = json!(unmerged));

    fixture.rejected(Issue::Sequence);
}

#[test]
fn migration_rollout_source_must_include_catalog_predecessor() {
    let fixture = Fixture::new();
    fixture.git(&["checkout", "--orphan", "unrebased"]);
    fixture.write_repo("unrebased.txt", b"same catalog bytes without predecessor");
    let source = fixture.commit("unrebased source");
    fixture.change(|input| input["source_sha"] = json!(source));

    fixture.rejected(Issue::Sequence);
}

#[test]
fn migration_rollout_source_catalog_whitespace_is_not_equivalent() {
    let fixture = Fixture::new();
    let path = fixture.repo.join("ci/slices.yml");
    let mut bytes = fs::read(&path).unwrap();
    bytes.push(b'\n');
    fs::write(path, bytes).unwrap();
    let source = fixture.commit("source catalog drift");
    fixture.change(|input| input["source_sha"] = json!(source));

    fixture.rejected(Issue::Catalog);
}

#[test]
fn migration_rollout_executable_catalog_blob_is_rejected() {
    let fixture = Fixture::new();
    fixture.git(&["update-index", "--chmod=+x", "ci/ownership.yml"]);
    fixture.git(&["commit", "-q", "-m", "catalog mode"]);
    let source = fixture.git(&["rev-parse", "HEAD"]);
    fixture.change(|input| input["source_sha"] = json!(source));

    fixture.rejected(Issue::Evidence);
}

#[test]
fn migration_rollout_branch_owned_planner_is_rejected() {
    let fixture = Fixture::new();
    fixture.change(|input| input["planner_sha"] = json!(fixture.source));

    fixture.rejected(Issue::Authority);
}

#[test]
fn migration_rollout_branch_owned_workspace_discovery_is_rejected() {
    let fixture = Fixture::new();
    fixture.change(|input| input["workspace_sha"] = json!(fixture.source));

    fixture.rejected(Issue::Authority);
}

#[test]
fn migration_rollout_stale_bootstrap_revision_is_rejected() {
    let fixture = Fixture::new();
    fixture.change(|input| input["bootstrap"]["source_sha"] = json!(fixture.catalog));

    fixture.rejected(Issue::Authority);
}

#[test]
fn migration_rollout_missing_protected_bootstrap_is_rejected() {
    let fixture = Fixture::new();
    fixture.change(|input| input["support_sha"] = json!(fixture.catalog));

    fixture.rejected(Issue::Evidence);
}

#[test]
fn migration_rollout_provider_change_cannot_be_authorized_by_local_input() {
    let fixture = Fixture::new();
    fixture.write_repo(
        ".github/actions/select-ci-runners/action.yml",
        b"changed provider authority",
    );
    let source = fixture.commit("provider policy change");
    fixture.change(|input| input["source_sha"] = json!(source));

    fixture.rejected(Issue::Policy);
}

#[test]
fn migration_rollout_image_pin_change_is_outside_migration_scope() {
    let fixture = Fixture::new();
    fixture.write_repo("ci/runner-images.json", b"changed image pin");
    let source = fixture.commit("image policy change");
    fixture.change(|input| input["source_sha"] = json!(source));

    fixture.rejected(Issue::Policy);
}

#[test]
fn migration_rollout_mutable_git_name_is_not_a_revision() {
    let fixture = Fixture::new();
    fixture.change(|input| input["protected_sha"] = json!("main"));

    fixture.rejected(Issue::Input);
}

#[test]
fn migration_rollout_worktree_bytes_do_not_replace_committed_catalogs() {
    let fixture = Fixture::new();
    fixture.write_repo(
        "ci/slices.yml",
        b"uncommitted and irrelevant to selected objects",
    );

    let result = rollout::validate_file(&fixture.input);

    assert!(result.is_ok(), "{result:?}");
}
