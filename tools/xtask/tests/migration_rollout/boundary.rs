use super::fixture::Fixture;
use super::rollout::Issue;
use serde_json::json;
use std::fs;

#[test]
fn migration_rollout_branch_owned_runner_authority_is_rejected() {
    let fixture = Fixture::new();
    fixture.change(|input| input["runner_policy_sha"] = json!(fixture.source));
    fixture.rejected(Issue::Authority);
}

#[test]
fn migration_rollout_bootstrap_report_must_be_available() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.root.join("bootstrap.txt")).unwrap();
    fixture.rejected(Issue::Evidence);
}

#[test]
fn migration_rollout_bootstrap_missing_binary_path_is_rejected() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("bootstrap.txt"), b"host=fixture\n").unwrap();
    fixture.change(|input| input["bootstrap"]["report"] = fixture.binding("bootstrap.txt"));
    fixture.rejected(Issue::Bootstrap);
}

#[test]
fn migration_rollout_malformed_digest_is_rejected_at_input_boundary() {
    let fixture = Fixture::new();
    fixture.change(|input| input["bootstrap"]["binary"]["sha256"] = json!("not-a-hash"));
    fixture.rejected(Issue::Input);
}

#[test]
fn migration_rollout_legacy_shadow_rows_are_not_accepted() {
    let fixture = Fixture::new();
    fixture.change(|input| input["shadow"] = json!({"legacy": {"enabled": true}}));
    fixture.rejected(Issue::Input);
}

#[cfg(unix)]
#[test]
fn migration_rollout_symlink_catalog_is_not_a_regular_git_blob() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.repo.join("ci/ownership.yml")).unwrap();
    std::os::unix::fs::symlink("slices.yml", fixture.repo.join("ci/ownership.yml")).unwrap();
    let source = fixture.commit("symlink catalog");
    fixture.change(|input| input["source_sha"] = json!(source));
    fixture.rejected(Issue::Evidence);
}
