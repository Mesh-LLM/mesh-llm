use super::*;
use crate::ci_governance::operations::preparation_fixtures::{ID, fixture};

struct Checkout {
    _directory: tempfile::TempDir,
    source: RecordSource,
    selector: RecordSelector,
    sha: String,
}

fn checkout() -> Checkout {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let source = RecordSource::new(&root, Cancellation::default()).unwrap();
    let (policy, expected, intent, _) = fixture();
    std::fs::create_dir_all(root.join("ci/governance/records")).unwrap();
    std::fs::create_dir(root.join("empty-hooks")).unwrap();
    let bytes = serde_json::to_vec_pretty(&intent).unwrap();
    std::fs::write(
        root.join(format!("ci/governance/records/{ID}.json")),
        &bytes,
    )
    .unwrap();
    std::fs::write(
        root.join("ci/governance/policy.json"),
        serde_json::to_vec(&policy).unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("ci/governance/expected-state.json"),
        serde_json::to_vec(&expected).unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("ci/governance/activation.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version":1,"repository":"Mesh-LLM/mesh-llm","active":false,
            "repository_id":null,"reviewer_team_id":null,"administration_environment_id":null,
            "observer_app_id":null,"observer_installation_id":null,"publisher_app_id":null,
        "publisher_installation_id":null,"native_ruleset_ids":[],"human_ids":{"micspiral":null,"ndizazzo":null,"i386":null}
        }))
        .unwrap(),
    )
    .unwrap();
    source.git(&["init", "-b", "main"]).unwrap();
    source
        .git(&["config", "user.name", "Governance fixture"])
        .unwrap();
    source
        .git(&["config", "user.email", "fixture@example.invalid"])
        .unwrap();
    source.git(&["config", "core.autocrlf", "false"]).unwrap();
    source
        .git(&[
            "config",
            "core.hooksPath",
            root.join("empty-hooks").to_str().unwrap(),
        ])
        .unwrap();
    source.git(&["add", "--", "ci"]).unwrap();
    source
        .git(&["commit", "-m", "fixture: immutable protected intent"])
        .unwrap();
    let sha = String::from_utf8(source.git(&["rev-parse", "HEAD"]).unwrap())
        .unwrap()
        .trim()
        .to_owned();
    Checkout {
        _directory: directory,
        source,
        selector: RecordSelector {
            id: ID.into(),
            raw_sha256: hex::encode(Sha256::digest(bytes)),
        },
        sha,
    }
}

#[test]
fn mutable_worktree_activation_cannot_replace_inactive_protected_anchors() {
    let checkout = checkout();
    let path = checkout.source.root.join("ci/governance/activation.json");
    std::fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({
            "schema_version":1,"repository":"Mesh-LLM/mesh-llm","active":true,
            "repository_id":1,"reviewer_team_id":2,"administration_environment_id":3,
            "observer_app_id":4,"observer_installation_id":5,"publisher_app_id":4,
        "publisher_installation_id":5,"native_ruleset_ids":[8],"human_ids":{"micspiral":101,"ndizazzo":102,"i386":103}
        }))
        .unwrap(),
    )
    .unwrap();
    let protected = checkout.source.activation_at(&checkout.sha).unwrap();
    assert!(!protected.active);
    assert!(protected.anchors().is_err());
}

#[test]
fn exact_current_main_blob_and_raw_byte_digest_are_required() {
    let mut checkout = checkout();
    let result = checkout
        .source
        .resolve(&checkout.selector, &checkout.sha)
        .unwrap();
    assert_eq!(result.intent.operation_id, ID);
    assert_eq!(result.protected_main_revision, checkout.sha);
    std::fs::write(
        checkout.source.root.join(checkout.selector.path()),
        b"unreviewed working-tree replacement",
    )
    .unwrap();
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &checkout.sha)
            .is_ok()
    );
    checkout.selector.raw_sha256 = "0".repeat(64);
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &checkout.sha)
            .is_err()
    );
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &"b".repeat(40))
            .unwrap_err()
            .to_string()
            .contains("exact current")
    );
}

#[test]
fn selectors_cannot_choose_paths_refs_or_uuid_aliases() {
    for id in [
        "../policy",
        "/tmp/intent",
        "main",
        "BA2E26E4-76A4-462F-894B-1A33D80D01AA",
        "ba2e26e476a4462f894b1a33d80d01aa",
    ] {
        assert!(
            RecordSelector {
                id: id.into(),
                raw_sha256: "a".repeat(64)
            }
            .validate()
            .is_err()
        );
    }
    assert!(
        RecordSelector {
            id: ID.into(),
            raw_sha256: "A".repeat(64)
        }
        .validate()
        .is_err()
    );
}

#[test]
fn executable_git_mode_and_oversized_blob_are_refused() {
    let mut checkout = checkout();
    let path = checkout.selector.path();
    checkout
        .source
        .git(&["update-index", "--chmod=+x", "--", &path])
        .unwrap();
    checkout
        .source
        .git(&["commit", "-m", "fixture: executable record mode"])
        .unwrap();
    checkout.sha = String::from_utf8(checkout.source.git(&["rev-parse", "HEAD"]).unwrap())
        .unwrap()
        .trim()
        .into();
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &checkout.sha)
            .is_err()
    );
    checkout
        .source
        .git(&["update-index", "--chmod=-x", "--", &path])
        .unwrap();
    std::fs::write(checkout.source.root.join(&path), vec![b'x'; LIMIT + 1]).unwrap();
    checkout.source.git(&["add", "--", &path]).unwrap();
    checkout
        .source
        .git(&["commit", "-m", "fixture: oversized record"])
        .unwrap();
    checkout.sha = String::from_utf8(checkout.source.git(&["rev-parse", "HEAD"]).unwrap())
        .unwrap()
        .trim()
        .into();
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &checkout.sha)
            .unwrap_err()
            .to_string()
            .contains("1 MiB")
    );
}

#[cfg(unix)]
#[test]
fn symlink_record_and_ancestor_are_refused_without_following_them() {
    use std::os::unix::fs::symlink;
    let checkout = checkout();
    let path = checkout.source.root.join(checkout.selector.path());
    let retained = checkout.source.root.join("retained-record");
    std::fs::rename(&path, &retained).unwrap();
    symlink(&retained, &path).unwrap();
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &checkout.sha)
            .is_err()
    );
    std::fs::remove_file(&path).unwrap();
    std::fs::rename(retained, &path).unwrap();
    let directory = checkout.source.root.join("ci/governance/records");
    let retained = checkout.source.root.join("retained-records");
    std::fs::rename(&directory, &retained).unwrap();
    symlink(retained, directory).unwrap();
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &checkout.sha)
            .is_err()
    );
}

#[test]
fn local_replace_objects_cannot_change_the_reviewed_blob() {
    let checkout = checkout();
    let path = checkout.selector.path();
    let original = String::from_utf8(
        checkout
            .source
            .git(&["rev-parse", &format!("{}:{path}", checkout.sha)])
            .unwrap(),
    )
    .unwrap()
    .trim()
    .to_owned();
    let alternate = checkout.source.root.join("alternate");
    std::fs::write(&alternate, "unreviewed replacement blob").unwrap();
    let replacement = String::from_utf8(
        checkout
            .source
            .git(&["hash-object", "-w", alternate.to_str().unwrap()])
            .unwrap(),
    )
    .unwrap()
    .trim()
    .to_owned();
    checkout
        .source
        .git(&["replace", &original, &replacement])
        .unwrap();
    assert!(
        checkout
            .source
            .resolve(&checkout.selector, &checkout.sha)
            .is_ok()
    );
}
