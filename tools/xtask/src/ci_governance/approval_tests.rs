use super::approval::{Account, AccountKind, PrSnapshot, ReviewState, evaluate_at};
use super::policy::{self, Policy};
use super::scope::{ChangeKind, ChangedPath};
use serde_json::{Value, json};

const HEAD: &str = "2222222222222222222222222222222222222222";
const BASE: &str = "1111111111111111111111111111111111111111";
const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn account(login: &str, id: u64) -> Account {
    Account {
        login: login.into(),
        id,
        kind: AccountKind::Human,
    }
}

fn snapshot() -> PrSnapshot {
    serde_json::from_value(json!({
        "schema_version": 1,
        "collection": {"source":"github-api", "observed_at":200,
            "files_complete":true, "reviews_complete":true,"team_complete":true,"truncated":false},
        "repository":"Mesh-LLM/mesh-llm", "base_ref":"main", "base_sha":BASE,
        "current_protected_sha":BASE,
        "head_sha":HEAD, "current_head_sha":HEAD, "policy_authority":"protected-base",
        "policy_revision":BASE, "policy_sha256":DIGEST, "pr_number":2781,
        "author":account("contributor", 4), "sponsor":account("contributor", 4),
        "changed_file_count":1,
        "changes":[{"kind":"modified", "path":"ci/governance/policy.json", "previous_path":null}],
        "reviews":[{"id":10, "reviewer":account("micspiral", 1), "state":"approved",
            "commit_sha":HEAD,"submitted_at":100}],
        "team":{"slug":"Mesh-LLM/ci-policy-reviewers", "visible":true,
            "repository":"Mesh-LLM/mesh-llm", "repository_permission":"write",
            "members":[account("micspiral",1),account("ndizazzo",2),account("i386",3)]}
    }))
    .unwrap()
}

fn allowed(policy: &Policy, snapshot: &PrSnapshot) -> bool {
    evaluate_at(policy, snapshot, DIGEST, 200).is_ok()
}

#[test]
fn each_eligible_human_alone_is_sufficient() {
    for (index, login) in policy::REVIEWERS.iter().enumerate() {
        let mut input = snapshot();
        input.reviews[0].reviewer = account(login, index as u64 + 1);
        let result = evaluate_at(&policy::fixture(), &input, DIGEST, 200).unwrap();
        assert_eq!(result.independent_approvers, [*login]);
    }
}

#[test]
fn author_and_sponsor_cannot_approve() {
    let mut input = snapshot();
    input.author = input.reviews[0].reviewer.clone();
    input.sponsor = input.author.clone();
    assert!(!allowed(&policy::fixture(), &input));
    input.author = account("contributor", 4);
    // A distinct asserted sponsor is itself unsupported authenticated evidence.
    assert!(!allowed(&policy::fixture(), &input));
}

#[test]
fn bot_authored_policy_requires_human_submission() {
    let mut input = snapshot();
    input.author.kind = AccountKind::Bot;
    assert!(!allowed(&policy::fixture(), &input));
    input.sponsor = input.author.clone();
    assert!(!allowed(&policy::fixture(), &input));
}

#[test]
fn missing_unlisted_bot_and_impersonated_approvals_fail() {
    let baseline = snapshot();
    let mut missing = baseline.clone();
    missing.reviews.clear();
    assert!(!allowed(&policy::fixture(), &missing));
    for reviewer in [
        account("unlisted", 5),
        account("micspiral", 99),
        Account {
            kind: AccountKind::Bot,
            ..account("micspiral", 1)
        },
    ] {
        let mut input = baseline.clone();
        input.reviews[0].reviewer = reviewer;
        assert!(!allowed(&policy::fixture(), &input));
    }
}

#[test]
fn stale_dismissed_and_rejected_reviews_fail() {
    let mut input = snapshot();
    input.reviews[0].commit_sha = Some(BASE.into());
    assert!(!allowed(&policy::fixture(), &input));
    for state in [
        ReviewState::Dismissed,
        ReviewState::ChangesRequested,
        ReviewState::Pending,
        ReviewState::Commented,
    ] {
        let mut input = snapshot();
        input.reviews[0].state = state;
        assert!(!allowed(&policy::fixture(), &input));
    }
}

#[test]
fn latest_effective_review_is_used_independently_of_input_order() {
    for state in [
        ReviewState::Dismissed,
        ReviewState::ChangesRequested,
        ReviewState::Approved,
    ] {
        let mut input = snapshot();
        let mut newer = input.reviews[0].clone();
        newer.id += 1;
        newer.submitted_at = Some(101);
        newer.state = state;
        if state == ReviewState::Approved {
            newer.commit_sha = Some(BASE.into());
        }
        input.reviews.insert(0, newer);
        assert!(!allowed(&policy::fixture(), &input));
    }
}

#[test]
fn comment_does_not_undo_effective_approval_and_fresh_reapproval_passes() {
    let mut input = snapshot();
    let mut later = input.reviews[0].clone();
    later.id = 11;
    later.submitted_at = Some(101);
    later.state = ReviewState::Commented;
    input.reviews.push(later);
    assert!(allowed(&policy::fixture(), &input));
    input.reviews[0].state = ReviewState::Dismissed;
    input.reviews[1].state = ReviewState::Approved;
    assert!(allowed(&policy::fixture(), &input));
}

#[test]
fn team_drift_fails_closed() {
    let baseline = snapshot();
    for variant in 0..6 {
        let mut input = baseline.clone();
        let team = input.team.as_mut().unwrap();
        match variant {
            0 => {
                team.members.pop();
            }
            1 => team.members.push(account("extra", 5)),
            2 => team.members[0].kind = AccountKind::Bot,
            3 => team.visible = false,
            4 => team.repository_permission = "read".into(),
            _ => team.members[1] = team.members[0].clone(),
        }
        assert!(!allowed(&policy::fixture(), &input), "variant {variant}");
    }
}

#[test]
fn incomplete_or_foreign_metadata_fails_closed() {
    let original = serde_json::to_value(snapshot()).unwrap();
    for (pointer, replacement) in [
        ("/collection/files_complete", json!(false)),
        ("/collection/reviews_complete", json!(false)),
        ("/collection/team_complete", json!(false)),
        ("/collection/truncated", json!(true)),
        ("/repository", json!("other/repo")),
        ("/base_ref", json!("feature")),
        ("/policy_revision", json!(HEAD)),
        ("/current_protected_sha", json!(HEAD)),
        ("/policy_sha256", json!("b".repeat(64))),
        ("/current_head_sha", json!(BASE)),
        ("/changed_file_count", json!(2)),
        ("/team", Value::Null),
        ("/reviews/0/id", json!(0)),
        ("/reviews/0/submitted_at", json!(300)),
        ("/reviews/0/commit_sha", json!("short")),
    ] {
        let mut value = original.clone();
        *value.pointer_mut(pointer).unwrap() = replacement;
        let input = serde_json::from_value(value).unwrap();
        assert!(!allowed(&policy::fixture(), &input), "pointer {pointer}");
    }
}

#[test]
fn coherent_historical_base_policy_is_rejected_by_current_main_observation() {
    let mut input = snapshot();
    input.current_protected_sha = HEAD.into();
    assert!(!allowed(&policy::fixture(), &input));
}

#[test]
fn conflicting_latest_same_second_reviews_do_not_use_creation_id_as_submission_order() {
    for reverse in [false, true] {
        let mut input = snapshot();
        let mut conflicting = input.reviews[0].clone();
        conflicting.id = 1;
        conflicting.state = ReviewState::ChangesRequested;
        input.reviews.push(conflicting);
        if reverse {
            input.reviews.reverse();
        }
        assert!(!allowed(&policy::fixture(), &input));
        let mut fresh = input.reviews[0].clone();
        fresh.id = 12;
        fresh.state = ReviewState::Approved;
        fresh.submitted_at = Some(101);
        input.reviews.push(fresh);
        assert!(allowed(&policy::fixture(), &input));
    }
}

#[test]
fn another_reviewers_ambiguity_does_not_undo_a_clear_independent_approval() {
    for reviewer in [account("unlisted", 5), account("i386", 3)] {
        let mut input = snapshot();
        let mut other = input.reviews[0].clone();
        other.id = 11;
        other.reviewer = reviewer;
        input.reviews.push(other.clone());
        other.id = 12;
        other.state = ReviewState::ChangesRequested;
        input.reviews.push(other);
        assert!(allowed(&policy::fixture(), &input));
    }
}

#[test]
fn web_commands_and_check_selectors_require_policy_approval() {
    let policy = policy::fixture();
    for path in [
        "mesh/crates/mesh-llm-ui/package.json",
        "mesh/crates/mesh-llm-ui/eslint.config.js",
        "mesh/crates/mesh-llm-ui/vitest.config.ts",
        "mesh/crates/mesh-llm-ui/vite.config.ts",
        "mesh/crates/mesh-llm-ui/playwright.config.ts",
        "mesh/crates/mesh-llm-ui/tsconfig.json",
        "mesh/crates/mesh-llm-ui/tsconfig.app.json",
        "mesh/crates/mesh-llm-ui/tsconfig.node.json",
        "mesh/crates/mesh-llm-ui/.prettierrc",
        "mesh/crates/mesh-llm-ui/.prettierignore",
        "mesh/crates/mesh-llm-ui/.npmrc",
        "mesh/crates/mesh-llm-ui/pnpm-workspace.yaml",
        "mesh/website/package.json",
        "mesh/crates/mesh-llm-ui/scripts/run-playwright.mjs",
        "mesh/website/scripts/test-cli-explorer.mjs",
    ] {
        let mut input = snapshot();
        input.changes[0].path = path.into();
        assert!(
            evaluate_at(&policy, &input, DIGEST, 200)
                .unwrap()
                .policy_change
        );
        input.reviews.clear();
        assert!(!allowed(&policy, &input), "{path}");
    }
}

#[test]
fn nullable_pending_review_does_not_undo_independent_current_approval() {
    let mut input = snapshot();
    let mut pending = input.reviews[0].clone();
    pending.id = 11;
    pending.state = ReviewState::Pending;
    pending.commit_sha = None;
    pending.submitted_at = None;
    input.reviews.push(pending);
    assert!(allowed(&policy::fixture(), &input));
}

#[test]
fn review_count_is_bounded_and_boundary_is_supported() {
    let mut input = snapshot();
    let review = input.reviews[0].clone();
    input.reviews = (1..=10_000)
        .map(|id| super::approval::Review {
            id,
            ..review.clone()
        })
        .collect();
    assert!(allowed(&policy::fixture(), &input));
    input.reviews.push(super::approval::Review {
        id: 10_001,
        ..review
    });
    assert!(!allowed(&policy::fixture(), &input));
}

#[test]
fn unknown_sources_authority_and_missing_fields_are_rejected() {
    let mut value = serde_json::to_value(snapshot()).unwrap();
    value["collection"]["source"] = json!("pull-request-body");
    assert!(serde_json::from_value::<PrSnapshot>(value).is_err());
    let mut value = serde_json::to_value(snapshot()).unwrap();
    value["policy_authority"] = json!("pull-request-head");
    assert!(serde_json::from_value::<PrSnapshot>(value).is_err());
    let mut value = serde_json::to_value(snapshot()).unwrap();
    value.as_object_mut().unwrap().remove("current_head_sha");
    assert!(serde_json::from_value::<PrSnapshot>(value).is_err());
}

#[test]
fn conflicting_review_identity_and_duplicate_ids_fail() {
    for variant in 0..3 {
        let mut input = snapshot();
        let mut other = input.reviews[0].clone();
        if variant != 0 {
            other.id = 11;
        }
        if variant == 1 {
            other.reviewer.id = 88;
        }
        if variant == 2 {
            other.reviewer.login = "another-login".into();
        }
        input.reviews.push(other);
        assert!(!allowed(&policy::fixture(), &input));
    }
}

#[test]
fn deletion_and_either_side_of_rename_cannot_remove_the_gate() {
    for (kind, path, previous) in [
        (ChangeKind::Deleted, "ci/governance/policy.json", None),
        (
            ChangeKind::Deleted,
            "tools/xtask/src/ci_governance/mod.rs",
            None,
        ),
        (
            ChangeKind::Renamed,
            "mesh/src/product.rs",
            Some(".github/CODEOWNERS"),
        ),
        (
            ChangeKind::Renamed,
            "ci/moved-policy.json",
            Some("mesh/src/product.rs"),
        ),
    ] {
        let mut input = snapshot();
        input.reviews.clear();
        input.changes = vec![ChangedPath {
            kind,
            path: path.into(),
            previous_path: previous.map(str::to_owned),
        }];
        assert!(!allowed(&policy::fixture(), &input));
    }
}

#[test]
fn product_only_prs_do_not_need_extra_approval_or_human_submission() {
    for path in [
        "mesh/crates/mesh-llm-ui/src/app.rs",
        "skippy/crates/skippy-cache/src/lib.rs",
        "mesh/docs/SDK.md",
        "README.md",
        "mesh/src/line\nbreak.rs",
        "nested/Dockerfile.test",
        "skippy/llama_cpp/patches/native-product.patch",
        "mesh/crates/mesh-llm-ui/src/example.test.tsx",
        "mesh/crates/mesh-llm-ui/e2e/example.spec.ts",
        "mesh/website/src/docs/example.md",
    ] {
        let mut input = snapshot();
        input.changes[0].path = path.into();
        input.reviews.clear();
        input.team = None;
        input.author.kind = AccountKind::Bot;
        input.collection.reviews_complete = false;
        input.collection.team_complete = false;
        let result = evaluate_at(&policy::fixture(), &input, DIGEST, 200).unwrap();
        assert!(!result.policy_change, "{path}");
    }
}

#[test]
fn snapshot_age_is_bounded_by_trusted_current_time() {
    let policy = policy::fixture();
    let input = snapshot();
    assert!(evaluate_at(&policy, &input, DIGEST, 199).is_err());
    assert!(evaluate_at(&policy, &input, DIGEST, 501).is_err());
    assert!(evaluate_at(&policy, &input, DIGEST, 500).is_ok());
}

#[test]
fn directory_boundaries_and_root_dockerfile_prefix_are_literal() {
    for path in [
        "ci-policy.json",
        "toolshed/main.rs",
        "mesh/scripts-other/x",
        "Dockerfiles",
    ] {
        assert!(
            !super::scope::is_policy_path(&policy::fixture(), path),
            "{path}"
        );
    }
    for path in [
        "ci/line\nbreak.json",
        "Dockerfile",
        "Dockerfile.gpu",
        "mesh/scripts/build.sh",
    ] {
        assert!(
            super::scope::is_policy_path(&policy::fixture(), path),
            "{path}"
        );
    }
}

#[test]
fn malformed_paths_renames_and_duplicate_files_fail() {
    for path in [
        "../ci/gate",
        "/ci/gate",
        "ci//gate",
        "ci/./gate",
        "ci\\gate",
    ] {
        let mut input = snapshot();
        input.changes[0].path = path.into();
        assert!(!allowed(&policy::fixture(), &input));
    }
    let mut input = snapshot();
    input.changes[0].kind = ChangeKind::Renamed;
    assert!(!allowed(&policy::fixture(), &input));
    let mut input = snapshot();
    input.changes.push(input.changes[0].clone());
    input.changed_file_count = 2;
    assert!(!allowed(&policy::fixture(), &input));
}

#[test]
fn policy_cannot_edit_away_its_identity_or_control_scope() {
    let mut policy = policy::fixture();
    policy.scope.directory_prefixes.retain(|path| path != "ci/");
    assert!(policy.validate().is_err());
    let mut policy = policy::fixture();
    policy.eligible_humans.push("unlisted".into());
    assert!(policy.validate().is_err());
    let mut policy = policy::fixture();
    policy.scope.exact_paths.push("mesh/**".into());
    assert!(policy.validate().is_err());
}

#[test]
fn shared_build_and_automation_controls_cannot_be_removed_from_policy_alone() {
    for path in [
        "scripts/",
        "just/",
        ".cargo/",
        ".config/",
        "mesh/deploy/",
        "skippy/scripts/",
        "mesh/crates/mesh-llm-ui/scripts/",
        "mesh/website/scripts/",
    ] {
        let mut policy = policy::fixture();
        policy.scope.directory_prefixes.retain(|item| item != path);
        assert!(policy.validate().is_err(), "removed directory {path}");
    }
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "Justfile",
        "Dockerfile",
        "mesh/crates/mesh-llm-ui/package.json",
        "mesh/crates/mesh-llm-ui/vitest.config.ts",
        "mesh/website/package.json",
    ] {
        let mut policy = policy::fixture();
        policy.scope.exact_paths.retain(|item| item != path);
        assert!(policy.validate().is_err(), "removed control {path}");
    }
    let mut policy = policy::fixture();
    policy.scope.root_filename_prefixes.clear();
    assert!(policy.validate().is_err(), "removed Dockerfile variants");
}

#[test]
fn scope_floor_requires_the_whole_directory_but_accepts_a_covering_prefix() {
    let mut narrowed = policy::fixture();
    narrowed
        .scope
        .directory_prefixes
        .retain(|path| path != "scripts/");
    narrowed
        .scope
        .directory_prefixes
        .push("scripts/build/".into());
    assert!(narrowed.validate().is_err());

    let mut covered = policy::fixture();
    covered
        .scope
        .directory_prefixes
        .retain(|path| path != ".agents/skills/manage-ci/");
    covered.scope.directory_prefixes.push(".agents/".into());
    assert!(covered.validate().is_ok());

    let mut covered_root = policy::fixture();
    covered_root
        .scope
        .exact_paths
        .retain(|path| path != "Dockerfile");
    covered_root.scope.root_filename_prefixes = vec!["Dockerfile".into()];
    assert!(covered_root.validate().is_ok());
}
