use super::record::*;
use super::state::*;
use super::*;
use crate::ci_governance::approval::{Account, AccountKind};

const NOW: u64 = 1_800_000_000;

fn human(login: &str, id: u64) -> Account {
    Account {
        login: login.into(),
        id,
        kind: AccountKind::Human,
    }
}

fn fixture() -> (Policy, ExpectedState, Observation, ChangeRecord) {
    let policy = crate::ci_governance::policy::fixture();
    let expected: ExpectedState = serde_json::from_str(include_str!(
        "../../../../../ci/governance/expected-state.json"
    ))
    .unwrap();
    let execution = ExecutionContext {
        repository: policy.repository.clone(),
        run_id: 101,
        run_attempt: 1,
        ref_name: "refs/heads/main".into(),
        protected_revision: "a".repeat(40),
        operator: human("micspiral", 1),
        triggerer: human("micspiral", 1),
        rerun_actor: human("micspiral", 1),
    };
    let mut before = expected.state.clone();
    before.native_rule.required_approvals = 0;
    let observed = Observation {
        schema_version: 1,
        normalization: Normalization::OperatorNormalizedV1,
        repository: policy.repository.clone(),
        complete: true,
        observed_at_unix: NOW - 10,
        protected_policy_revision: "a".repeat(40),
        current_protected_main_revision: "a".repeat(40),
        policy_sha256: digest(&policy).unwrap(),
        readback_link: evidence_link(),
        replay_ledger_complete: true,
        replay_ledger_link: evidence_link(),
        consumed_operation_ids: Vec::new(),
        observer: ObserverContext {
            repository: policy.repository.clone(),
            run_id: 101,
            run_attempt: 1,
            ref_name: "refs/heads/main".into(),
            protected_revision: "a".repeat(40),
            principal: human("micspiral", 1),
        },
        execution: Some(execution.clone()),
        state: before.clone(),
    };
    let request = ChangeRequest {
        operation_id: "ba2e26e4-76a4-462f-894b-1a33d80d01aa".into(),
        repository: policy.repository.clone(),
        policy_revision: "a".repeat(40),
        policy_sha256: digest(&policy).unwrap(),
        expected_before_sha256: before.canonical_digest().unwrap(),
        before,
        desired_after: expected.state.clone(),
        controls: vec![Control::CiRuleset],
        execution_method: ExecutionMethod::ManualProviderOperation,
        execution: execution.clone(),
        owner: human("micspiral", 1),
        author: human("micspiral", 1),
        sponsor: human("micspiral", 1),
        rationale: "Require a current independent CI policy review".into(),
        affected_jobs: vec!["CI policy / independent approval".into()],
        tests: vec!["local deterministic governance acceptance fixtures".into()],
        rollout: "Authorized human applies the exact native ruleset and retains readback".into(),
        rollback: "Authorized human restores the captured before state with separate approval"
            .into(),
        requested_at_unix: NOW - 100,
        expires_at_unix: NOW + 600,
    };
    let approval = Approval {
        approver: human("ndizazzo", 2),
        state: ApprovalState::Approved,
        request_sha256: request.canonical_digest().unwrap(),
        policy_revision: request.policy_revision.clone(),
        execution,
        approved_at_unix: NOW - 20,
        expires_at_unix: NOW + 300,
        link: evidence_link(),
        environment: "ci-governance-administration".into(),
        environment_prevented_self_review: true,
        environment_admin_bypass: false,
    };
    let record = ChangeRecord {
        schema_version: 1,
        request,
        approval,
        manual_readback: None,
    };
    (policy, expected, observed, record)
}

fn evidence_link() -> String {
    "https://github.com/Mesh-LLM/mesh-llm/issues/2781".into()
}

fn resign(record: &mut ChangeRecord) {
    record.approval.request_sha256 = record.request.canonical_digest().unwrap();
}

#[test]
fn valid_plan_has_no_execution_authority() {
    let (policy, expected, observed, record) = fixture();
    let result = validate_record(&policy, &expected, &record, &observed, NOW).unwrap();
    assert_eq!(result["status"], "VALIDATED_PLAN");
    assert_eq!(result["execution_authorized"], false);
    assert_eq!(result["evidence_authenticated"], false);
}

#[test]
fn exact_reviewer_membership_drift_is_rejected() {
    let (policy, expected, mut observed, _) = fixture();
    observed.state = expected.state.clone();
    assert!(drift(&policy, &expected, &observed, NOW).is_ok());
    observed.state.reviewer_team.members.push("unlisted".into());
    let error = drift(&policy, &expected, &observed, NOW)
        .unwrap_err()
        .to_string();
    assert!(error.contains("reviewer-team"));
}

#[test]
fn native_and_environment_bypasses_are_drift() {
    let (policy, expected, mut observed, _) = fixture();
    observed.state = expected.state.clone();
    observed
        .state
        .native_rule
        .bypass_actors
        .push("administrator".into());
    assert!(
        drift(&policy, &expected, &observed, NOW)
            .unwrap_err()
            .to_string()
            .contains("ci-ruleset")
    );
    observed.state = expected.state.clone();
    observed.state.administration_environment.admin_bypass = true;
    assert!(
        drift(&policy, &expected, &observed, NOW)
            .unwrap_err()
            .to_string()
            .contains("administration-environment")
    );
}

#[test]
fn github_actions_identity_cannot_stand_in_for_dedicated_app() {
    let (policy, expected, mut observed, _) = fixture();
    observed.state = expected.state.clone();
    observed.state.credentials.observer_kind = CheckSource::GithubActions;
    assert!(
        drift(&policy, &expected, &observed, NOW)
            .unwrap_err()
            .to_string()
            .contains("credential-boundaries")
    );
}

#[test]
fn product_only_scope_remains_unaffected() {
    let (policy, mut expected, mut observed, _) = fixture();
    observed.state = expected.state.clone();
    observed.state.native_rule.product_only_requires_team_review = true;
    assert!(drift(&policy, &expected, &observed, NOW).is_err());
    expected.state.native_rule.product_only_requires_team_review = true;
    assert!(expected.validate(&policy).is_err());
    expected.state.native_rule.product_only_requires_team_review = false;
    expected
        .state
        .native_rule
        .scope
        .exact_paths
        .push("mesh/crates/product.rs".into());
    assert!(expected.validate(&policy).is_err());
}

#[test]
fn stale_before_state_and_reviewed_digest_are_rejected() {
    let (policy, expected, mut observed, mut record) = fixture();
    observed.state.native_rule.required_approvals = 2;
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
    observed.state = record.request.before.clone();
    record.request.rationale = "Changed after approval".into();
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
}

#[test]
fn exact_policy_revision_required_even_for_main_ancestor() {
    let (policy, expected, observed, mut record) = fixture();
    record.request.policy_revision = "b".repeat(40);
    resign(&mut record);
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
    record.request.policy_revision = observed.protected_policy_revision.clone();
    record.request.policy_sha256 = "0".repeat(64);
    resign(&mut record);
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
}

#[test]
fn every_responsible_actor_is_excluded_from_approval() {
    let (policy, expected, observed, record) = fixture();
    for role in ["author", "sponsor", "operator", "triggerer", "rerun-actor"] {
        let mut candidate = record.clone();
        let mut current = observed.clone();
        let approver = candidate.approval.approver.clone();
        match role {
            "author" | "sponsor" => {
                candidate.request.owner = approver.clone();
                candidate.request.author = approver.clone();
                candidate.request.sponsor = approver;
            }
            "operator" => candidate.request.execution.operator = approver,
            "triggerer" => candidate.request.execution.triggerer = approver,
            _ => candidate.request.execution.rerun_actor = approver,
        }
        current.execution = Some(candidate.request.execution.clone());
        candidate.approval.execution = candidate.request.execution.clone();
        resign(&mut candidate);
        assert!(
            validate_record(&policy, &expected, &candidate, &current, NOW)
                .unwrap_err()
                .to_string()
                .contains("approver must differ"),
            "{role}"
        );
    }
}

#[test]
fn bots_and_unlisted_reviewers_cannot_approve_or_sponsor() {
    let (policy, expected, observed, record) = fixture();
    let mut candidate = record.clone();
    candidate.approval.approver.kind = AccountKind::Bot;
    assert!(validate_record(&policy, &expected, &candidate, &observed, NOW).is_err());
    candidate = record.clone();
    candidate.approval.approver = human("unlisted", 10);
    assert!(validate_record(&policy, &expected, &candidate, &observed, NOW).is_err());
    candidate = record;
    candidate.request.sponsor.kind = AccountKind::Bot;
    resign(&mut candidate);
    assert!(validate_record(&policy, &expected, &candidate, &observed, NOW).is_err());
}

#[test]
fn expiry_replay_and_wrong_run_are_rejected() {
    let (policy, expected, observed, record) = fixture();
    let mut candidate = record.clone();
    candidate.approval.expires_at_unix = NOW;
    assert!(validate_record(&policy, &expected, &candidate, &observed, NOW).is_err());
    let mut current = observed.clone();
    current
        .consumed_operation_ids
        .push(record.request.operation_id.clone());
    assert!(
        validate_record(&policy, &expected, &record, &current, NOW)
            .unwrap_err()
            .to_string()
            .contains("replay")
    );
    candidate = record;
    candidate.request.execution.run_attempt = 2;
    resign(&mut candidate);
    assert!(validate_record(&policy, &expected, &candidate, &observed, NOW).is_err());
}

#[test]
fn incomplete_stale_unknown_observations_fail_closed() {
    let (policy, expected, observed, record) = fixture();
    let mut current = observed.clone();
    current.complete = false;
    assert!(validate_record(&policy, &expected, &record, &current, NOW).is_err());
    current = observed.clone();
    current.observed_at_unix = NOW - 901;
    assert!(drift(&policy, &expected, &current, NOW).is_err());
    let mut unknown = serde_json::to_value(observed).unwrap();
    unknown["state"]["credentials"]["secret_value"] = json!("never admitted");
    assert!(serde_json::from_value::<Observation>(unknown).is_err());
}

#[test]
fn arbitrary_operations_and_extra_changes_are_rejected() {
    let (policy, expected, observed, mut record) = fixture();
    let mut unknown = serde_json::to_value(&record).unwrap();
    unknown["request"]["controls"] = json!(["arbitrary-shell"]);
    assert!(serde_json::from_value::<ChangeRecord>(unknown).is_err());
    record.request.controls.push(Control::ReviewerTeam);
    resign(&mut record);
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
}

#[test]
fn manual_readback_requires_exact_after_state_and_consumption() {
    let (policy, expected, observed, mut record) = fixture();
    record.manual_readback = Some(ManualReadback {
        operation_id: record.request.operation_id.clone(),
        request_sha256: record.request.canonical_digest().unwrap(),
        execution: record.request.execution.clone(),
        operated_at_unix: NOW - 5,
        observed_at_unix: NOW - 1,
        operator: record.request.execution.operator.clone(),
        actual_state: expected.state.clone(),
        operation_link: evidence_link(),
        readback_link: evidence_link(),
        replay_ledger_link: evidence_link(),
        consumed_operation_ids: vec![record.request.operation_id.clone()],
        replay_ledger_complete: true,
    });
    assert_eq!(
        validate_record(&policy, &expected, &record, &observed, NOW).unwrap()["status"],
        "RECORDED_MANUAL_READBACK"
    );
    assert!(validate_record(&policy, &expected, &record, &observed, NOW + 86_400).is_err());
    let valid_receipt = record.manual_readback.clone();
    record.manual_readback.as_mut().unwrap().operated_at_unix = observed.observed_at_unix;
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
    record.manual_readback = valid_receipt.clone();
    let operated_at = record.manual_readback.as_ref().unwrap().operated_at_unix;
    record.manual_readback.as_mut().unwrap().observed_at_unix = operated_at;
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
    record.manual_readback = valid_receipt;
    record
        .manual_readback
        .as_mut()
        .unwrap()
        .actual_state
        .native_rule
        .required_approvals = 0;
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
    record.manual_readback.as_mut().unwrap().actual_state = expected.state.clone();
    record
        .manual_readback
        .as_mut()
        .unwrap()
        .consumed_operation_ids
        .clear();
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_err());
}

#[test]
fn post_approval_readback_rejects_approval_wait_and_same_second() {
    let (policy, expected, mut observed, record) = fixture();
    for offset in [30, 20] {
        observed.observed_at_unix = NOW - offset;
        assert!(
            validate_record(&policy, &expected, &record, &observed, NOW)
                .unwrap_err()
                .to_string()
                .contains("after independent approval")
        );
    }
}

#[test]
fn coherent_historical_revision_cannot_replace_current_main() {
    let (policy, expected, mut observed, record) = fixture();
    observed.current_protected_main_revision = "b".repeat(40);
    assert!(
        validate_record(&policy, &expected, &record, &observed, NOW)
            .unwrap_err()
            .to_string()
            .contains("protected policy")
    );
}

#[test]
fn reordered_state_lists_have_equal_comparison_and_reviewed_digests() {
    let (policy, expected, mut observed, mut record) = fixture();
    observed.state = expected.state.clone();
    observed.state.reviewer_team.members.reverse();
    observed
        .state
        .administration_environment
        .reviewers
        .reverse();
    observed
        .state
        .administration_environment
        .allowed_refs
        .reverse();
    observed
        .state
        .native_rule
        .scope
        .directory_prefixes
        .reverse();
    observed.state.native_rule.scope.exact_paths.reverse();
    observed
        .state
        .native_rule
        .scope
        .root_filename_prefixes
        .reverse();
    observed.state.credentials.observer_permissions.reverse();
    assert!(drift(&policy, &expected, &observed, NOW).is_ok());
    assert_eq!(
        observed.state.canonical_digest().unwrap(),
        expected.state.canonical_digest().unwrap()
    );
    let request_digest = record.request.canonical_digest().unwrap();
    record.request.before.reviewer_team.members.reverse();
    record
        .request
        .before
        .native_rule
        .scope
        .exact_paths
        .reverse();
    record
        .request
        .desired_after
        .credentials
        .observer_permissions
        .reverse();
    assert_eq!(record.request.canonical_digest().unwrap(), request_digest);
    observed.state = record.request.before.clone();
    assert!(validate_record(&policy, &expected, &record, &observed, NOW).is_ok());
}

#[test]
fn duplicate_unordered_state_entries_are_rejected() {
    let (policy, expected, observed, _) = fixture();
    for list in 0..4 {
        let mut current = observed.clone();
        current.state = expected.state.clone();
        match list {
            0 => current.state.reviewer_team.members.push("micspiral".into()),
            1 => current
                .state
                .administration_environment
                .reviewers
                .push("micspiral".into()),
            2 => current
                .state
                .native_rule
                .scope
                .exact_paths
                .push("AGENTS.md".into()),
            _ => current
                .state
                .credentials
                .observer_permissions
                .push(ObserverPermission::MetadataRead),
        }
        assert!(
            drift(&policy, &expected, &current, NOW)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
    }
}

#[test]
fn read_only_service_can_report_drift_but_has_no_mutation_execution_context() {
    let (policy, expected, mut observed, record) = fixture();
    observed.state = expected.state.clone();
    observed.execution = None;
    observed.observer.principal = Account {
        login: "governance-observer[bot]".into(),
        id: 99,
        kind: AccountKind::Bot,
    };
    let result = drift(&policy, &expected, &observed, NOW).unwrap();
    assert_eq!(result["identity_binding_status"], "logical-only-unverified");
    assert_eq!(result["evidence_authenticated"], false);
    assert!(
        validate_record(&policy, &expected, &record, &observed, NOW)
            .unwrap_err()
            .to_string()
            .contains("execution context")
    );
}

#[test]
fn deterministic_fixture_illustrates_local_drift_detection_timing() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../ci/governance/examples/drift-latency.fixture.json"
    ))
    .unwrap();
    let (policy, expected, mut observed, _) = self::fixture();
    observed.state = expected.state.clone();
    assert!(drift(&policy, &expected, &observed, NOW).is_ok());
    let changed_at = fixture["drift_started_offset_seconds"].as_u64().unwrap();
    let detected_at = fixture["first_detection_offset_seconds"].as_u64().unwrap();
    observed.state.reviewer_team.members.pop();
    observed.observed_at_unix = NOW + detected_at;
    assert!(drift(&policy, &expected, &observed, NOW + detected_at).is_err());
    let illustrative_latency = detected_at - changed_at;
    assert_eq!(
        illustrative_latency,
        fixture["illustrative_fixture_latency_seconds"]
            .as_u64()
            .unwrap()
    );
    assert!(illustrative_latency <= expected.drift_target_seconds);
    assert_eq!(fixture["live_measurement"], false);
}
