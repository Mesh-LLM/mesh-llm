use super::*;
use crate::ci_validation::lane_results::workflow_yaml;
use std::{fs, path::Path};

fn actual() -> BTreeMap<String, Node> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    fs::read_dir(root.join(".github/workflows"))
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            (
                path.file_name().unwrap().to_str().unwrap().to_owned(),
                workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap(),
            )
        })
        .collect()
}
fn field_mut<'a>(node: &'a mut Node, key: &str) -> &'a mut Node {
    let Node::Map(fields) = node else {
        panic!("mapping required")
    };
    &mut fields.iter_mut().find(|(name, _)| name == key).unwrap().1
}
fn job_mut<'a>(
    workflows: &'a mut BTreeMap<String, Node>,
    workflow: &str,
    job: &str,
) -> &'a mut Node {
    field_mut(field_mut(workflows.get_mut(workflow).unwrap(), "jobs"), job)
}
fn steps_mut(job: &mut Node) -> &mut Vec<Node> {
    let Node::Seq(steps) = field_mut(job, "steps") else {
        panic!("steps required")
    };
    steps
}
fn last(workflows: &mut BTreeMap<String, Node>) -> &mut Node {
    steps_mut(job_mut(workflows, "agentic-replay-nightly.yml", "replay"))
        .last_mut()
        .unwrap()
}

#[test]
fn actual_persistent_jobs_finish_after_uploads_and_release_cache_save() {
    check(&actual()).unwrap();
}
#[test]
fn actual_job_missing_cleanup_or_with_a_later_step_is_rejected() {
    let mut workflows = actual();
    steps_mut(job_mut(
        &mut workflows,
        "agentic-replay-nightly.yml",
        "replay",
    ))
    .pop();
    assert!(check(&workflows).is_err());
    let mut workflows = actual();
    steps_mut(job_mut(
        &mut workflows,
        "agentic-replay-nightly.yml",
        "replay",
    ))
    .push(Node::Map(vec![(
        "run".into(),
        Node::Scalar("echo late".into()),
    )]));
    assert!(check(&workflows).is_err());
}
#[test]
fn actual_cleanup_rejects_omitted_outcomes_false_gates_and_unbounded_budget() {
    for condition in [
        "success()",
        "success() || failure()",
        "(success() || failure() || cancelled()) && false",
    ] {
        let mut workflows = actual();
        *field_mut(last(&mut workflows), "if") = Node::Scalar(condition.into());
        assert!(check(&workflows).is_err(), "accepted {condition}");
    }
    let mut workflows = actual();
    *field_mut(last(&mut workflows), "timeout-minutes") = Node::Scalar("30".into());
    assert!(check(&workflows).is_err());
}
#[test]
fn actual_cleanup_requires_execution_not_echo_or_masked_failure() {
    for run in [
        "echo cargo xtool ci-ops runner-cleanup --job replay",
        "cargo xtool ci-ops runner-cleanup --job replay || true",
        "python3 scripts/cleanup-self-hosted.py --job replay",
        "if false; then\ncargo xtool ci-ops runner-cleanup --job replay\nfi",
        "exit 0\ncargo xtool ci-ops runner-cleanup --job replay",
        "case x in x) exit 0 ;; esac\ncargo xtool ci-ops runner-cleanup --job replay",
        "while false; do\ncargo xtool ci-ops runner-cleanup --job replay\ndone",
        "cargo xtool ci-ops runner-cleanup --job replay\necho success",
    ] {
        let mut workflows = actual();
        *field_mut(last(&mut workflows), "run") = Node::Scalar(run.into());
        assert!(check(&workflows).is_err(), "accepted {run}");
    }
}
#[test]
fn actual_cuda_release_requires_save_before_cleanup_and_restore() {
    let mut workflows = actual();
    let steps = steps_mut(job_mut(
        &mut workflows,
        "release.yml",
        "build_native_runtime_linux_x86_64_cuda",
    ));
    let length = steps.len();
    steps.swap(length - 1, length - 2);
    assert!(check(&workflows).is_err());
    let mut workflows = actual();
    steps_mut(job_mut(
        &mut workflows,
        "release.yml",
        "build_native_runtime_linux_x86_64_cuda",
    ))
    .retain(|step| {
        !field(step, "uses").is_some_and(|action| action.starts_with("actions/cache/restore@"))
    });
    assert!(check(&workflows).is_err());
}
#[test]
fn outcome_order_and_whitespace_are_semantic_and_known_hosted_selectors_remain() {
    assert!(all_outcomes(
        "${{ (cancelled() || success() || failure()) }}"
    ));
    assert!(all_outcomes(
        "${{ (failure() || cancelled() || success()) && inputs.runner == 'gpu-nvidia' }}"
    ));
    assert!(cleanup_invocation(
        "if [[ -n \"${MESH_LLM_AUTOMATION_BIN:-}\" ]]; then\n\"$MESH_LLM_AUTOMATION_BIN\" ci-ops runner-cleanup --job smoke --evidence-uploaded false\nelse\ncargo xtool ci-ops runner-cleanup --job smoke --evidence-uploaded false\nfi"
    ));
    assert!(cleanup_invocation(
        "set -euo pipefail\ncargo xtool ci-ops runner-cleanup --job replay --evidence-uploaded false"
    ));
    assert!(cleanup_invocation(
        "\"$MESH_LLM_AUTOMATION_BIN\" ci-ops runner-cleanup --job build \\\n --evidence-uploaded \"$EVIDENCE_UPLOADED\""
    ));
}
#[test]
fn actual_matrix_and_literal_runner_routes_are_included_without_freezing_job_count() {
    let workflows = actual();
    let jobs = workflows["ci-runner-contract-slice.yml"]
        .get("jobs")
        .unwrap();
    assert!(persistent_runner(jobs.get("trusted_runner_image").unwrap()));
    assert!(!persistent_runner(jobs.get("runner_contract").unwrap()));
    let jobs = workflows["llama-canary-family-pass.yml"]
        .get("jobs")
        .unwrap();
    assert!(persistent_runner(jobs.get("family").unwrap()));
    assert!(!persistent_runner(jobs.get("aggregate").unwrap()));
}
