use super::super::workflow_yaml;
use super::*;
use std::{fs, path::Path};

fn actual() -> BTreeMap<String, Node> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    ["ci-rust-tests-slice.yml", "claude-live-model-gate.yml"]
        .into_iter()
        .map(|name| {
            let source = fs::read_to_string(root.join(".github/workflows").join(name)).unwrap();
            (name.into(), workflow_yaml::parse(&source).unwrap())
        })
        .collect()
}
fn step<'a>(document: &'a mut Node, job: &str, name: &str) -> &'a mut Node {
    let job = h::mutable(h::mutable(document, "jobs"), job);
    let Node::Seq(steps) = h::mutable(job, "steps") else {
        panic!("step sequence");
    };
    steps
        .iter_mut()
        .find(|step| field(step, "name") == Some(name))
        .expect("owning step")
}
fn entry(node: &mut Node, key: &str, value: &str) {
    let Node::Map(entries) = node else {
        panic!("mapping");
    };
    if let Some((_, old)) = entries.iter_mut().find(|(name, _)| name == key) {
        *old = Node::Scalar(value.into());
    } else {
        entries.push((key.into(), Node::Scalar(value.into())));
    }
}

#[test]
fn actual_required_and_manual_claude_workflows_retain_real_client_contract() {
    check(&actual()).unwrap();
}
const RUST_TESTS: &str = "ci-rust-tests-slice.yml";
const BATCH: &str = "Run isolated Cargo tests for the batch";
const CLIENT_TEST: &str = "cargo test --locked -p \"$crate\" --no-default-features --features claude-code-integration claude_cli_executes_read_tool_through_host_ingress";

fn batch_script(client: &str) -> String {
    format!(
        "set -euo pipefail\nmapfile -t crates < <(jq -r '.[]' <<<\"$TEST_CRATES\")\nfor crate in ${{crates[@]+\"${{crates[@]}}\"}}; do\n  cargo test --locked -p \"$crate\"\n  if [[ \"$crate\" == \"mesh-llm-host-runtime\" ]]; then\n    echo \"::group::Claude Code 2.1.273 real-client integration\"\n    {client}\n    echo \"::endgroup::\"\n  fi\ndone\n"
    )
}
fn with_batch(run: &str) -> BTreeMap<String, Node> {
    let mut workflows = actual();
    let gate = step(workflows.get_mut(RUST_TESTS).unwrap(), "rust_tests", BATCH);
    entry(gate, "run", run);
    workflows
}

#[test]
fn actual_batch_script_shape_is_admissible() {
    check(&with_batch(&batch_script(CLIENT_TEST))).unwrap();
}
#[test]
fn missing_echoed_nested_or_ignored_only_batch_execution_rejects() {
    for client in [
        "echo claude_cli_executes_read_tool_through_host_ingress",
        "cargo test --locked -p \"$crate\"",
        "cargo test --locked -p \"$crate\" --no-default-features --features claude-code-integration claude_cli_executes_read_tool_through_host_ingress -- --ignored",
        "if false; then\ncargo test --locked -p \"$crate\" --no-default-features --features claude-code-integration claude_cli_executes_read_tool_through_host_ingress\nfi",
        "cargo test --locked -p skippy-inference-api --no-default-features --features claude-code-integration claude_cli_executes_read_tool_through_host_ingress",
    ] {
        assert!(
            check(&with_batch(&batch_script(client))).is_err(),
            "accepted {client}"
        );
    }
}
#[test]
fn batch_must_keep_the_host_branch_and_every_planned_crate() {
    let without_branch = batch_script(CLIENT_TEST).replace("mesh-llm-host-runtime", "mesh-llm");
    let without_crates =
        batch_script(CLIENT_TEST).replace("  cargo test --locked -p \"$crate\"\n", "");
    for run in [without_branch, without_crates] {
        assert!(check(&with_batch(&run)).is_err(), "accepted {run}");
    }
}
#[test]
fn batch_gate_requires_planned_crates_and_failure_propagation() {
    for (key, value) in [("if", "${{ false }}"), ("continue-on-error", "true")] {
        let mut workflows = actual();
        let gate = step(workflows.get_mut(RUST_TESTS).unwrap(), "rust_tests", BATCH);
        entry(gate, key, value);
        assert!(check(&workflows).is_err(), "accepted {key}");
    }
    let mut workflows = actual();
    let gate = step(workflows.get_mut(RUST_TESTS).unwrap(), "rust_tests", BATCH);
    h::replace(gate, &["env", "TEST_CRATES"], Node::Scalar("[]".into()));
    assert!(check(&workflows).is_err());
}
#[test]
fn batch_memory_settings_are_bound_to_the_owning_workflow() {
    for key in ["CARGO_INCREMENTAL", "CARGO_PROFILE_TEST_DEBUG"] {
        let mut workflows = actual();
        h::replace(
            workflows.get_mut(RUST_TESTS).unwrap(),
            &["env", key],
            Node::Scalar("1".into()),
        );
        assert!(check(&workflows).is_err(), "accepted {key}");
    }
}
#[test]
fn pinned_client_install_cannot_be_replaced_by_a_label_or_skipped_for_host_batches() {
    for (key, value) in [
        ("run", "echo @anthropic-ai/claude-code@2.1.273"),
        ("if", "${{ false }}"),
    ] {
        let mut workflows = actual();
        let install = step(
            workflows.get_mut(RUST_TESTS).unwrap(),
            "rust_tests",
            "Install pinned Claude Code integration client",
        );
        entry(install, key, value);
        assert!(check(&workflows).is_err(), "accepted {key}");
    }
}

#[test]
fn manual_live_gate_requires_approval_environment_and_owned_secret_binding() {
    for (path, value) in [
        (vec!["jobs", "live_claude", "environment"], "unprotected"),
        (
            vec!["jobs", "live_claude", "env", "ANTHROPIC_API_KEY"],
            "${{ inputs.key }}",
        ),
        (
            vec!["jobs", "live_claude", "env", "CARGO_PROFILE_TEST_DEBUG"],
            "1",
        ),
    ] {
        let mut workflows = actual();
        h::replace(
            workflows.get_mut("claude-live-model-gate.yml").unwrap(),
            &path,
            Node::Scalar(value.into()),
        );
        assert!(check(&workflows).is_err());
    }
    let mut workflows = actual();
    let events = h::mutable(
        workflows.get_mut("claude-live-model-gate.yml").unwrap(),
        "on",
    );
    entry(events, "pull_request", "");
    assert!(check(&workflows).is_err());
}
#[test]
fn manual_compiler_execution_requires_accelerator_and_direct_nonignored_round_trip() {
    for (key, value) in [
        ("run", "echo --features claude-live-model-integration"),
        (
            "run",
            "cargo test --locked -p mesh-llm-host-runtime --no-default-features --features claude-live-model-integration claude_cli_round_trips_through_host_ingress_and_live_claude_model -- --ignored",
        ),
        ("if", "${{ false }}"),
        ("continue-on-error", "true"),
    ] {
        let mut workflows = actual();
        let gate = step(
            workflows.get_mut("claude-live-model-gate.yml").unwrap(),
            "live_claude",
            "Run live Claude model round trip through Mesh",
        );
        entry(gate, key, value);
        assert!(check(&workflows).is_err());
    }
    let mut workflows = actual();
    let accelerator = step(
        workflows.get_mut("claude-live-model-gate.yml").unwrap(),
        "live_claude",
        "Install repository build accelerator",
    );
    entry(accelerator, "uses", "actions/checkout@not-a-compiler-cache");
    assert!(check(&workflows).is_err());
}

#[test]
fn equivalent_flag_order_comments_continuations_and_just_wrapper_remain_admissible() {
    let mut workflows = with_batch(&batch_script(
        "# Real client, equivalent order\n    just with-lld cargo test --features claude-code-integration --no-default-features -p \"$crate\" --locked \\\n      claude_cli_executes_read_tool_through_host_ingress -- --nocapture",
    ));
    let manual = workflows.get_mut("claude-live-model-gate.yml").unwrap();
    let job = h::mutable(h::mutable(manual, "jobs"), "live_claude");
    entry(job, "if", "${{ github.repository == 'Mesh-LLM/mesh-llm' }}");
    let gate = step(
        manual,
        "live_claude",
        "Run live Claude model round trip through Mesh",
    );
    entry(
        gate,
        "run",
        "just with-lld cargo test --features claude-live-model-integration --locked --package mesh-llm-host-runtime --no-default-features claude_cli_round_trips_through_host_ingress_and_live_claude_model -- --nocapture\n",
    );
    check(&workflows).unwrap();
}

#[test]
fn cargo_invocation_cannot_be_list_only_unlocked_or_success_masked() {
    for tail in ["-- --list", "|| true", "-- --ignored"] {
        let run = batch_script(&format!("{CLIENT_TEST} {tail}"));
        assert!(check(&with_batch(&run)).is_err(), "accepted {tail}");
    }
    let unlocked = batch_script(&CLIENT_TEST.replace("--locked ", ""));
    assert!(check(&with_batch(&unlocked)).is_err());
    assert!(invocation(&["cargo", "test", "-p", "skippy-inference-api"]).is_err());
}

#[test]
fn manual_job_literal_false_cannot_replace_real_authorized_execution() {
    for condition in ["false", "${{ false }}", "${{\n false\n }}"] {
        let mut workflows = actual();
        let manual = workflows.get_mut("claude-live-model-gate.yml").unwrap();
        let job = h::mutable(h::mutable(manual, "jobs"), "live_claude");
        entry(job, "if", condition);
        assert!(check(&workflows).is_err(), "{condition}");
    }
}
