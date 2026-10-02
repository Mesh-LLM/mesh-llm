use super::super::workflow_yaml;
use super::*;
const SOURCE: &str = include_str!("../../../../../.github/workflows/agentic-replay-nightly.yml");
fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn validate(source: &str) -> DynResult<()> {
    check(
        &root(),
        &BTreeMap::from([(WORKFLOW.into(), workflow_yaml::parse(source)?)]),
    )
}
fn replace(node: &mut Node, key: &str, value: Node) {
    let Node::Map(entries) = node else {
        panic!("mapping")
    };
    if let Some((_, old)) = entries.iter_mut().find(|(name, _)| name == key) {
        *old = value;
    } else {
        entries.push((key.into(), value));
    }
}
fn replay(document: &mut Node) -> &mut Node {
    let Node::Map(entries) = document else {
        panic!("document")
    };
    let jobs = &mut entries
        .iter_mut()
        .find(|(name, _)| name == "jobs")
        .unwrap()
        .1;
    let Node::Map(entries) = jobs else {
        panic!("jobs")
    };
    &mut entries
        .iter_mut()
        .find(|(name, _)| name == "replay")
        .unwrap()
        .1
}
fn steps(document: &mut Node) -> &mut Vec<Node> {
    let Node::Map(entries) = replay(document) else {
        panic!("replay")
    };
    let Node::Seq(steps) = &mut entries
        .iter_mut()
        .find(|(name, _)| name == "steps")
        .unwrap()
        .1
    else {
        panic!("steps")
    };
    steps
}
fn check_tree(document: Node) -> DynResult<()> {
    check(&root(), &BTreeMap::from([(WORKFLOW.into(), document)]))
}
#[test]
fn current_manual_replay_retains_admission_locked_reader_and_execution_headroom() {
    validate(SOURCE).unwrap();
    let wrapped = SOURCE
        .replace("github.repository ==", "${{ github.repository ==")
        .replace(
            "github.event_name == 'schedule')",
            "github.event_name == 'schedule') }}",
        );
    validate(&wrapped).unwrap();
}
#[test]
fn fork_feature_push_disabled_or_disjunctive_admission_cannot_select_runner() {
    for condition in [
        "true",
        "false",
        "github.ref == 'refs/heads/main'",
        "github.repository == 'fork/mesh-llm' && github.ref == 'refs/heads/main' && (github.event_name == 'workflow_dispatch' || github.event_name == 'schedule')",
        "github.repository == 'Mesh-LLM/mesh-llm' && github.ref == 'refs/heads/main' && (github.event_name == 'push' || github.event_name == 'schedule')",
        "github.repository == 'Mesh-LLM/mesh-llm' && github.ref == 'refs/heads/main' && (github.event_name == 'workflow_dispatch' || github.event_name == 'schedule') || true",
    ] {
        let mut document = workflow_yaml::parse(SOURCE).unwrap();
        replace(replay(&mut document), "if", Node::Scalar(condition.into()));
        assert!(check_tree(document).is_err(), "{condition}");
    }
    for (from, to) in [
        ("cancel-in-progress: false", "cancel-in-progress: true"),
        ("  workflow_dispatch:", "  pull_request:"),
    ] {
        assert!(validate(&SOURCE.replace(from, to)).is_err());
    }
}
#[test]
fn runner_guard_must_be_unconditional_before_immutable_credential_free_checkout() {
    let mut document = workflow_yaml::parse(SOURCE).unwrap();
    steps(&mut document).swap(0, 1);
    assert!(check_tree(document).is_err());
    for (from, to) in [
        (
            "EXPECTED_REPLAY_RUNNER_NAME: micstudio",
            "EXPECTED_REPLAY_RUNNER_NAME: studio54",
        ),
        ("ref: ${{ github.sha }}", "ref: refs/heads/feature"),
        ("persist-credentials: false", "persist-credentials: true"),
    ] {
        assert!(validate(&SOURCE.replace(from, to)).is_err());
    }
    for key in ["if", "continue-on-error"] {
        let mut document = workflow_yaml::parse(SOURCE).unwrap();
        replace(
            &mut steps(&mut document)[0],
            key,
            Node::Scalar("false".into()),
        );
        assert!(check_tree(document).is_err());
    }
}
#[test]
fn component_lock_and_reader_exports_cannot_be_skipped_reordered_or_globalized() {
    for (from, to) in [
        (
            "uv sync --locked --project ci/agentic-replay-nightly",
            "uv sync --project ci/agentic-replay-nightly",
        ),
        (
            "uv sync --locked --project ci/agentic-replay-nightly",
            "echo uv sync --locked --project ci/agentic-replay-nightly",
        ),
        (
            "uv sync --locked --project ci/agentic-replay-nightly",
            "uv sync --locked --project .",
        ),
        ("import duckdb;", "import unrelated;"),
        (
            "echo \"$REPLAY_PYTHON_BIN\" >> \"$GITHUB_PATH\"",
            "echo \"/usr/bin\" >> \"$GITHUB_PATH\"",
        ),
    ] {
        assert!(validate(&SOURCE.replace(from, to)).is_err(), "{from}");
    }
    let mut document = workflow_yaml::parse(SOURCE).unwrap();
    let sequence = steps(&mut document);
    let reader = sequence
        .iter()
        .position(|step| field(step, "name") == Some(READER))
        .unwrap();
    let inputs = sequence
        .iter()
        .position(|step| field(step, "id") == Some("inputs"))
        .unwrap();
    sequence.swap(reader, inputs);
    assert!(check_tree(document).is_err());
    let temporary = tempfile::tempdir().unwrap();
    let document = workflow_yaml::parse(SOURCE).unwrap();
    assert!(
        check(
            temporary.path(),
            &BTreeMap::from([(WORKFLOW.into(), document)])
        )
        .is_err()
    );
}
#[test]
fn timeout_domain_and_all_three_models_plus_repair_leave_real_job_headroom() {
    for timeout in ["0", "-1", "true", "1.5", "361"] {
        let mut document = workflow_yaml::parse(SOURCE).unwrap();
        let model = steps(&mut document)
            .iter_mut()
            .find(|step| field(step, "id") == Some("model_0"))
            .unwrap();
        replace(model, "timeout-minutes", Node::Scalar(timeout.into()));
        assert!(check_tree(document).is_err(), "{timeout}");
    }
    for (from, to) in [
        ("timeout-minutes: 1800", "timeout-minutes: 1440"),
        ("ulimit -n 65536", "echo ulimit -n 65536"),
    ] {
        assert!(validate(&SOURCE.replace(from, to)).is_err());
    }
}
