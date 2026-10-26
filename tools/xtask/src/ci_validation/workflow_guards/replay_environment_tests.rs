use super::super::workflow_yaml;
use super::*;
const SOURCE: &str = include_str!("../../../../../.github/workflows/agentic-replay-nightly.yml");
fn validate(source: &str) -> DynResult<()> {
    check(&BTreeMap::from([(
        "agentic-replay-nightly.yml".into(),
        workflow_yaml::parse(source)?,
    )]))
}
#[test]
fn actual_replay_shared_cache_anonymous_history_and_repair_bindings_are_admitted() {
    validate(SOURCE).unwrap();
}
#[test]
fn shared_cache_offline_implicit_token_exports_and_writability_cannot_drift() {
    for (from, to) in [
        ("HF_HUB_OFFLINE: \"0\"", "HF_HUB_OFFLINE: \"1\""),
        (
            "HF_HUB_DISABLE_IMPLICIT_TOKEN: \"1\"",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN: \"0\"",
        ),
        (
            "export HF_HOME=\"$HF_CACHE\"",
            "export HF_HOME=\"$RUNNER_TEMP/private-cache\"",
        ),
        (
            "export HF_HUB_CACHE=\"$HF_CACHE/hub\"",
            "echo export HF_HUB_CACHE=\"$HF_CACHE/hub\"",
        ),
        ("! -w \"$HF_CACHE/hub\"", "! -d \"$HF_CACHE/hub\""),
        (
            "if [[ \"$missing\" == \"1\" ]]; then exit 1; fi",
            "if [[ \"$missing\" == \"1\" ]]; then true; fi",
        ),
        ("} >> \"$GITHUB_ENV\"", "} > /dev/null"),
    ] {
        assert!(SOURCE.contains(from));
        assert!(validate(&SOURCE.replace(from, to)).is_err(), "{from}");
    }
}
#[test]
fn pinned_cli_downloads_require_revision_dataset_kind_exact_digest_and_shared_cache() {
    for (from, to) in [
        (
            "--repo-type dataset --revision \"$dataset_revision\"",
            "--repo-type model --revision \"$dataset_revision\"",
        ),
        (
            "--revision \"$revision\" --format quiet",
            "--revision main --format quiet",
        ),
        ("model_path=$(hf download", "model_path=$(echo hf download"),
        ("--sha256 \"$dataset_sha\"", "--sha256 \"$expected_sha\""),
        (
            "--format quiet)",
            "--cache-dir \"$RUNNER_TEMP\" --format quiet)",
        ),
    ] {
        assert!(SOURCE.contains(from));
        assert!(validate(&SOURCE.replace(from, to)).is_err(), "{from}");
    }
}
#[test]
fn sccache_socket_requires_attempt_identity_in_runner_temporary_directory() {
    for (from, to) in [
        (
            "agentic-replay-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.sock",
            "agentic-replay-${GITHUB_RUN_ID}.sock",
        ),
        (
            "SCCACHE_SERVER_UDS=$RUNNER_TEMP/",
            "SCCACHE_SERVER_UDS=$HOME/",
        ),
        (
            "echo \"SCCACHE_SERVER_UDS=",
            "echo ignored \"SCCACHE_SERVER_UDS=",
        ),
    ] {
        assert!(validate(&SOURCE.replace(from, to)).is_err(), "{from}");
    }
}
#[test]
fn repair_requires_explicit_regression_owned_provider_model_log_and_outputs() {
    for (from, to) in [
        ("steps.history.outputs.repair_required == 'true'", "true"),
        ("vars.LLAMA_CANARY_GOOSE_PROVIDER", "inputs.provider"),
        ("vars.LLAMA_CANARY_GOOSE_MODEL", "inputs.model"),
        (
            "tee \"$RUNNER_TEMP/agentic-replay-artifacts/repair.log\"",
            "tee /dev/null",
        ),
        ("prepared=false", "prepared=true"),
        ("set -euo pipefail", "set -eu"),
    ] {
        assert!(SOURCE.contains(from));
        assert!(validate(&SOURCE.replace(from, to)).is_err(), "{from}");
    }
}
#[test]
fn anonymous_history_cannot_acquire_step_tokens_or_precede_cache_setup() {
    let mut document = workflow_yaml::parse(SOURCE).unwrap();
    let Node::Map(jobs) = h::mutable(&mut document, "jobs") else {
        panic!("jobs")
    };
    let job = &mut jobs
        .iter_mut()
        .find(|(name, _)| name == "replay")
        .unwrap()
        .1;
    let Node::Seq(steps) = h::mutable(job, "steps") else {
        panic!("steps")
    };
    let (_, baseline) = steps
        .iter_mut()
        .enumerate()
        .find(|(_, step)| field(step, "id") == Some("baseline"))
        .unwrap();
    let Node::Map(entries) = baseline else {
        panic!("baseline")
    };
    entries.push((
        "env".into(),
        Node::Map(vec![(
            "HF_TOKEN".into(),
            Node::Scalar("${{ secrets.HF_TOKEN }}".into()),
        )]),
    ));
    assert!(
        check(&BTreeMap::from([(
            "agentic-replay-nightly.yml".into(),
            document
        )]))
        .is_err()
    );
    let mut document = workflow_yaml::parse(SOURCE).unwrap();
    let Node::Map(jobs) = h::mutable(&mut document, "jobs") else {
        panic!("jobs")
    };
    let job = &mut jobs
        .iter_mut()
        .find(|(name, _)| name == "replay")
        .unwrap()
        .1;
    let Node::Seq(steps) = h::mutable(job, "steps") else {
        panic!("steps")
    };
    let first = steps
        .iter()
        .position(|step| field(step, "name") == Some("Verify runner toolchain"))
        .unwrap();
    let second = steps
        .iter()
        .position(|step| field(step, "id") == Some("inputs"))
        .unwrap();
    steps.swap(first, second);
    assert!(
        check(&BTreeMap::from([(
            "agentic-replay-nightly.yml".into(),
            document
        )]))
        .is_err()
    );
}
