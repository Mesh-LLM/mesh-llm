//! Parsed current canary graph and authority contracts; no hosted execution proof.
use crate::workflow_yaml::{self, Node};
use std::{collections::BTreeSet, fs, path::PathBuf};
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn document(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(root().join(path)).unwrap()).unwrap()
}
fn field<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap()
}
fn steps(job: &Node) -> &[Node] {
    let Node::Seq(items) = job.get("steps").unwrap() else {
        panic!("steps");
    };
    items
}
fn named<'a>(job: &'a Node, key: &str, value: &str) -> (usize, &'a Node) {
    let matches: Vec<_> = steps(job)
        .iter()
        .enumerate()
        .filter(|(_, step)| step.get(key).and_then(Node::text) == Some(value))
        .collect();
    assert_eq!(matches.len(), 1, "unique structural step {key}={value}");
    matches[0]
}
fn needs(job: &Node) -> BTreeSet<&str> {
    job.get("needs").unwrap().list().into_iter().collect()
}
fn reject_secrets(node: &Node) -> Result<(), String> {
    match node {
        Node::Scalar(value)
            if ["secrets.", "github.token", "CANARY_REPAIR_TOKEN"]
                .iter()
                .any(|key| value.contains(key)) =>
        {
            Err("publication credentials in worker/controller node".into())
        }
        Node::Scalar(_) => Ok(()),
        Node::Seq(values) => values.iter().try_for_each(reject_secrets),
        Node::Map(values) => values.iter().try_for_each(|(key, value)| {
            if matches!(
                key.as_str(),
                "GH_TOKEN" | "GITHUB_TOKEN" | "CANARY_REPAIR_TOKEN"
            ) {
                return Err("publication credential key in worker/controller node".into());
            }
            reject_secrets(value)
        }),
    }
}
fn read_only(node: &Node) -> Result<(), String> {
    match node {
        Node::Map(entries)
            if entries
                .iter()
                .all(|(_, value)| matches!(value.text(), Some("read" | "none"))) =>
        {
            Ok(())
        }
        _ => Err("worker permissions are not an explicit read/none map".into()),
    }
}
fn security(controller: &Node, worker: &Node) -> Result<(), String> {
    read_only(
        controller
            .get("permissions")
            .ok_or("controller permissions absent")?,
    )?;
    read_only(
        worker
            .get("permissions")
            .ok_or("worker permissions absent")?,
    )?;
    if worker
        .get("on")
        .and_then(|node| node.get("workflow_call"))
        .and_then(|node| node.get("secrets"))
        .is_some()
    {
        return Err("worker declares secrets".into());
    }
    reject_secrets(worker)?;
    let jobs = controller.get("jobs").ok_or("controller jobs absent")?;
    for (name, job) in jobs.entries() {
        if name != "publish-certified-canary" {
            reject_secrets(job)?;
        }
        if let Some(permissions) = job.get("permissions")
            && permissions
                .get("contents")
                .is_some_and(|value| value.text() != Some("read"))
        {
            return Err("job contents permission escalation".into());
        }
        if job.get("secrets").is_some() {
            return Err("reusable invocation forwards secrets".into());
        }
    }
    for (_, job) in worker.get("jobs").ok_or("worker jobs absent")?.entries() {
        if let Some(permissions) = job.get("permissions") {
            read_only(permissions)?;
        }
        if job.get("secrets").is_some() {
            return Err("worker forwards secrets".into());
        }
    }
    Ok(())
}
fn change<'a>(node: &'a mut Node, keys: &[&str]) -> &'a mut Node {
    if let Some((key, tail)) = keys.split_first() {
        let Node::Map(entries) = node else {
            panic!("map");
        };
        change(
            &mut entries
                .iter_mut()
                .find(|(name, _)| name.as_str() == *key)
                .unwrap()
                .1,
            tail,
        )
    } else {
        node
    }
}
#[test]
fn daily_manual_canary_queues_behind_active_work_without_worker_cancellation() {
    let controller = document(".github/workflows/llama-upstream-canary.yml");
    let triggers = controller.get("on").unwrap();
    assert_eq!(
        triggers
            .entries()
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["schedule", "workflow_dispatch"])
    );
    let Node::Seq(schedules) = triggers.get("schedule").unwrap() else {
        panic!("schedule");
    };
    assert_eq!(
        schedules
            .iter()
            .map(|row| field(row, "cron"))
            .collect::<Vec<_>>(),
        ["47 3 * * *"]
    );
    let concurrency = controller.get("concurrency").unwrap();
    assert_eq!(field(concurrency, "group"), "llama-upstream-canary");
    assert_eq!(field(concurrency, "cancel-in-progress"), "false");
    let worker = document(".github/workflows/llama-canary-family-pass.yml");
    for (_, job) in worker.get("jobs").unwrap().entries() {
        assert!(job.get("concurrency").is_none());
    }
}
#[test]
fn protected_controller_and_worker_credentials_refuse_permission_or_token_mutations() {
    let controller = document(".github/workflows/llama-upstream-canary.yml");
    let worker = document(".github/workflows/llama-canary-family-pass.yml");
    security(&controller, &worker).unwrap();
    assert_eq!(
        field(controller.get("permissions").unwrap(), "contents"),
        "read"
    );
    assert_eq!(
        field(worker.get("permissions").unwrap(), "contents"),
        "read"
    );
    let resolve = controller.get("jobs").unwrap().get("resolve").unwrap();
    assert_eq!(
        field(resolve, "if"),
        "github.repository == 'Mesh-LLM/mesh-llm' && github.ref == 'refs/heads/main'"
    );
    assert_eq!(
        field(steps(resolve)[0].get("with").unwrap(), "ref"),
        "${{ github.sha }}"
    );
    assert_eq!(
        field(
            steps(resolve)[0].get("with").unwrap(),
            "persist-credentials"
        ),
        "false"
    );
    let build = worker.get("jobs").unwrap().get("build").unwrap();
    assert_eq!(field(build, "if"), field(resolve, "if"));
    assert_eq!(
        field(steps(build)[0].get("with").unwrap(), "ref"),
        "${{ inputs.source }}"
    );
    for (_, job) in worker.get("jobs").unwrap().entries() {
        for step in steps(job) {
            if step
                .get("uses")
                .and_then(Node::text)
                .is_some_and(|value| value.starts_with("actions/checkout@"))
            {
                assert_eq!(
                    field(step.get("with").unwrap(), "persist-credentials"),
                    "false"
                );
            }
        }
    }
    let mut escalated = worker.clone();
    *change(&mut escalated, &["permissions", "contents"]) = Node::Scalar("write".into());
    assert!(security(&controller, &escalated).is_err());
    let mut token = worker.clone();
    *change(&mut token, &["env", "CANARY_CONTROLLER_SHA"]) =
        Node::Scalar("${{ github.token }}".into());
    assert!(security(&controller, &token).is_err());
    let mut forwarded = controller.clone();
    let Node::Map(candidate) = change(&mut forwarded, &["jobs", "candidate"]) else {
        panic!("job");
    };
    candidate.push(("secrets".into(), Node::Scalar("inherit".into())));
    assert!(security(&forwarded, &worker).is_err());
}
#[test]
fn changed_pin_graph_requires_preflight_candidate_independent_verification_and_publish_decision() {
    let controller = document(".github/workflows/llama-upstream-canary.yml");
    let jobs = controller.get("jobs").unwrap();
    for (name, dependencies) in [
        ("preflight", vec!["resolve"]),
        ("candidate", vec!["resolve", "preflight"]),
        ("verification", vec!["resolve", "candidate"]),
        (
            "result",
            vec!["resolve", "preflight", "candidate", "verification"],
        ),
        ("publish-certified-canary", vec!["resolve", "result"]),
    ] {
        assert_eq!(
            needs(jobs.get(name).unwrap()),
            dependencies.into_iter().collect()
        );
    }
    let candidate = jobs.get("candidate").unwrap();
    let verification = jobs.get("verification").unwrap();
    assert_eq!(
        field(candidate, "uses"),
        "./.github/workflows/llama-canary-family-pass.yml"
    );
    assert_eq!(field(verification, "uses"), field(candidate, "uses"));
    assert_eq!(
        field(candidate, "if"),
        "${{ !cancelled() && needs.preflight.result == 'success' }}"
    );
    assert_eq!(
        field(verification, "if"),
        "${{ !cancelled() && needs.resolve.outputs.changed == 'true' && needs.candidate.outputs.green == 'true' }}"
    );
    let with = verification.get("with").unwrap();
    assert_eq!(field(with, "mode"), "verify-build");
    for (key, value) in [("source", "source"), ("upstream", "upstream")] {
        assert_eq!(
            field(with, key),
            format!("${{{{ needs.resolve.outputs.{value} }}}}")
        );
    }
    for key in ["package", "identity", "head"] {
        assert_eq!(
            field(with, &format!("previous_{key}")),
            format!("${{{{ needs.candidate.outputs.{key} }}}}")
        );
    }
    assert_eq!(
        jobs.entries()
            .iter()
            .filter(
                |(_, job)| job.get("uses").and_then(Node::text) == Some(field(candidate, "uses"))
            )
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["candidate", "verification"])
    );
    let publish = jobs.get("publish-certified-canary").unwrap();
    assert_eq!(
        field(publish, "if"),
        "${{ needs.result.outputs.publish == 'true' }}"
    );
    assert_eq!(field(publish, "runs-on"), "ubuntu-24.04");
    let (_, action) = named(publish, "id", "publish");
    assert_eq!(field(action, "run"), "scripts/llama-canary-publish.sh");
    assert_eq!(
        field(action.get("env").unwrap(), "CANARY_REPAIR_TOKEN"),
        "${{ secrets.CANARY_REPAIR_TOKEN }}"
    );
    let worker = document(".github/workflows/llama-canary-family-pass.yml");
    let build = worker.get("jobs").unwrap().get("build").unwrap();
    assert_eq!(field(build, "timeout-minutes"), "1430");
    let (_, action) = named(build, "id", "build");
    let env = action.get("env").unwrap();
    assert_eq!(field(env, "CANARY_AGENT_TIMEOUT_SECONDS"), "41400");
    assert_eq!(field(env, "CANARY_VERIFICATION_TIMEOUT_SECONDS"), "43200");
    assert!(field(env, "CANARY_AGENT_PROVIDER").contains("vars.LLAMA_CANARY_GOOSE_PROVIDER"));
    assert!(field(env, "CANARY_AGENT_MODEL").contains("glm-5.3-flash"));
    let (_, evidence) = named(build, "id", "upload_evidence");
    assert_eq!(field(evidence.get("with").unwrap(), "retention-days"), "14");
}
#[test]
fn frozen_prebuilt_workers_load_cache_after_controller_and_before_consumption() {
    let controller = document(".github/workflows/llama-upstream-canary.yml");
    let worker = document(".github/workflows/llama-canary-family-pass.yml");
    let worker_jobs = worker.get("jobs").unwrap();
    for (job, consumer) in [
        (
            controller.get("jobs").unwrap().get("preflight").unwrap(),
            "Verify immutable family plan and pinned cache",
        ),
        (
            worker_jobs.get("build").unwrap(),
            "Build exact candidate for distributed certification",
        ),
        (worker_jobs.get("family").unwrap(), "Certify one family"),
    ] {
        let (prepare, step) = named(job, "uses", "./.github/actions/prepare-automation");
        let (cache, _) = named(job, "uses", "./.github/actions/use-canary-cache");
        let (consume, _) = named(job, "name", consumer);
        assert!(prepare < cache && cache < consume);
        let config = step.get("with").unwrap();
        assert_eq!(field(config, "allow_depot_remote_cache"), "false");
        assert_eq!(field(config, "allow_native_github_cache"), "false");
        assert!(
            job.get("env")
                .unwrap()
                .entries()
                .iter()
                .all(|(key, _)| !key.starts_with("HF_"))
        );
    }
    let family = worker_jobs.get("family").unwrap();
    assert_eq!(needs(family), BTreeSet::from(["build"]));
    let strategy = family.get("strategy").unwrap();
    assert_eq!(field(strategy, "fail-fast"), "false");
    assert_eq!(
        field(strategy, "matrix"),
        "${{ fromJSON(needs.build.outputs.matrix) }}"
    );
    let (restore, _) = named(
        family,
        "name",
        "Verify immutable handoff and restore producer executables",
    );
    let (certify, _) = named(family, "id", "certify");
    assert!(restore < certify);
    let setup = document(".github/actions/setup-canary-runner/action.yml");
    assert!(
        steps(setup.get("runs").unwrap())
            .iter()
            .all(|step| step.get("uses").and_then(Node::text)
                != Some("./.github/actions/use-canary-cache"))
    );
    let cache = document(".github/actions/use-canary-cache/action.yml");
    let (_, step) = named(
        cache.get("runs").unwrap(),
        "name",
        "Read configured model cache",
    );
    assert_eq!(field(step, "shell"), "/bin/zsh -il {0}");
    assert_eq!(
        field(step, "run").trim(),
        "set -eu\n\"${MESH_LLM_AUTOMATION_BIN:?protected controller automation required}\" ci-ops configure-canary-cache"
    );
}
