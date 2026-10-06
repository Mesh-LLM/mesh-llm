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

// Local component execution of the retained action declarations; no GitHub or toolchain calls.
fn component(
    root: &std::path::Path,
    executable: &std::path::Path,
    args: &[std::ffi::OsString],
) -> (i32, Vec<u8>, Vec<u8>) {
    use crate::process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value};
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: executable.into(),
            cwd: root.into(),
            arguments: args.iter().cloned().map(Value::Public).collect(),
            environment: std::collections::BTreeMap::from([(
                "PATH".into(),
                Value::Public("/usr/bin:/bin".into()),
            )]),
        },
        &Limits {
            execution: std::time::Duration::from_secs(10),
            graceful_shutdown: std::time::Duration::from_secs(1),
            forced_shutdown: std::time::Duration::from_secs(1),
            retained_bytes_per_stream: 262144,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(262144),
            stderr: std::num::NonZeroUsize::new(262144),
        },
    )
    .unwrap();
    let p = &report.process;
    assert_eq!(p.outcome, process::Outcome::Exited, "{p:?}");
    assert!(
        p.failure.is_none()
            && p.cleanup.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed,
        "{p:?}"
    );
    assert!(
        p.stdout.line_capture_complete
            && p.stderr.line_capture_complete
            && !p.stdout.truncated
            && !p.stderr.truncated,
        "{p:?}"
    );
    let stdout = report.stdout.unwrap().as_bytes().to_vec();
    let stderr = report.stderr.unwrap().as_bytes().to_vec();
    assert_eq!(u64::try_from(stdout.len()).unwrap(), p.stdout.bytes_seen);
    assert_eq!(u64::try_from(stderr.len()).unwrap(), p.stderr.bytes_seen);
    (p.status.unwrap().code().unwrap(), stdout, stderr)
}
fn action_shell(root: &std::path::Path, setup: &str, production: &str) -> (i32, Vec<u8>, Vec<u8>) {
    component(
        root,
        std::path::Path::new("/bin/bash"),
        &[
            "-c".into(),
            format!("set -euo pipefail\n{setup}\n{production}").into(),
        ],
    )
}
#[test]
fn parsed_family_worker_preserves_bounded_fanout_and_prebuilt_restore_before_certification() {
    let worker = document(".github/workflows/llama-canary-family-pass.yml");
    let jobs = worker.get("jobs").unwrap();
    let family = jobs.get("family").unwrap();
    assert_eq!(needs(family), BTreeSet::from(["build"]));
    let strategy = family.get("strategy").unwrap();
    assert_eq!(field(strategy, "max-parallel"), "8");
    assert_eq!(field(strategy, "fail-fast"), "false");
    assert_eq!(
        field(strategy, "matrix"),
        "${{ fromJSON(needs.build.outputs.matrix) }}"
    );
    let (restore, restored) = named(
        family,
        "name",
        "Verify immutable handoff and restore producer executables",
    );
    let (certify, certified) = named(family, "id", "certify");
    assert!(restore < certify);
    assert!(field(restored, "run").contains("automation canary-receipts restore --input"));
    let env = certified.get("env").unwrap();
    assert_eq!(field(env, "SHARD_INDEX"), "${{ matrix.shard_index }}");
    assert_eq!(field(env, "MEMORY_TIER"), "${{ matrix.memory_tier }}");
    let run = field(certified, "run");
    assert!(run.lines().any(|l| l.trim()
        == "\"$MESH_LLM_AUTOMATION_BIN\" automation canary-receipts certify --input \"$input\""));
    assert!(run.contains("shard_index:$shard_index,memory_tier:$memory_tier"));
    assert!(steps(family).iter().all(|s| {
        s.get("run")
            .and_then(Node::text)
            .is_none_or(|s| !s.contains("cargo "))
    }));
    let build = jobs.get("build").unwrap().get("env").unwrap();
    assert_eq!(field(build, "LLAMA_STAGE_BACKEND"), "metal");
    assert!(field(build, "LLAMA_STAGE_BUILD_DIR").contains("inputs.pass_id"));
    let certification = fs::read_to_string(
        root().join("tools/xtask/src/automation/canary_receipts/package_closure/certification.rs"),
    )
    .unwrap();
    assert!(
        certification.contains("\"--skip-build\".into()")
            && certification.contains("\"--plan\".into()")
            && certification.contains("\"--shard-index\".into()")
    );
}
fn setup_observers() -> &'static str {
    r#"
export GITHUB_ENV="$PWD/action.env"
export MACOSX_DEPLOYMENT_TARGET=13.3
for tool in cargo just sccache jq python3 uv cmake ninja hf git-lfs lipo xcrun; do eval "$tool() { :; }"; done
uname() { printf 'Darwin\n'; }
git() { printf '%s\n' "$*" >> "$PWD/git.observed"; case "$*" in 'config --local user.name mesh-llama-canary-bot'|'config --local user.email llama-canary-bot@meshllm.invalid'|'lfs install --force') return 0;; *) return 91;; esac; }
brew() { case "$*" in '--prefix llvm@22') printf '%s\n' "$PWD/pinned";; '--prefix llvm') printf '%s\n' "$PWD/fallback";; *) return 92;; esac; }
"#
}
#[test]
fn actual_setup_action_preserves_local_git_identity_and_llvm_fallback_refusal_without_installing() {
    let action = document(".github/actions/setup-canary-runner/action.yml");
    let runs = action.get("runs").unwrap();
    let (identity, git) = named(runs, "name", "Configure canary Git identity");
    let (toolchain, tools) = named(runs, "name", "Verify runner toolchain");
    assert!(identity < toolchain);
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().canonicalize().unwrap();
    fs::create_dir_all(directory.join("scripts/lib")).unwrap();
    for name in ["macos-deployment-target.sh", "macos-deployment-target.txt"] {
        fs::copy(
            root().join("scripts/lib").join(name),
            directory.join("scripts/lib").join(name),
        )
        .unwrap();
    }
    for name in ["pinned", "fallback"] {
        let prefix = directory.join(name);
        fs::create_dir_all(prefix.join("bin")).unwrap();
        fs::create_dir_all(prefix.join("lib/cmake/clang")).unwrap();
        let clang = prefix.join("bin/clang");
        fs::write(&clang, "#!/bin/sh\nexit 91\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&clang, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(prefix.join("lib/cmake/clang/ClangConfig.cmake"), "").unwrap();
    }
    let production = format!("{}\n{}", field(git, "run"), field(tools, "run"));
    let (status, _, stderr) = action_shell(&directory, setup_observers(), &production);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    let observed = fs::read_to_string(directory.join("git.observed")).unwrap();
    assert!(
        observed.contains("config --local user.name mesh-llama-canary-bot")
            && observed.contains("config --local user.email llama-canary-bot@meshllm.invalid")
    );
    assert!(
        fs::read_to_string(directory.join("action.env"))
            .unwrap()
            .contains(&format!(
                "SKIPPY_REWRITER_LLVM_PREFIX={}/pinned",
                directory.display()
            ))
    );
    fs::remove_file(directory.join("pinned/lib/cmake/clang/ClangConfig.cmake")).unwrap();
    fs::write(directory.join("action.env"), "").unwrap();
    let (status, _, _) = action_shell(&directory, setup_observers(), &production);
    assert_eq!(status, 0);
    assert!(
        fs::read_to_string(directory.join("action.env"))
            .unwrap()
            .contains(&format!(
                "SKIPPY_REWRITER_LLVM_PREFIX={}/fallback",
                directory.display()
            ))
    );
    fs::remove_file(directory.join("fallback/lib/cmake/clang/ClangConfig.cmake")).unwrap();
    fs::write(directory.join("action.env"), "").unwrap();
    let (status, _, stderr) = action_shell(&directory, setup_observers(), &production);
    assert_eq!(status, 1);
    assert!(String::from_utf8_lossy(&stderr).contains("installs nothing"));
    assert!(
        !fs::read_to_string(directory.join("action.env"))
            .unwrap()
            .contains("SKIPPY_REWRITER_LLVM_PREFIX=")
    );
    temp.close().unwrap();
}
#[test]
fn actual_scheduled_alert_reconciles_first_failure_repeated_failure_and_recovery_with_finite_api() {
    let controller = document(".github/workflows/llama-upstream-canary.yml");
    let alert = controller
        .get("jobs")
        .unwrap()
        .get("alert-consecutive-failures")
        .unwrap();
    assert_eq!(
        field(alert, "if"),
        "${{ !cancelled() && github.event_name == 'schedule' }}"
    );
    assert_eq!(
        needs(alert),
        BTreeSet::from(["resolve", "result", "publish-certified-canary"])
    );
    assert_eq!(field(alert, "runs-on"), "ubuntu-24.04");
    assert_eq!(field(alert, "timeout-minutes"), "5");
    let permissions = alert.get("permissions").unwrap();
    assert_eq!(field(permissions, "actions"), "read");
    assert_eq!(field(permissions, "issues"), "write");
    assert!(steps(alert).iter().all(|s| {
        s.get("uses")
            .and_then(Node::text)
            .is_none_or(|s| !s.starts_with("actions/checkout@"))
    }));
    let (_, step) = named(alert, "name", "Reconcile consecutive-failure alert");
    assert_eq!(field(step, "continue-on-error"), "true");
    let script = field(step.get("with").unwrap(), "script");
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().canonicalize().unwrap();
    fs::write(directory.join("script.js"), script).unwrap();
    fs::write(directory.join("fixture.cjs"), ALERT_FIXTURE).unwrap();
    let node = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|p| p.join("node"))
        .find(|p| p.is_file())
        .expect("required Node component")
        .canonicalize()
        .unwrap();
    let (status, stdout, stderr) =
        component(&directory, &node, &[directory.join("fixture.cjs").into()]);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    let cases: serde_json::Value = serde_json::from_slice(&stdout).unwrap();
    for (index, expected) in [
        (0, vec![]),
        (1, vec![]),
        (2, vec!["create"]),
        (3, vec!["comment"]),
        (4, vec!["comment", "update"]),
        (5, vec!["create"]),
        (6, vec![]),
    ] {
        let calls = cases[index]["calls"].as_array().unwrap();
        assert_eq!(
            calls
                .iter()
                .map(|r| r["kind"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
    }
    assert_eq!(cases[4]["calls"][1]["options"]["state"], "closed");
    assert_eq!(
        cases[2]["calls"][0]["options"]["title"],
        "[CI] llama.cpp upstream canary failing repeatedly"
    );
    assert_eq!(cases[3]["calls"][0]["options"]["issue_number"], 7);
    assert_eq!(cases[4]["runReads"], 0);
    temp.close().unwrap();
}
const ALERT_FIXTURE: &str = r#"
'use strict';
const fs=require('node:fs');
global.fetch=async()=>{throw Error('fixture forbids network');};
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
const source=fs.readFileSync('script.js','utf8');
(async()=>{
const outputs=[];
for(const config of [
 {previous:null,alert:false,repair:'failure'},
 {previous:'success',alert:false,repair:'failure'},
 {previous:'failure',alert:false,repair:'failure'},
 {previous:'failure',alert:true,repair:'failure'},
 {previous:'failure',alert:true,repair:'success'},
 {previous:'failure',alert:false,repair:'success',changed:'true',verify:'failure'},
 {previous:'failure',alert:false,repair:'success'},
]) {
 const calls=[];let runReads=0;
 const record=kind=>async options=>{calls.push({kind,options});};
 const issueList=()=>{throw Error('pagination only');},runList=()=>{throw Error('pagination only');};
 const github={rest:{issues:{listForRepo:issueList,create:record('create'),createComment:record('comment'),update:record('update')},actions:{listWorkflowRuns:runList}},paginate:async(fn,options)=>{
   if(fn===issueList)return config.alert?[{number:7,body:'<!-- llama-upstream-canary-consecutive-failure-alert -->'}]:[];
   if(fn===runList){runReads++;if(options.event!=='schedule'||options.status!=='completed')throw Error('wrong history');return config.previous?[{id:1,conclusion:config.previous,html_url:'https://fixture.invalid/previous'}]:[];}
   throw Error('unowned API');
 }};
 const context={repo:{owner:'fixture',repo:'repo'},runId:2};
 const fakeProcess={env:{REPAIR_RESULT:config.repair,VERIFY_RESULT:config.verify||'success',PUBLISH_RESULT:'success',CHANGED:config.changed||'false'}};
 await new AsyncFunction('github','context','process',source)(github,context,fakeProcess);
 outputs.push({calls,runReads});
}
process.stdout.write(JSON.stringify(outputs));
})().catch(e=>{process.stderr.write(e.message);process.exitCode=1;});
"#;
fn inert_executable(path: &std::path::Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
#[test]
fn actual_rewriter_preamble_and_cache_slice_reenter_native_arch_and_invalidate_only_mismatched_tool()
 {
    let source =
        fs::read_to_string(root().join("scripts/check-skippy-generated-family-patch.sh")).unwrap();
    let prefix = source.split_once("\nROOT=").unwrap().0;
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().canonicalize().unwrap();
    let bin = directory.join("bin");
    fs::create_dir(&bin).unwrap();
    inert_executable(
        &bin.join("uname"),
        "#!/bin/sh\ncase \"$1\" in -s) printf 'Darwin\\n';; -m) printf 'x86_64\\n';; *) exit 91;; esac\n",
    );
    inert_executable(
        &bin.join("sysctl"),
        "#!/bin/sh\n[ \"$*\" = '-n hw.optional.arm64' ] || exit 92\nprintf '1\\n'\n",
    );
    inert_executable(
        &bin.join("arch"),
        "#!/bin/sh\nprintf '%s\\0' \"$@\" > \"$PWD/arch.argv\"\n[ \"$1\" = '-arm64' ] || exit 93\n",
    );
    let preamble = directory.join("preamble.sh");
    fs::write(&preamble, prefix).unwrap();
    let command = "PATH=\"$PWD/bin\" /bin/bash ./preamble.sh --fixture 'bytes with spaces'";
    let (status, _, stderr) = action_shell(&directory, "", command);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    let argv = fs::read(directory.join("arch.argv")).unwrap();
    assert_eq!(
        argv.split(|b| *b == 0)
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>(),
        [
            b"-arm64".as_slice(),
            b"./preamble.sh",
            b"--fixture",
            b"bytes with spaces"
        ]
    );
    let cache = source
        .split_once("\nmkdir -p \"$ARTIFACT_ROOT\"\n")
        .unwrap()
        .1
        .split_once("\nEXTRA_ARGS=")
        .unwrap()
        .0;
    let setup = r#"
ROOT="$PWD"
ARTIFACT_ROOT="$PWD/artifacts"
TOOL_BUILD="$ARTIFACT_ROOT/tool-build"
LLVM_PREFIX="$PWD/llvm"
NATIVE_ARCH=arm64
uname() { printf 'Darwin\n'; }
cmake() { printf '%s\0' "$@" >> "$PWD/cmake.argv"; }
ctest() { printf '%s\0' "$@" > "$PWD/ctest.argv"; }
"#;
    for arch in ["x86_64", "arm64"] {
        let tool = directory.join("artifacts/tool-build");
        fs::create_dir_all(&tool).unwrap();
        fs::write(
            tool.join("CMakeCache.txt"),
            format!("CMAKE_OSX_ARCHITECTURES:STRING={arch}\n"),
        )
        .unwrap();
        fs::write(tool.join("sentinel"), "fixture").unwrap();
        fs::write(directory.join("sibling-preserve"), "untouched").unwrap();
        fs::write(directory.join("cmake.argv"), "").unwrap();
        let (status, _, stderr) = action_shell(&directory, setup, cache);
        assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
        assert_eq!(tool.join("sentinel").exists(), arch == "arm64");
        assert_eq!(
            fs::read(directory.join("sibling-preserve")).unwrap(),
            b"untouched"
        );
        assert!(
            fs::read(directory.join("cmake.argv"))
                .unwrap()
                .split(|b| *b == 0)
                .any(|a| a == b"-DCMAKE_OSX_ARCHITECTURES=arm64")
        );
    }
    temp.close().unwrap();
}
