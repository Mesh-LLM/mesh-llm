//! Component-owned contracts for the existing Node action, with no live GitHub access.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Duration};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn node_executable() -> PathBuf {
    let name = if cfg!(windows) { "node.exe" } else { "node" };
    std::env::split_paths(&std::env::var_os("PATH").expect("Node fixtures require PATH"))
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
        .expect("Node component fixture executable is required")
        .canonicalize()
        .unwrap()
}

fn action_case(body: &str) -> Json {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let action = root.join("action.js");
    fs::copy(
        repository().join(".github/actions/cancel-pr-sibling-runs/index.js"),
        &action,
    )
    .unwrap();
    let harness = root.join("fixture.js");
    fs::write(&harness, format!(r#"
'use strict';
global.fetch = async () => {{ throw Error('unconfigured fixture HTTP request'); }};
const action = require(process.argv[2]);
const sha = 'b'.repeat(40);
const trigger = {{createdAt: Date.parse('2026-08-20T12:00:00Z'), headSha: sha, pullNumber: 42, triggerRunId: 201}};
const base = {{event:'pull_request',head_sha:sha,created_at:'2026-08-20T12:00:20Z',pull_requests:[{{number:42}}],status:'in_progress'}};
(async () => {{ const result = await (async () => {{ {body} }})(); process.stdout.write(JSON.stringify(result)); }})().catch(error => {{ console.error(error.message); process.exitCode = 1; }});
"#)).unwrap();
    let report = process::supervise(
        &ProcessSpec {
            executable: node_executable(),
            cwd: root,
            arguments: vec![Value::Public(harness.into()), Value::Public(action.into())],
            environment: BTreeMap::from([(
                "PATH".into(),
                Value::Public(std::env::var_os("PATH").unwrap()),
            )]),
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    assert!(
        report.success(),
        "{}",
        String::from_utf8_lossy(&report.stderr.bytes_retained)
    );
    serde_json::from_slice(&report.stdout.bytes_retained).unwrap()
}

#[test]
fn sibling_trigger_requires_exact_protected_quality_pr_identity() {
    let result = action_case(
        r#"
const payload = {repository:{full_name:'Mesh-LLM/mesh-llm'},workflow_run:{id:201,name:'PR · Quality',event:'pull_request',head_sha:sha,created_at:'2026-08-20T12:00:00Z',pull_requests:[{number:42}]}};
const selected = action.parseTrigger(payload,'Mesh-LLM/mesh-llm');
const rejected=[];
for (const patch of [{name:'PR · Linux'},{id:0},{head_sha:'main'},{created_at:'invalid'},{pull_requests:[]},{pull_requests:[{number:42},{number:43}]}]) {
  try { action.parseTrigger({...payload,workflow_run:{...payload.workflow_run,...patch}},'Mesh-LLM/mesh-llm'); rejected.push(false); } catch { rejected.push(true); }
}
let foreign=false;try {action.parseTrigger(payload,'foreign/repo');}catch{foreign=true;}
return {selected,rejected,foreign,nonPr:action.parseTrigger({...payload,workflow_run:{...payload.workflow_run,event:'push'}},'Mesh-LLM/mesh-llm')};
"#,
    );
    assert_eq!(result["selected"]["pullNumber"], 42);
    assert_eq!(result["selected"]["triggerRunId"], 201);
    assert_eq!(result["selected"]["headSha"], "b".repeat(40));
    assert!(
        result["rejected"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v == &json!(true))
    );
    assert_eq!(result["foreign"], true);
    assert!(result["nonPr"].is_null());
}

#[test]
fn sibling_selection_excludes_foreign_revision_epoch_and_wrong_quality_run() {
    let result = action_case(
        r#"
const runs=[{...base,id:201,name:'PR · Quality'},{...base,id:202,name:'PR · Linux'},{...base,id:203,name:'PR · Windows',head_sha:'c'.repeat(40)},{...base,id:204,name:'PR · macOS',pull_requests:[{number:43}]},{...base,id:205,name:'PR · Website',created_at:'2026-08-20T11:40:00Z'},{...base,id:206,name:'Release'},{...base,id:207,name:'PR · Quality'},{...base,id:208,name:'PR · Linux',event:'push'}];
return action.selectTargetRuns(runs,trigger).map(run=>[run.id,run.name]);
"#,
    );
    assert_eq!(result, json!([[202, "PR · Linux"]]));
}

#[test]
fn sibling_first_failure_is_preserved_and_only_active_other_runs_cancel() {
    let result = action_case(
        r#"
const runs=[{id:301,name:'PR · Quality',status:'in_progress'},{id:302,name:'PR · Website',status:'completed'},{id:303,name:'PR · Linux',status:'in_progress'},{id:304,name:'PR · macOS',status:'queued'},{id:305,name:'PR · Windows',status:'completed'}];
const failure=action.findEarliestFailure([{run:runs[2],jobs:[{id:2,name:'later',conclusion:'failure',completed_at:'2026-08-20T12:02:00Z'}]},{run:runs[0],jobs:[{id:1,name:'first',conclusion:'failure',completed_at:'2026-08-20T12:01:00Z'}]}]);
return {failure,cancel:action.cancellableSiblingRuns(runs,failure.runId).map(run=>run.id)};
"#,
    );
    assert_eq!(result["failure"]["runId"], 301);
    assert_eq!(result["cancel"], json!([303, 304]));
}

#[test]
fn sibling_clean_completion_requires_all_five_terminal_lanes() {
    let result = action_case(
        r#"
const complete=action.TARGET_WORKFLOWS.map((name,index)=>({id:index+1,name,status:'completed'}));
return {complete:action.allTargetsTerminal(complete),missing:action.allTargetsTerminal(complete.slice(0,4)),active:action.allTargetsTerminal(complete.map((run,index)=>index===2?{...run,status:'in_progress'}:run))};
"#,
    );
    assert_eq!(
        result,
        json!({"complete":true,"missing":false,"active":false})
    );
}

#[test]
fn sibling_api_empty_202_and_terminal_races_preserve_cancellation_semantics() {
    let result = action_case(
        r#"
let status=202;const methods=[];
global.fetch=async (url,options)=>{methods.push([new URL(url).pathname,options.method,options.signal instanceof AbortSignal]);return {ok:status===202,status,text:async()=>''};};
const api=action.githubApi('fixture-token','owner','repo');
const accepted=await api.cancelRun(123);status=409;const conflict=await api.cancelRun(123);status=422;const terminal=await api.cancelRun(123);status=403;let forbidden=false;try{await api.cancelRun(123);}catch(error){forbidden=error.status===403;}
return {accepted,conflict,terminal,forbidden,methods};
"#,
    );
    assert_eq!(result["accepted"], true);
    assert_eq!(result["conflict"], false);
    assert_eq!(result["terminal"], false);
    assert_eq!(result["forbidden"], true);
    for row in result["methods"].as_array().unwrap() {
        assert_eq!(
            row,
            &json!(["/repos/owner/repo/actions/runs/123/cancel", "POST", true])
        );
    }
}

#[test]
fn sibling_api_paginates_both_runs_and_jobs_with_request_deadlines() {
    let result = action_case(
        r#"
const pages=[];
global.fetch=async(rawUrl,options)=>{const url=new URL(rawUrl),page=Number(url.searchParams.get('page')),field=url.pathname.endsWith('/jobs')?'jobs':'workflow_runs';pages.push([field,page,options.signal instanceof AbortSignal]);return {ok:true,status:200,text:async()=>JSON.stringify({[field]:Array.from({length:page===1?100:1},(_,index)=>({id:(page-1)*100+index+1}))})};};
const api=action.githubApi('fixture-token','owner','repo');const runs=await api.listRuns(sha);const jobs=await api.listJobs(123);return {pages,runs:runs.length,jobs:jobs.length};
"#,
    );
    assert_eq!(result["runs"], 101);
    assert_eq!(result["jobs"], 101);
    assert_eq!(
        result["pages"],
        json!([
            ["workflow_runs", 1, true],
            ["workflow_runs", 2, true],
            ["jobs", 1, true],
            ["jobs", 2, true]
        ])
    );
}

#[test]
fn sibling_monitor_cancels_late_created_lanes_and_preserves_first_failure() {
    let result = action_case(
        r#"
const quality={...base,id:201,name:'PR · Quality'};const all=action.TARGET_WORKFLOWS.map((name,index)=>({...base,id:201+index,name}));let polls=0;const cancelled=[];
const outcome=await action.monitor({api:{listRuns:async()=>++polls===1?[quality]:all,listJobs:async id=>id===201?[{id:501,name:'Plan',conclusion:'failure',completed_at:'2026-08-20T12:00:06Z'}]:[],cancelRun:async id=>{cancelled.push(id);return true;}},trigger,pollSeconds:0,maxMinutes:1,log:()=>{}});return {cancelled,failed:outcome.failure.runId,polls};
"#,
    );
    assert_eq!(
        result,
        json!({"cancelled":[202,203,204,205],"failed":201,"polls":2})
    );
}

#[test]
fn sibling_late_window_is_elapsed_time_instead_of_fixed_poll_count() {
    let result = action_case(
        r#"
const quality={...base,id:201,name:'PR · Quality'};let polls=0,clock=0;
const outcome=await action.monitor({api:{listRuns:async()=>{polls++;return [quality];},listJobs:async()=>[{id:701,name:'Plan',conclusion:'failure',completed_at:'2026-08-20T12:00:06Z'}],cancelRun:async()=>true},trigger,pollSeconds:30,maxMinutes:10,log:()=>{},now:()=>clock,sleepFn:async ms=>{clock+=ms;}});return {failed:outcome.failure.runId,polls,clock};
"#,
    );
    assert_eq!(result, json!({"failed":201,"polls":5,"clock":120000}));
}

#[test]
fn sibling_slow_successful_poll_cannot_extend_absolute_monitor_deadline() {
    let result = action_case(
        r#"
let clock=0,polls=0,error=null;const sleeps=[];
try{await action.monitor({api:{listRuns:async(_sha,options)=>{polls++;if(options.remainingMs()<=0)throw Error('late API operation');clock+=40000;return [];},listJobs:async()=>[],cancelRun:async()=>true},trigger,pollSeconds:30,maxMinutes:1,log:()=>{},now:()=>clock,sleepFn:async ms=>{sleeps.push(ms);clock+=ms;}});}catch(caught){error=caught.message;}return {error,polls,sleeps};
"#,
    );
    assert!(
        result["error"]
            .as_str()
            .unwrap()
            .contains("timed out after 1 minutes")
    );
    assert_eq!(result["polls"], 1);
    assert_eq!(result["sleeps"], json!([20000]));
}

fn deny_actions(
    permissions: Option<&crate::workflow_yaml::Node>,
    inherited: Option<&crate::workflow_yaml::Node>,
    require_explicit: bool,
) -> Result<(), &'static str> {
    use crate::workflow_yaml::Node;
    if require_explicit && permissions.is_none() {
        return Err("workflow permissions must be explicit");
    }
    let effective = permissions
        .or(inherited)
        .ok_or("permissions mapping missing")?;
    if !matches!(effective, Node::Map(_)) {
        return Err("permissions must be a mapping");
    }
    if effective
        .get("actions")
        .is_some_and(|value| value.text() != Some("none"))
    {
        return Err("PR entry may not grant Actions API access");
    }
    Ok(())
}

#[test]
fn sibling_monitor_is_protected_and_pr_entrypoints_have_no_actions_authority() {
    use crate::workflow_yaml::{self, Node};
    let root = repository();
    let monitor = workflow_yaml::parse(
        &fs::read_to_string(root.join(".github/workflows/pr-cancel-sibling-runs.yml")).unwrap(),
    )
    .unwrap();
    let events = monitor.get("on").unwrap();
    assert_eq!(events.entries().len(), 1);
    let event = events.get("workflow_run").unwrap();
    assert_eq!(event.get("workflows").unwrap().list(), ["PR · Quality"]);
    assert_eq!(event.get("types").unwrap().list(), ["in_progress"]);
    assert!(matches!(monitor.get("permissions"), Some(Node::Map(entries)) if entries.is_empty()));
    let concurrency = monitor.get("concurrency").unwrap();
    assert_eq!(
        concurrency.get("group").and_then(Node::text),
        Some("pr-sibling-cancel-${{ github.event.workflow_run.id }}")
    );
    let job = monitor.get("jobs").unwrap().get("monitor").unwrap();
    assert!(monitor.get("secrets").is_none() && job.get("secrets").is_none());
    assert_eq!(
        job.get("permissions")
            .unwrap()
            .get("actions")
            .and_then(Node::text),
        Some("write")
    );
    let condition = job.get("if").and_then(Node::text).unwrap();
    assert_eq!(
        condition.split_whitespace().collect::<Vec<_>>(),
        [
            "${{",
            "github.event.workflow_run.event",
            "==",
            "'pull_request'",
            "&&",
            "github.event.workflow_run.pull_requests[0]",
            "!=",
            "null",
            "}}",
        ]
    );
    let Node::Seq(steps) = job.get("steps").unwrap() else {
        panic!("monitor steps missing")
    };
    let checkouts = steps
        .iter()
        .filter(|step| {
            step.get("uses")
                .and_then(Node::text)
                .is_some_and(|name| name.starts_with("actions/checkout@"))
        })
        .collect::<Vec<_>>();
    assert_eq!(checkouts.len(), 1, "one protected source checkout");
    let checkout = checkouts[0];
    let inputs = checkout.get("with").unwrap();
    assert_eq!(
        inputs.get("ref").and_then(Node::text),
        Some("${{ github.event.repository.default_branch }}")
    );
    assert_eq!(
        inputs.get("persist-credentials").and_then(Node::text),
        Some("false")
    );
    assert!(steps.iter().all(|step| step.get("secrets").is_none()));
    for lane in ["quality", "website", "linux", "macos", "windows"] {
        let workflow = workflow_yaml::parse(
            &fs::read_to_string(root.join(format!(".github/workflows/pr_{lane}.yml"))).unwrap(),
        )
        .unwrap();
        let permissions = workflow.get("permissions");
        deny_actions(permissions, None, true).unwrap();
        for (_, job) in workflow.get("jobs").unwrap().entries() {
            deny_actions(job.get("permissions"), permissions, false).unwrap();
        }
    }
}

#[test]
fn sibling_permission_guard_rejects_implicit_scalars_and_actions_access() {
    use crate::workflow_yaml::{self, Node};
    for ambiguous in [
        "actions: none\nactions: write\n",
        "permissions: {}\npermissions: write-all\n",
    ] {
        assert!(workflow_yaml::parse(ambiguous).is_err());
    }
    let inherited = workflow_yaml::parse("contents: read\nactions: none\n").unwrap();
    assert!(deny_actions(None, Some(&inherited), true).is_err());
    assert!(deny_actions(None, Some(&inherited), false).is_ok());
    assert!(deny_actions(None, None, false).is_err());
    for scalar in ["write-all", "read-all"] {
        assert!(deny_actions(Some(&Node::Scalar(scalar.into())), Some(&inherited), false).is_err());
    }
    for text in ["actions: write", "actions: read", "actions: {}"] {
        let permissions = workflow_yaml::parse(text).unwrap();
        assert!(
            deny_actions(Some(&permissions), Some(&inherited), false).is_err(),
            "{text}"
        );
    }
    assert!(deny_actions(Some(&Node::Map(Vec::new())), None, true).is_ok());
    for text in ["contents: read", "actions: none"] {
        let permissions = workflow_yaml::parse(text).unwrap();
        assert!(
            deny_actions(Some(&permissions), None, true).is_ok(),
            "{text}"
        );
    }
}
