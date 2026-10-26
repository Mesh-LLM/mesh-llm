//! Exact publication logic with unchanged action scalars and no cache/API calls.
use super::{
    support::{Fixture, action},
    workflow_yaml::Node,
};
use serde_json::{Value, json};
use std::{fs, process::Command};

const SAVE: &str = "actions/cache/save@caa296126883cff596d87d8935842f9db880ef25";
const RESTORE: &str = "actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25";
const SCRIPT: &str = "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd";
const RETRY: &str = "steps.initial-publication.outputs.result != 'published'";
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn exact(node: &Node, key: &str, expected: &str) -> Result<(), String> {
    if text(node, key) != expected {
        return Err(format!("{key} must bind {expected}"));
    }
    Ok(())
}
fn steps(action: &Node) -> Result<&[Node], String> {
    match action.get("runs").and_then(|n| n.get("steps")) {
        Some(Node::Seq(steps)) => Ok(steps),
        _ => Err("composite steps missing".into()),
    }
}
fn unique<'a>(steps: &'a [Node], key: &str, value: &str) -> Result<(usize, &'a Node), String> {
    let selected: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| text(s, key) == value)
        .collect();
    match selected.as_slice() {
        [found] => Ok(*found),
        _ => Err(format!("one {key}={value} required")),
    }
}
fn fields<'a>(node: &'a Node, key: &str) -> Result<&'a Node, String> {
    node.get(key).ok_or_else(|| format!("{key} missing"))
}
fn check(document: &Node) -> Result<(), String> {
    let inputs = fields(document, "inputs")?;
    for key in ["path", "cache-key", "cache-ref", "cache-label"] {
        exact(fields(inputs, key)?, "required", "true")?;
    }
    let steps = steps(document)?;
    if steps
        .iter()
        .filter(|step| text(step, "uses").starts_with("actions/cache/"))
        .count()
        != 3
    {
        return Err("only the two owned saves and exact lookup are admitted".into());
    }
    let (snapshot_index, snapshot) = unique(steps, "id", "existing")?;
    let (initial_index, initial) = unique(steps, "id", "initial-publication")?;
    let saves: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| text(s, "uses").starts_with("actions/cache/save@"))
        .collect();
    let [(_, save), (retry_index, retry)] = saves.as_slice() else {
        return Err("one save and one conditional retry required".into());
    };
    let (save_index, _) = saves[0];
    let (final_index, final_probe) = unique(steps, "name", "Verify new exact cache publication")?;
    let (lookup_index, lookup) = unique(steps, "name", "Verify current cache version lookup")?;
    if !(snapshot_index < save_index
        && save_index < initial_index
        && initial_index < *retry_index
        && *retry_index < final_index
        && final_index < lookup_index)
    {
        return Err("snapshot/save/probe/retry/proof/lookup order changed".into());
    }
    for probe in [snapshot, initial, final_probe] {
        exact(probe, "uses", SCRIPT)?;
        if probe.get("if").is_some() || probe.get("continue-on-error").is_some() {
            return Err("required probe cannot be bypassed".into());
        }
        let env = fields(probe, "env")?;
        exact(env, "CACHE_KEY", "${{ inputs.cache-key }}")?;
        exact(env, "CACHE_REF", "${{ inputs.cache-ref }}")?;
        if text(fields(probe, "with")?, "script").is_empty() {
            return Err("script missing".into());
        }
    }
    for probe in [initial, final_probe] {
        let env = fields(probe, "env")?;
        exact(
            env,
            "EXISTING_CACHE_IDS",
            "${{ steps.existing.outputs.result }}",
        )?;
        exact(env, "CACHE_LABEL", "${{ inputs.cache-label }}")?;
    }
    for probe in [snapshot, initial] {
        exact(fields(probe, "with")?, "result-encoding", "string")?;
    }
    for save in [*save, *retry] {
        exact(save, "uses", SAVE)?;
        exact(fields(save, "with")?, "path", "${{ inputs.path }}")?;
        exact(fields(save, "with")?, "key", "${{ inputs.cache-key }}")?;
        if save.get("continue-on-error").is_some() {
            return Err("cache save cannot mask failure".into());
        }
    }
    if save.get("if").is_some() {
        return Err("initial save must run".into());
    }
    exact(retry, "if", RETRY)?;
    exact(lookup, "uses", RESTORE)?;
    if lookup.get("if").is_some() || lookup.get("continue-on-error").is_some() {
        return Err("exact lookup cannot be bypassed".into());
    }
    let lookup = fields(lookup, "with")?;
    for (k, v) in [
        ("path", "${{ inputs.path }}"),
        ("key", "${{ inputs.cache-key }}"),
        ("lookup-only", "true"),
        ("fail-on-cache-miss", "true"),
    ] {
        exact(lookup, k, v)?;
    }
    Ok(())
}
fn probe_script(document: &Node, key: &str, value: &str) -> String {
    text(
        unique(steps(document).unwrap(), key, value)
            .unwrap()
            .1
            .get("with")
            .unwrap(),
        "script",
    )
    .to_owned()
}
fn run(initial: Value, final_rows: Value) -> Value {
    let fixture = Fixture::new();
    let document = action("save-and-verify-actions-cache");
    check(&document).unwrap();
    let scripts = json!({
        "snapshot":probe_script(&document,"id","existing"),
        "initial":probe_script(&document,"id","initial-publication"),
        "final":probe_script(&document,"name","Verify new exact cache publication"),
    });
    let scripts_path = fixture.path().join("scripts.json");
    fs::write(&scripts_path, serde_json::to_vec(&scripts).unwrap()).unwrap();
    let harness = fixture.path().join("publication.js");
    fs::write(&harness, HARNESS).unwrap();
    let mut command = Command::new("node");
    command
        .env_clear()
        .env(
            "PATH",
            std::env::var_os("PATH").expect("Node requires explicit PATH"),
        )
        .env("CACHE_KEY", "fixture-exact-key")
        .env("CACHE_REF", "refs/heads/main")
        .env("CACHE_LABEL", "finite cache fixture")
        .env(
            "PROBE_ROWS",
            json!({"initial":initial,"final":final_rows}).to_string(),
        )
        .arg(harness)
        .arg(scripts_path);
    // Fixture::run resolves node and fails loudly if absent, then uses the
    // existing eight-second bounded/cancellable process owner with closed stdin.
    let result = fixture.run(command);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}
fn entry(id: Value, key: &str, reference: &str, size: i64) -> Value {
    json!({"id":id,"key":key,"ref":reference,"size_in_bytes":size})
}
fn valid() -> Value {
    entry(json!(456), "fixture-exact-key", "refs/heads/main", 17)
}
fn events(value: &Value, kind: &str) -> usize {
    value["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["kind"] == kind)
        .count()
}
fn assert_published(value: &Value) {
    assert_eq!(value["failures"], json!([]));
    assert_eq!(events(value, "lookup"), 1);
    assert_eq!(value["snapshot"], json!(["123"]));
}
#[test]
fn cache_publication_early_exact_entry_avoids_retry_and_cooldown() {
    let output = run(json!([[valid()]]), json!([[valid()]]));
    assert_published(&output);
    assert_eq!(output["initial"], "published");
    assert_eq!(events(&output, "save"), 1);
    assert_eq!(events(&output, "timer"), 0);
    assert_eq!(output["requests"]["initial"], 1);
    assert_eq!(output["requests"]["final"], 1);
}
#[test]
fn cache_publication_nonmatching_or_existing_entries_do_not_count() {
    for invalid in [
        entry(json!(123), "fixture-exact-key", "refs/heads/main", 17),
        entry(json!(456), "other-key", "refs/heads/main", 17),
        entry(json!(456), "fixture-exact-key", "refs/pull/7/merge", 17),
        entry(json!(456), "fixture-exact-key", "refs/heads/main", 0),
    ] {
        let output = run(json!([[invalid.clone()], [valid()]]), json!([[valid()]]));
        assert_published(&output);
        assert_eq!(output["requests"]["initial"], 2);
        assert_eq!(events(&output, "timer"), 1);
        assert_eq!(events(&output, "save"), 1);
        let output = run(json!([[valid()]]), json!([[invalid], [valid()]]));
        assert_published(&output);
        assert_eq!(output["requests"]["final"], 2);
        assert_eq!(events(&output, "timer"), 1);
        assert_eq!(events(&output, "save"), 1);
    }
}
#[test]
fn cache_publication_cooldown_precedes_retry_and_terminal_missing_fails() {
    for final_rows in [json!([[valid()]]), json!([[]])] {
        let output = run(json!([[]]), final_rows.clone());
        assert_eq!(output["initial"], "missing");
        assert_eq!(events(&output, "save"), 2);
        assert!(events(&output, "timer") > 0);
        let trace = output["events"].as_array().unwrap();
        let retry = trace.iter().rposition(|e| e["kind"] == "save").unwrap();
        assert!(trace[..retry].iter().any(|e| e["kind"] == "timer"));
        assert_eq!(trace[retry - 1]["kind"], "initial-result");
        assert_eq!(trace[retry - 1]["value"], "missing");
        if final_rows[0].as_array().unwrap().is_empty() {
            assert_eq!(output["failures"].as_array().unwrap().len(), 1);
            assert!(
                output["failures"][0]
                    .as_str()
                    .unwrap()
                    .contains("No new exact")
            );
            assert_eq!(events(&output, "lookup"), 0);
        } else {
            assert_published(&output);
        }
    }
}
fn mutate(node: &mut Node, key: &str, new: &str) {
    let Node::Map(entries) = node else {
        panic!("map")
    };
    if let Some((_, value)) = entries.iter_mut().find(|(k, _)| k == key) {
        *value = Node::Scalar(new.into());
    } else {
        entries.push((key.into(), Node::Scalar(new.into())));
    }
}
#[test]
fn cache_publication_required_bindings_and_lookup_refusal_cannot_drift() {
    check(&action("save-and-verify-actions-cache")).unwrap();
    for (name, section, key, value) in [
        (
            "Probe initial exact cache publication",
            "env",
            "EXISTING_CACHE_IDS",
            "[]",
        ),
        (
            "Probe initial exact cache publication",
            "env",
            "CACHE_REF",
            "refs/heads/other",
        ),
        (
            "Retry exact cache after cache-service cooldown",
            "",
            "if",
            "true",
        ),
        (
            "Verify current cache version lookup",
            "with",
            "fail-on-cache-miss",
            "false",
        ),
        (
            "Verify current cache version lookup",
            "with",
            "key",
            "other-key",
        ),
        ("Save exact cache", "with", "path", "other-path"),
        (
            "Snapshot existing exact cache entries",
            "with",
            "result-encoding",
            "json",
        ),
    ] {
        let mut doc = action("save-and-verify-actions-cache");
        let Node::Map(root) = &mut doc else {
            panic!("map")
        };
        let runs = &mut root.iter_mut().find(|(k, _)| k == "runs").unwrap().1;
        let Node::Map(runs) = runs else {
            panic!("runs")
        };
        let Node::Seq(steps) = &mut runs.iter_mut().find(|(k, _)| k == "steps").unwrap().1 else {
            panic!("steps")
        };
        let step = steps.iter_mut().find(|s| text(s, "name") == name).unwrap();
        let target = if section.is_empty() {
            step
        } else {
            let Node::Map(fields) = step else {
                panic!("step")
            };
            &mut fields.iter_mut().find(|(k, _)| k == section).unwrap().1
        };
        mutate(target, key, value);
        assert!(check(&doc).is_err(), "{name}/{key}");
    }
}
const HARNESS: &str = r#"'use strict';
const fs=require('fs');
const scripts=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const rows=JSON.parse(process.env.PROBE_ROWS);
const events=[],failures=[],notices=[],warnings=[];
const requests={snapshot:0,initial:0,final:0};
let phase='snapshot',budget=0;
global.fetch=()=>{throw Error('network forbidden');};
global.setTimeout=(callback,delay)=>{
  if(++budget>96 || !(delay>0 && delay<=60000)) throw Error('unbounded cooldown');
  events.push({kind:'timer',phase,delay});callback();
};
const context={repo:{owner:'fixture-owner',repo:'fixture-repository'}};
const core={notice:v=>notices.push(v),warning:v=>warnings.push(v),setFailed:v=>failures.push(v)};
const github={request:async(route,args)=>{
  if(route!=='GET /repos/{owner}/{repo}/actions/caches' || args.owner!==context.repo.owner || args.repo!==context.repo.repo || args.key!==process.env.CACHE_KEY || args.ref!==process.env.CACHE_REF || args.per_page!==100) throw Error('cache identity/request changed');
  if(++budget>96) throw Error('unbounded probes');
  const at=requests[phase]++;
  events.push({kind:'request',phase});
  const candidates=phase==='snapshot'?[{id:123,key:args.key,ref:args.ref,size_in_bytes:17}]:rows[phase][Math.min(at,rows[phase].length-1)];
  return {data:{actions_caches:candidates}};
}};
async function scalar(name){phase=name;const fn=new Function('github','core','context',`return (async()=>{${scripts[name]}\n})()`);return fn(github,core,context);}
(async()=>{
  const snapshot=await scalar('snapshot');
  process.env.EXISTING_CACHE_IDS=snapshot;
  events.push({kind:'save'}); // cache action remains inert; YAML binding is checked by Rust.
  const initial=await scalar('initial');
  events.push({kind:'initial-result',value:initial});
  if(initial!=='published')events.push({kind:'save'});
  await scalar('final');
  if(!failures.length)events.push({kind:'lookup'}); // exact lookup-only/fail-on-miss YAML checked by Rust.
  console.log(JSON.stringify({snapshot:JSON.parse(snapshot),initial,requests,events,failures,notices,warnings}));
})().catch(e=>{console.error(e);process.exitCode=1});
"#;
