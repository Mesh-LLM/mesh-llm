//! Certify the payload selected by every stage of a pinned two-node smoke run.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    message.into().into()
}

fn read_json(path: &Path) -> DynResult<Value> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    if !value.is_object() {
        return Err(error(format!(
            "{} must contain a JSON object",
            path.display()
        )));
    }
    Ok(value)
}

fn required<'a>(value: &'a Value, key: &str) -> DynResult<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| error(format!("missing {key}")))
}

fn string<'a>(value: &'a Value, key: &str) -> DynResult<&'a str> {
    required(value, key)?
        .as_str()
        .ok_or_else(|| error(format!("{key} must be a string")))
}

fn array<'a>(value: &'a Value, key: &str) -> DynResult<&'a Vec<Value>> {
    required(value, key)?
        .as_array()
        .ok_or_else(|| error(format!("{key} must be an array")))
}

fn load_events(path: &Path) -> DynResult<Vec<Value>> {
    let bytes = fs::read(path)?;
    let contents = String::from_utf8_lossy(&bytes);
    Ok(contents
        .lines()
        .filter_map(|line| {
            let event: Value = serde_json::from_str(line).ok()?;
            event.get("attributes")?.as_object()?;
            Some(event)
        })
        .collect())
}

fn observer(stage: &Value, observers: &Value) -> DynResult<&'static str> {
    let node_id = string(stage, "node_id")?;
    let matches = ["seed", "worker"]
        .into_iter()
        .filter(|node| {
            observers
                .get(*node)
                .and_then(|value| value.get("node_id"))
                .and_then(Value::as_str)
                .is_some_and(|prefix| node_id.starts_with(prefix))
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(error(format!(
            "stage {} has ambiguous node: {matches:?}",
            string(stage, "stage_id")?
        )));
    }
    Ok(matches[0])
}

const SELECTION_FIELDS: [(&str, &str); 8] = [
    ("payload", "skippy.kv.payload"),
    ("reason", "skippy.kv.payload_selection_reason"),
    ("fallbacks", "skippy.kv.payload_fallbacks"),
    ("admitted_graph_state", "skippy.kv.admitted_graph_state"),
    ("loaded_state_kind", "skippy.kv.loaded_state_kind"),
    ("resident", "skippy.kv.loaded_memory_cache_resident"),
    ("kv_recurrent", "skippy.kv.loaded_memory_cache_kv_recurrent"),
    (
        "graph_loaded_state_mismatches",
        "skippy.kv.graph_loaded_state_mismatches",
    ),
];

fn selection(attrs: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for (name, attr) in SELECTION_FIELDS {
        result.insert(name.into(), attrs.get(attr).cloned().unwrap_or(Value::Null));
    }
    Value::Object(result)
}

fn certify(
    evidence: &Value,
    expectation: &Value,
    logs: &BTreeMap<&str, Vec<Value>>,
    artifact_id: &str,
    model_sha256: &str,
) -> DynResult<Value> {
    if expectation["artifact_id"] != artifact_id || expectation["sha256"] != model_sha256 {
        return Err(error(
            "model artifact ID or SHA-256 differs from the pinned expectation",
        ));
    }
    let topology = required(evidence, "topology")?;
    if evidence["status"] != "ready" || evidence["model_id"] != topology["model_id"] {
        return Err(error("split topology evidence is not ready for one model"));
    }
    let stages = array(topology, "stages")?;
    let expected_stages = array(expectation, "stages")?;
    let indices = stages
        .iter()
        .map(|stage| stage["stage_index"].as_u64())
        .collect::<Option<BTreeSet<_>>>();
    if stages.len() != expected_stages.len()
        || indices != Some((0..expected_stages.len() as u64).collect())
    {
        return Err(error(
            "topology stages differ from the explicit expectation",
        ));
    }
    let mut by_id = BTreeMap::new();
    for stage in stages {
        let id = string(stage, "stage_id")?;
        if by_id.insert(id, stage).is_some() {
            return Err(error("duplicate stage ID in topology"));
        }
    }
    let mut matched: BTreeMap<&str, Vec<Value>> =
        by_id.keys().map(|id| (*id, Vec::new())).collect();
    let mut exact_kinds: BTreeMap<&str, BTreeSet<String>> =
        by_id.keys().map(|id| (*id, BTreeSet::new())).collect();
    let mut lookups: BTreeMap<&str, Vec<Value>> =
        by_id.keys().map(|id| (*id, Vec::new())).collect();
    let observers = required(evidence, "observers")?;
    for (node, events) in logs {
        for event in events {
            let attrs = &event["attributes"];
            if [
                ("skippy.run_id", &topology["run_id"]),
                ("skippy.model_id", &topology["model_id"]),
                ("skippy.topology_id", &topology["topology_id"]),
            ]
            .iter()
            .any(|(key, value)| attrs[*key] != **value)
            {
                continue;
            }
            let name = event["event"].as_str().unwrap_or_default();
            let is_selection = name == "stage.kv_payload_selected";
            let is_lookup = matches!(
                name,
                "stage.binary_kv_lookup_decision" | "stage.openai_kv_lookup_decision"
            );
            if !is_selection && !is_lookup && attrs.get("skippy.exact_cache.payload_kind").is_none()
            {
                continue;
            }
            let id = attrs["skippy.stage_id"]
                .as_str()
                .ok_or_else(|| error("missing stage ID on matched event"))?;
            let stage = by_id
                .get(id)
                .ok_or_else(|| error(format!("unexpected stage selection for {id:?}")))?;
            if *node != observer(stage, observers)? {
                return Err(error(format!("stage {id} emitted on wrong node {node}")));
            }
            if attrs["skippy.stage_index"] != stage["stage_index"] {
                return Err(error(format!("stage index mismatch for {id}")));
            }
            if let Some(kind) = attrs["skippy.exact_cache.payload_kind"].as_str() {
                exact_kinds.get_mut(id).unwrap().insert(kind.into());
            }
            if is_lookup {
                lookups.get_mut(id).unwrap().push(json!({
                    "event": event["event"], "request_id": attrs["skippy.request_id"],
                    "decision": attrs["skippy.kv.decision"],
                    "restored_tokens": attrs["skippy.kv.restored_tokens"]
                }));
            }
            if is_selection {
                matched.get_mut(id).unwrap().push(selection(attrs));
            }
        }
    }
    let mut results = Vec::new();
    for stage in stages {
        let id = string(stage, "stage_id")?;
        let index = stage["stage_index"]
            .as_u64()
            .ok_or_else(|| error("invalid stage index"))? as usize;
        let outcomes = &matched[id];
        if outcomes.is_empty() {
            return Err(error(format!("missing selection for stage {id}")));
        }
        let unique = outcomes
            .iter()
            .map(Value::to_string)
            .collect::<BTreeSet<_>>();
        if unique.len() != 1 {
            return Err(error(format!("contradictory selections for stage {id}")));
        }
        if outcomes.len() != 1 {
            return Err(error(format!("duplicate selection for stage {id}")));
        }
        let observed = &outcomes[0];
        let expected = &expected_stages[index];
        let fields = expected
            .as_object()
            .ok_or_else(|| error("stage expectation must be an object"))?;
        for (field, value) in fields {
            if field == "exact_payload_kind" {
                if exact_kinds[id].len() != 1
                    || !exact_kinds[id].contains(value.as_str().unwrap_or_default())
                {
                    return Err(error(format!(
                        "stage {id} exact payload: expected {value}, observed {:?}",
                        exact_kinds[id]
                    )));
                }
            } else if observed[field] != *value {
                return Err(error(format!(
                    "stage {id} {field}: expected {value}, observed {}",
                    observed[field]
                )));
            }
        }
        results.push(json!({
            "stage_id": id, "stage_index": index, "node": observer(stage, observers)?,
            "expected": expected, "observed": observed,
            "exact_payload_kinds": exact_kinds[id], "selection_event_count": outcomes.len(),
            "lookups": lookups[id]
        }));
    }
    Ok(
        json!({"status":"pass", "artifact_id":artifact_id, "sha256":model_sha256,
        "model_id":topology["model_id"], "run_id":topology["run_id"],
        "topology_id":topology["topology_id"], "stages":results}),
    )
}

fn option<'a>(args: &'a [String], name: &str) -> DynResult<&'a str> {
    let position = args
        .iter()
        .position(|arg| arg == name)
        .ok_or_else(|| error(format!("missing {name}")))?;
    args.get(position + 1)
        .map(String::as_str)
        .ok_or_else(|| error(format!("missing value for {name}")))
}

fn validate_sha(value: &str) -> DynResult<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(error(
            "tested commit must be a lowercase 40-character source SHA",
        ));
    }
    Ok(())
}

pub(crate) fn artifact_for_sha256(args: &[String]) -> DynResult<()> {
    let sha = option(args, "--sha256")?;
    let expectations = read_json(Path::new(option(args, "--expectations")?))?;
    let models = expectations["models"]
        .as_object()
        .ok_or_else(|| error("missing models"))?;
    let matches = models
        .iter()
        .filter(|(_, model)| model["sha256"] == sha)
        .map(|(id, _)| id)
        .collect::<Vec<_>>();
    println!(
        "{}",
        if matches.len() == 1 {
            matches[0].as_str()
        } else {
            "unspecified"
        }
    );
    Ok(())
}

fn certify_files(args: &[String]) -> DynResult<Value> {
    let artifact_id = option(args, "--artifact-id")?;
    let sha = option(args, "--model-sha256")?;
    let tested_commit = option(args, "--tested-commit")?;
    validate_sha(tested_commit)?;
    let expectations = read_json(Path::new(option(args, "--expectations")?))?;
    let expectation = expectations["models"]
        .get(artifact_id)
        .ok_or_else(|| error(format!("no pinned Auto expectation for {artifact_id}")))?;
    let manifest = read_json(Path::new(option(args, "--model-manifest")?))?;
    let artifact = array(&manifest, "artifacts")?
        .iter()
        .find(|value| value["id"] == artifact_id)
        .ok_or_else(|| error("model artifact absent from immutable manifest"))?;
    if ["revision", "sha256"]
        .iter()
        .any(|key| expectation[*key] != artifact[*key])
    {
        return Err(error(
            "Auto expectation differs from the immutable model manifest",
        ));
    }
    let evidence = read_json(Path::new(option(args, "--evidence")?))?;
    let logs = BTreeMap::from([
        ("seed", load_events(Path::new(option(args, "--seed-log")?))?),
        (
            "worker",
            load_events(Path::new(option(args, "--worker-log")?))?,
        ),
    ]);
    let mut result = certify(&evidence, expectation, &logs, artifact_id, sha)?;
    result["revision"] = expectation["revision"].clone();
    result["tested_commit"] = json!(tested_commit);
    let roster = read_json(Path::new(option(args, "--roster")?))?;
    result["native_recipe"] = required(&roster, "native_recipe")?.clone();
    let mut manifests = Vec::new();
    for entry in fs::read_dir(option(args, "--runtime-bundle")?)? {
        let path = entry?.path().join("manifest.json");
        if path.is_file() {
            manifests.push(path);
        }
    }
    if manifests.len() != 1 {
        return Err(error(format!(
            "expected one packaged native runtime, found {}",
            manifests.len()
        )));
    }
    let runtime_manifest = read_json(&manifests[0])?;
    let runtime = required(&runtime_manifest, "runtime")?;
    if runtime["skippy_abi"] != result["native_recipe"]["skippy_abi"] {
        return Err(error(
            "loaded native runtime ABI differs from the checked-in roster",
        ));
    }
    result["native_runtime"] = json!({"id":runtime["id"], "skippy_abi":runtime["skippy_abi"]});
    let responses = Path::new(option(args, "--responses-dir")?);
    let mut cold_warm = Vec::new();
    for index in 1..=2 {
        let response = read_json(&responses.join(format!("response-{index}.json")))?;
        let cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .ok_or_else(|| error("cold/warm response omitted cache usage or output"))?;
        let output = response["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| error("cold/warm response omitted cache usage or output"))?;
        cold_warm.push(json!({"request_id":response["id"], "cached_tokens":cached,
            "prompt_tokens":response["usage"]["prompt_tokens"], "output":output}));
    }
    if cold_warm[0]["cached_tokens"] != 0 || cold_warm[1]["cached_tokens"].as_u64().unwrap() == 0 {
        return Err(error("expected a cold miss followed by a warm restore"));
    }
    if cold_warm[0]["output"] != cold_warm[1]["output"] {
        return Err(error(
            "warm continuation differs from clean cold continuation",
        ));
    }
    result["cold_warm"] = json!(cold_warm);
    Ok(result)
}

pub(crate) fn certify_command(args: &[String]) -> DynResult<()> {
    let output = PathBuf::from(option(args, "--output")?);
    match certify_files(args) {
        Ok(result) => {
            fs::write(
                &output,
                format!("{}\n", serde_json::to_string_pretty(&result)?),
            )?;
            println!(
                "certified Auto payload for {} loaded stages",
                result["stages"].as_array().unwrap().len()
            );
            Ok(())
        }
        Err(problem) => {
            let evidence = option(args, "--evidence")
                .ok()
                .and_then(|path| read_json(Path::new(path)).ok());
            let expectation = option(args, "--expectations")
                .ok()
                .and_then(|path| read_json(Path::new(path)).ok());
            let roster = option(args, "--roster")
                .ok()
                .and_then(|path| read_json(Path::new(path)).ok());
            let artifact_id = option(args, "--artifact-id").ok();
            let pin = artifact_id.and_then(|id| expectation.as_ref()?.get("models")?.get(id));
            let topology = evidence.as_ref().and_then(|value| value.get("topology"));
            let failure = json!({"status":"fail", "error":problem.to_string(),
                "artifact_id":artifact_id,
                "sha256":option(args,"--model-sha256").ok(),
                "revision":pin.and_then(|value| value.get("revision")),
                "tested_commit":option(args,"--tested-commit").ok(),
                "native_recipe":roster.as_ref().and_then(|value| value.get("native_recipe")),
                "model_id":topology.and_then(|value| value.get("model_id")),
                "run_id":topology.and_then(|value| value.get("run_id")),
                "stages":topology.and_then(|value| value.get("stages"))});
            fs::write(
                &output,
                format!("{}\n", serde_json::to_string_pretty(&failure)?),
            )?;
            Err(problem)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::unique_temp_dir;

    fn fixture_file(root: &Path, name: &str, value: &Value) -> DynResult<()> {
        fs::write(root.join(name), serde_json::to_vec(value)?)?;
        Ok(())
    }
    #[test]
    fn tested_commit_requires_immutable_sha() {
        assert!(validate_sha("a".repeat(40).as_str()).is_ok());
        assert!(validate_sha("").is_err());
        assert!(validate_sha("HEAD").is_err());
    }
    #[test]
    fn wrong_second_stage_is_rejected() {
        let evidence = json!({"status":"ready","model_id":"pinned", "topology":{
            "model_id":"pinned","run_id":"run","topology_id":"topology","stages":[
                {"stage_id":"stage-0","stage_index":0,"node_id":"seed-node-full"},
                {"stage_id":"stage-1","stage_index":1,"node_id":"worker-node-full"}]},
            "observers":{"seed":{"node_id":"seed-node"},"worker":{"node_id":"worker-node"}}});
        let expectation = json!({"artifact_id":"artifact","sha256":"sha","stages":[
            {"payload":"ResidentKv"},{"payload":"ResidentKv"}]});
        let event = |index: usize, payload: &str| {
            json!({"event":"stage.kv_payload_selected","attributes":{
            "skippy.run_id":"run","skippy.model_id":"pinned","skippy.topology_id":"topology",
            "skippy.stage_id":format!("stage-{index}"),"skippy.stage_index":index,
            "skippy.kv.payload":payload}})
        };
        let mut logs = BTreeMap::from([
            ("seed", vec![event(0, "ResidentKv")]),
            ("worker", vec![event(1, "ResidentKv")]),
        ]);
        assert!(certify(&evidence, &expectation, &logs, "artifact", "sha").is_ok());
        logs.get_mut("worker").unwrap()[0] = event(1, "FullState");
        let problem = certify(&evidence, &expectation, &logs, "artifact", "sha").unwrap_err();
        assert!(problem.to_string().contains("stage stage-1 payload"));
        logs.get_mut("worker")
            .unwrap()
            .insert(0, event(1, "ResidentKv"));
        let problem = certify(&evidence, &expectation, &logs, "artifact", "sha").unwrap_err();
        assert!(problem.to_string().contains("contradictory selections"));
        logs.get_mut("worker").unwrap().clear();
        assert!(
            certify(&evidence, &expectation, &logs, "artifact", "sha")
                .unwrap_err()
                .to_string()
                .contains("missing selection")
        );
    }

    #[test]
    fn pinned_expectations_match_immutable_model_manifest() -> DynResult<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let expectations =
            read_json(&root.join("ci/model-artifacts/kv-auto-smoke-expectations.json"))?;
        let manifest =
            read_json(&root.join("ci/model-artifacts/manifests/scripted-binary-smoke.json"))?;
        for (id, expected) in expectations["models"].as_object().ok_or("missing models")? {
            let artifact = array(&manifest, "artifacts")?
                .iter()
                .find(|artifact| artifact["id"] == *id)
                .ok_or("missing pinned artifact")?;
            assert_eq!(expected["artifact_id"], *id);
            assert_eq!(expected["revision"], artifact["revision"]);
            assert_eq!(expected["sha256"], artifact["sha256"]);
        }
        Ok(())
    }

    #[test]
    fn command_writes_revision_bound_success_and_failure_evidence() -> DynResult<()> {
        let root = unique_temp_dir("split-payload-certifier");
        fs::create_dir_all(root.join("runtime/cpu"))?;
        fs::create_dir_all(root.join("responses"))?;
        fixture_file(
            &root,
            "evidence.json",
            &json!({
            "status":"ready","model_id":"pinned","topology":{
                "model_id":"pinned","run_id":"run","topology_id":"topology","stages":[
                    {"stage_id":"stage-0","stage_index":0,"node_id":"seed-node-full"},
                    {"stage_id":"stage-1","stage_index":1,"node_id":"worker-node-full"}]},
            "observers":{"seed":{"node_id":"seed-node"},"worker":{"node_id":"worker-node"}}}),
        )?;
        fixture_file(
            &root,
            "expectations.json",
            &json!({"models":{"artifact":{
            "artifact_id":"artifact","revision":"pinned-revision","sha256":"digest",
            "stages":[{"payload":"ResidentKv"},{"payload":"ResidentKv"}]}}}),
        )?;
        fixture_file(
            &root,
            "manifest.json",
            &json!({"artifacts":[{
            "id":"artifact","revision":"pinned-revision","sha256":"digest"}]}),
        )?;
        fixture_file(
            &root,
            "roster.json",
            &json!({"native_recipe":{"skippy_abi":"0.1.67"}}),
        )?;
        fixture_file(
            &root,
            "runtime/cpu/manifest.json",
            &json!({"runtime":{
            "id":"cpu","skippy_abi":"0.1.67"}}),
        )?;
        for (index, cached) in [(1, 0), (2, 8)] {
            fixture_file(
                &root,
                &format!("responses/response-{index}.json"),
                &json!({
                "id":format!("request-{index}"),
                "choices":[{"message":{"content":"same continuation"}}],
                "usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":cached}}}),
            )?;
        }
        let event = |index: usize, payload: &str| {
            json!({
            "event":"stage.kv_payload_selected","attributes":{
                "skippy.run_id":"run","skippy.model_id":"pinned",
                "skippy.topology_id":"topology",
                "skippy.stage_id":format!("stage-{index}"),"skippy.stage_index":index,
                "skippy.kv.payload":payload}})
        };
        fixture_file(&root, "seed.jsonl", &event(0, "ResidentKv"))?;
        fixture_file(&root, "worker.jsonl", &event(1, "ResidentKv"))?;
        let options = [
            ("--evidence", "evidence.json"),
            ("--expectations", "expectations.json"),
            ("--model-manifest", "manifest.json"),
            ("--roster", "roster.json"),
            ("--runtime-bundle", "runtime"),
            ("--seed-log", "seed.jsonl"),
            ("--worker-log", "worker.jsonl"),
            ("--responses-dir", "responses"),
            ("--output", "result.json"),
        ];
        let mut args = Vec::new();
        for (flag, path) in options {
            args.push(flag.to_string());
            args.push(root.join(path).display().to_string());
        }
        args.extend(
            [
                "--artifact-id",
                "artifact",
                "--model-sha256",
                "digest",
                "--tested-commit",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ]
            .into_iter()
            .map(str::to_string),
        );
        certify_command(&args)?;
        let passed = read_json(&root.join("result.json"))?;
        assert_eq!(passed["status"], "pass");
        assert_eq!(
            passed["tested_commit"],
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert_eq!(passed["cold_warm"][1]["cached_tokens"], 8);
        fixture_file(&root, "worker.jsonl", &event(1, "FullState"))?;
        assert!(certify_command(&args).is_err());
        let failed = read_json(&root.join("result.json"))?;
        assert_eq!(failed["status"], "fail");
        assert!(
            failed["error"]
                .as_str()
                .unwrap()
                .contains("stage stage-1 payload")
        );
        assert_eq!(failed["native_recipe"]["skippy_abi"], "0.1.67");
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
