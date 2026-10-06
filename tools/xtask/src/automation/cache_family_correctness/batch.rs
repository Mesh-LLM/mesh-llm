//! Catalog-backed batch/table adapter over the existing correctness producer.
use super::*;
use serde::Deserialize;
use std::collections::BTreeSet;
#[path = "batch_publication.rs"]
mod publication;
#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
const DEFAULT_CASES: [&str; 7] = [
    "qwen3_dense",
    "llama",
    "glm4",
    "gemma3",
    "falcon_h1",
    "olmo",
    "qwen3next",
];
fn topologies() -> Vec<Topology> {
    vec![
        Topology::OneStage,
        Topology::SplitMiddle,
        Topology::SplitFinal,
    ]
}
fn lanes() -> u32 {
    4
}
fn repeats() -> u32 {
    3
}
fn seconds() -> u64 {
    900
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u64,
    /// Existing prepare-full output; tool/model byte observations are not build proof.
    prepared_input: PathBuf,
    #[serde(default)]
    cases: Vec<String>,
    #[serde(default = "topologies")]
    topologies: Vec<Topology>,
    prefix_tokens: Option<u32>,
    #[serde(default = "repeats")]
    cache_hit_repeats: u32,
    #[serde(default = "lanes")]
    runtime_lane_count: u32,
    n_gpu_layers: Option<i32>,
    #[serde(default = "seconds")]
    execution_seconds: u64,
}
impl Request {
    fn validate(&self) -> DynResult<()> {
        let known: Value =
            serde_json::from_slice(include_bytes!("../cache_family_plan/catalog.json"))?;
        let keys: BTreeSet<_> = known
            .as_array()
            .ok_or("catalog")?
            .iter()
            .filter_map(|c| c["key"].as_str())
            .collect();
        if self.schema_version != 1
            || !self.prepared_input.is_absolute()
            || self.cases.len() > 14
            || self.cases.iter().collect::<BTreeSet<_>>().len() != self.cases.len()
            || self.cases.iter().any(|c| !keys.contains(c.as_str()))
            || self.topologies.is_empty()
            || self.topologies.len() > 4
            || self
                .topologies
                .iter()
                .any(|t| matches!(t, Topology::PackageStage1))
            || self
                .topologies
                .iter()
                .map(|t| format!("{t:?}"))
                .collect::<BTreeSet<_>>()
                .len()
                != self.topologies.len()
            || self.prefix_tokens == Some(0)
            || !(1..=1024).contains(&self.cache_hit_repeats)
            || !(1..=1024).contains(&self.runtime_lane_count)
            || !(4..=86400).contains(&self.execution_seconds)
            || self.n_gpu_layers.is_some_and(|v| v < -1)
        {
            return Err("invalid closed correctness batch selection/budget".into());
        }
        Ok(())
    }
}
fn profile(request: &Request, case: &Value, value: &Value) -> DynResult<Input> {
    let mut input: Input = serde_json::from_value(value.clone())?;
    if input.case_key != case["key"] || input.model_id != case["model_id"] {
        return Err("prepared correctness case/model correlation refused".into());
    }
    input.prefix_tokens = request.prefix_tokens.unwrap_or(
        case["prefix_tokens"]
            .as_u64()
            .ok_or("catalog prefix")?
            .min(32) as u32,
    );
    input.ctx_size = input.ctx_size.max(
        input
            .prefix_tokens
            .checked_add(129)
            .ok_or("context overflow")?,
    );
    input.runtime_lane_count = request.runtime_lane_count;
    input.cache_hit_repeats = request.cache_hit_repeats;
    input.n_gpu_layers = request.n_gpu_layers.unwrap_or(input.n_gpu_layers);
    input.borrow_resident_hits = true;
    input.cache_decoded_result_hits = false;
    input.topologies = request.topologies.clone();
    input.validate()?;
    Ok(input)
}
fn row(case: &Value, topology: Value, evidence: &Value) -> Value {
    let r = &evidence["skippy"];
    let status = evidence["status"].as_str().unwrap_or("failed");
    let suffix = r["suffix_token_count"].as_u64().unwrap_or(1);
    json!({"family":case["family"],"model_id":case["model_id"],"payload":case["payload"],"topology":topology,"status":status,"matches":r["matches"].as_bool().unwrap_or(false),"native_seq_remapped":r["native_seq_remapped"],"source_native_seq_id":r["source_native_seq_id"],"restore_native_seq_id":r["restore_native_seq_id"],"prompt_tokens":r["prompt_token_count"],"suffix_tokens":if status=="pass"{json!(suffix)}else{Value::Null},"suffix_prefill_matches":r["suffix_prefill_matches"],"state_bytes":r["state_bytes"],"cache_storage_bytes":r["cache_storage_bytes"],"recurrent_bytes":r["payload_digest"]["recurrent_bytes"],"kv_bytes":r["payload_digest"]["kv_bytes"],"cache_hit_repeats":r["cache_hit_repeats"].as_u64().unwrap_or(0),"cache_hit_matches":r["cache_hit_matches"].as_bool().unwrap_or(false),"promotion_decision":if status=="pass"{"pass"}else{"disabled-or-recompute"}})
}
fn markdown(rows: &[Value]) -> String {
    let fields = [
        "family",
        "model_id",
        "payload",
        "topology",
        "status",
        "native_seq_remapped",
        "source_native_seq_id",
        "restore_native_seq_id",
        "prompt_tokens",
        "suffix_prefill_matches",
        "state_bytes",
        "recurrent_bytes",
        "cache_hit_matches",
        "promotion_decision",
    ];
    let mut text = String::from(
        "# Cache correctness gate\n\n| Family | Model ref | Payload | Topology | Result | Seq remap | Source seq | Target seq | Tokens | Suffix | Payload bytes | Recurrent bytes | Hits | Promotion |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for row in rows {
        text.push_str("| ");
        for field in fields {
            let value = &row[field];
            let s = value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string());
            text.push_str(&s.replace('|', "\\|").replace(['\n', '\r'], " "));
            text.push_str(" | ");
        }
        text.push('\n');
    }
    text
}
fn terminal(rows: &mut [Value], cancelled: bool, expired: bool, finish_ok: bool) -> bool {
    if cancelled || expired || !finish_ok {
        for row in rows {
            if row["status"] == "pass" {
                row["status"] = json!("failed-terminal");
                row["promotion_decision"] = json!("disabled-or-recompute");
            }
        }
        return false;
    }
    rows.iter()
        .all(|r| r["status"] == "pass" || r["status"] == "missing-model")
}
fn collect(
    request: &Request,
    prepared: &crate::automation::cache_family_run::contract::Input,
    plan: &Value,
    output: &Path,
    execution: &Execution<'_>,
) -> DynResult<(Vec<Value>, Vec<Value>)> {
    let mut raw = Vec::new();
    let mut table = Vec::new();
    for (index, cell) in plan["cells"]
        .as_array()
        .ok_or("batch plan cells")?
        .iter()
        .enumerate()
    {
        let case = &cell["case"];
        let key = case["key"].as_str().ok_or("case key")?;
        if cell["model_observation"]["status"] == "missing-model" {
            table.push(row(case, json!("all"), &json!({"status":"missing-model"})));
            continue;
        }
        let input: DynResult<Input> = match prepared.profiles.get(key) {
            Some(value) => profile(request, case, &value.correctness),
            None => Err("present model lacks observed correctness profile".into()),
        };
        for (n, topology) in request.topologies.iter().enumerate() {
            let directory = output.join(format!("case-{index:02}-topology-{n:02}"));
            std::fs::create_dir(&directory)?;
            let result = match &input {
                Ok(input)
                    if !execution.cancellation.is_cancelled()
                        && Instant::now() < execution.deadline =>
                {
                    if serde_json::to_value(&input.model)?
                        != cell["model_observation"]["runtime_entrypoint"]
                    {
                        json!({"status":"refused","reason":"prepared_model_path_correlation"})
                    } else {
                        one(input,*topology,&directory,execution).unwrap_or_else(|_|json!({"status":"refused","reason":"existing_identity_report_or_deadline_admission"}))
                    }
                }
                _ => json!({"status":"refused","reason":"profile_or_terminal_admission"}),
            };
            table.push(row(case, serde_json::to_value(topology)?, &result));
            raw.push(json!({"case":case,"topology":topology,"report":result["skippy"],"evidence":result}));
            publish(
                &directory.join("trial.json"),
                raw.last().ok_or("batch trial")?,
            )?;
        }
    }
    Ok((raw, table))
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let (path, output) = paths(args)?;
    let request: Request = serde_json::from_slice(
        &crate::automation::waiting_prefix::adaptive_identity::bounded(&path, 1024 * 1024)?,
    )?;
    request.validate()?;
    let prepared: crate::automation::cache_family_run::contract::Input = serde_json::from_slice(
        &crate::automation::waiting_prefix::adaptive_identity::bounded(
            &request.prepared_input,
            4 * 1024 * 1024,
        )?,
    )?;
    prepared.validate()?;
    let mut plan_input = prepared.plan.clone();
    plan_input["cases"] = if request.cases.is_empty() {
        json!(DEFAULT_CASES)
    } else {
        json!(request.cases)
    };
    plan_input["use_cases"] = json!([]);
    plan_input["corpus"] = Value::Null;
    plan_input["prefix_sweep"] = json!([]);
    plan_input["prefix_tokens"] = Value::Null;
    let parent = output.parent().ok_or("batch parent")?;
    if !std::fs::symlink_metadata(parent)?.is_dir() {
        return Err("batch parent must be regular directory".into());
    }
    let deadline = Instant::now() + Duration::from_secs(request.execution_seconds);
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let plan = crate::automation::cache_family_plan::plan_value(plan_input)?;
    std::fs::create_dir(&output)?;
    publish(&output.join("plan.json"), &plan)?;
    let execution = Execution {
        deadline,
        cancellation: &cancellation,
    };
    let result = collect(&request, &prepared, &plan, &output, &execution);
    let (raw, mut table) = result?;
    bounded_publish(
        &output.join("cache-correctness-gate.json"),
        &json!(raw),
        64 * 1024 * 1024,
    )?;
    let mut summary = json!({"schema_version":1,"completed":false,"rows":table.len(),"scope":"correctness_only_observed_models_missing_models_are_unqualified","historical_suffix_default":1,"toolkit_directories":prepared.profiles.iter().map(|(k,p)|(k,&p.correctness["toolkit_directories"])).collect::<BTreeMap<_,_>>(),"effective_settings":prepared.profiles.iter().map(|(k,p)|(k,&p.correctness["settings"])).collect::<BTreeMap<_,_>>()});
    let mut files = publication::Files::new(&output);
    let completed = publication::finish_owned(
        &mut table,
        &mut summary,
        interrupt,
        deadline,
        &mut |i, bytes| files.write(i, bytes),
    )?;
    if !completed {
        return Err("correctness batch failed; partial table/raw observations retained".into());
    }
    Ok(())
}

fn bounded_publish(path: &Path, value: &Value, limit: usize) -> DynResult<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > limit {
        return Err("correctness batch projection exceeds bound".into());
    }
    crate::automation::waiting_prefix::adaptive_identity::fresh(path, &bytes)
}
