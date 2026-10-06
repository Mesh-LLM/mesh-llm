//! Manual parity frontdoor composes current source admission, offline cache and retained execution.
use super::{parity_local_plan, parity_local_run};
use crate::{command::DynResult, process::Cancellation};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[path = "offline_cache.rs"]
mod offline_cache;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    authority: Value,
    cache_root: PathBuf,
    mode: String,
    statuses: Vec<String>,
    families: Vec<String>,
    llama_models: Vec<String>,
    priorities: Vec<String>,
    limit: Option<usize>,
    missing_only: bool,
    local_only: bool,
    policy: Value,
    run: Option<Value>,
    admission_seconds: u64,
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let [flag, path] = args else {
        return Err("parity-local requires --input PATH".into());
    };
    if flag != "--input" {
        return Err("parity-local requires --input PATH".into());
    }
    let input: Input = serde_json::from_slice(&parity_local_plan::document(Path::new(path))?)?;
    if !input.cache_root.is_absolute()
        || !input.cache_root.is_dir()
        || !["inventory", "run"].contains(&input.mode.as_str())
        || input.missing_only && input.local_only
        || !(1..=3600).contains(&input.admission_seconds)
        || !input.policy.is_object()
    {
        return Err("invalid local parity frontdoor input".into());
    }
    let deadline = Instant::now() + Duration::from_secs(input.admission_seconds);
    let authority = serde_json::to_vec(&json!({"authority":input.authority}))?;
    let mut measured = None;
    let output = crate::automation::canary_package_closure::with_local_parity(
        &authority,
        |root, admitted, manifest, registry, cancellation| {
            let (plan, rows, actual) = compose(
                &input,
                root,
                admitted,
                manifest,
                registry,
                deadline,
                cancellation,
            )?;
            if input.mode == "inventory" {
                let shown: Vec<_> = rows
                    .into_iter()
                    .filter(|row| {
                        (!input.missing_only || row["status"] == "missing")
                            && (!input.local_only || row["local_path"].is_string())
                            && (input.priorities.is_empty()
                                || input.priorities.iter().any(|p| row["priority"] == *p))
                    })
                    .collect();
                return Ok(
                    json!({"schema_version":1,"scope":"current_source_admission_and_offline_cache_observation_no_model_certification","admission":admitted,"inventory":shown,"prepared_plan":plan,"invocation_plan":actual}),
                );
            }
            let settings = input
                .run
                .as_ref()
                .and_then(Value::as_object)
                .ok_or("run requires execution settings")?;
            if settings.contains_key("plan") {
                return Err("execution settings may not replace admitted plan".into());
            }
            let mut request = Value::Object(settings.clone());
            request["plan"] = plan;
            let observation = parity_local_run::execute_document(&request, cancellation)?;
            let output = observation.output.clone();
            measured = Some(observation);
            Ok(output)
        },
    );
    let output = match measured {
        Some(observation) => observation.publish(output.map(|_| ()))?,
        None => output?,
    };
    crate::repository::check_report::CheckReport::success(format!(
        "{}\n",
        serde_json::to_string_pretty(&output)?
    ))
    .emit()
}
fn guard(deadline: Instant, cancellation: &Cancellation) -> std::io::Result<()> {
    if cancellation.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "parity composition cancelled",
        ));
    }
    if Instant::now() >= deadline {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "parity composition deadline",
        ));
    }
    Ok(())
}
fn policy(input: &Input, manifest: &Value) -> DynResult<Value> {
    let mut value = json!({"ctx_size":manifest["defaults"]["ctx_size"].as_u64().unwrap_or(128),"n_gpu_layers":manifest["defaults"]["n_gpu_layers"].as_i64().unwrap_or(999),"prompt":manifest["defaults"]["prompt"].as_str().unwrap_or("Hello"),"run_id":"llama-parity-cheap","skip_build":false,"skip_state":false,"state_payload_kind":null,"dense_state_payload_kind":manifest["defaults"]["state_payload_kind"].as_str().unwrap_or("resident-kv"),"prefix_token_count":null,"cache_hit_repeats":null,"borrow_resident_hits":false,"cache_decoded_result_hits":false,"startup_timeout_secs":null});
    for (key, field) in input.policy.as_object().ok_or("policy")? {
        if value.get(key).is_none() {
            return Err(format!("unknown parity policy override {key}").into());
        }
        value[key] = field.clone();
    }
    Ok(value)
}
fn candidate(row: &Value, pin: Value, observed: bool) -> DynResult<Value> {
    let ranges = match row.get("recurrent_ranges") {
        None | Some(Value::Null) => json!([]),
        Some(Value::Array(a)) => Value::Array(a.clone()),
        Some(Value::String(s)) => json!(s.split(',').collect::<Vec<_>>()),
        _ => return Err("recurrent_ranges must be array/CSV".into()),
    };
    let splits = match row.get("splits") {
        None | Some(Value::Null) => Value::Null,
        Some(Value::Array(a)) => Value::Array(a.clone()),
        Some(Value::String(s)) => json!(
            s.split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()?
        ),
        _ => return Err("splits must be array/CSV".into()),
    };
    Ok(
        json!({"family":row["family"],"llama_model":row["llama_model"],"status":row["status"],"priority":row["priority"],"model_pin":{"repo":pin["repo"],"revision":pin["revision"],"file":pin["file"],"blob_sha256":pin["blob_sha256"],"size_bytes":pin["size_bytes"]},"layer_end":row.get("layer_end").cloned().unwrap_or(Value::Null),"split_layer":row.get("split_layer").cloned().unwrap_or(Value::Null),"splits":splits,"recurrent_all":row["recurrent"]=="all","recurrent_ranges":ranges,"observation_only":observed}),
    )
}
fn compose(
    input: &Input,
    root: &Path,
    admitted: &Value,
    manifest: &Value,
    registry: &Value,
    deadline: Instant,
    cancellation: &Cancellation,
) -> DynResult<(Value, Vec<Value>, Value)> {
    let classifications = admitted["classifications"]
        .as_array()
        .ok_or("current native admission lacks complete classifications")?;
    let mut candidates = Vec::new();
    let mut observations = Vec::new();
    for row in classifications {
        guard(deadline, cancellation)?;
        let mut observation = row.clone();
        observation["classification"] = row["status"].clone();
        let pin = if row.get("model_pin").is_some_and(|pin| !pin.is_null()) {
            Ok(Some((row["model_pin"].clone(), false)))
        } else if let Some(id) = row["artifact_id"].as_str().filter(|s| !s.is_empty()) {
            crate::model_registry::parity_download::source_pin(registry, id)
                .map(|pin| Some((pin, false)))
        } else {
            offline_cache::discover(&input.cache_root, row, &mut || {
                guard(deadline, cancellation)
            })
            .map(|pin| pin.map(|pin| (pin, true)))
        };
        match pin {
            Ok(Some((pin, observed))) => {
                let candidate = candidate(row, pin, observed)?;
                observation["candidate_index"] = json!(candidates.len());
                candidates.push(candidate);
            }
            Ok(None) => {
                observation["status"] = "missing".into();
                observation["local_path"] = Value::Null;
                observation["discovery_scope"] = "offline_only_no_provider_request".into();
            }
            Err(error) => {
                guard(deadline, cancellation)?;
                observation["status"] = "inspect_error".into();
                observation["inspect_error"] = error.to_string().into();
            }
        }
        observations.push(observation);
    }
    let plan = json!({"cache_root":input.cache_root,"source_root":root,"candidates":candidates,"statuses":input.statuses,"families":input.families,"llama_models":input.llama_models,"priorities":input.priorities,"limit":input.limit,"policy":policy(input,manifest)?});
    let parsed: parity_local_plan::Input = serde_json::from_value(plan.clone())?;
    let actual =
        parity_local_plan::plan_with_guard(&parsed, &mut || guard(deadline, cancellation))?;
    for observation in &mut observations {
        if let Some(index) = observation.get("candidate_index").and_then(Value::as_u64) {
            let row = &actual["rows"][usize::try_from(index)?];
            for (key, value) in row.as_object().ok_or("observed row")? {
                if key != "classification" {
                    observation[key] = value.clone();
                }
            }
        }
    }
    guard(deadline, cancellation)?;
    Ok((plan, observations, actual))
}
