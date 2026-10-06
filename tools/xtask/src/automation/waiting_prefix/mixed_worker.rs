//! Concurrent owned request futures for the mixed anchor/decode + delayed prefill workload.
use super::{
    Prompt,
    acceptance::Version,
    adaptive_identity as identity,
    mixed_workload::{PromptRecord, Request, Role},
    options, requests,
};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub round: u64,
    pub version: Version,
    pub base_url: String,
    pub model: String,
    pub request_timeout_secs: f64,
    pub timeout_secs: u64,
    pub readiness_timeout_secs: u64,
    pub warmup: PromptRecord,
    pub requests: Vec<Request>,
    pub suppressed_token_ids: Vec<u32>,
    pub manifest_metadata: Map<String, Value>,
    pub provenance: Map<String, Value>,
    pub workload_sha256: String,
}
impl Input {
    pub fn workload_sha(&self) -> DynResult<String> {
        Ok(identity::digest(&serde_json::to_vec(&(
            &self.warmup,
            &self.requests,
            &self.suppressed_token_ids,
            &self.manifest_metadata,
        ))?))
    }
    pub fn validate(&self) -> DynResult<()> {
        requests::Input {
            schema_version: self.schema_version,
            round: self.round,
            version: self.version,
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            output_tokens: 4,
            request_timeout_secs: self.request_timeout_secs,
            stagger_ms: 0.0,
            prompts: vec![Prompt {
                family: self.warmup.family.clone(),
                prompt: self.warmup.prompt.clone(),
            }],
        }
        .validate()?;
        if !(1..=86400).contains(&self.timeout_secs)
            || self.readiness_timeout_secs > self.timeout_secs
            || self.model.len() > 4096
            || self.requests.is_empty()
            || self.requests.len() > 16
            || self.suppressed_token_ids.len() > 256
            || self.workload_sha256 != self.workload_sha()?
            || self.warmup.prompt.len() > 256 * 1024
            || !self.requests.iter().any(|r| r.role == Role::Anchor)
            || !self.requests.iter().any(|r| r.role == Role::Prefill)
            || self.requests.iter().enumerate().any(|(i, r)| {
                r.request_index != i as u32
                    || r.prompt.family.trim().is_empty()
                    || r.prompt.prompt.trim().is_empty()
                    || r.prompt.prompt.len() > 256 * 1024
                    || !(1..=4096).contains(&r.output_tokens)
                    || !r.delay_ms.is_finite()
                    || r.delay_ms < 0.0
                    || r.delay_ms > 3_600_000.0
            })
        {
            return Err(
                "mixed worker requires bound exact role roster, workload SHA and timing".into(),
            );
        }
        Ok(())
    }
}
struct Endpoint {
    input: Input,
    epoch: Instant,
    until: Instant,
    cancel: Cancellation,
}
async fn cancelled(cancel: &Cancellation) {
    while !cancel.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
fn body(input: &Input, request: &Request) -> Value {
    let mut body = json!({"model":input.model,"messages":[{"role":"user","content":request.prompt.prompt}],"max_tokens":request.output_tokens,"temperature":0,"seed":0,"stream":true,"stream_options":{"include_usage":true}});
    if !input.suppressed_token_ids.is_empty() {
        body["logit_bias"] = json!(
            input
                .suppressed_token_ids
                .iter()
                .map(|id| (id.to_string(), -100))
                .collect::<std::collections::BTreeMap<_, _>>()
        );
    }
    body
}
async fn request(endpoint: Arc<Endpoint>, request: Request) -> Value {
    let mut provenance = request.prompt.provenance.clone();
    provenance.insert("family".into(), json!(request.prompt.family));
    let mut row = json!({"role":request.role,"request_index":request.request_index,"prompt_sha256":identity::digest(request.prompt.prompt.as_bytes()),"prompt_provenance":provenance,"scheduled_ms":request.delay_ms});
    let mut submitted_ms = None;
    let exchange = async {
        tokio::time::sleep_until(tokio::time::Instant::from_std(
            endpoint.epoch + Duration::from_secs_f64(request.delay_ms / 1000.0),
        ))
        .await;
        if Instant::now() >= endpoint.until {
            return Err("mixed whole-cell deadline expired before request".into());
        }
        let submitted = endpoint.epoch.elapsed().as_secs_f64() * 1000.0;
        submitted_ms = Some(submitted);
        let budget = Duration::from_secs_f64(endpoint.input.request_timeout_secs)
            .min(endpoint.until.saturating_duration_since(Instant::now()));
        let evidence = tokio::time::timeout(
            budget,
            crate::automation::openai_exchange::request(
                &endpoint.input.base_url,
                &body(&endpoint.input, &request),
                false,
            ),
        )
        .await
        .map_err(|_| "mixed request/cell deadline expired".to_owned())?
        .map_err(|e| e.to_string())?;
        Ok::<_, String>(
            json!({"submitted_ms":submitted,"first_token_ms":submitted+evidence.ttft_seconds*1000.0,"completed_ms":endpoint.epoch.elapsed().as_secs_f64()*1000.0,"ttft_ms":evidence.ttft_seconds*1000.0,"elapsed_ms":evidence.elapsed_seconds*1000.0,"completion_tokens":evidence.completion_tokens,"content_sha256":evidence.content_sha256,"content_gaps_ms":evidence.decode_inter_token_seconds.iter().map(|v|v*1000.0).collect::<Vec<_>>()}),
        )
    };
    let result = tokio::select! {biased;()=cancelled(&endpoint.cancel)=>Err("mixed request interrupted".into()),()=tokio::time::sleep_until(tokio::time::Instant::from_std(endpoint.until))=>Err("mixed whole-cell deadline expired".into()),result=exchange=>result};
    if let Some(submitted) = submitted_ms {
        row["submitted_ms"] = json!(submitted);
    }
    match result {
        Ok(value) => row
            .as_object_mut()
            .unwrap()
            .extend(value.as_object().unwrap().clone()),
        Err(e) => row["error"] = json!(e.chars().take(1024).collect::<String>()),
    };
    row
}
async fn ready(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    if input.readiness_timeout_secs == 0 {
        return Ok(());
    }
    let ready_until =
        (Instant::now() + Duration::from_secs(input.readiness_timeout_secs)).min(until);
    loop {
        if cancel.is_cancelled() {
            return Err("mixed readiness interrupted".into());
        }
        let remaining = ready_until
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs_f64(input.request_timeout_secs))
            .min(Duration::from_millis(250));
        if remaining.is_zero() {
            return Err("mixed exact-model readiness expired".into());
        }
        if let Ok(Ok(bytes)) = tokio::time::timeout(
            remaining,
            crate::automation::openai_exchange::get(&format!("{}/models", input.base_url)),
        )
        .await
            && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
            && value["data"].as_array().is_some_and(|rows| {
                rows.iter()
                    .any(|r| r["id"].as_str() == Some(input.model.as_str()))
            })
        {
            return Ok(());
        }
        tokio::time::sleep(
            Duration::from_millis(10).min(ready_until.saturating_duration_since(Instant::now())),
        )
        .await;
    }
}
pub(super) async fn execute(input: &Input, cancel: Cancellation) -> Value {
    let started = Instant::now();
    let mut output = json!({"schema_version":1,"round":input.round,"version":input.version,"model":input.model,"input_sha256":identity::digest(&serde_json::to_vec(input).unwrap()),"workload_sha256":input.workload_sha256,"manifest_metadata":input.manifest_metadata,"provenance":input.provenance,"warmup":null,"requests":[],"makespan_ms":null,"successful_requests":0,"timing_origin":"measured-cell-start","gap_boundary":"SSE-content-event-arrival","completion_usage_required":true,"error":null});
    let result = input.validate();
    if let Err(e) = result {
        output["error"] = json!(e.to_string());
        return output;
    }
    let until = started + Duration::from_secs(input.timeout_secs);
    if let Err(e) = ready(input, until, &cancel).await {
        output["error"] = json!(e.to_string());
        return output;
    }
    let warmup = Request {
        role: Role::Prefill,
        request_index: 0,
        prompt: input.warmup.clone(),
        output_tokens: 4,
        delay_ms: 0.0,
    };
    let calibration = request(
        Arc::new(Endpoint {
            input: input.clone(),
            epoch: Instant::now(),
            until,
            cancel: cancel.clone(),
        }),
        warmup,
    )
    .await;
    let failed = !calibration["error"].is_null();
    output["warmup"] = calibration;
    if failed {
        output["error"] = json!("mixed warmup failed; measurement withheld");
        return output;
    }
    let wall_start = unix_nanos();
    let epoch = Instant::now();
    let endpoint = Arc::new(Endpoint {
        input: input.clone(),
        epoch,
        until,
        cancel,
    });
    let mut pending = tokio::task::JoinSet::new();
    for request_spec in &input.requests {
        let endpoint = endpoint.clone();
        let request_spec = request_spec.clone();
        pending.spawn(async move { request(endpoint, request_spec).await });
    }
    let mut rows = Vec::new();
    while let Some(result) = pending.join_next().await {
        match result {
            Ok(row) => rows.push(row),
            Err(e) => {
                output["error"] = json!(format!("mixed owned task failed: {e}"));
                pending.abort_all();
                while pending.join_next().await.is_some() {}
                break;
            }
        }
    }
    let elapsed = epoch.elapsed();
    let wall_end = unix_nanos();
    output["measured_start_unix_nanos"] = json!(wall_start);
    output["measured_end_unix_nanos"] = json!(wall_end);
    output["local_clock_consistent"] = json!(
        wall_start
            .zip(wall_end)
            .is_some_and(|(start, end)| end > start
                && (u128::from(end - start)).abs_diff(elapsed.as_nanos()) <= 10_000_000)
    );
    rows.sort_by_key(|r| r["request_index"].as_u64());
    let successful = rows.iter().filter(|r| r["error"].is_null()).count();
    output["makespan_ms"] = json!(epoch.elapsed().as_secs_f64() * 1000.0);
    output["successful_requests"] = json!(successful);
    output["requests"] = json!(rows);
    if successful != input.requests.len()
        || endpoint.cancel.is_cancelled()
        || Instant::now() >= until
    {
        output["error"] =
            json!("mixed measured requests failed, interrupted or expired; partial rows retained");
    }
    match super::mixed_summary::requests(&output) {
        Ok(summary) => output["summary"] = summary,
        Err(error) => {
            output["summary"] = Value::Null;
            if output["error"].is_null() {
                output["error"] = json!(error.to_string());
            }
        }
    }
    output
}
fn unix_nanos() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|v| u64::try_from(v.as_nanos()).ok())
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let output = Path::new(opts["--output"]);
    if !output.is_absolute() || !Path::new(opts["--input"]).is_absolute() {
        return Err("mixed worker paths must be absolute".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("mixed worker receipt must be fresh".into()),
    }
    let input: Input = serde_json::from_slice(&identity::bounded(
        Path::new(opts["--input"]),
        16 * 1024 * 1024,
    )?)?;
    input.validate()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let receipt = runtime.block_on(execute(&input, interrupt.cancellation()));
    let publication = identity::fresh(output, &serde_json::to_vec_pretty(&receipt)?);
    let finished = interrupt.finish();
    publication?;
    finished?;
    if !receipt["error"].is_null() {
        return Err("mixed worker failed; partial evidence retained".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "mixed_worker_tests.rs"]
mod tests;
