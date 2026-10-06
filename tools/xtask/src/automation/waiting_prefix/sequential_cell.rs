//! One calibration and serial measured requests against an already-owned local arm.
//! This worker owns request futures; it does not launch or qualify the server.
use super::{Prompt, acceptance::Version, options, publish, requests};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[derive(Clone, Deserialize, Serialize)]
struct PromptRecord {
    family: String,
    prompt: String,
    #[serde(flatten)]
    provenance: Map<String, Value>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    #[serde(default)]
    metadata: Map<String, Value>,
    prompts: Vec<PromptRecord>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    schema_version: u64,
    round: u64,
    version: Version,
    base_url: String,
    model: String,
    output_tokens: u64,
    request_timeout_secs: f64,
    timeout_secs: u64,
    #[serde(default)]
    readiness_timeout_secs: u64,
    prompt_manifest_sha256: String,
    manifest: Manifest,
    /// Caller-declared arm/model identity, retained without claiming verification.
    provenance: Map<String, Value>,
}
impl Input {
    fn phase(&self, prompt: &PromptRecord, timeout: f64) -> requests::Input {
        requests::Input {
            schema_version: 1,
            round: self.round,
            version: self.version,
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            output_tokens: self.output_tokens,
            request_timeout_secs: timeout,
            stagger_ms: 0.0,
            prompts: vec![Prompt {
                family: prompt.family.clone(),
                prompt: prompt.prompt.clone(),
            }],
        }
    }
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.readiness_timeout_secs > self.timeout_secs
            || !(1..=86400).contains(&self.timeout_secs)
            || !(1..=4096).contains(&self.output_tokens)
            || self.manifest.prompts.is_empty()
            || self.manifest.prompts.len() > 1000
            || self
                .manifest
                .prompts
                .iter()
                .any(|p| p.prompt.len() > 16 * 1024)
            || self.prompt_manifest_sha256
                != hex::encode(Sha256::digest(serde_json::to_vec(&self.manifest)?))
        {
            return Err("invalid serial A/B cell schema, bounds or typed manifest hash".into());
        }
        for prompt in &self.manifest.prompts {
            self.phase(prompt, self.request_timeout_secs).validate()?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Evidence {
    schema_version: u64,
    round: u64,
    version: Version,
    model: String,
    prompt_manifest_sha256: String,
    prompt_manifest_metadata: Map<String, Value>,
    provenance: Map<String, Value>,
    discarded_calibration_requests_per_cell: u64,
    calibration_request: Option<Value>,
    requests: Vec<Value>,
    makespan_ms: Option<f64>,
    timing_origin: &'static str,
    successful_requests: usize,
    prefill_telemetry_available: bool,
    error: Option<String>,
    input_sha256: String,
}
impl Evidence {
    fn new(input: &Input) -> Self {
        Self {
            schema_version: 1,
            round: input.round,
            version: input.version,
            model: input.model.clone(),
            prompt_manifest_sha256: input.prompt_manifest_sha256.clone(),
            prompt_manifest_metadata: input.manifest.metadata.clone(),
            provenance: input.provenance.clone(),
            discarded_calibration_requests_per_cell: 1,
            calibration_request: None,
            requests: Vec::new(),
            makespan_ms: None,
            timing_origin: "measured-cell-start",
            successful_requests: 0,
            prefill_telemetry_available: false,
            error: None,
            input_sha256: hex::encode(Sha256::digest(
                serde_json::to_vec(input).expect("validated JSON input"),
            )),
        }
    }
}
fn remaining(input: &Input, started: Instant, cancellation: &Cancellation) -> DynResult<f64> {
    if cancellation.is_cancelled() {
        return Err("serial A/B cell interrupted".into());
    }
    let remaining = Duration::from_secs(input.timeout_secs).saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err("serial A/B cell deadline expired".into());
    }
    Ok(remaining.as_secs_f64().min(input.request_timeout_secs))
}

async fn execute_with<F, Fut>(input: &Input, cancellation: Cancellation, mut request: F) -> Evidence
where
    F: FnMut(requests::Input, Cancellation) -> Fut,
    Fut: std::future::Future<Output = DynResult<requests::Phase>>,
{
    let mut evidence = Evidence::new(input);
    let started = Instant::now();
    let result: DynResult<()> = async {
        input.validate()?;
        ready(input, started, &cancellation).await?;
        let first = &input.manifest.prompts[0];
        let calibration = request(
            input.phase(first, remaining(input, started, &cancellation)?),
            cancellation.clone(),
        )
        .await?;
        let passed = calibration.passed();
        evidence.calibration_request = Some(serde_json::to_value(calibration)?);
        if !passed {
            return Err("calibration request failed; measurement withheld".into());
        }
        let measured_started = Instant::now();
        for (index, prompt) in input.manifest.prompts.iter().enumerate() {
            let request_offset_ms = measured_started.elapsed().as_secs_f64() * 1000.0;
            let phase = request(
                input.phase(prompt, remaining(input, started, &cancellation)?),
                cancellation.clone(),
            )
            .await?;
            let passed = phase.passed();
            let mut phase = serde_json::to_value(phase)?;
            let mut row = phase["requests"]
                .as_array_mut()
                .and_then(|rows| rows.pop())
                .ok_or("serial request receipt absent")?;
            row["request_id"] = json!(index);
            for key in ["submitted_ms", "first_token_ms", "completed_ms"] {
                if let Some(value) = row[key].as_f64() {
                    row[key] = json!(value + request_offset_ms);
                }
            }
            let mut provenance = prompt.provenance.clone();
            provenance.insert("family".into(), json!(prompt.family));
            row["prompt_provenance"] = Value::Object(provenance);
            evidence.requests.push(row);
            evidence.successful_requests += usize::from(passed);
        }
        evidence.makespan_ms = Some(measured_started.elapsed().as_secs_f64() * 1000.0);
        remaining(input, started, &cancellation)?;
        if evidence.successful_requests != input.manifest.prompts.len() {
            return Err("measured request failed; partial evidence retained".into());
        }
        Ok(())
    }
    .await;
    if let Err(error) = result {
        evidence.error = Some(error.to_string().chars().take(1024).collect());
    }
    evidence
}

async fn ready(input: &Input, started: Instant, cancellation: &Cancellation) -> DynResult<()> {
    if input.readiness_timeout_secs == 0 {
        return Ok(());
    }
    let until = Instant::now() + Duration::from_secs(input.readiness_timeout_secs);
    loop {
        let budget = until
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs_f64(remaining(
                input,
                started,
                cancellation,
            )?))
            .min(Duration::from_millis(250));
        if budget.is_zero() {
            return Err("adaptive OpenAI model readiness expired".into());
        }
        if let Ok(Ok(bytes)) = tokio::time::timeout(
            budget,
            crate::automation::openai_exchange::get(&format!("{}/models", input.base_url)),
        )
        .await
            && let Ok(document) = serde_json::from_slice::<Value>(&bytes)
            && document["data"].as_array().is_some_and(|rows| {
                rows.iter()
                    .any(|row| row["id"].as_str() == Some(input.model.as_str()))
            })
        {
            return Ok(());
        }
        tokio::time::sleep(
            Duration::from_millis(10).min(until.saturating_duration_since(Instant::now())),
        )
        .await;
    }
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let bytes = super::adaptive_identity::bounded(Path::new(opts["--input"]), 16 * 1024 * 1024)?;
    let input: Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let evidence = runtime.block_on(execute_with(
        &input,
        interrupt.cancellation(),
        requests::execute,
    ));
    let failed = evidence.error.is_some();
    let publication = publish(
        Path::new(opts["--output"]),
        &serde_json::to_vec_pretty(&evidence)?,
    );
    let finished = interrupt.finish();
    publication?;
    finished?;
    if failed {
        Err("serial A/B cell failed; evidence retained".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "sequential_cell_tests.rs"]
mod tests;
