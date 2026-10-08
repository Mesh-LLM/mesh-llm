//! Bounded real concurrent request cohort; retained parent owns all child lifecycle and telemetry.
use super::{
    Prompt, acceptance::Version, adaptive_identity as io, options, radix_workload::Batch, requests,
};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub batch: Batch,
    pub model: String,
    pub base_url: String,
    pub output_tokens: u32,
    pub request_timeout_secs: f64,
    pub readiness_timeout_secs: u64,
    pub timeout_secs: u64,
}
impl Input {
    pub fn validate(&self) -> DynResult<()> {
        requests::Input {
            schema_version: self.schema_version,
            round: 1,
            version: Version::Old,
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            output_tokens: self.output_tokens.into(),
            request_timeout_secs: self.request_timeout_secs,
            stagger_ms: 0.0,
            prompts: vec![Prompt {
                family: "radix".into(),
                prompt: "validation".into(),
            }],
        }
        .validate()?;
        if !(1..=16).contains(&self.batch.concurrency)
            || self.batch.prompts.is_empty()
            || self.batch.prompts.len() > 128
            || self
                .batch
                .prompts
                .iter()
                .any(|p| p.trim().is_empty() || p.len() > 512 * 1024)
            || self.batch.warmup && (self.batch.concurrency != 1 || self.batch.prompts.len() != 1)
            || !(1..=4096).contains(&self.output_tokens)
            || !(2..=86400).contains(&self.timeout_secs)
            || self.readiness_timeout_secs == 0
            || self.readiness_timeout_secs >= self.timeout_secs
            || self.model.len() > 4096
        {
            return Err("radix HTTP cohort counts/prompt/budget invalid".into());
        }
        Ok(())
    }
}
async fn cancelled(cancel: &Cancellation) {
    while !cancel.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
async fn ready(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    let until = until.min(Instant::now() + Duration::from_secs(input.readiness_timeout_secs));
    loop {
        let budget = until
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(250))
            .min(Duration::from_secs_f64(input.request_timeout_secs));
        if budget.is_zero() || cancel.is_cancelled() {
            return Err("radix advertised model readiness expired/interrupted".into());
        }
        if let Ok(Ok(bytes)) = tokio::time::timeout(
            budget,
            crate::automation::openai_exchange::get(&format!("{}/models", input.base_url)),
        )
        .await
            && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
            && value["data"]
                .as_array()
                .is_some_and(|a| a.len() <= 64 && a.iter().any(|m| m["id"] == input.model))
        {
            return Ok(());
        }
        tokio::time::sleep(
            Duration::from_millis(5).min(until.saturating_duration_since(Instant::now())),
        )
        .await;
    }
}
async fn request(input: Arc<Input>, index: usize, until: Instant) -> Value {
    let prompt = &input.batch.prompts[index];
    let mut row =
        json!({"request_id":index,"prompt_sha256":io::digest(prompt.as_bytes()),"error":null});
    let budget = until
        .saturating_duration_since(Instant::now())
        .min(Duration::from_secs_f64(input.request_timeout_secs));
    let body = json!({"model":input.model,"messages":[{"role":"user","content":prompt}],"max_tokens":input.output_tokens,"temperature":0,"seed":0,"stream":true,"stream_options":{"include_usage":true}});
    let result = tokio::time::timeout(
        budget,
        crate::automation::openai_exchange::request(&input.base_url, &body, false),
    )
    .await;
    match result {
        Ok(Ok(e)) => {
            row["ttft_ms"] = json!(e.ttft_seconds * 1000.0);
            row["elapsed_ms"] = json!(e.elapsed_seconds * 1000.0);
            row["tpot_ms"] = json!(
                e.generation_seconds * 1000.0 / e.completion_tokens.saturating_sub(1).max(1) as f64
            );
            row["completion_tokens"] = json!(e.completion_tokens);
            row["prompt_tokens"] = json!(e.prompt_tokens);
            row["cached_tokens"] = json!(e.cached_tokens);
            row["content_sha256"] = json!(e.content_sha256);
        }
        Ok(Err(error)) => {
            row["error"] = json!(error.to_string().chars().take(1024).collect::<String>())
        }
        Err(_) => row["error"] = json!("radix request deadline expired"),
    };
    row
}
pub(super) async fn execute(input: &Input, cancel: Cancellation, directory: &Path) -> Value {
    let mut output = json!({"schema_version":1,"request_sha256":io::digest(&serde_json::to_vec(input).unwrap()),"batch":input.batch,"requests":[],"makespan_ms":null,"error":null});
    if let Err(error) = input.validate() {
        output["error"] = json!(error.to_string());
        return output;
    }
    let until = Instant::now() + Duration::from_secs(input.timeout_secs);
    if let Err(error) = ready(input, until, &cancel).await {
        output["error"] = json!(error.to_string());
        return output;
    }
    let input = Arc::new(input.clone());
    let epoch = Instant::now();
    let mut tasks = tokio::task::JoinSet::new();
    let mut next = 0;
    let mut rows = vec![];
    while next < input.batch.prompts.len() || !tasks.is_empty() {
        while next < input.batch.prompts.len() && tasks.len() < input.batch.concurrency as usize {
            let i = next;
            let input = input.clone();
            tasks.spawn(async move { request(input, i, until).await });
            next += 1;
        }
        let budget = until.saturating_duration_since(Instant::now());
        let result = tokio::select! {biased;()=cancelled(&cancel)=>Err("radix batch interrupted"),value=tokio::time::timeout(budget,tasks.join_next())=>match value{Ok(Some(Ok(row)))=>Ok(row),Ok(_)=>Err("radix owned request task failed"),Err(_)=>Err("radix whole batch deadline expired")}};
        match result {
            Ok(row) => {
                let index = row["request_id"].as_u64().unwrap();
                let journal =
                    json!({"schema_version":1,"request_sha256":output["request_sha256"],"row":row});
                if let Err(error) =
                    serde_json::to_vec(&journal)
                        .map_err(|e| e.into())
                        .and_then(|bytes| {
                            io::fresh(&directory.join(format!("request-{index}.json")), &bytes)
                        })
                {
                    output["error"] = json!(format!("radix journal publication failed: {error}"));
                    rows.push(row);
                    break;
                }
                rows.push(row);
            }
            Err(error) => {
                output["error"] = json!(error);
                break;
            }
        }
    }
    tasks.abort_all();
    while let Some(result) = tasks.join_next().await {
        if let Ok(row) = result {
            rows.push(row);
        }
    }
    rows.sort_by_key(|r| r["request_id"].as_u64());
    if (rows.len() != input.batch.prompts.len() || rows.iter().any(|r| !r["error"].is_null()))
        && output["error"].is_null()
    {
        output["error"] = json!("radix request roster incomplete/failed");
    }
    output["makespan_ms"] = json!(epoch.elapsed().as_secs_f64() * 1000.0);
    output["requests"] = json!(rows);
    output
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let flags = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let output = Path::new(flags["--output"]);
    if !output.is_absolute() || !Path::new(flags["--input"]).is_absolute() {
        return Err("radix worker requires absolute owned paths".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("radix worker output must be fresh".into()),
    };
    let input: Input =
        serde_json::from_slice(&io::bounded(Path::new(flags["--input"]), 64 * 1024 * 1024)?)?;
    input.validate()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let cancellation = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut value = runtime.block_on(execute(
        &input,
        cancellation.clone(),
        output.parent().ok_or("radix output parent absent")?,
    ));
    let finished = interrupt.finish().map_err(Into::into);
    let terminal = super::radix_terminal::finalize(&mut value, finished, &cancellation, deadline);
    io::fresh(output, &serde_json::to_vec_pretty(&value)?)?;
    terminal
}
