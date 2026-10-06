//! Owned full-message HTTP cohort worker; no server launch or state replacement here.
use super::{adaptive_identity as io, kv_manifest::Manifest, options};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(super) const BASE: &str = "http://127.0.0.1:9337/v1";
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Fill,
    Replay,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub phase: Phase,
    pub manifest: Manifest,
    pub manifest_sha256: String,
    pub expected_model: Option<String>,
    pub baseline_sha256: Option<String>,
    pub restore_repeats: u32,
    pub max_output_tokens: u64,
    pub request_timeout_secs: f64,
    pub ready_timeout_secs: u64,
    pub timeout_secs: u64,
}
impl Input {
    pub fn validate(&self) -> DynResult<()> {
        self.manifest.validate()?;
        if self.schema_version != 1
            || self.manifest_sha256 != io::digest(&serde_json::to_vec(&self.manifest)?)
            || !(1..=128).contains(&self.restore_repeats)
            || !(1..=4096).contains(&self.max_output_tokens)
            || !self.request_timeout_secs.is_finite()
            || self.request_timeout_secs <= 0.0
            || self.request_timeout_secs > 86400.0
            || self.ready_timeout_secs == 0
            || self.ready_timeout_secs >= self.timeout_secs
            || !(2..=86400).contains(&self.timeout_secs)
            || self
                .expected_model
                .as_ref()
                .is_some_and(|v| v.trim().is_empty() || v.len() > 4096)
            || self.phase == Phase::Replay
                && self
                    .baseline_sha256
                    .as_ref()
                    .is_none_or(|v| v.len() != 64 || !v.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err("restart worker manifest/cohort/timing identity invalid".into());
        }
        Ok(())
    }
}
async fn cancelled(cancel: &Cancellation) {
    while !cancel.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
async fn ready(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<String> {
    let deadline = (Instant::now() + Duration::from_secs(input.ready_timeout_secs)).min(until);
    loop {
        if cancel.is_cancelled() {
            return Err("restart model readiness interrupted".into());
        }
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(250))
            .min(Duration::from_secs_f64(input.request_timeout_secs));
        if remaining.is_zero() {
            return Err("restart advertised model readiness expired".into());
        }
        if let Ok(Ok(bytes)) = tokio::time::timeout(
            remaining,
            crate::automation::openai_exchange::get(&format!("{BASE}/models")),
        )
        .await
            && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
            && let Some(models) = value["data"]
                .as_array()
                .filter(|v| !v.is_empty() && v.len() <= 64)
            && let Some(id) = models.iter().filter_map(|v| v["id"].as_str()).find(|id| {
                !id.trim().is_empty()
                    && id.len() <= 4096
                    && input
                        .expected_model
                        .as_ref()
                        .is_none_or(|expected| expected == id)
            })
        {
            return Ok(id.to_owned());
        }
        tokio::time::sleep(
            Duration::from_millis(10).min(deadline.saturating_duration_since(Instant::now())),
        )
        .await;
    }
}
async fn measured(
    input: &Input,
    model: &str,
    messages: &[Value],
    until: Instant,
    cancel: &Cancellation,
) -> Result<Value, String> {
    let budget = until
        .saturating_duration_since(Instant::now())
        .min(Duration::from_secs_f64(input.request_timeout_secs));
    if budget.is_zero() {
        return Err("restart request whole-worker deadline expired".into());
    }
    let body = json!({"model":model,"messages":messages,"max_tokens":input.max_output_tokens,"temperature":0,"seed":42,"stream":true,"stream_options":{"include_usage":true}});
    let exchange = async {
        tokio::time::timeout(
            budget,
            crate::automation::openai_exchange::request(BASE, &body, false),
        )
        .await
        .map_err(|_| "restart request deadline expired".to_owned())?
        .map_err(|e| e.to_string())
    };
    let evidence = tokio::select! {biased;()=cancelled(cancel)=>return Err("restart request interrupted".into()),value=exchange=>value?};
    Ok(
        json!({"ttft_seconds":evidence.ttft_seconds,"total_seconds":evidence.elapsed_seconds,"prompt_tokens":evidence.prompt_tokens,"completion_tokens":evidence.completion_tokens,"cached_tokens":evidence.cached_tokens,"decode_tokens_per_second":if evidence.generation_seconds>0.0 {Some(evidence.completion_tokens as f64/evidence.generation_seconds)}else{None},"content_sha256":evidence.content_sha256}),
    )
}
pub(super) async fn execute(input: &Input, cancel: Cancellation, journal: Option<&Path>) -> Value {
    let started = Instant::now();
    let mut output = json!({"schema_version":1,"request_sha256":io::digest(&serde_json::to_vec(input).unwrap()),"manifest_sha256":input.manifest_sha256,"phase":input.phase,"model_id":null,"ready_seconds":null,"requests":[],"baseline_sha256":input.baseline_sha256,"error":null});
    if let Err(error) = input.validate() {
        output["error"] = json!(error.to_string());
        return output;
    }
    let until = started + Duration::from_secs(input.timeout_secs);
    let model = match ready(input, until, &cancel).await {
        Ok(id) => id,
        Err(e) => {
            output["error"] = json!(e.to_string());
            return output;
        }
    };
    output["model_id"] = json!(model);
    output["ready_seconds"] = json!(started.elapsed().as_secs_f64());
    if journal.is_some() {
        use std::io::Write as _;
        println!(
            "{}",
            json!({"event":"kv_restart_model_ready","request_sha256":output["request_sha256"]})
        );
        if let Err(error) = std::io::stdout().flush() {
            output["error"] = json!(format!("restart readiness marker failed: {error}"));
            return output;
        }
    }
    let count = if input.phase == Phase::Fill {
        input.manifest.turns.len()
    } else {
        input.restore_repeats as usize
    };
    let mut rows = Vec::new();
    for index in 0..count {
        let turn = if input.phase == Phase::Fill {
            index
        } else {
            input.manifest.turns.len() - 1
        };
        let messages = input.manifest.messages(turn).unwrap();
        let cohort = if input.phase == Phase::Fill {
            "fill"
        } else if index == 0 {
            "restore"
        } else {
            "warm"
        };
        let number = if cohort == "warm" { index } else { index + 1 };
        let mut row = json!({"cohort":cohort,"request_id":format!("{cohort}-{number}"),"request_index":if cohort=="warm"{index-1}else{index},"prompt_sha256":io::digest(&serde_json::to_vec(&messages).unwrap()),"messages_count":messages.len(),"messages_end_role":"user","error":null});
        match measured(input, &model, &messages, until, &cancel).await {
            Ok(value) => {
                row.as_object_mut()
                    .unwrap()
                    .extend(value.as_object().unwrap().clone());
                if input.phase == Phase::Replay
                    && row["content_sha256"] != input.baseline_sha256.as_deref().unwrap()
                {
                    row["error"] =
                        json!("restart canonical completion differs from last fill baseline");
                }
            }
            Err(e) => row["error"] = json!(e.chars().take(1024).collect::<String>()),
        }
        let failed = !row["error"].is_null();
        if let Some(directory) = journal {
            let record = json!({"schema_version":1,"request_sha256":output["request_sha256"],"manifest_sha256":input.manifest_sha256,"phase":input.phase,"row":row});
            if let Err(error) = serde_json::to_vec_pretty(&record)
                .map_err(|e| e.into())
                .and_then(|bytes| {
                    io::fresh(&directory.join(format!("request-{index}.json")), &bytes)
                })
            {
                output["error"] = json!(format!("restart request journal failed: {error}"));
                rows.push(row);
                break;
            }
        }
        rows.push(row);
        if failed {
            output["error"] = json!("restart cohort request failed; partial rows retained");
            break;
        }
    }
    if input.phase == Phase::Fill && rows.len() == count && output["error"].is_null() {
        output["baseline_sha256"] = rows.last().unwrap()["content_sha256"].clone();
    }
    output["requests"] = json!(rows);
    output
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let flags = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let input: Input =
        serde_json::from_slice(&io::bounded(Path::new(flags["--input"]), 16 * 1024 * 1024)?)?;
    input.validate()?;
    let output = Path::new(flags["--output"]);
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("restart worker receipt must be fresh".into()),
    };
    if !output.is_absolute() || !Path::new(flags["--input"]).is_absolute() {
        return Err("restart worker input/output paths must be absolute".into());
    }
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let value = runtime.block_on(execute(&input, interrupt.cancellation(), output.parent()));
    let publication = io::fresh(output, &serde_json::to_vec_pretty(&value)?);
    let finished = interrupt.finish();
    publication?;
    finished?;
    if !value["error"].is_null() {
        return Err("restart cohort failed; receipt retained".into());
    }
    Ok(())
}
