//! Existing llama-benchy adapter; no prompt/tokenizer reimplementation.
use crate::{
    command::DynResult,
    process::{self, Value as Argument},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    config: PathBuf,
    config_sha256: String,
    cell: Value,
    base_url: String,
    served_model: String,
    launch_provenance: Value,
    output: PathBuf,
    timeout_seconds: u64,
    request_timeout_seconds: u64,
    benchy: super::competitive_launch::Artifact,
    tokenizer: PathBuf,
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let [flag, path] = args else {
        return Err("synthetic worker requires --input PATH".into());
    };
    if flag != "--input" {
        return Err("synthetic worker requires --input PATH".into());
    }
    let started = Instant::now();
    let input: Input = serde_json::from_slice(&super::competitive_cell::read(
        Path::new(path),
        8 * 1024 * 1024,
    )?)?;
    let bytes = super::competitive_cell::read(&input.config, 8 * 1024 * 1024)?;
    if hex::encode(Sha256::digest(&bytes)) != input.config_sha256 {
        return Err("synthetic config pin mismatch".into());
    }
    let document: Value = serde_json::from_slice(&bytes)?;
    super::competitive_plan::admit_cell(&document, &bytes, &input.cell)?;
    if input.cell["workload"] != "synthetic"
        || !input.output.is_absolute()
        || !input.tokenizer.is_absolute()
        || !(4..=86400).contains(&input.timeout_seconds)
        || input.request_timeout_seconds == 0
    {
        return Err("invalid synthetic worker boundaries".into());
    }
    super::competitive_launch::file(&input.benchy)?;
    let uri: hyper::Uri = input.base_url.parse()?;
    if uri.scheme_str() != Some("http")
        || uri.host() != Some("127.0.0.1")
        || uri.port_u16().is_none_or(|port| port == 0)
        || uri.path() != "/v1"
        || uri.query().is_some()
    {
        return Err("synthetic worker requires owned loopback endpoint".into());
    }
    std::fs::create_dir(&input.output)?;
    let deadline = started + Duration::from_secs(input.timeout_seconds);
    let result = execute(&input, &document, deadline);
    let summary = match &result {
        Ok(summary) => summary.clone(),
        Err(error) => {
            json!({"schema_version":1,"scope":"competitive_synthetic_worker","completed":false,"passed":false,"error":error.to_string(),"cell":input.cell,"config_sha256":input.config_sha256,"launch_provenance":input.launch_provenance})
        }
    };
    write_new(&input.output.join("worker-summary.json"), &summary)?;
    result.map(|_| ())
}
fn execute(input: &Input, document: &Value, deadline: Instant) -> DynResult<Value> {
    readiness(input, deadline)?;
    let version = invoke(input, vec!["--version".into()], deadline, "version")?;
    if std::str::from_utf8(&version)?.trim()
        != document["baseline"]["llama_benchy_version"]
            .as_str()
            .ok_or("benchy version")?
    {
        return Err("benchy version differs from pinned baseline".into());
    }
    let mut warm = common(input, document)?;
    warm.extend(["--tg", "8", "--concurrency", "4", "--format", "json"].map(str::to_owned));
    invoke(input, warm, deadline, "warmup")?;
    let mut measured = common(input, document)?;
    let tokens = input.cell["output_tokens"]
        .as_u64()
        .ok_or("output tokens")?;
    let concurrency = input.cell["concurrency"].as_u64().ok_or("concurrency")?;
    measured.extend([
        "--tg".into(),
        tokens.to_string(),
        "--concurrency".into(),
        concurrency.to_string(),
        "--format".into(),
        "json".into(),
        "--save-result".into(),
        input
            .output
            .join("result.json")
            .to_str()
            .ok_or("result Unicode")?
            .into(),
        "--emit-progress".into(),
        input
            .output
            .join("progress.jsonl")
            .to_str()
            .ok_or("progress Unicode")?
            .into(),
    ]);
    invoke(input, measured, deadline, "measured")?;
    let result = super::competitive_cell::read(&input.output.join("result.json"), 8 * 1024 * 1024)?;
    let progress =
        super::competitive_cell::read(&input.output.join("progress.jsonl"), 64 * 1024 * 1024)?;
    let requests = concurrency
        .checked_mul(document["synthetic"]["runs"].as_u64().ok_or("runs")?)
        .ok_or("request count overflow")?;
    validate(&progress, &result, requests, tokens)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let probed = runtime.block_on(super::competitive_parity::probe(
        &input.base_url,
        &input.served_model,
        usize::try_from(concurrency)?,
        deadline,
        interrupt.cancellation(),
    ));
    let finish = interrupt.finish();
    let parity = probed?;
    finish?;
    write_new(&input.output.join("parity.json"), &parity)?;
    if parity["passed"] != true {
        return Err("synthetic scheduler parity probe failed; partial evidence retained".into());
    }
    Ok(
        json!({"schema_version":1,"scope":"competitive_synthetic_worker","completed":true,"passed":true,"cell":input.cell,"config_sha256":input.config_sha256,"launch_provenance":input.launch_provenance,"benchy_sha256":input.benchy.sha256,"result_sha256":hex::encode(Sha256::digest(result)),"progress_sha256":hex::encode(Sha256::digest(progress)),"completed_requests":requests,"parity_sha256":crate::product::digest::file_sha256(&input.output.join("parity.json")).map_err(|error|error.error)?}),
    )
}
fn readiness(input: &Input, deadline: Instant) -> DynResult<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = (|| {
        while !interrupt.cancellation().is_cancelled() {
            let budget = deadline
                .saturating_duration_since(Instant::now())
                .saturating_sub(Duration::from_secs(3))
                .min(Duration::from_secs(input.request_timeout_seconds))
                .min(Duration::from_secs(2));
            if budget.is_zero() {
                break;
            }
            let cancellation = interrupt.cancellation();
            if let Some(bytes) = runtime.block_on(super::competitive_cell::ready_get(
                &format!("{}/models", input.base_url),
                budget,
                &cancellation,
            ))? && let Ok(models) = serde_json::from_slice::<Value>(&bytes)
                && models["data"].as_array().is_some_and(|models| {
                    models.iter().any(|model| model["id"] == input.served_model)
                })
            {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Err("synthetic server readiness deadline or cancellation".into())
    })();
    let finish = interrupt.finish();
    finish?;
    result
}
fn common(input: &Input, document: &Value) -> DynResult<Vec<String>> {
    let synthetic = &document["synthetic"];
    let mut extra = vec![
        format!(
            "temperature={}",
            synthetic["temperature"].as_f64().ok_or("temperature")?
        ),
        format!("seed={}", synthetic["seed"].as_u64().ok_or("seed")?),
    ];
    if ["vllm", "sglang"].contains(&input.cell["arm"].as_str().ok_or("arm")?) {
        extra.push("return_token_ids=false".into());
    }
    Ok(vec![
        "--base-url".into(),
        input.base_url.clone(),
        "--api-key".into(),
        "EMPTY".into(),
        "--model".into(),
        input.served_model.clone(),
        "--served-model-name".into(),
        input.served_model.clone(),
        "--tokenizer".into(),
        input.tokenizer.to_str().ok_or("tokenizer Unicode")?.into(),
        "--pp".into(),
        input.cell["prompt_tokens"]
            .as_u64()
            .ok_or("prompt tokens")?
            .to_string(),
        "--exact-tg".into(),
        "--extra-body".into(),
        extra.join(","),
        "--depth".into(),
        "0".into(),
        "--runs".into(),
        synthetic["runs"].as_u64().ok_or("runs")?.to_string(),
        "--warmup-runs".into(),
        "0".into(),
        "--latency-mode".into(),
        "none".into(),
        "--skip-coherence".into(),
        "--no-adapt-prompt".into(),
        "--no-cache".into(),
        "--no-warmup".into(),
        "--exit-on-first-fail".into(),
        "--no-results-on-fail".into(),
    ])
}
fn invoke(input: &Input, args: Vec<String>, deadline: Instant, label: &str) -> DynResult<Vec<u8>> {
    let execution = deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(3))
        .ok_or("synthetic command lacks cleanup reserve")?;
    if execution.is_zero() {
        return Err("synthetic deadline exceeded".into());
    }
    let spec = process::ProcessSpec {
        executable: input.benchy.path.clone(),
        arguments: args
            .into_iter()
            .map(|value| Argument::Public(value.into()))
            .collect(),
        cwd: input.output.clone(),
        environment: super::competitive_launch::environment(None, false),
    };
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = process::supervise_raw(
        &spec,
        &super::competitive_run_cell::limits(execution),
        &interrupt.cancellation(),
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(8 * 1024 * 1024),
            stderr: std::num::NonZeroUsize::new(8 * 1024 * 1024),
        },
    );
    let finish = interrupt.finish();
    let report = result?;
    finish?;
    let stdout = report
        .stdout
        .as_ref()
        .ok_or("synthetic raw stdout absent")?
        .as_bytes();
    let stderr = report
        .stderr
        .as_ref()
        .ok_or("synthetic raw stderr absent")?
        .as_bytes();
    std::fs::write(input.output.join(format!("{label}.stdout.log")), stdout)?;
    std::fs::write(input.output.join(format!("{label}.stderr.log")), stderr)?;
    if !report.process.success() {
        return Err(
            format!("synthetic {label} failed or required forced/incomplete cleanup").into(),
        );
    }
    Ok(stdout.to_vec())
}
pub(super) fn validate(
    progress: &[u8],
    result: &[u8],
    requests: u64,
    tokens: u64,
) -> DynResult<()> {
    let mut count = 0;
    for line in std::str::from_utf8(progress)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let event: Value = serde_json::from_str(line)?;
        if event["type"] == "request_end" {
            count += 1;
            if !event["error"].is_null() && event["error"] != false && event["error"] != "" {
                return Err("benchy request contained hidden error".into());
            }
            if event["total_tokens"].as_u64() != Some(tokens) {
                return Err("benchy response shorter than output budget".into());
            }
        }
    }
    if count != requests {
        return Err("benchy request completion count differs".into());
    }
    let document: Value = serde_json::from_slice(result)?;
    let rows = document["benchmarks"]
        .as_array()
        .ok_or("benchy benchmark rows missing")?;
    let [benchmark] = rows.as_slice() else {
        return Err("benchy requires one measured result".into());
    };
    if benchmark["response_size"].as_u64() != Some(tokens)
        || !benchmark["tg_throughput"]["mean"]
            .as_f64()
            .is_some_and(|value| value.is_finite() && value > 0.0)
    {
        return Err(
            "benchy measured result lacks finite positive throughput or output budget".into(),
        );
    }
    Ok(())
}
pub(super) fn write_new(path: &Path, value: &Value) -> DynResult<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}
