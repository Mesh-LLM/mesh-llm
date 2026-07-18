use std::{
    fs,
    net::TcpStream,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use model_artifact::ModelIdentity;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, recv_reply,
    write_stage_message,
};
use skippy_runtime::{
    RuntimeConfig, RuntimeLoadMode, StageModel,
    package::{PackageStageRequest, materialize_layer_package_details},
};

use crate::{
    cli::{DEFAULT_RUN_MAX_NEW_TOKENS, FocusedRuntimeArgs, RunArgs},
    model_identity::model_identity_for_path,
    support::{ChildGuard, ensure_release_skippy_server_bin, parse_wire_dtype, retry},
};

#[path = "deployment.rs"]
mod deployment;
#[path = "focused_runtime.rs"]
mod focused_runtime;

use deployment::{
    DeploymentPlan, StageAssignment, build_deployment_plan, collect_and_cleanup_remote,
    configure_child_logs, connect_endpoint_ready, execute_remote_plan, model_ref_for_configs,
    parse_hosts, parse_stage_ranges, validate_balanced_stage_ranges, validate_distinct_stage_hosts,
    validate_topology_plan, wait_remote_readiness, write_stage_configs, write_stage_topology,
};

pub(super) struct DistributedRunOutcome {
    run_id: String,
    topology_id: String,
    model_id: String,
    model_identity: ModelIdentity,
    run_dir: PathBuf,
    plan_path: PathBuf,
    report_path: PathBuf,
    execute_remote: bool,
    stage_count: usize,
    hosts: Vec<String>,
    report_counts: Value,
    remote_status_path: Option<PathBuf>,
    driver_result_path: Option<PathBuf>,
    driver_report: Option<PromptDriverReport>,
    startup_elapsed: Option<Duration>,
    run_elapsed: Duration,
}

#[derive(Debug, Serialize)]
pub(super) struct PromptDriverReport {
    first_stage_endpoint: String,
    prompt_count: usize,
    max_new_tokens: usize,
    prefill_chunk_size: Option<usize>,
    prefill_chunk_threshold: Option<usize>,
    prefill_chunk_schedule: Option<String>,
    corpus: Option<PathBuf>,
    summary: PromptDriverSummary,
    results: Vec<PromptDriverResult>,
}

#[derive(Debug, Serialize)]
pub(super) struct PromptDriverSummary {
    prompt_tokens_total: usize,
    generated_tokens_total: usize,
    elapsed_ms_total: u128,
    elapsed_ms_mean: f64,
    elapsed_ms_p50: u128,
    elapsed_ms_p95: u128,
    elapsed_ms_p99: u128,
    wire_elapsed_ms_mean: f64,
    wire_elapsed_ms_p50: u128,
    wire_elapsed_ms_p95: u128,
    wire_elapsed_ms_p99: u128,
    prefill_elapsed_ms_mean: f64,
    prefill_elapsed_ms_p50: u128,
    prefill_elapsed_ms_p95: u128,
    prefill_elapsed_ms_p99: u128,
    ttft_ms_mean: f64,
    ttft_ms_p50: u128,
    ttft_ms_p95: u128,
    ttft_ms_p99: u128,
    decode_elapsed_ms_mean: f64,
    decode_elapsed_ms_p50: u128,
    decode_elapsed_ms_p95: u128,
    decode_elapsed_ms_p99: u128,
    total_tokens_per_second: f64,
    generated_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct PromptDriverResult {
    prompt_id: Option<String>,
    category: Option<String>,
    prompt: String,
    token_ids: Vec<i32>,
    prefill_token_count: usize,
    prefill_chunk_count: usize,
    effective_prefill_chunk_size: Option<usize>,
    predicted_tokens: Vec<i32>,
    max_new_tokens: usize,
    elapsed_ms: u128,
    wire_elapsed_ms: u128,
    prefill_elapsed_ms: u128,
    ttft_ms: u128,
    decode_elapsed_ms: u128,
}

#[derive(Debug, Clone)]
struct PromptCase {
    prompt_id: Option<String>,
    category: Option<String>,
    prompt: String,
}

struct DriverTokenizer {
    model: StageModel,
    _materialized_model_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct CreateRunResponse {
    run_id: String,
}

pub fn run_distributed(args: RunArgs) -> Result<()> {
    let outcome = run_distributed_collect(args)?;
    print_distributed_run_outcome(&outcome)
}

pub(super) fn run_distributed_collect(args: RunArgs) -> Result<DistributedRunOutcome> {
    ensure_release_skippy_server_bin(&args.stage_server_bin)?;
    let run_started = Instant::now();
    let hosts = parse_hosts(&args.hosts)?;
    let ranges = parse_stage_ranges(&args.splits, args.layer_end)?;
    validate_distinct_stage_hosts(&hosts, ranges.len())?;
    validate_topology_plan(&args, &hosts, &ranges)?;
    validate_balanced_stage_ranges(&ranges)?;
    let run_id = args.run_id.clone().unwrap_or_else(generate_bench_run_id);
    let run_dir = args.work_dir.join(&run_id);
    let config_dir = run_dir.join("configs");
    let topology_path = config_dir.join("topology.json");
    fs::create_dir_all(&config_dir)
        .with_context(|| format!("create benchmark work dir {}", config_dir.display()))?;

    let metrics_http = format!("http://{}", args.metrics_http_addr);
    let db = args
        .db
        .clone()
        .unwrap_or_else(|| run_dir.join("metrics.sqlite"));
    let model_ref = model_ref_for_configs(&args)?;
    let fallback_model_identity =
        model_identity_for_path(&args.model_id, args.model_path.as_deref())?;
    let plan = build_deployment_plan(
        &args,
        &run_id,
        &hosts,
        &ranges,
        &config_dir,
        &model_ref,
        fallback_model_identity,
    )?;
    write_stage_configs(&args, &plan, &model_ref)?;
    write_stage_topology(&args, &plan, &topology_path)?;
    write_json_file(&run_dir.join("deployment-plan.json"), &plan)?;

    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;

    let mut metrics_command = Command::new(&args.metrics_server_bin);
    metrics_command.args([
        "serve",
        "--db",
        db.to_str().context("db path is not valid UTF-8")?,
        "--http-addr",
        &args.metrics_http_addr.to_string(),
        "--otlp-grpc-addr",
        &args.metrics_otlp_grpc_addr.to_string(),
    ]);
    configure_child_logs(&mut metrics_command, args.child_logs);
    let _metrics = ChildGuard::spawn(metrics_command)?;

    let run_config = json!({
        "run_id": run_id,
        "topology_id": args.topology_id,
        "model_id": plan.model_identity.model_id,
        "model_identity": plan.model_identity,
        "mode": "distributed-run",
        "hosts": hosts,
        "stage_load_mode": args.stage_load_mode,
        "stage_count": plan.stages.len(),
        "prompt_corpus": args.prompt_corpus.clone(),
        "prompt_limit": args.prompt_limit,
        "prefill_chunk_size": args.prefill_chunk_size,
        "prefill_chunk_threshold": args.prefill_chunk_threshold,
        "prefill_chunk_schedule": args.prefill_chunk_schedule,
        "max_new_tokens": effective_run_max_new_tokens(&args),
        "stage_max_inflight": args.stage_max_inflight,
        "stage_reply_credit_limit": args.stage_reply_credit_limit,
        "stage_async_prefill_forward": args.stage_async_prefill_forward,
        "stage_downstream_wire_delay_ms": args.stage_downstream_wire_delay_ms,
        "stage_downstream_wire_mbps": args.stage_downstream_wire_mbps,
        "stage_telemetry_queue_capacity": args.stage_telemetry_queue_capacity,
        "stage_telemetry_level": args.stage_telemetry_level,
        "stages": plan
            .stages
            .iter()
            .map(|stage| {
                json!({
                    "stage_id": stage.stage_id,
                    "stage_index": stage.stage_index,
                    "host": stage.host,
                    "layer_start": stage.layer_start,
                    "layer_end": stage.layer_end,
                    "bind_addr": stage.bind_addr,
                    "endpoint": stage.endpoint,
                })
            })
            .collect::<Vec<_>>(),
        "execute_remote": args.execute_remote,
        "keep_remote": args.keep_remote,
        "rsync_model_artifacts": args.rsync_model_artifacts,
    });
    retry(args.startup_timeout_secs, || {
        let response = client
            .post(format!("{metrics_http}/v1/runs"))
            .json(&run_config)
            .send()
            .and_then(|response| response.error_for_status())?
            .json::<CreateRunResponse>()?;
        if response.run_id == run_id {
            Ok(())
        } else {
            Err(anyhow!(
                "metrics-server returned unexpected run_id {}",
                response.run_id
            ))
        }
    })
    .context("metrics-server did not become ready")?;

    let mut protocol_ready = false;
    let mut startup_elapsed = None;
    let mut remote_sessions = Vec::new();
    let run_result = (|| -> Result<(Value, PathBuf, Option<PromptDriverReport>)> {
        let mut driver_result = None;
        if args.execute_remote {
            remote_sessions = execute_remote_plan(&args, &plan)?;
            wait_remote_readiness(&args, &plan)?;
            protocol_ready = true;
            startup_elapsed = Some(run_started.elapsed());
            let result = run_remote_prompt_driver(&args, &plan)?;
            driver_result = Some(result);
        }

        thread::sleep(Duration::from_secs(1));
        client
            .post(format!("{metrics_http}/v1/runs/{run_id}/finalize"))
            .send()
            .context("failed to finalize run")?
            .error_for_status()
            .context("metrics-server rejected finalize")?;
        let report: Value = client
            .get(format!("{metrics_http}/v1/runs/{run_id}/report.json"))
            .send()
            .context("failed to fetch report")?
            .error_for_status()
            .context("metrics-server rejected report fetch")?
            .json()
            .context("failed to parse report")?;

        let output = args
            .output
            .clone()
            .unwrap_or_else(|| run_dir.join("report.json"));
        write_json_file(&output, &report)?;
        if let Some(driver_result) = driver_result.as_ref() {
            write_json_file(&run_dir.join("driver-result.json"), driver_result)?;
        }
        Ok((report, output, driver_result))
    })();

    let mut remote_status_path = None;
    if args.execute_remote {
        let cleanup_statuses = collect_and_cleanup_remote(&args, &plan, &run_dir, protocol_ready)
            .context("collect remote logs and cleanup")?;
        let path = run_dir.join("remote-status.json");
        write_json_file(&path, &cleanup_statuses)?;
        remote_status_path = Some(path);
        if args.keep_remote {
            for session in remote_sessions.drain(..) {
                session.keep_alive();
            }
        }
    }

    let (report, output, driver_report) = run_result?;
    let driver_result_path = driver_report
        .as_ref()
        .map(|_| run_dir.join("driver-result.json"));

    Ok(DistributedRunOutcome {
        run_id,
        topology_id: plan.topology_id.clone(),
        model_id: plan.model_id.clone(),
        model_identity: plan.model_identity.clone(),
        run_dir: run_dir.clone(),
        plan_path: run_dir.join("deployment-plan.json"),
        report_path: output,
        execute_remote: args.execute_remote,
        stage_count: plan.stages.len(),
        hosts: plan.hosts.clone(),
        report_counts: report["counts"].clone(),
        remote_status_path,
        driver_result_path,
        driver_report,
        startup_elapsed,
        run_elapsed: run_started.elapsed(),
    })
}

fn print_distributed_run_outcome(outcome: &DistributedRunOutcome) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": outcome.run_id.clone(),
            "model_identity": outcome.model_identity.clone(),
            "plan": outcome.plan_path.clone(),
            "report": outcome.report_path.clone(),
            "execute_remote": outcome.execute_remote,
            "stage_count": outcome.stage_count,
            "hosts": outcome.hosts.clone(),
            "report_counts": outcome.report_counts.clone(),
            "remote_status": outcome.remote_status_path.clone(),
            "driver_result": outcome.driver_result_path.clone(),
        }))?
    );

    Ok(())
}

pub fn focused_runtime(args: FocusedRuntimeArgs) -> Result<()> {
    focused_runtime::focused_runtime(args)
}

pub(super) fn effective_run_max_new_tokens(args: &RunArgs) -> usize {
    args.max_new_tokens.unwrap_or(DEFAULT_RUN_MAX_NEW_TOKENS)
}

fn run_remote_prompt_driver(args: &RunArgs, plan: &DeploymentPlan) -> Result<PromptDriverReport> {
    let first = plan
        .stages
        .first()
        .context("deployment plan has no stages")?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let prompt_cases = prompt_cases(args)?;
    if prompt_cases.is_empty() {
        bail!("prompt corpus is empty");
    }
    if args.prompt_corpus.is_some() && args.prompt_token_ids.is_some() {
        bail!("--prompt-token-ids cannot be used with --prompt-corpus");
    }
    let tokenizer = if args.prompt_token_ids.is_some() {
        None
    } else {
        Some(DriverTokenizer::open(args, plan)?)
    };
    let mut results = Vec::with_capacity(prompt_cases.len());
    for (index, prompt_case) in prompt_cases.iter().enumerate() {
        let started = Instant::now();
        let token_ids = if let Some(token_ids) = args.prompt_token_ids.as_ref() {
            parse_prompt_token_ids(token_ids)?
        } else {
            tokenizer
                .as_ref()
                .expect("tokenizer is present without explicit prompt tokens")
                .tokenize(&prompt_case.prompt)?
        };
        let mut result =
            run_remote_prompt_case(args, first, wire_dtype, prompt_case, token_ids, index)?;
        result.elapsed_ms = started.elapsed().as_millis();
        results.push(result);
    }

    Ok(PromptDriverReport {
        first_stage_endpoint: first.endpoint.clone(),
        prompt_count: results.len(),
        max_new_tokens: effective_run_max_new_tokens(args),
        prefill_chunk_size: args.prefill_chunk_size,
        prefill_chunk_threshold: args.prefill_chunk_threshold,
        prefill_chunk_schedule: args.prefill_chunk_schedule.clone(),
        corpus: args.prompt_corpus.clone(),
        summary: prompt_driver_summary(&results),
        results,
    })
}

pub(super) fn prompt_driver_summary(results: &[PromptDriverResult]) -> PromptDriverSummary {
    let prompt_tokens_total = results.iter().map(|result| result.token_ids.len()).sum();
    let generated_tokens_total = results
        .iter()
        .map(|result| result.predicted_tokens.len())
        .sum();
    let elapsed_ms_total = results.iter().map(|result| result.elapsed_ms).sum();
    let elapsed_seconds = elapsed_ms_total as f64 / 1000.0;
    PromptDriverSummary {
        prompt_tokens_total,
        generated_tokens_total,
        elapsed_ms_total,
        elapsed_ms_mean: if results.is_empty() {
            0.0
        } else {
            elapsed_ms_total as f64 / results.len() as f64
        },
        elapsed_ms_p50: percentile_ms(results, 0.50),
        elapsed_ms_p95: percentile_ms(results, 0.95),
        elapsed_ms_p99: percentile_ms(results, 0.99),
        wire_elapsed_ms_mean: mean_ms(results, |result| result.wire_elapsed_ms),
        wire_elapsed_ms_p50: percentile_ms_by(results, 0.50, |result| result.wire_elapsed_ms),
        wire_elapsed_ms_p95: percentile_ms_by(results, 0.95, |result| result.wire_elapsed_ms),
        wire_elapsed_ms_p99: percentile_ms_by(results, 0.99, |result| result.wire_elapsed_ms),
        prefill_elapsed_ms_mean: mean_ms(results, |result| result.prefill_elapsed_ms),
        prefill_elapsed_ms_p50: percentile_ms_by(results, 0.50, |result| result.prefill_elapsed_ms),
        prefill_elapsed_ms_p95: percentile_ms_by(results, 0.95, |result| result.prefill_elapsed_ms),
        prefill_elapsed_ms_p99: percentile_ms_by(results, 0.99, |result| result.prefill_elapsed_ms),
        ttft_ms_mean: mean_ms(results, |result| result.ttft_ms),
        ttft_ms_p50: percentile_ms_by(results, 0.50, |result| result.ttft_ms),
        ttft_ms_p95: percentile_ms_by(results, 0.95, |result| result.ttft_ms),
        ttft_ms_p99: percentile_ms_by(results, 0.99, |result| result.ttft_ms),
        decode_elapsed_ms_mean: mean_ms(results, |result| result.decode_elapsed_ms),
        decode_elapsed_ms_p50: percentile_ms_by(results, 0.50, |result| result.decode_elapsed_ms),
        decode_elapsed_ms_p95: percentile_ms_by(results, 0.95, |result| result.decode_elapsed_ms),
        decode_elapsed_ms_p99: percentile_ms_by(results, 0.99, |result| result.decode_elapsed_ms),
        total_tokens_per_second: if elapsed_seconds > 0.0 {
            (prompt_tokens_total + generated_tokens_total) as f64 / elapsed_seconds
        } else {
            0.0
        },
        generated_tokens_per_second: if elapsed_seconds > 0.0 {
            generated_tokens_total as f64 / elapsed_seconds
        } else {
            0.0
        },
    }
}

fn percentile_ms(results: &[PromptDriverResult], percentile: f64) -> u128 {
    percentile_ms_by(results, percentile, |result| result.elapsed_ms)
}

fn percentile_ms_by(
    results: &[PromptDriverResult],
    percentile: f64,
    value: impl Fn(&PromptDriverResult) -> u128,
) -> u128 {
    if results.is_empty() {
        return 0;
    }
    let mut values = results.iter().map(value).collect::<Vec<_>>();
    values.sort_unstable();
    let rank = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[rank.min(values.len() - 1)]
}

fn mean_ms(results: &[PromptDriverResult], value: impl Fn(&PromptDriverResult) -> u128) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(value).sum::<u128>() as f64 / results.len() as f64
}

fn ensure_reply_kind(
    reply: &skippy_protocol::binary::StageReply,
    expected: WireReplyKind,
) -> Result<()> {
    if reply.kind != expected {
        bail!("expected {expected:?} reply, got {:?}", reply.kind);
    }
    Ok(())
}

fn run_remote_prompt_case(
    args: &RunArgs,
    first: &StageAssignment,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    prompt_case: &PromptCase,
    token_ids: Vec<i32>,
    prompt_index: usize,
) -> Result<PromptDriverResult> {
    if token_ids.is_empty() {
        bail!("prompt produced no tokens");
    }

    let mut stream = connect_endpoint_ready(&first.endpoint, args.startup_timeout_secs)
        .with_context(|| {
            format!(
                "connect prompt {prompt_index} to first binary stage {}",
                first.endpoint
            )
        })?;

    let wire_started = Instant::now();
    let request_id = 10_000_u64 + prompt_index as u64;
    let session_id = 20_000_u64 + prompt_index as u64;
    send_generation_config(
        &mut stream,
        wire_dtype,
        request_id,
        session_id,
        token_ids.len(),
    )
    .with_context(|| format!("send generation config for prompt {prompt_index}"))?;
    let prefill_token_count = token_ids.len().saturating_sub(1);
    let mut prefill_chunk_count = 0usize;
    let mut effective_chunk_size = None;
    let prefill_started = Instant::now();
    if prefill_token_count > 0 {
        let prefill_tokens = token_ids[..prefill_token_count].to_vec();
        let chunk_size = effective_prefill_chunk_size(args, prefill_tokens.len());
        effective_chunk_size = Some(chunk_size);
        for (chunk_index, chunk) in prefill_tokens.chunks(chunk_size).enumerate() {
            prefill_chunk_count += 1;
            let pos_start = chunk_index
                .checked_mul(chunk_size)
                .context("prefill chunk position overflow")?;
            send_prefill_chunk(
                &mut stream,
                wire_dtype,
                PrefillChunk {
                    prompt_index,
                    request_id,
                    session_id,
                    pos_start,
                    prefill_token_count,
                    tokens: chunk,
                },
            )?;
        }
    }
    let prefill_elapsed_ms = prefill_started.elapsed().as_millis();

    let max_new_tokens = effective_run_max_new_tokens(args);
    let mut predicted_tokens = Vec::with_capacity(max_new_tokens);
    let mut current = *token_ids.last().expect("checked non-empty tokens");
    let decode_started = Instant::now();
    let mut ttft_ms = 0;
    for decode_step in 0..max_new_tokens {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
        state.seq_id = i32::try_from(prompt_index).context("prompt index exceeds i32")?;
        state.prompt_token_count =
            i32::try_from(token_ids.len()).context("prompt token count exceeds i32")?;
        state.decode_step = i32::try_from(decode_step).context("decode step exceeds i32")?;
        state.current_token = current;
        state.source_stage_index = -1;
        let decode_pos = i32::try_from(prefill_token_count + decode_step)
            .context("decode position exceeds i32")?;
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: decode_pos,
            token_count: 1,
            state,
            request_id,
            session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![current],
            positions: vec![decode_pos],
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        write_stage_message(&mut stream, &message, wire_dtype).with_context(|| {
            format!("send remote decode step {decode_step} for prompt {prompt_index}")
        })?;
        let reply = recv_reply(&mut stream).with_context(|| {
            format!("receive decode step {decode_step} reply for prompt {prompt_index}")
        })?;
        ensure_reply_kind(&reply, WireReplyKind::PredictedToken)?;
        if decode_step == 0 {
            ttft_ms = wire_started.elapsed().as_millis();
        }
        current = reply.predicted;
        predicted_tokens.push(reply.predicted);
    }
    let decode_elapsed_ms = decode_started.elapsed().as_millis();

    write_stage_message(
        &mut stream,
        &StageWireMessage::stop_with_identity(wire_dtype, request_id, session_id),
        wire_dtype,
    )
    .context("send remote stop")?;
    let wire_elapsed_ms = wire_started.elapsed().as_millis();

    Ok(PromptDriverResult {
        prompt_id: prompt_case.prompt_id.clone(),
        category: prompt_case.category.clone(),
        prompt: prompt_case.prompt.clone(),
        token_ids,
        prefill_token_count,
        prefill_chunk_count,
        effective_prefill_chunk_size: effective_chunk_size,
        predicted_tokens,
        max_new_tokens,
        elapsed_ms: 0,
        wire_elapsed_ms,
        prefill_elapsed_ms,
        ttft_ms,
        decode_elapsed_ms,
    })
}

fn effective_prefill_chunk_size(args: &RunArgs, prefill_token_count: usize) -> usize {
    let Some(chunk_size) = args.prefill_chunk_size else {
        return prefill_token_count.max(1);
    };
    if args
        .prefill_chunk_threshold
        .is_some_and(|threshold| prefill_token_count <= threshold)
    {
        return prefill_token_count.max(1);
    }
    adaptive_prefill_chunk_size(args, prefill_token_count)
        .unwrap_or(chunk_size)
        .max(1)
}

fn adaptive_prefill_chunk_size(args: &RunArgs, prefill_token_count: usize) -> Option<usize> {
    let schedule = args.prefill_chunk_schedule.as_deref()?;
    let mut selected = None;
    for entry in schedule.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (min_tokens, chunk_size) = entry.split_once(':')?;
        let min_tokens = min_tokens.trim().parse::<usize>().ok()?;
        let chunk_size = chunk_size.trim().parse::<usize>().ok()?;
        if prefill_token_count >= min_tokens {
            selected = Some(match selected {
                Some((selected_min, selected_chunk)) if selected_min > min_tokens => {
                    (selected_min, selected_chunk)
                }
                _ => (min_tokens, chunk_size),
            });
        }
    }
    selected.map(|(_, chunk_size)| chunk_size)
}

fn send_generation_config(
    stream: &mut TcpStream,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
) -> Result<()> {
    let message = StageWireMessage::configure_generation(
        wire_dtype,
        request_id,
        session_id,
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?,
        None,
        None,
    );
    write_stage_message(&mut *stream, &message, wire_dtype).context("send configure-generation")?;
    let reply = recv_reply(&mut *stream).context("receive configure-generation ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected configure-generation ACK, got {:?}", reply.kind);
    }
    Ok(())
}

struct PrefillChunk<'a> {
    prompt_index: usize,
    request_id: u64,
    session_id: u64,
    pos_start: usize,
    prefill_token_count: usize,
    tokens: &'a [i32],
}

fn send_prefill_chunk(
    stream: &mut TcpStream,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    chunk: PrefillChunk<'_>,
) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id = i32::try_from(chunk.prompt_index).context("prompt index exceeds i32")?;
    state.prompt_token_count =
        i32::try_from(chunk.prefill_token_count).context("prompt token count exceeds i32")?;
    state.current_token = *chunk.tokens.last().context("prefill chunk is empty")?;
    state.source_stage_index = -1;
    let pos_start = i32::try_from(chunk.pos_start).context("prefill chunk position exceeds i32")?;
    let token_count =
        i32::try_from(chunk.tokens.len()).context("prefill token count exceeds i32")?;
    let positions: Vec<i32> = (pos_start..pos_start + token_count).collect();
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start,
        token_count,
        state,
        request_id: chunk.request_id,
        session_id: chunk.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: chunk.tokens.to_vec(),
        positions,
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, wire_dtype).with_context(|| {
        format!(
            "send remote prefill chunk for prompt {}",
            chunk.prompt_index
        )
    })?;
    let reply = recv_reply(&mut *stream).with_context(|| {
        format!(
            "receive remote prefill chunk ACK for prompt {}",
            chunk.prompt_index
        )
    })?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

impl DriverTokenizer {
    fn open(args: &RunArgs, plan: &DeploymentPlan) -> Result<Self> {
        let first = plan
            .stages
            .first()
            .context("deployment plan has no stages")?;
        let mut materialized_model_path = None;
        let (model_path, load_mode) = if let Some(model_path) = args.model_path.as_ref() {
            (model_path.clone(), RuntimeLoadMode::RuntimeSlice)
        } else if args.stage_load_mode == "layer-package" {
            let missing_model =
                "--model-path is required unless --stage-model is a local layer-package directory";
            let stage_model = args
                .stage_model
                .as_ref()
                .filter(|path| path.is_dir())
                .context(missing_model)?;
            let package = materialize_layer_package_details(&PackageStageRequest {
                model_id: args.model_id.clone(),
                topology_id: args.topology_id.clone(),
                package_ref: path_string(stage_model),
                stage_id: "driver-tokenizer".to_string(),
                layer_start: first.layer_start,
                layer_end: first.layer_end,
                include_embeddings: true,
                include_output: plan.stages.len() == 1,
            })
            .context("materialize local layer-package tokenizer model")?;
            materialized_model_path = Some(package.output_path.clone());
            (package.output_path, RuntimeLoadMode::LayerPackage)
        } else {
            bail!(
                "--model-path or a local layer-package --stage-model is required for prompt tokenization"
            );
        };

        let model = StageModel::open(
            &model_path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: first.layer_start,
                layer_end: first.layer_end,
                ctx_size: args.ctx_size,
                lane_count: 1,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: args.n_gpu_layers,
                mmap: None,
                mlock: false,
                selected_backend_device: None,
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
                load_mode,
                projector_path: None,
                include_embeddings: true,
                include_output: plan.stages.len() == 1,
                filter_tensors_on_load: args.stage_load_mode != "runtime-slice",
            },
        )
        .with_context(|| format!("open tokenizer model {}", model_path.display()))?;
        Ok(Self {
            model,
            _materialized_model_path: materialized_model_path,
        })
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        self.model
            .tokenize(prompt, true)
            .with_context(|| format!("tokenize prompt {prompt:?}"))
    }
}

fn prompt_cases(args: &RunArgs) -> Result<Vec<PromptCase>> {
    if let Some(path) = args.prompt_corpus.as_ref() {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read prompt corpus {}", path.display()))?;
        let mut cases = Vec::new();
        for (line_index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "parse prompt corpus JSONL line {} in {}",
                    line_index + 1,
                    path.display()
                )
            })?;
            cases.push(prompt_case_from_value(&value).with_context(|| {
                format!(
                    "read prompt corpus line {} in {}",
                    line_index + 1,
                    path.display()
                )
            })?);
            if args.prompt_limit.is_some_and(|limit| cases.len() >= limit) {
                break;
            }
        }
        Ok(cases)
    } else {
        Ok(vec![PromptCase {
            prompt_id: None,
            category: None,
            prompt: args.prompt.clone(),
        }])
    }
}

fn prompt_case_from_value(value: &Value) -> Result<PromptCase> {
    let prompt_id = value
        .get("id")
        .or_else(|| value.get("prompt_id"))
        .and_then(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .or_else(|| value.as_i64().map(|id| id.to_string()))
        });
    let category = value
        .get("category")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let prompt = if let Some(prompt) = value.get("prompt").and_then(Value::as_str) {
        prompt.to_string()
    } else if let Some(turns) = value.get("turns").and_then(Value::as_array) {
        turns
            .iter()
            .find_map(Value::as_str)
            .context("turns did not contain a string prompt")?
            .to_string()
    } else if let Some(messages) = value.get("messages").and_then(Value::as_array) {
        messages
            .iter()
            .filter_map(|message| {
                let role = message.get("role").and_then(Value::as_str)?;
                let content = message.get("content").and_then(Value::as_str)?;
                (role == "user").then_some(content)
            })
            .next()
            .context("messages did not contain a user prompt")?
            .to_string()
    } else {
        bail!("prompt corpus row must include prompt, turns, or messages");
    };
    Ok(PromptCase {
        prompt_id,
        category,
        prompt,
    })
}

fn parse_prompt_token_ids(value: &str) -> Result<Vec<i32>> {
    let mut tokens = Vec::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        tokens.push(
            token
                .parse::<i32>()
                .with_context(|| format!("invalid prompt token id {token}"))?,
        );
    }
    if tokens.is_empty() {
        bail!("--prompt-token-ids must contain at least one token id");
    }
    Ok(tokens)
}

pub(super) fn write_json_file(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))
        .with_context(|| format!("write {}", path.display()))
}

pub(super) fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

pub(super) fn generate_bench_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis();
    format!("run-bench-{millis}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_prompt_token_ids() {
        assert_eq!(parse_prompt_token_ids("1, 2,3").unwrap(), vec![1, 2, 3]);
        assert!(parse_prompt_token_ids("").is_err());
        assert!(parse_prompt_token_ids("1,nope").is_err());
    }

    #[test]
    fn applies_prefill_chunk_threshold() {
        let mut args = test_run_args();
        args.prefill_chunk_size = Some(128);
        assert_eq!(effective_prefill_chunk_size(&args, 64), 128);
        assert_eq!(effective_prefill_chunk_size(&args, 256), 128);

        args.prefill_chunk_threshold = Some(128);
        assert_eq!(effective_prefill_chunk_size(&args, 64), 64);
        assert_eq!(effective_prefill_chunk_size(&args, 128), 128);
        assert_eq!(effective_prefill_chunk_size(&args, 129), 128);
    }

    #[test]
    fn applies_prefill_chunk_schedule() {
        let mut args = test_run_args();
        args.prefill_chunk_size = Some(256);
        args.prefill_chunk_schedule = Some("513:512,1025:768".to_string());

        assert_eq!(effective_prefill_chunk_size(&args, 512), 256);
        assert_eq!(effective_prefill_chunk_size(&args, 513), 512);
        assert_eq!(effective_prefill_chunk_size(&args, 1024), 512);
        assert_eq!(effective_prefill_chunk_size(&args, 1025), 768);
    }

    #[test]
    fn parses_prompt_cases_from_corpus_shapes() {
        let turns = prompt_case_from_value(&json!({
            "prompt_id": 42,
            "category": "math",
            "turns": ["first turn", "second turn"]
        }))
        .unwrap();
        assert_eq!(turns.prompt_id.as_deref(), Some("42"));
        assert_eq!(turns.category.as_deref(), Some("math"));
        assert_eq!(turns.prompt, "first turn");

        let messages = prompt_case_from_value(&json!({
            "id": "mt_bench_1",
            "messages": [
                {"role": "system", "content": "ignore"},
                {"role": "user", "content": "hello"}
            ]
        }))
        .unwrap();
        assert_eq!(messages.prompt_id.as_deref(), Some("mt_bench_1"));
        assert_eq!(messages.prompt, "hello");
    }

    #[test]
    fn summarizes_prompt_driver_percentiles() {
        let results = [100_u128, 200, 300, 400]
            .into_iter()
            .map(|elapsed_ms| PromptDriverResult {
                prompt_id: None,
                category: None,
                prompt: "hello".to_string(),
                token_ids: vec![1, 2],
                prefill_token_count: 1,
                prefill_chunk_count: 1,
                effective_prefill_chunk_size: Some(1),
                predicted_tokens: vec![3],
                max_new_tokens: 1,
                elapsed_ms,
                wire_elapsed_ms: elapsed_ms - 10,
                prefill_elapsed_ms: elapsed_ms - 20,
                ttft_ms: elapsed_ms - 15,
                decode_elapsed_ms: 10,
            })
            .collect::<Vec<_>>();

        let summary = prompt_driver_summary(&results);
        assert_eq!(summary.prompt_tokens_total, 8);
        assert_eq!(summary.generated_tokens_total, 4);
        assert_eq!(summary.elapsed_ms_total, 1000);
        assert_eq!(summary.elapsed_ms_p50, 300);
        assert_eq!(summary.elapsed_ms_p95, 400);
        assert_eq!(summary.elapsed_ms_p99, 400);
        assert_eq!(summary.wire_elapsed_ms_p50, 290);
        assert_eq!(summary.prefill_elapsed_ms_p50, 280);
        assert_eq!(summary.ttft_ms_p50, 285);
        assert_eq!(summary.decode_elapsed_ms_p50, 10);
    }

    fn test_run_args() -> RunArgs {
        RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "host.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_path: Some(PathBuf::from("model.gguf")),
            stage_model: None,
            stage_load_mode: "runtime-slice".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: None,
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: None,
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: false,
            keep_remote: false,
            rsync_model_artifacts: false,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: None,
            stage_async_prefill_forward: false,
            stage_downstream_wire_delay_ms: 0.0,
            stage_downstream_wire_mbps: None,
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        }
    }
}
