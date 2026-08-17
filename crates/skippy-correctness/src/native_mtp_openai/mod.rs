mod metrics;
mod remote;
mod reporting;
mod stage_process;

use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::{
    cli::{NativeMtpOpenAiAbArgs, StageLoadMode},
    report::{NativeMtpOpenAiAbReport, NativeMtpOpenAiCaseReport},
    support::generate_run_id,
};

use metrics::read_metrics;
use reporting::{emit_report, status};
use stage_process::{
    case_addr, spawn_stage, start_stage1, wait_openai_ready, wait_stage1_ready, write_stage_config,
};

pub(super) const BATCHED_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_BATCHED_VERIFY";
pub(super) const NATIVE_MTP_ENABLED_ENV: &str = "SKIPPY_NATIVE_MTP_ENABLED";

struct OpenAiCaseConfig {
    case: &'static str,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
    run_id: String,
    root: PathBuf,
    openai_bind_addr: SocketAddr,
    stage0_bind_addr: SocketAddr,
    stage0_endpoint_addr: SocketAddr,
    stage1_bind_addr: SocketAddr,
    stage1_endpoint_addr: SocketAddr,
}

struct OpenAiStageConfig<'a> {
    run_id: &'a str,
    model_id: &'a str,
    model_path: &'a Path,
    stage_id: &'a str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    bind_addr: SocketAddr,
    upstream: Option<Value>,
    downstream: Option<Value>,
}

pub fn native_mtp_openai_ab(args: NativeMtpOpenAiAbArgs) -> Result<()> {
    if args.runtime.stage_load_mode != StageLoadMode::RuntimeSlice {
        bail!("native-mtp-open-ai-ab currently supports --stage-load-mode runtime-slice only");
    }
    if args.split_layer == 0 || args.split_layer >= args.runtime.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    if args.external_stage1 && args.stage1_ssh_host.is_some() {
        bail!("--external-stage1 cannot be combined with --stage1-ssh-host");
    }
    if args.stage1_ssh_host.is_some() && args.stage1_remote_stage_server_bin.is_none() {
        bail!("--stage1-remote-stage-server-bin is required with --stage1-ssh-host");
    }

    let model_id = args.runtime.model_id.clone().unwrap_or_else(|| {
        args.runtime
            .model
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("local-model")
            .to_string()
    });
    let root = args
        .case_root
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(generate_run_id()));
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let client = Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs.max(1)))
        .build()
        .context("failed to build HTTP client")?;

    let baseline = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "baseline",
            native_mtp_enabled: false,
            batched_verify_enabled: false,
            run_id: format!("{}-baseline", generate_run_id()),
            root: root.join("baseline"),
            openai_bind_addr: case_addr(args.openai_bind_addr, args.batched_port_offset, 0)?,
            stage0_bind_addr: case_addr(args.stage0_bind_addr, args.batched_port_offset, 0)?,
            stage0_endpoint_addr: case_addr(
                args.stage0_endpoint_addr.unwrap_or(args.stage0_bind_addr),
                args.batched_port_offset,
                0,
            )?,
            stage1_bind_addr: case_addr(args.stage1_bind_addr, args.batched_port_offset, 0)?,
            stage1_endpoint_addr: case_addr(
                args.stage1_endpoint_addr.unwrap_or(args.stage1_bind_addr),
                args.batched_port_offset,
                0,
            )?,
        },
    )?;
    let n1 = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "n1",
            native_mtp_enabled: true,
            batched_verify_enabled: false,
            run_id: format!("{}-n1", generate_run_id()),
            root: root.join("n1"),
            openai_bind_addr: case_addr(args.openai_bind_addr, args.batched_port_offset, 1)?,
            stage0_bind_addr: case_addr(args.stage0_bind_addr, args.batched_port_offset, 1)?,
            stage0_endpoint_addr: case_addr(
                args.stage0_endpoint_addr.unwrap_or(args.stage0_bind_addr),
                args.batched_port_offset,
                1,
            )?,
            stage1_bind_addr: case_addr(args.stage1_bind_addr, args.batched_port_offset, 1)?,
            stage1_endpoint_addr: case_addr(
                args.stage1_endpoint_addr.unwrap_or(args.stage1_bind_addr),
                args.batched_port_offset,
                1,
            )?,
        },
    )?;
    let batched = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "batched",
            native_mtp_enabled: true,
            batched_verify_enabled: true,
            run_id: format!("{}-batched", generate_run_id()),
            root: root.join("batched"),
            openai_bind_addr: case_addr(args.openai_bind_addr, args.batched_port_offset, 2)?,
            stage0_bind_addr: case_addr(args.stage0_bind_addr, args.batched_port_offset, 2)?,
            stage0_endpoint_addr: case_addr(
                args.stage0_endpoint_addr.unwrap_or(args.stage0_bind_addr),
                args.batched_port_offset,
                2,
            )?,
            stage1_bind_addr: case_addr(args.stage1_bind_addr, args.batched_port_offset, 2)?,
            stage1_endpoint_addr: case_addr(
                args.stage1_endpoint_addr.unwrap_or(args.stage1_bind_addr),
                args.batched_port_offset,
                2,
            )?,
        },
    )?;

    let exact_content_match = baseline.content == n1.content && baseline.content == batched.content;
    let batched_events_present = batched.metrics.batched_verify_events > 0;
    let require_batched_events = !args.allow_missing_batched_events;
    let matches = baseline.http_status == 200
        && n1.http_status == 200
        && batched.http_status == 200
        && exact_content_match
        && !baseline.metrics.native_mtp_enabled
        && n1.metrics.native_mtp_enabled
        && batched.metrics.native_mtp_enabled
        && baseline.metrics.fatal_error_events == 0
        && n1.metrics.fatal_error_events == 0
        && batched.metrics.fatal_error_events == 0
        && (!require_batched_events || batched_events_present);

    let report = NativeMtpOpenAiAbReport {
        mode: "native-mtp-open-ai-ab",
        status: status(matches),
        model_id,
        model_path: args.runtime.model.display().to_string(),
        prompt: args.runtime.prompt,
        max_tokens: args.max_tokens,
        split_layer: args.split_layer,
        layer_end: args.runtime.layer_end,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        exact_content_match,
        batched_events_required: require_batched_events,
        batched_events_present,
        matches,
        baseline,
        n1,
        batched,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    if !report.matches && !args.allow_mismatch {
        bail!("native MTP OpenAI n=1 and batched verification did not match");
    }
    Ok(())
}

fn run_openai_case(
    args: &NativeMtpOpenAiAbArgs,
    client: &Client,
    model_id: &str,
    case: OpenAiCaseConfig,
) -> Result<NativeMtpOpenAiCaseReport> {
    fs::create_dir_all(&case.root)
        .with_context(|| format!("failed to create {}", case.root.display()))?;
    let stage0_config_path = case.root.join("stage0.json");
    let stage1_config_path = case.root.join("stage1.json");
    let topology_path = case.root.join("topology.json");
    let stage0_log = case.root.join("stage0.log");
    let stage1_log = case.root.join("stage1.log");
    let stage0_model_path = args.stage0_model.as_deref().unwrap_or(&args.runtime.model);
    let stage1_model_path = args.stage1_model.as_deref().unwrap_or(&args.runtime.model);

    write_stage_config(
        &stage0_config_path,
        &stage_config_json(
            args,
            OpenAiStageConfig {
                run_id: &case.run_id,
                model_id,
                model_path: stage0_model_path,
                stage_id: "stage-0",
                stage_index: 0,
                layer_start: 0,
                layer_end: args.split_layer,
                bind_addr: case.stage0_bind_addr,
                upstream: None,
                downstream: Some(json!({
                    "stage_id": "stage-1",
                    "stage_index": 1,
                    "endpoint": format!("tcp://{}", case.stage1_endpoint_addr),
                })),
            },
        ),
    )?;
    write_stage_config(
        &stage1_config_path,
        &stage_config_json(
            args,
            OpenAiStageConfig {
                run_id: &case.run_id,
                model_id,
                model_path: stage1_model_path,
                stage_id: "stage-1",
                stage_index: 1,
                layer_start: args.split_layer,
                layer_end: args.runtime.layer_end,
                bind_addr: case.stage1_bind_addr,
                upstream: Some(json!({
                    "stage_id": "stage-0",
                    "stage_index": 0,
                    "endpoint": format!("tcp://{}", case.stage0_endpoint_addr),
                })),
                downstream: None,
            },
        ),
    )?;
    write_stage_config(
        &topology_path,
        &json!({
            "topology_id": "native-mtp-open-ai-ab",
            "model_id": model_id,
            "stages": [
                {
                    "stage_id": "stage-0",
                    "stage_index": 0,
                    "host": "localhost",
                    "endpoint": format!("tcp://{}", case.stage0_endpoint_addr),
                    "layer_start": 0,
                    "layer_end": args.split_layer,
                    "load_mode": "runtime-slice",
                },
                {
                    "stage_id": "stage-1",
                    "stage_index": 1,
                    "host": "localhost",
                    "endpoint": format!("tcp://{}", case.stage1_endpoint_addr),
                    "layer_start": args.split_layer,
                    "layer_end": args.runtime.layer_end,
                    "load_mode": "runtime-slice",
                },
            ],
        }),
    )?;

    let (stage1, stage1_launch) = start_stage1(
        args,
        &case.run_id,
        &stage1_config_path,
        &topology_path,
        &stage1_log,
        case.native_mtp_enabled,
        case.batched_verify_enabled,
    )?;
    wait_stage1_ready(
        &stage1,
        case.stage1_endpoint_addr,
        args.server.startup_timeout_secs,
    )
    .context("stage 1 binary server did not become ready")?;
    let stage0 = spawn_stage(
        args,
        &stage0_config_path,
        Some(case.openai_bind_addr),
        &topology_path,
        &stage0_log,
        case.native_mtp_enabled,
        case.batched_verify_enabled,
    )?;
    wait_openai_ready(
        client,
        case.openai_bind_addr,
        args.server.startup_timeout_secs,
    )
    .context("stage 0 OpenAI server did not become ready")?;

    let response = client
        .post(format!(
            "http://{}/v1/chat/completions",
            case.openai_bind_addr
        ))
        .json(&json!({
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": args.runtime.prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }))
        .send()
        .context("failed to send OpenAI chat completion request")?;
    let http_status = response.status().as_u16();
    let body: Value = response.json().context("failed to parse OpenAI response")?;
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let completion_tokens = body
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64);

    drop(stage0);
    stage1.stop_and_collect(&stage1_log)?;

    let metrics = read_metrics(&stage0_log, &stage1_log)?;
    Ok(NativeMtpOpenAiCaseReport {
        case: case.case,
        native_mtp_enabled: case.native_mtp_enabled,
        batched_verify_enabled: case.batched_verify_enabled,
        http_status,
        content,
        completion_tokens,
        openai_bind_addr: case.openai_bind_addr.to_string(),
        stage0_bind_addr: case.stage0_bind_addr.to_string(),
        stage0_endpoint_addr: case.stage0_endpoint_addr.to_string(),
        stage1_bind_addr: case.stage1_bind_addr.to_string(),
        stage1_endpoint_addr: case.stage1_endpoint_addr.to_string(),
        stage0_config: stage0_config_path.display().to_string(),
        stage1_config: stage1_config_path.display().to_string(),
        topology_config: topology_path.display().to_string(),
        stage0_log: stage0_log.display().to_string(),
        stage1_log: stage1_log.display().to_string(),
        stage1_launch_mode: stage1_launch.launch_mode.to_string(),
        stage1_remote_config: stage1_launch.remote_config,
        stage1_remote_topology: stage1_launch.remote_topology,
        stage1_remote_log: stage1_launch.remote_log,
        metrics,
    })
}

fn stage_config_json(args: &NativeMtpOpenAiAbArgs, stage: OpenAiStageConfig<'_>) -> Value {
    json!({
        "run_id": stage.run_id,
        "topology_id": "native-mtp-open-ai-ab",
        "model_id": stage.model_id,
        "model_path": stage.model_path,
        "stage_id": stage.stage_id,
        "stage_index": stage.stage_index,
        "layer_start": stage.layer_start,
        "layer_end": stage.layer_end,
        "ctx_size": args.runtime.ctx_size,
        "lane_count": 1,
        "n_batch": args.runtime.n_batch,
        "n_ubatch": args.runtime.n_ubatch,
        "n_gpu_layers": args.runtime.n_gpu_layers,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn_type": protocol_flash_attn(args.runtime.flash_attn),
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": stage.bind_addr,
        "upstream": stage.upstream,
        "downstream": stage.downstream,
    })
}

fn protocol_flash_attn(value: crate::cli::FlashAttentionArg) -> &'static str {
    match value {
        crate::cli::FlashAttentionArg::Auto => "auto",
        crate::cli::FlashAttentionArg::Disabled => "disabled",
        crate::cli::FlashAttentionArg::Enabled => "enabled",
    }
}
