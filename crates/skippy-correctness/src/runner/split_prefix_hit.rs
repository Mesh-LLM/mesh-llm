use std::{
    fs::{self, File},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    cli::SplitPrefixHitArgs,
    report::{SplitPrefixHitCaseReport, SplitPrefixHitReport},
    support::{ChildGuard, connect_ready, generate_run_id},
};

use super::{
    native_mtp::emit_report,
    stage_execution::{
        CorrectnessTopologyStage, correctness_topology, ensure_matches, protocol_flash_attn,
        protocol_load_mode, runtime_model_identity, status,
    },
};

const CASE_PORT_OFFSET: u16 = 10;

pub fn split_prefix_hit(args: SplitPrefixHitArgs) -> Result<()> {
    if args.split_layer == 0 || args.split_layer >= args.runtime.layer_end {
        bail!(
            "split_layer must be greater than zero and less than layer_end {}",
            args.runtime.layer_end
        );
    }
    if args.prompt_extension.trim().is_empty() {
        bail!("prompt_extension must be a non-empty token suffix");
    }
    let model_identity = runtime_model_identity(&args.runtime)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs.max(1)))
        .build()
        .context("failed to build HTTP client")?;
    let root = args
        .case_root
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(generate_run_id()));
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let extended_prompt = format!("{}{}", args.runtime.prompt, args.prompt_extension);

    let warm = run_split_prefix_hit_case(
        &args,
        &client,
        &model_identity.model_id,
        true,
        &extended_prompt,
        root.join("warm"),
    )
    .context("warm-cache case failed")?;
    let control = run_split_prefix_hit_case(
        &args,
        &client,
        &model_identity.model_id,
        false,
        &extended_prompt,
        root.join("control"),
    )
    .context("cold-cache control case failed")?;

    let content_matches = warm.content == control.content;
    let partial_hit_observed = warm.partial_prefix_hits > 0 && control.partial_prefix_hits == 0;
    let matches = warm.http_status == 200
        && control.http_status == 200
        && warm.warmup_http_status.unwrap_or(200) == 200
        && content_matches
        && partial_hit_observed
        && warm.fatal_error_events == 0
        && control.fatal_error_events == 0;

    let report = SplitPrefixHitReport {
        mode: "split-prefix-hit",
        status: status(matches),
        model_identity,
        split_layer: args.split_layer,
        layer_end: args.runtime.layer_end,
        prompt: args.runtime.prompt.clone(),
        extended_prompt,
        matches,
        partial_hit_observed,
        warm,
        control,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(report.matches, args.allow_mismatch)?;
    Ok(())
}

fn run_split_prefix_hit_case(
    args: &SplitPrefixHitArgs,
    client: &Client,
    model_id: &str,
    warmup: bool,
    extended_prompt: &str,
    root: PathBuf,
) -> Result<SplitPrefixHitCaseReport> {
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let case_index = if warmup { 0 } else { 1 };
    let openai_bind_addr = offset_port(args.openai_bind_addr, CASE_PORT_OFFSET * case_index)?;
    let stage0_bind_addr = offset_port(args.stage0_bind_addr, CASE_PORT_OFFSET * case_index)?;
    let stage1_bind_addr = offset_port(args.stage1_bind_addr, CASE_PORT_OFFSET * case_index)?;
    let stage0_config_path = root.join("stage0.json");
    let stage1_config_path = root.join("stage1.json");
    let topology_path = root.join("topology.json");
    let stage0_log = root.join("stage0.log");
    let stage1_log = root.join("stage1.log");
    let run_id = format!(
        "{}-{}",
        generate_run_id(),
        if warmup { "warm" } else { "control" }
    );

    let common_stage_fields = json!({
        "run_id": run_id,
        "topology_id": "correctness-split-prefix-hit",
        "model_id": model_id,
        "model_path": args.runtime.model,
        "source_model_sha256": source_model_sha256(&args.runtime.model)?,
        "ctx_size": args.runtime.ctx_size,
        "lane_count": 1,
        "n_batch": args.runtime.n_batch,
        "n_ubatch": args.runtime.n_ubatch,
        "n_gpu_layers": args.runtime.n_gpu_layers,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn_type": protocol_flash_attn(args.runtime.flash_attn),
        "filter_tensors_on_load": true,
        "load_mode": protocol_load_mode(args.runtime.stage_load_mode),
    });
    let stage0_config = merge_stage_config(
        &common_stage_fields,
        json!({
            "stage_id": "stage-0",
            "stage_index": 0,
            "layer_start": 0,
            "layer_end": args.split_layer,
            "bind_addr": stage0_bind_addr,
            "upstream": null,
            "downstream": {
                "stage_id": "stage-1",
                "stage_index": 1,
                "endpoint": format!("tcp://{}", stage1_bind_addr),
            },
        }),
    );
    let stage1_config = merge_stage_config(
        &common_stage_fields,
        json!({
            "stage_id": "stage-1",
            "stage_index": 1,
            "layer_start": args.split_layer,
            "layer_end": args.runtime.layer_end,
            "bind_addr": stage1_bind_addr,
            "upstream": {
                "stage_id": "stage-0",
                "stage_index": 0,
                "endpoint": format!("tcp://{}", stage0_bind_addr),
            },
            "downstream": null,
        }),
    );
    let topology = correctness_topology(
        "correctness-split-prefix-hit",
        model_id,
        &[
            CorrectnessTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: format!("tcp://{}", stage0_bind_addr),
                layer_start: 0,
                layer_end: args.split_layer,
                load_mode: protocol_load_mode(args.runtime.stage_load_mode),
            },
            CorrectnessTopologyStage {
                stage_id: "stage-1",
                stage_index: 1,
                endpoint: format!("tcp://{}", stage1_bind_addr),
                layer_start: args.split_layer,
                layer_end: args.runtime.layer_end,
                load_mode: protocol_load_mode(args.runtime.stage_load_mode),
            },
        ],
    );
    write_config(&stage0_config_path, &stage0_config)?;
    write_config(&stage1_config_path, &stage1_config)?;
    write_config(&topology_path, &topology)?;

    let _stage1 = spawn_prefix_hit_stage(
        args,
        &stage1_config_path,
        &topology_path,
        &stage1_log,
        None,
        true,
    )?;
    drop(
        connect_ready(stage1_bind_addr, args.server.startup_timeout_secs)
            .context("stage 1 binary server did not become ready")?,
    );
    let _stage0 = spawn_prefix_hit_stage(
        args,
        &stage0_config_path,
        &topology_path,
        &stage0_log,
        Some(openai_bind_addr),
        false,
    )?;
    wait_openai_ready(client, openai_bind_addr, args.server.startup_timeout_secs)
        .context("stage 0 OpenAI server did not become ready")?;

    let warmup_http_status = if warmup {
        Some(
            send_chat_completion(
                client,
                openai_bind_addr,
                model_id,
                &args.runtime.prompt,
                args.max_tokens,
            )
            .context("warmup chat completion failed")?
            .0,
        )
    } else {
        None
    };
    let (http_status, content, completion_tokens) = send_chat_completion(
        client,
        openai_bind_addr,
        model_id,
        extended_prompt,
        args.max_tokens,
    )
    .context("extended-prompt chat completion failed")?;

    drop(_stage0);
    drop(_stage1);

    let metrics = read_prefix_hit_metrics(&stage0_log, &stage1_log)?;
    Ok(SplitPrefixHitCaseReport {
        case: if warmup { "warm-cache" } else { "control" },
        warmup_http_status,
        http_status,
        content,
        completion_tokens,
        partial_prefix_hits: metrics.partial_prefix_hits,
        full_prefix_hits: metrics.full_prefix_hits,
        fatal_error_events: metrics.fatal_error_events,
        openai_bind_addr: openai_bind_addr.to_string(),
        stage1_bind_addr: stage1_bind_addr.to_string(),
        stage0_log: stage0_log.display().to_string(),
        stage1_log: stage1_log.display().to_string(),
    })
}

fn source_model_sha256(model: &Path) -> Result<String> {
    let mut file =
        fs::File::open(model).with_context(|| format!("failed to open {}", model.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("failed to hash {}", model.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn merge_stage_config(base: &Value, stage: Value) -> Value {
    let mut merged = base.clone();
    if let (Some(merged_object), Some(stage_object)) = (merged.as_object_mut(), stage.as_object()) {
        for (key, value) in stage_object {
            merged_object.insert(key.clone(), value.clone());
        }
    }
    merged
}

fn write_config(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn spawn_prefix_hit_stage(
    args: &SplitPrefixHitArgs,
    config_path: &Path,
    topology_path: &Path,
    log_path: &Path,
    openai_bind_addr: Option<SocketAddr>,
    cache_enabled: bool,
) -> Result<ChildGuard> {
    let log = File::create(log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let mut command = Command::new(&args.server.stage_server_bin);
    command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("topology path is not valid UTF-8")?,
        "--activation-width",
        &args.activation_width.to_string(),
        "--telemetry-level",
        "debug",
    ]);
    if let Some(openai_bind_addr) = openai_bind_addr {
        command.args(["--openai-bind-addr", &openai_bind_addr.to_string()]);
        command.args(["--openai-prefill-chunk-policy", "fixed"]);
        command.args(["--openai-prefill-chunk-size", "4096"]);
    }
    command.env("SKIPPY_TELEMETRY_STDERR", "1");
    if cache_enabled {
        command.env("SKIPPY_KV_CACHE", "on");
        command.env("SKIPPY_KV_CACHE_MIN_TOKENS", "8");
    } else {
        command.env("SKIPPY_KV_CACHE", "off");
    }
    command.env("SKIPPY_NATIVE_MTP_ENABLED", "0");
    command.stdout(Stdio::from(log.try_clone()?));
    command.stderr(Stdio::from(log));
    ChildGuard::spawn(command)
}

fn send_chat_completion(
    client: &Client,
    openai_bind_addr: SocketAddr,
    model_id: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<(u16, String, Option<u64>)> {
    let response = client
        .post(format!("http://{}/v1/chat/completions", openai_bind_addr))
        .json(&json!({
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
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
    Ok((http_status, content, completion_tokens))
}

fn wait_openai_ready(client: &Client, addr: SocketAddr, timeout_secs: u64) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(4).max(1);
    let url = format!("http://{addr}/v1/models");
    let mut last_error = None;
    for _ in 0..attempts {
        match client.get(&url).send() {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => last_error = Some(format!("HTTP {}", response.status())),
            Err(error) => last_error = Some(error.to_string()),
        }
        thread::sleep(Duration::from_millis(250));
    }
    bail!(
        "timed out waiting for {url}: {}",
        last_error.unwrap_or_else(|| "no attempts made".to_string())
    )
}

struct PrefixHitMetrics {
    partial_prefix_hits: u64,
    full_prefix_hits: u64,
    fatal_error_events: u64,
}

fn read_prefix_hit_metrics(stage0_log: &Path, stage1_log: &Path) -> Result<PrefixHitMetrics> {
    let mut metrics = PrefixHitMetrics {
        partial_prefix_hits: 0,
        full_prefix_hits: 0,
        fatal_error_events: 0,
    };
    for (path, is_final_stage) in [(stage0_log, false), (stage1_log, true)] {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        metrics.fatal_error_events += count_fatal_log_lines(&text);
        if !is_final_stage {
            continue;
        }
        for line in text.lines() {
            let Ok(event) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if event.get("event").and_then(Value::as_str) != Some("stage.binary_kv_lookup_decision")
            {
                continue;
            }
            let attrs = event.get("attributes").cloned().unwrap_or(Value::Null);
            let decision = attrs
                .get("skippy.kv.decision")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let suffix_tokens = attrs
                .get("skippy.kv.suffix_prefill_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if decision == "resident_hit" && suffix_tokens > 0 {
                metrics.partial_prefix_hits += 1;
            } else if decision == "resident_hit" || decision.ends_with("_hit") {
                metrics.full_prefix_hits += 1;
            }
        }
    }
    Ok(metrics)
}

fn count_fatal_log_lines(text: &str) -> u64 {
    text.lines()
        .filter(|line| {
            line.contains("panicked")
                || line.contains("service_unavailable")
                || line.contains("INVALID_ARGUMENT")
        })
        .count() as u64
}

fn offset_port(mut addr: SocketAddr, offset: u16) -> Result<SocketAddr> {
    let port = addr
        .port()
        .checked_add(offset)
        .context("case port offset exceeds u16")?;
    addr.set_port(port);
    Ok(addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_port_shifts_case_ports() {
        let addr: SocketAddr = "127.0.0.1:19290".parse().unwrap();
        assert_eq!(
            offset_port(addr, CASE_PORT_OFFSET).unwrap().to_string(),
            "127.0.0.1:19300"
        );
    }

    #[test]
    fn offset_port_rejects_overflow() {
        let addr: SocketAddr = "127.0.0.1:65530".parse().unwrap();
        assert!(offset_port(addr, CASE_PORT_OFFSET).is_err());
    }

    #[test]
    fn partial_hits_require_final_stage_resident_hit_with_suffix() {
        let text = concat!(
            "{\"event\":\"stage.binary_kv_lookup_decision\",\"attributes\":",
            "{\"skippy.kv.decision\":\"resident_hit\",",
            "\"skippy.kv.suffix_prefill_tokens\":4}}\n",
            "{\"event\":\"stage.binary_kv_lookup_decision\",\"attributes\":",
            "{\"skippy.kv.decision\":\"resident_hit\",",
            "\"skippy.kv.suffix_prefill_tokens\":0}}\n",
            "{\"event\":\"stage.binary_kv_lookup_decision\",\"attributes\":",
            "{\"skippy.kv.decision\":\"try_restore_miss\"}}\n"
        );
        let dir = std::env::temp_dir().join("split-prefix-hit-metrics-test");
        fs::create_dir_all(&dir).unwrap();
        let stage0 = dir.join("stage0.log");
        let stage1 = dir.join("stage1.log");
        fs::write(&stage0, "ready\n").unwrap();
        fs::write(&stage1, text).unwrap();
        let metrics = read_prefix_hit_metrics(&stage0, &stage1).unwrap();
        assert_eq!(metrics.partial_prefix_hits, 1);
        assert_eq!(metrics.full_prefix_hits, 1);
        assert_eq!(metrics.fatal_error_events, 0);
    }
}
