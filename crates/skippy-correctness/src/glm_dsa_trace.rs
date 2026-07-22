use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, ErrorKind, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use sha2::{Digest, Sha256};
use skippy_protocol::binary::{
    StageReplyStats, WireMessageKind, read_stage_message, recv_ready, send_ready,
    send_reply_ack_with_stats, send_reply_predicted_with_stats,
};

use crate::{
    cli::{FlashAttentionArg, GlmDsaStage0TraceArgs, StageLoadMode},
    report::{
        GlmDsaActivationErrorReport, GlmDsaActivationStatsReport, GlmDsaDownstreamComparisonReport,
        GlmDsaDownstreamMessageReport, GlmDsaDownstreamParityReport, GlmDsaSemanticParityReport,
        GlmDsaStage0TraceReport, GlmDsaTimingChunkReport, GlmDsaTimingGroupChunkReport,
        GlmDsaTimingReport, GlmDsaTopKComparisonReport, GlmDsaTraceKeyReport,
        GlmDsaTraceParityMismatchReport, GlmDsaTraceParityReport, GlmDsaTraceVariantReport,
    },
    support::{ChildGuard, parse_wire_dtype},
};

#[derive(Debug, Clone)]
struct FakeDownstreamMessage {
    kind: WireMessageKind,
    pos_start: i32,
    token_count: i32,
    activation_bytes: usize,
    activation_sha256: String,
    activation_f32: Option<GlmDsaActivationStatsReport>,
    activation_f32_payload: Option<Vec<u8>>,
    top_k_count: usize,
    top_k_sha256: String,
    top_k_values: Vec<i32>,
}

struct FakeDownstreamSummary {
    message_count: usize,
    prefill_message_count: usize,
    decode_message_count: usize,
    prefill_token_count: usize,
    top_k_message_count: usize,
    max_top_k_count: usize,
    total_top_k_count: usize,
    total_causal_visible_top_k_count: usize,
    total_active_top_k_window_count: usize,
    total_finite_top_k_count: usize,
    total_padded_top_k_count: usize,
    avg_top_k_per_token: Option<f64>,
    avg_causal_visible_top_k_per_token: Option<f64>,
    avg_active_top_k_window_per_token: Option<f64>,
    avg_finite_top_k_per_token: Option<f64>,
    max_top_k_per_token: Option<f64>,
    top_k_padding_ratio: Option<f64>,
    top_k_sideband_to_hidden_ratio: Option<f64>,
    messages: Vec<GlmDsaDownstreamMessageReport>,
}

struct FakeDownstreamGuard {
    stop: Arc<AtomicBool>,
    messages: Arc<Mutex<Vec<FakeDownstreamMessage>>>,
    handle: Option<JoinHandle<Result<()>>>,
}

struct StopAwareReader<'a> {
    stream: &'a mut TcpStream,
    stop: &'a AtomicBool,
}

impl Read for StopAwareReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return Err(io::Error::new(
                    ErrorKind::ConnectionAborted,
                    "fake downstream stopped",
                ));
            }
            match self.stream.read(buffer) {
                Err(error)
                    if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
}

struct TraceVariantRun {
    report: GlmDsaTraceVariantReport,
    fake_messages: Vec<FakeDownstreamMessage>,
}

#[derive(Clone, Copy)]
struct TraceVariant {
    name: &'static str,
    direct_sparse_attn: bool,
    fused_sparse_mask: bool,
}

pub fn glm_dsa_stage0_trace(args: GlmDsaStage0TraceArgs) -> Result<()> {
    ensure_supported_args(&args)?;
    let run_id = generate_glm_dsa_run_id();
    let case_root = args
        .case_root
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(&run_id));
    fs::create_dir_all(&case_root)
        .with_context(|| format!("create case root {}", case_root.display()))?;

    let fused = run_variant(
        &args,
        &run_id,
        &case_root,
        TraceVariant {
            name: "fused",
            direct_sparse_attn: false,
            fused_sparse_mask: true,
        },
    )?;
    let direct = run_variant(
        &args,
        &run_id,
        &case_root,
        TraceVariant {
            name: "direct",
            direct_sparse_attn: true,
            fused_sparse_mask: false,
        },
    )?;

    let both_variants_completed = fused.report.prompt_success && direct.report.prompt_success;
    let trace_parity = compare_variant_traces(
        &fused.report.stage_log,
        &direct.report.stage_log,
        !args.allow_trace_mismatch,
    )?;
    let downstream_parity =
        compare_downstream_messages(&fused.fake_messages, &direct.fake_messages);
    let semantic_parity = compare_semantic_parity(
        &downstream_parity,
        args.activation_atol,
        args.activation_relative_rmse_tolerance,
    );
    let parity_matched = parity_gate_matched(trace_parity.matched, semantic_parity.matched);
    let status = if both_variants_completed && parity_matched {
        "pass"
    } else {
        "fail"
    };
    let fused_prefill_speedup_vs_direct = speedup(
        fused.report.prompt_prefill_tok_s,
        direct.report.prompt_prefill_tok_s,
    );
    let fused_glm_dsa_op_speedup_vs_direct = timing_speedup(
        fused.report.avg_128_token_timing.as_ref(),
        direct.report.avg_128_token_timing.as_ref(),
    );
    let report = GlmDsaStage0TraceReport {
        mode: "glm-dsa-stage0-trace",
        status,
        run_id,
        model_id: args
            .runtime
            .model_id
            .clone()
            .unwrap_or_else(|| "local/glm-dsa-stage0-trace".to_string()),
        model_path: stage_model_path(&args)?.to_string_lossy().into_owned(),
        case_root: case_root.to_string_lossy().into_owned(),
        stage_layer_end: args.stage_layer_end,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        prefill_chunk_size: args.prefill_chunk_size,
        max_new_tokens: args.max_new_tokens,
        trace_filter: args.trace_filter,
        both_variants_completed,
        fused_prefill_speedup_vs_direct,
        fused_glm_dsa_op_speedup_vs_direct,
        trace_parity,
        downstream_parity,
        semantic_parity,
        variants: vec![fused.report, direct.report],
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    if !report.both_variants_completed {
        bail!("GLM-DSA stage0 trace did not complete both variants");
    }
    if !parity_matched {
        bail!("GLM-DSA fused/direct parity failed");
    }
    Ok(())
}

fn parity_gate_matched(trace_matched: bool, semantic_matched: bool) -> bool {
    trace_matched && semantic_matched
}

fn ensure_supported_args(args: &GlmDsaStage0TraceArgs) -> Result<()> {
    if args.runtime.stage_load_mode != StageLoadMode::LayerPackage {
        bail!("glm-dsa-stage0-trace currently requires --stage-load-mode layer-package");
    }
    if args.stage_layer_end == 0 {
        bail!("--stage-layer-end must be greater than zero");
    }
    parse_wire_dtype(&args.activation_wire_dtype)?;
    Ok(())
}

fn run_variant(
    args: &GlmDsaStage0TraceArgs,
    run_id: &str,
    case_root: &Path,
    variant: TraceVariant,
) -> Result<TraceVariantRun> {
    let variant_root = case_root.join(variant.name);
    fs::create_dir_all(&variant_root)
        .with_context(|| format!("create variant root {}", variant_root.display()))?;
    let stage_config_path = variant_root.join("stage0.json");
    let stage_log_path = variant_root.join("stage0.log");
    let prompt_log_path = variant_root.join("prompt.log");
    write_stage_config(args, run_id, variant.name, &stage_config_path)?;

    let fake = FakeDownstreamGuard::start(args.fake_downstream_bind_addr, args.activation_width)
        .context("start fake downstream")?;
    let mut stage =
        start_stage0(args, &stage_config_path, &stage_log_path, variant).context("start stage0")?;
    drop(wait_for_stage_ready_or_exit(
        &mut stage,
        args.stage0_bind_addr,
        args.server.startup_timeout_secs,
        &stage_log_path,
    )?);

    let prompt_output = run_prompt(args, &prompt_log_path).context("run skippy-prompt")?;
    let fake_messages = fake.finish()?;
    let stage_log = fs::read_to_string(&stage_log_path).unwrap_or_default();
    let prompt_log = fs::read_to_string(&prompt_log_path).unwrap_or_default();
    let timing_chunks = parse_timing_chunks(&stage_log);
    let timing_group_chunks = parse_timing_group_chunks(&stage_log);
    let avg_128_token_timing = avg_128_token_timing(&timing_chunks);
    let max_128_token_timing = max_128_token_timing(&timing_chunks);
    let last_128_token_timing = last_128_token_timing(&timing_chunks);
    let (prompt_prefill_tok_s, prompt_decode_tok_s) = parse_prompt_speeds(&prompt_log);
    let trace_line_count = stage_log.matches("glm_dsa_tensor_trace").count();
    let timing_line_count = stage_log.matches("glm_dsa_op_timing").count();
    let fake_downstream = summarize_fake_downstream(&fake_messages);

    let report = GlmDsaTraceVariantReport {
        variant: variant.name,
        direct_sparse_attn: variant.direct_sparse_attn,
        fused_sparse_mask: variant.fused_sparse_mask,
        prompt_exit_code: prompt_output.status.code(),
        prompt_success: prompt_output.status.success(),
        stage_log: stage_log_path.to_string_lossy().into_owned(),
        prompt_log: prompt_log_path.to_string_lossy().into_owned(),
        fake_downstream_message_count: fake_downstream.message_count,
        fake_downstream_prefill_message_count: fake_downstream.prefill_message_count,
        fake_downstream_decode_message_count: fake_downstream.decode_message_count,
        fake_downstream_prefill_token_count: fake_downstream.prefill_token_count,
        fake_downstream_top_k_message_count: fake_downstream.top_k_message_count,
        fake_downstream_max_top_k_count: fake_downstream.max_top_k_count,
        fake_downstream_total_top_k_count: fake_downstream.total_top_k_count,
        fake_downstream_total_causal_visible_top_k_count: fake_downstream
            .total_causal_visible_top_k_count,
        fake_downstream_total_active_top_k_window_count: fake_downstream
            .total_active_top_k_window_count,
        fake_downstream_total_finite_top_k_count: fake_downstream.total_finite_top_k_count,
        fake_downstream_total_padded_top_k_count: fake_downstream.total_padded_top_k_count,
        fake_downstream_avg_top_k_per_token: fake_downstream.avg_top_k_per_token,
        fake_downstream_avg_causal_visible_top_k_per_token: fake_downstream
            .avg_causal_visible_top_k_per_token,
        fake_downstream_avg_active_top_k_window_per_token: fake_downstream
            .avg_active_top_k_window_per_token,
        fake_downstream_avg_finite_top_k_per_token: fake_downstream.avg_finite_top_k_per_token,
        fake_downstream_max_top_k_per_token: fake_downstream.max_top_k_per_token,
        fake_downstream_top_k_padding_ratio: fake_downstream.top_k_padding_ratio,
        fake_downstream_top_k_sideband_to_hidden_ratio: fake_downstream
            .top_k_sideband_to_hidden_ratio,
        fake_downstream_messages: fake_downstream.messages,
        trace_line_count,
        timing_line_count,
        prompt_prefill_tok_s,
        prompt_decode_tok_s,
        avg_128_token_timing,
        max_128_token_timing,
        last_128_token_timing,
        timing_chunks,
        timing_group_chunks,
    };
    Ok(TraceVariantRun {
        report,
        fake_messages,
    })
}

fn summarize_fake_downstream(messages: &[FakeDownstreamMessage]) -> FakeDownstreamSummary {
    let top_k_message_count = messages
        .iter()
        .filter(|message| message.top_k_count > 0)
        .count();
    let prefill_message_count = messages
        .iter()
        .filter(|message| message.kind.is_prefill())
        .count();
    let decode_message_count = messages
        .iter()
        .filter(|message| message.kind == WireMessageKind::DecodeEmbd)
        .count();
    let prefill_token_count = messages
        .iter()
        .filter(|message| message.kind.is_prefill())
        .map(|message| usize::try_from(message.token_count.max(0)).unwrap_or(0))
        .sum();
    let max_top_k_count = messages
        .iter()
        .map(|message| message.top_k_count)
        .max()
        .unwrap_or(0);
    let total_top_k_count: usize = messages.iter().map(|message| message.top_k_count).sum();
    let total_causal_visible_top_k_count = messages
        .iter()
        .map(causal_visible_top_k_count)
        .sum::<usize>();
    let total_active_top_k_window_count = messages
        .iter()
        .map(active_top_k_window_count)
        .sum::<usize>();
    let total_finite_top_k_count = messages.iter().map(finite_top_k_count).sum::<usize>();
    let total_padded_top_k_count =
        total_top_k_count.saturating_sub(total_causal_visible_top_k_count);
    let top_k_token_count = messages
        .iter()
        .filter(|message| message.top_k_count > 0)
        .map(message_token_count)
        .sum::<usize>();
    let top_k_activation_bytes = messages
        .iter()
        .filter(|message| message.top_k_count > 0)
        .map(|message| message.activation_bytes)
        .sum::<usize>();
    let avg_top_k_per_token = nonzero_div_usize(total_top_k_count, top_k_token_count);
    let avg_causal_visible_top_k_per_token =
        nonzero_div_usize(total_causal_visible_top_k_count, top_k_token_count);
    let avg_active_top_k_window_per_token =
        nonzero_div_usize(total_active_top_k_window_count, top_k_token_count);
    let avg_finite_top_k_per_token = nonzero_div_usize(total_finite_top_k_count, top_k_token_count);
    let max_top_k_per_token = messages
        .iter()
        .filter(|message| message.top_k_count > 0)
        .filter_map(|message| nonzero_div_usize(message.top_k_count, message_token_count(message)))
        .reduce(f64::max);
    let top_k_padding_ratio = nonzero_div_usize(total_padded_top_k_count, total_top_k_count);
    let top_k_sideband_to_hidden_ratio = nonzero_div_usize(
        total_top_k_count * std::mem::size_of::<i32>(),
        top_k_activation_bytes,
    );
    let message_reports = messages
        .iter()
        .map(fake_downstream_message_report)
        .collect::<Vec<_>>();

    FakeDownstreamSummary {
        message_count: messages.len(),
        prefill_message_count,
        decode_message_count,
        prefill_token_count,
        top_k_message_count,
        max_top_k_count,
        total_top_k_count,
        total_causal_visible_top_k_count,
        total_active_top_k_window_count,
        total_finite_top_k_count,
        total_padded_top_k_count,
        avg_top_k_per_token,
        avg_causal_visible_top_k_per_token,
        avg_active_top_k_window_per_token,
        avg_finite_top_k_per_token,
        max_top_k_per_token,
        top_k_padding_ratio,
        top_k_sideband_to_hidden_ratio,
        messages: message_reports,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorTraceRecord {
    key: TensorTraceKey,
    tensor_type: String,
    shape: [i64; 4],
    stats: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TensorTraceKey {
    tokens: u64,
    name: String,
    occurrence: usize,
}

fn compare_variant_traces(
    fused_log_path: &str,
    direct_log_path: &str,
    required: bool,
) -> Result<GlmDsaTraceParityReport> {
    let fused_log =
        fs::read_to_string(fused_log_path).with_context(|| format!("read {fused_log_path}"))?;
    let direct_log =
        fs::read_to_string(direct_log_path).with_context(|| format!("read {direct_log_path}"))?;
    let fused = parse_tensor_trace_records(&fused_log);
    let direct = parse_tensor_trace_records(&direct_log);
    Ok(compare_tensor_traces(required, fused, direct))
}

fn parse_tensor_trace_records(log: &str) -> BTreeMap<TensorTraceKey, TensorTraceRecord> {
    let mut seen = BTreeMap::<(u64, String), usize>::new();
    let mut records = BTreeMap::new();
    for line in log
        .lines()
        .filter(|line| line.contains("glm_dsa_tensor_trace"))
    {
        let Some(record) = parse_tensor_trace_record(line, &mut seen) else {
            continue;
        };
        records.insert(record.key.clone(), record);
    }
    records
}

fn parse_tensor_trace_record(
    line: &str,
    seen: &mut BTreeMap<(u64, String), usize>,
) -> Option<TensorTraceRecord> {
    let fields = parse_trace_fields(line);
    let tokens = fields.get("tokens")?.parse::<u64>().ok()?;
    let name = fields.get("name")?.to_string();
    let occurrence_key = (tokens, name.clone());
    let occurrence = *seen
        .entry(occurrence_key)
        .and_modify(|count| *count += 1)
        .or_insert(0);
    Some(TensorTraceRecord {
        key: TensorTraceKey {
            tokens,
            name,
            occurrence,
        },
        tensor_type: fields.get("type")?.to_string(),
        shape: parse_trace_shape(fields.get("ne")?)?,
        stats: fields.get("stats").map(|value| value.to_string()),
    })
}

fn parse_trace_fields(line: &str) -> BTreeMap<&str, &str> {
    line.split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect()
}

fn parse_trace_shape(value: &str) -> Option<[i64; 4]> {
    let values = value
        .strip_prefix('[')?
        .strip_suffix(']')?
        .split(',')
        .map(str::parse::<i64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    <[i64; 4]>::try_from(values).ok()
}

fn compare_tensor_traces(
    required: bool,
    fused: BTreeMap<TensorTraceKey, TensorTraceRecord>,
    direct: BTreeMap<TensorTraceKey, TensorTraceRecord>,
) -> GlmDsaTraceParityReport {
    let fused_keys = fused.keys().cloned().collect::<BTreeSet<_>>();
    let direct_keys = direct.keys().cloned().collect::<BTreeSet<_>>();
    let shared_keys = fused_keys
        .intersection(&direct_keys)
        .cloned()
        .collect::<Vec<_>>();
    let missing_in_fused = direct_keys
        .difference(&fused_keys)
        .map(trace_key_report)
        .collect::<Vec<_>>();
    let missing_in_direct = fused_keys
        .difference(&direct_keys)
        .map(trace_key_report)
        .collect::<Vec<_>>();
    let mismatches = shared_keys
        .iter()
        .filter_map(|key| compare_tensor_trace_record(key, &fused[key], &direct[key]))
        .collect::<Vec<_>>();
    let compared_trace_count = shared_keys.len();
    let matched = !required || (compared_trace_count > 0 && mismatches.is_empty());
    GlmDsaTraceParityReport {
        required,
        matched,
        fused_trace_count: fused.len(),
        direct_trace_count: direct.len(),
        compared_trace_count,
        mismatched_trace_count: mismatches.len(),
        missing_in_fused_count: missing_in_fused.len(),
        missing_in_direct_count: missing_in_direct.len(),
        mismatches,
        missing_in_fused,
        missing_in_direct,
    }
}

fn compare_tensor_trace_record(
    key: &TensorTraceKey,
    fused: &TensorTraceRecord,
    direct: &TensorTraceRecord,
) -> Option<GlmDsaTraceParityMismatchReport> {
    let reason = if fused.tensor_type != direct.tensor_type {
        Some("type mismatch")
    } else if fused.shape != direct.shape {
        Some("shape mismatch")
    } else if fused.stats.is_none() || direct.stats.is_none() {
        Some("missing stats")
    } else if fused.stats != direct.stats {
        Some("stats digest mismatch")
    } else {
        None
    }?;
    Some(GlmDsaTraceParityMismatchReport {
        key: trace_key_report(key),
        reason: reason.to_string(),
        fused_stats: fused.stats.clone(),
        direct_stats: direct.stats.clone(),
        fused_type: fused.tensor_type.clone(),
        direct_type: direct.tensor_type.clone(),
        fused_shape: fused.shape,
        direct_shape: direct.shape,
    })
}

fn trace_key_report(key: &TensorTraceKey) -> GlmDsaTraceKeyReport {
    GlmDsaTraceKeyReport {
        tokens: key.tokens,
        name: key.name.clone(),
        occurrence: key.occurrence,
    }
}

fn message_token_count(message: &FakeDownstreamMessage) -> usize {
    usize::try_from(message.token_count.max(0)).unwrap_or(0)
}

fn causal_visible_top_k_count(message: &FakeDownstreamMessage) -> usize {
    let token_count = message_token_count(message);
    if token_count == 0 || message.top_k_count == 0 {
        return 0;
    }
    let sideband_width = message.top_k_count / token_count;
    (0..token_count)
        .map(|token_index| sideband_width.min(causal_visible_width(message.pos_start, token_index)))
        .sum()
}

fn active_top_k_window_count(message: &FakeDownstreamMessage) -> usize {
    let token_count = message_token_count(message);
    if token_count == 0 || message.top_k_values.is_empty() {
        return 0;
    }
    let sideband_width = message.top_k_values.len() / token_count;
    (0..token_count)
        .map(|token_index| {
            let visible_width = i32::try_from(causal_visible_width(message.pos_start, token_index))
                .unwrap_or(i32::MAX);
            let row_start = token_index * sideband_width;
            message.top_k_values[row_start..row_start + sideband_width]
                .iter()
                .rposition(|i_kv| *i_kv >= 0 && *i_kv < visible_width)
                .map_or(0, |i_top| i_top + 1)
        })
        .sum()
}

fn finite_top_k_count(message: &FakeDownstreamMessage) -> usize {
    let token_count = message_token_count(message);
    if token_count == 0 || message.top_k_values.is_empty() {
        return 0;
    }
    let sideband_width = message.top_k_values.len() / token_count;
    (0..token_count)
        .map(|token_index| {
            let visible_width = i32::try_from(causal_visible_width(message.pos_start, token_index))
                .unwrap_or(i32::MAX);
            let row_start = token_index * sideband_width;
            message.top_k_values[row_start..row_start + sideband_width]
                .iter()
                .filter(|i_kv| **i_kv >= 0 && **i_kv < visible_width)
                .count()
        })
        .sum()
}

fn causal_visible_width(pos_start: i32, token_index: usize) -> usize {
    let token_offset = i32::try_from(token_index).unwrap_or(i32::MAX);
    usize::try_from(
        pos_start
            .saturating_add(token_offset)
            .saturating_add(1)
            .max(0),
    )
    .unwrap_or(0)
}

fn decode_i32_values(raw_bytes: &[u8]) -> Vec<i32> {
    raw_bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("exact i32 chunk")))
        .collect()
}

fn write_stage_config(
    args: &GlmDsaStage0TraceArgs,
    run_id: &str,
    variant: &str,
    path: &Path,
) -> Result<()> {
    let model_path = stage_model_path(args)?;
    let model_id = args
        .runtime
        .model_id
        .clone()
        .unwrap_or_else(|| "local/glm-dsa-stage0-trace".to_string());
    let config = json!({
        "run_id": run_id,
        "topology_id": format!("glm-dsa-stage0-trace-{variant}"),
        "model_id": model_id,
        "model_path": model_path,
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": args.stage_layer_end,
        "ctx_size": args.runtime.ctx_size,
        "n_batch": args.runtime.n_batch,
        "n_ubatch": args.runtime.n_ubatch,
        "n_gpu_layers": args.runtime.n_gpu_layers,
        "flash_attn_type": protocol_flash_attn(args.runtime.flash_attn),
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "filter_tensors_on_load": true,
        "use_mmap": true,
        "load_mode": "layer-package",
        "bind_addr": args.stage0_bind_addr,
        "upstream": null,
        "downstream": {
            "stage_id": "fake-stage-1",
            "stage_index": 1,
            "endpoint": format!("tcp://{}", args.fake_downstream_bind_addr),
        },
    });
    fs::write(path, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("write stage config {}", path.display()))
}

fn start_stage0(
    args: &GlmDsaStage0TraceArgs,
    config_path: &Path,
    log_path: &Path,
    variant: TraceVariant,
) -> Result<ChildGuard> {
    let log = File::create(log_path).with_context(|| format!("create {}", log_path.display()))?;
    let err_log = log
        .try_clone()
        .with_context(|| format!("clone {}", log_path.display()))?;
    let mut command = Command::new(&args.server.stage_server_bin);
    command
        .args([
            "serve-binary",
            "--config",
            path_str(config_path)?,
            "--activation-width",
            &args.activation_width.to_string(),
            "--activation-wire-dtype",
            &args.activation_wire_dtype,
            "--max-inflight",
            &args.server.max_inflight.to_string(),
        ])
        .env("SKIPPY_GLM_DSA_OP_TIMING", "1")
        .env("SKIPPY_GLM_DSA_TENSOR_TRACE", "1")
        .env("SKIPPY_GLM_DSA_TENSOR_TRACE_STATS", "1")
        .env(
            "SKIPPY_GLM_DSA_TENSOR_TRACE_VALUES",
            args.trace_values.to_string(),
        )
        .env(
            "SKIPPY_GLM_DSA_TENSOR_TRACE_NODES",
            args.trace_nodes.to_string(),
        )
        .env("SKIPPY_GLM_DSA_TENSOR_TRACE_FILTER", &args.trace_filter)
        .env(
            "SKIPPY_GLM_DSA_TENSOR_TRACE_STATS_MAX_BYTES",
            args.trace_stats_max_bytes.to_string(),
        )
        .env(
            "SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_ATTN",
            if variant.direct_sparse_attn { "1" } else { "0" },
        )
        .env(
            "SKIPPY_GLM_DSA_ENABLE_FUSED_SPARSE_MASK",
            if variant.fused_sparse_mask { "1" } else { "0" },
        )
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log));
    ChildGuard::spawn(command)
}

fn run_prompt(
    args: &GlmDsaStage0TraceArgs,
    prompt_log_path: &Path,
) -> Result<std::process::Output> {
    let model_path = stage_model_path(args)?;
    let mut child = Command::new(&args.prompt_bin)
        .args([
            "binary",
            "--model-path",
            path_str(&model_path)?,
            "--tokenizer-load-mode",
            "layer-package",
            "--tokenizer-layer-start",
            "0",
            "--tokenizer-layer-end",
            "1",
            "--first-stage-addr",
            &args.stage0_bind_addr.to_string(),
            "--ctx-size",
            &args.runtime.ctx_size.to_string(),
            "--n-gpu-layers",
            "0",
            "--activation-width",
            &args.activation_width.to_string(),
            "--activation-wire-dtype",
            &args.activation_wire_dtype,
            "--prefill-chunk-size",
            &args.prefill_chunk_size.to_string(),
            "--max-new-tokens",
            &args.max_new_tokens.to_string(),
            "--decode-timeout-secs",
            &args.server.startup_timeout_secs.to_string(),
            "--trace-token-ids",
            "--no-think",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {}", args.prompt_bin.display()))?;
    {
        let stdin = child.stdin.as_mut().context("open skippy-prompt stdin")?;
        let prompt = args.runtime.prompt.replace(['\r', '\n'], " ");
        writeln!(stdin, "{prompt}")?;
        writeln!(stdin, ":quit")?;
    }
    let output = child.wait_with_output().context("wait for skippy-prompt")?;
    let mut log = Vec::new();
    log.extend_from_slice(&output.stdout);
    log.extend_from_slice(&output.stderr);
    fs::write(prompt_log_path, log)
        .with_context(|| format!("write prompt log {}", prompt_log_path.display()))?;
    Ok(output)
}

fn wait_for_stage_ready_or_exit(
    stage: &mut ChildGuard,
    addr: SocketAddr,
    timeout_secs: u64,
    log_path: &Path,
) -> Result<TcpStream> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        if let Some(status) = stage.try_wait()? {
            bail!(
                "stage0 exited before ready with status {status}; log tail:\n{}",
                log_tail(log_path, 40)
            );
        }
        match try_connect_ready(addr) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out"))
        .context("stage0 did not become ready"))
}

fn try_connect_ready(addr: SocketAddr) -> Result<TcpStream> {
    let mut stream = TcpStream::connect(addr).context("connect failed")?;
    stream.set_nodelay(true).ok();
    stream
        .set_read_timeout(Some(Duration::from_millis(500)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_millis(500)))
        .ok();
    recv_ready(&mut stream).context("ready handshake failed")?;
    Ok(stream)
}

fn log_tail(path: &Path, max_lines: usize) -> String {
    let Ok(log) = fs::read_to_string(path) else {
        return format!("unable to read {}", path.display());
    };
    let mut lines = log.lines().rev().take(max_lines).collect::<Vec<_>>();
    lines.reverse();
    lines.join("\n")
}

impl FakeDownstreamGuard {
    fn start(addr: SocketAddr, activation_width: i32) -> Result<Self> {
        let listener = TcpListener::bind(addr).with_context(|| format!("bind fake {addr}"))?;
        listener
            .set_nonblocking(true)
            .with_context(|| format!("set fake {addr} nonblocking"))?;
        let stop = Arc::new(AtomicBool::new(false));
        let messages = Arc::new(Mutex::new(Vec::new()));
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread_stop = Arc::clone(&stop);
        let thread_messages = Arc::clone(&messages);
        let handle = thread::spawn(move || -> Result<()> {
            ready_tx.send(()).ok();
            while !thread_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_nonblocking(false)
                            .context("set fake downstream stream blocking")?;
                        stream
                            .set_read_timeout(Some(Duration::from_millis(100)))
                            .context("set fake downstream stream read timeout")?;
                        send_ready(&mut stream).context("send fake downstream ready")?;
                        loop {
                            let reader = StopAwareReader {
                                stream: &mut stream,
                                stop: &thread_stop,
                            };
                            let message = match read_stage_message(reader, activation_width) {
                                Ok(message) => message,
                                Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
                                Err(error)
                                    if error.kind() == ErrorKind::ConnectionAborted
                                        && thread_stop.load(Ordering::Relaxed) =>
                                {
                                    break;
                                }
                                Err(error) => {
                                    return Err(anyhow!(error).context("read stage message"));
                                }
                            };
                            let activation_f32_payload =
                                message.activation_f32_payload(activation_width).ok();
                            let summary = FakeDownstreamMessage {
                                kind: message.kind,
                                pos_start: message.pos_start,
                                token_count: message.token_count,
                                activation_bytes: message.activation.len(),
                                activation_sha256: sha256_hex(&message.activation),
                                activation_f32: activation_f32_payload
                                    .as_ref()
                                    .and_then(|payload| activation_f32_stats(payload)),
                                activation_f32_payload,
                                top_k_count: message.raw_bytes.len() / std::mem::size_of::<i32>(),
                                top_k_sha256: sha256_hex(&message.raw_bytes),
                                top_k_values: decode_i32_values(&message.raw_bytes),
                            };
                            thread_messages
                                .lock()
                                .expect("fake downstream messages lock poisoned")
                                .push(summary.clone());
                            match message.kind {
                                WireMessageKind::Stop => {
                                    send_reply_ack_with_stats(
                                        &mut stream,
                                        StageReplyStats::default(),
                                    )?;
                                    return Ok(());
                                }
                                kind if kind.requires_predicted_reply() => {
                                    send_reply_predicted_with_stats(
                                        &mut stream,
                                        2,
                                        StageReplyStats::default(),
                                    )?;
                                }
                                _ => {
                                    send_reply_ack_with_stats(
                                        &mut stream,
                                        StageReplyStats::default(),
                                    )?;
                                }
                            }
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(error) => return Err(anyhow!(error).context("accept fake downstream")),
                }
            }
            Ok(())
        });
        ready_rx
            .recv_timeout(Duration::from_secs(2))
            .context("fake downstream thread did not start")?;
        Ok(Self {
            stop,
            messages,
            handle: Some(handle),
        })
    }

    fn finish(mut self) -> Result<Vec<FakeDownstreamMessage>> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("fake downstream thread panicked"))??;
        }
        Ok(self
            .messages
            .lock()
            .expect("fake downstream messages lock poisoned")
            .clone())
    }
}

impl Drop for FakeDownstreamGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn parse_timing_chunks(stage_log: &str) -> Vec<GlmDsaTimingChunkReport> {
    stage_log
        .lines()
        .filter(|line| line.contains("glm_dsa_op_timing"))
        .enumerate()
        .filter_map(|(index, line)| parse_timing_chunk(index, line))
        .collect()
}

fn parse_timing_chunk(index: usize, line: &str) -> Option<GlmDsaTimingChunkReport> {
    Some(GlmDsaTimingChunkReport {
        index,
        tokens: timing_value(line, "tokens")? as u32,
        total_us: timing_value(line, "total_us")?,
        indexer_topk_us: timing_value(line, "indexer_topk_us")?,
        sparse_mask_us: timing_value(line, "sparse_mask_us")?,
        dsa_sparse_attn_us: timing_value(line, "dsa_sparse_attn_us").unwrap_or(0.0),
        mla_attention_us: timing_value(line, "mla_attention_us")?,
    })
}

fn parse_timing_group_chunks(stage_log: &str) -> Vec<GlmDsaTimingGroupChunkReport> {
    stage_log
        .lines()
        .filter(|line| line.contains("glm_dsa_group_timing"))
        .enumerate()
        .filter_map(|(index, line)| parse_timing_group_chunk(index, line))
        .collect()
}

fn parse_timing_group_chunk(index: usize, line: &str) -> Option<GlmDsaTimingGroupChunkReport> {
    Some(GlmDsaTimingGroupChunkReport {
        index,
        tokens: timing_value(line, "tokens")? as u32,
        group: timing_string_value(line, "group")?.to_owned(),
        total_us: timing_value(line, "total_us")?,
        indexer_topk_us: timing_value(line, "indexer_topk_us")?,
        sparse_mask_us: timing_value(line, "sparse_mask_us")?,
        dsa_sparse_attn_us: timing_value(line, "dsa_sparse_attn_us").unwrap_or(0.0),
        mla_attention_us: timing_value(line, "mla_attention_us")?,
    })
}

fn avg_128_token_timing(chunks: &[GlmDsaTimingChunkReport]) -> Option<GlmDsaTimingReport> {
    let chunks = chunks
        .iter()
        .filter(|chunk| chunk.tokens == 128)
        .collect::<Vec<_>>();
    let count = chunks.len();
    if count == 0 {
        return None;
    }
    let mut total_us = 0.0;
    let mut indexer_topk_us = 0.0;
    let mut sparse_mask_us = 0.0;
    let mut dsa_sparse_attn_us = 0.0;
    let mut mla_attention_us = 0.0;
    for chunk in &chunks {
        total_us += chunk.total_us;
        indexer_topk_us += chunk.indexer_topk_us;
        sparse_mask_us += chunk.sparse_mask_us;
        dsa_sparse_attn_us += chunk.dsa_sparse_attn_us;
        mla_attention_us += chunk.mla_attention_us;
    }
    let count_f = count as f64;
    Some(GlmDsaTimingReport {
        chunk_count: count,
        total_us: total_us / count_f,
        indexer_topk_us: indexer_topk_us / count_f,
        sparse_mask_us: sparse_mask_us / count_f,
        dsa_sparse_attn_us: dsa_sparse_attn_us / count_f,
        mla_attention_us: mla_attention_us / count_f,
    })
}

fn max_128_token_timing(chunks: &[GlmDsaTimingChunkReport]) -> Option<GlmDsaTimingChunkReport> {
    chunks
        .iter()
        .filter(|chunk| chunk.tokens == 128)
        .max_by(|left, right| left.total_us.total_cmp(&right.total_us))
        .cloned()
}

fn last_128_token_timing(chunks: &[GlmDsaTimingChunkReport]) -> Option<GlmDsaTimingChunkReport> {
    chunks
        .iter()
        .rev()
        .find(|chunk| chunk.tokens == 128)
        .cloned()
}

fn timing_value(line: &str, key: &str) -> Option<f64> {
    line.split_whitespace().find_map(|field| {
        let (field_key, value) = field.split_once('=')?;
        (field_key == key)
            .then(|| value.parse::<f64>().ok())
            .flatten()
    })
}

fn timing_string_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_whitespace().find_map(|field| {
        let (field_key, value) = field.split_once('=')?;
        (field_key == key).then_some(value)
    })
}

fn parse_prompt_speeds(prompt_log: &str) -> (Option<f64>, Option<f64>) {
    let mut prefill = None;
    let mut decode = None;
    for line in prompt_log.lines() {
        if !line.trim_start().starts_with("speed") {
            continue;
        }
        prefill = speed_value(line, "prefill");
        decode = speed_value(line, "decode");
    }
    (prefill, decode)
}

fn speed_value(line: &str, key: &str) -> Option<f64> {
    line.split_whitespace().find_map(|field| {
        let (field_key, value) = field.split_once('=')?;
        (field_key == key)
            .then(|| value.parse::<f64>().ok())
            .flatten()
    })
}

fn speedup(fused: Option<f64>, direct: Option<f64>) -> Option<f64> {
    let fused = fused?;
    let direct = direct?;
    (direct > 0.0).then_some(fused / direct)
}

fn timing_speedup(
    fused: Option<&GlmDsaTimingReport>,
    direct: Option<&GlmDsaTimingReport>,
) -> Option<f64> {
    let fused = fused?;
    let direct = direct?;
    (fused.total_us > 0.0).then_some(direct.total_us / fused.total_us)
}

fn nonzero_div_usize(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn compare_downstream_messages(
    fused: &[FakeDownstreamMessage],
    direct: &[FakeDownstreamMessage],
) -> GlmDsaDownstreamParityReport {
    let compared_message_count = fused.len().min(direct.len());
    let messages = (0..compared_message_count)
        .map(|index| compare_downstream_message(index, &fused[index], &direct[index]))
        .collect::<Vec<_>>();
    let activation_mismatch_count = messages
        .iter()
        .filter(|message| !message.activation_sha256_equal)
        .count()
        + fused.len().abs_diff(direct.len());
    let top_k_mismatch_count = messages
        .iter()
        .filter(|message| !message.top_k_sha256_equal || !message.top_k_count_equal)
        .count();
    let mismatched_message_count = messages
        .iter()
        .filter(|message| {
            message.fused_kind != message.direct_kind
                || message.fused_pos_start != message.direct_pos_start
                || message.fused_token_count != message.direct_token_count
                || !message.activation_sha256_equal
                || !message.top_k_sha256_equal
                || !message.top_k_count_equal
        })
        .count()
        + fused.len().abs_diff(direct.len());
    GlmDsaDownstreamParityReport {
        matched: fused.len() == direct.len() && mismatched_message_count == 0,
        fused_message_count: fused.len(),
        direct_message_count: direct.len(),
        compared_message_count,
        mismatched_message_count,
        activation_mismatch_count,
        top_k_mismatch_count,
        messages,
    }
}

fn compare_semantic_parity(
    downstream: &GlmDsaDownstreamParityReport,
    activation_atol: f32,
    activation_relative_rmse_tolerance: f64,
) -> GlmDsaSemanticParityReport {
    let message_metadata_exact = downstream.fused_message_count == downstream.direct_message_count
        && downstream.messages.iter().all(|message| {
            message.fused_kind == message.direct_kind
                && message.fused_pos_start == message.direct_pos_start
                && message.fused_token_count == message.direct_token_count
        });
    let top_k_exact = downstream.top_k_mismatch_count == 0
        && downstream
            .messages
            .iter()
            .all(|message| message.top_k_sha256_equal && message.top_k_count_equal);
    let activation_out_of_tolerance_count = downstream
        .messages
        .iter()
        .filter(|message| {
            !activation_message_within_tolerance(
                message,
                activation_atol,
                activation_relative_rmse_tolerance,
            )
        })
        .count();
    let activation_within_tolerance = downstream.fused_message_count
        == downstream.direct_message_count
        && activation_out_of_tolerance_count == 0;
    GlmDsaSemanticParityReport {
        matched: message_metadata_exact && top_k_exact && activation_within_tolerance,
        activation_atol,
        activation_relative_rmse_tolerance,
        activation_within_tolerance,
        activation_out_of_tolerance_count,
        top_k_exact,
        message_metadata_exact,
        compared_message_count: downstream.compared_message_count,
    }
}

fn activation_message_within_tolerance(
    message: &GlmDsaDownstreamComparisonReport,
    activation_atol: f32,
    activation_relative_rmse_tolerance: f64,
) -> bool {
    if message.activation_sha256_equal {
        return true;
    }

    let Some(error) = message.activation_error.as_ref() else {
        return false;
    };
    let relative_rmse_ok = error
        .relative_rmse
        .is_some_and(|relative_rmse| relative_rmse <= activation_relative_rmse_tolerance);
    error.non_finite_pair_count == 0 && error.max_abs_error <= activation_atol && relative_rmse_ok
}

fn compare_downstream_message(
    index: usize,
    fused: &FakeDownstreamMessage,
    direct: &FakeDownstreamMessage,
) -> GlmDsaDownstreamComparisonReport {
    GlmDsaDownstreamComparisonReport {
        index,
        fused_kind: format!("{:?}", fused.kind),
        direct_kind: format!("{:?}", direct.kind),
        fused_pos_start: fused.pos_start,
        direct_pos_start: direct.pos_start,
        fused_token_count: fused.token_count,
        direct_token_count: direct.token_count,
        activation_sha256_equal: fused.activation_sha256 == direct.activation_sha256,
        top_k_sha256_equal: fused.top_k_sha256 == direct.top_k_sha256,
        top_k_count_equal: fused.top_k_count == direct.top_k_count,
        top_k_comparison: compare_top_k_values(fused, direct),
        activation_error: compare_activation_payloads(
            fused.activation_f32_payload.as_deref(),
            direct.activation_f32_payload.as_deref(),
        ),
    }
}

fn compare_top_k_values(
    fused: &FakeDownstreamMessage,
    direct: &FakeDownstreamMessage,
) -> Option<GlmDsaTopKComparisonReport> {
    if fused.top_k_values.is_empty() || fused.top_k_values.len() != direct.top_k_values.len() {
        return None;
    }

    let mut mismatch_count = 0usize;
    let mut first_mismatch = None;
    let mut active_compared_count = 0usize;
    let mut active_mismatch_count = 0usize;
    let mut first_active_mismatch = None;
    for (index, (fused_value, direct_value)) in fused
        .top_k_values
        .iter()
        .zip(direct.top_k_values.iter())
        .enumerate()
    {
        if fused_value != direct_value {
            mismatch_count += 1;
            first_mismatch.get_or_insert((index, *fused_value, *direct_value));
        }
        if top_k_index_is_active(fused, index, *fused_value)
            || top_k_index_is_active(direct, index, *direct_value)
        {
            active_compared_count += 1;
            if fused_value != direct_value {
                active_mismatch_count += 1;
                first_active_mismatch.get_or_insert((index, *fused_value, *direct_value));
            }
        }
    }

    let compared_count = fused.top_k_values.len();
    Some(GlmDsaTopKComparisonReport {
        compared_count,
        mismatch_count,
        mismatch_ratio: ratio(mismatch_count, compared_count),
        active_compared_count,
        active_mismatch_count,
        active_mismatch_ratio: ratio(active_mismatch_count, active_compared_count),
        first_mismatch_index: first_mismatch.map(|(index, _, _)| index),
        first_mismatch_fused: first_mismatch.map(|(_, value, _)| value),
        first_mismatch_direct: first_mismatch.map(|(_, _, value)| value),
        first_active_mismatch_index: first_active_mismatch.map(|(index, _, _)| index),
        first_active_mismatch_fused: first_active_mismatch.map(|(_, value, _)| value),
        first_active_mismatch_direct: first_active_mismatch.map(|(_, _, value)| value),
    })
}

fn top_k_index_is_active(message: &FakeDownstreamMessage, index: usize, value: i32) -> bool {
    let token_count = message_token_count(message);
    if token_count == 0 {
        return false;
    }
    let sideband_width = message.top_k_values.len() / token_count;
    if sideband_width == 0 {
        return false;
    }
    let token_index = index / sideband_width;
    let visible_width =
        i32::try_from(causal_visible_width(message.pos_start, token_index)).unwrap_or(i32::MAX);
    value >= 0 && value < visible_width
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn compare_activation_payloads(
    fused: Option<&[u8]>,
    direct: Option<&[u8]>,
) -> Option<GlmDsaActivationErrorReport> {
    let fused = fused?;
    let direct = direct?;
    if fused.is_empty()
        || fused.len() != direct.len()
        || !fused.len().is_multiple_of(std::mem::size_of::<f32>())
    {
        return None;
    }

    let mut count = 0usize;
    let mut non_finite_pair_count = 0usize;
    let mut max_abs_error = 0.0f32;
    let mut max_reference_abs = 0.0f32;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut ref_sq_sum = 0.0f64;
    for (fused_chunk, direct_chunk) in fused
        .chunks_exact(std::mem::size_of::<f32>())
        .zip(direct.chunks_exact(std::mem::size_of::<f32>()))
    {
        let fused_value = f32::from_le_bytes(fused_chunk.try_into().ok()?);
        let direct_value = f32::from_le_bytes(direct_chunk.try_into().ok()?);
        count += 1;
        if !fused_value.is_finite() || !direct_value.is_finite() {
            non_finite_pair_count += 1;
            continue;
        }
        let error = (fused_value - direct_value).abs();
        max_abs_error = max_abs_error.max(error);
        max_reference_abs = max_reference_abs.max(fused_value.abs());
        abs_sum += f64::from(error);
        sq_sum += f64::from(error) * f64::from(error);
        ref_sq_sum += f64::from(fused_value) * f64::from(fused_value);
    }
    let finite_count = count.saturating_sub(non_finite_pair_count);
    if finite_count == 0 {
        return None;
    }
    let rmse = (sq_sum / finite_count as f64).sqrt();
    let reference_rmse = (ref_sq_sum / finite_count as f64).sqrt();
    Some(GlmDsaActivationErrorReport {
        count,
        max_abs_error,
        mean_abs_error: abs_sum / finite_count as f64,
        rmse,
        relative_rmse: (reference_rmse > 0.0).then_some(rmse / reference_rmse),
        max_reference_abs,
        non_finite_pair_count,
    })
}

fn fake_downstream_message_report(
    message: &FakeDownstreamMessage,
) -> GlmDsaDownstreamMessageReport {
    GlmDsaDownstreamMessageReport {
        kind: format!("{:?}", message.kind),
        pos_start: message.pos_start,
        token_count: message.token_count,
        activation_bytes: message.activation_bytes,
        activation_sha256: message.activation_sha256.clone(),
        activation_f32: message.activation_f32.clone(),
        top_k_count: message.top_k_count,
        top_k_sha256: message.top_k_sha256.clone(),
    }
}

fn activation_f32_stats(bytes: &[u8]) -> Option<GlmDsaActivationStatsReport> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return None;
    }
    let mut count = 0usize;
    let mut non_finite_count = 0usize;
    let mut sum = 0.0f64;
    let mut max_abs = 0.0f32;
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        let value = f32::from_le_bytes(chunk.try_into().ok()?);
        count += 1;
        if value.is_finite() {
            sum += f64::from(value);
            max_abs = max_abs.max(value.abs());
        } else {
            non_finite_count += 1;
        }
    }
    let finite_count = count.saturating_sub(non_finite_count);
    Some(GlmDsaActivationStatsReport {
        count,
        sum,
        mean: if finite_count == 0 {
            f64::NAN
        } else {
            sum / finite_count as f64
        },
        max_abs,
        non_finite_count,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_sha256_finish(hasher)
}

fn hex_sha256_finish(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn stage_model_path(args: &GlmDsaStage0TraceArgs) -> Result<PathBuf> {
    Ok(args
        .runtime
        .stage_model
        .clone()
        .unwrap_or_else(|| args.runtime.model.clone()))
}

fn protocol_flash_attn(value: FlashAttentionArg) -> &'static str {
    match value {
        FlashAttentionArg::Auto => "auto",
        FlashAttentionArg::Disabled => "disabled",
        FlashAttentionArg::Enabled => "enabled",
    }
}

fn emit_report<T: serde::Serialize>(report: &T, report_out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");
    if let Some(path) = report_out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write correctness report {}", path.display()))?;
    }
    Ok(())
}

fn generate_glm_dsa_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis();
    format!("glm-dsa-stage0-trace-{millis}")
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::{
        net::{TcpListener, TcpStream},
        sync::mpsc,
        thread,
        time::Duration,
    };

    use skippy_protocol::binary::recv_ready;

    use super::{
        FakeDownstreamGuard, FakeDownstreamMessage, WireMessageKind, active_top_k_window_count,
        causal_visible_top_k_count, compare_tensor_traces, finite_top_k_count, parity_gate_matched,
        parse_tensor_trace_records, parse_timing_group_chunks, summarize_fake_downstream,
    };

    #[test]
    fn parity_gate_requires_trace_and_semantic_matches() {
        assert!(parity_gate_matched(true, true));
        assert!(!parity_gate_matched(true, false));
        assert!(!parity_gate_matched(false, true));
    }

    #[test]
    fn causal_visible_top_k_counts_prefill_positions_per_token() {
        let message = FakeDownstreamMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 512,
            token_count: 128,
            activation_bytes: 0,
            activation_sha256: String::new(),
            activation_f32: None,
            activation_f32_payload: None,
            top_k_count: 98_304,
            top_k_sha256: String::new(),
            top_k_values: Vec::new(),
        };

        assert_eq!(causal_visible_top_k_count(&message), 73_792);
    }

    #[test]
    fn active_top_k_window_counts_last_visible_index_per_token() {
        let message = FakeDownstreamMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 2,
            token_count: 2,
            activation_bytes: 0,
            activation_sha256: String::new(),
            activation_f32: None,
            activation_f32_payload: None,
            top_k_count: 8,
            top_k_sha256: String::new(),
            top_k_values: vec![
                7, 1, 9, 2, //
                3, 8, 9, 7,
            ],
        };

        assert_eq!(causal_visible_top_k_count(&message), 7);
        assert_eq!(finite_top_k_count(&message), 3);
        assert_eq!(active_top_k_window_count(&message), 5);

        let summary = summarize_fake_downstream(std::slice::from_ref(&message));
        assert_eq!(summary.message_count, 1);
        assert_eq!(summary.prefill_message_count, 1);
        assert_eq!(summary.total_top_k_count, 8);
        assert_eq!(summary.total_causal_visible_top_k_count, 7);
        assert_eq!(summary.total_active_top_k_window_count, 5);
        assert_eq!(summary.total_finite_top_k_count, 3);
        assert_eq!(summary.total_padded_top_k_count, 1);
        assert_eq!(summary.messages.len(), 1);
    }

    #[test]
    fn fake_downstream_finish_interrupts_idle_connection() {
        let probe = TcpListener::bind("127.0.0.1:0").expect("bind free port probe");
        let addr = probe.local_addr().expect("free port address");
        drop(probe);

        let fake = FakeDownstreamGuard::start(addr, 1).expect("start fake downstream");
        let mut stream = TcpStream::connect(addr).expect("connect fake downstream");
        recv_ready(&mut stream).expect("receive fake downstream ready");

        let (finished_tx, finished_rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = finished_tx.send(fake.finish());
        });
        let messages = finished_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("fake downstream finish timed out")
            .expect("finish fake downstream");
        assert!(messages.is_empty());
    }

    #[test]
    fn parses_tensor_trace_records_with_digest_stats() {
        let traces = parse_tensor_trace_records(
            "skippy: glm_dsa_tensor_trace stage=0 tokens=128 op=mla_attention node=1 name=kqv_out-0 type=f16 ne=[6144,128,1,1] nb=[2,12288,1572864,1572864] contiguous=1 nbytes=1572864 values=[0,1] stats=fnv64:abcd count=786432 sum=1 mean=2 max_abs=3\n",
        );

        let record = traces.values().next().expect("trace record");
        assert_eq!(record.key.tokens, 128);
        assert_eq!(record.key.name, "kqv_out-0");
        assert_eq!(record.tensor_type, "f16");
        assert_eq!(record.shape, [6144, 128, 1, 1]);
        assert_eq!(record.stats.as_deref(), Some("fnv64:abcd"));
    }

    #[test]
    fn parses_glm_dsa_group_timing_chunks() {
        let chunks = parse_timing_group_chunks(
            "skippy: glm_dsa_group_timing stage=0 tokens=128 group=layer_0 total_us=123 indexer_topk_nodes=4 indexer_topk_us=12 indexer_nodes=3 indexer_us=8 top_k_nodes=1 top_k_us=4 sparse_mask_nodes=2 sparse_mask_us=7 sparse_mask_fill_nodes=1 sparse_mask_fill_us=2 sparse_mask_topk_nodes=0 sparse_mask_topk_us=0 sparse_mask_add_nodes=1 sparse_mask_add_us=5 dsa_sparse_attn_nodes=0 dsa_sparse_attn_us=0 mla_attention_nodes=1 mla_attention_us=99 routed_moe_nodes=0 routed_moe_us=0 shared_expert_nodes=0 shared_expert_us=0\n",
        );

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].tokens, 128);
        assert_eq!(chunks[0].group, "layer_0");
        assert_eq!(chunks[0].indexer_topk_us, 12.0);
        assert_eq!(chunks[0].sparse_mask_us, 7.0);
        assert_eq!(chunks[0].mla_attention_us, 99.0);
    }

    #[test]
    fn trace_parity_passes_with_direct_only_trace_points() {
        let fused = parse_tensor_trace_records(
            "skippy: glm_dsa_tensor_trace stage=0 tokens=128 op=mla_attention node=1 name=kqv_out-0 type=f16 ne=[6144,128,1,1] nb=[2,12288,1572864,1572864] contiguous=1 nbytes=1572864 values=[0] stats=fnv64:aaaa count=1 sum=0 mean=0 max_abs=0\n",
        );
        let direct = parse_tensor_trace_records(
            "skippy: glm_dsa_tensor_trace stage=0 tokens=128 op=dsa_sparse_attn node=1 name=dsa_sparse_attn-0 type=f16 ne=[6144,128,1,1] nb=[2,12288,1572864,1572864] contiguous=1 nbytes=1572864 values=[0] stats=fnv64:bbbb count=1 sum=0 mean=0 max_abs=0\n\
             skippy: glm_dsa_tensor_trace stage=0 tokens=128 op=mla_attention node=2 name=kqv_out-0 type=f16 ne=[6144,128,1,1] nb=[2,12288,1572864,1572864] contiguous=1 nbytes=1572864 values=[0] stats=fnv64:aaaa count=1 sum=0 mean=0 max_abs=0\n",
        );

        let report = compare_tensor_traces(true, fused, direct);
        assert!(report.matched);
        assert_eq!(report.compared_trace_count, 1);
        assert_eq!(report.missing_in_fused_count, 1);
        assert_eq!(report.mismatched_trace_count, 0);
    }

    #[test]
    fn trace_parity_fails_on_shared_digest_mismatch() {
        let fused = parse_tensor_trace_records(
            "skippy: glm_dsa_tensor_trace stage=0 tokens=128 op=mla_attention node=1 name=kqv_out-0 type=f16 ne=[6144,128,1,1] nb=[2,12288,1572864,1572864] contiguous=1 nbytes=1572864 values=[0] stats=fnv64:aaaa count=1 sum=0 mean=0 max_abs=0\n",
        );
        let direct = parse_tensor_trace_records(
            "skippy: glm_dsa_tensor_trace stage=0 tokens=128 op=mla_attention node=2 name=kqv_out-0 type=f16 ne=[6144,128,1,1] nb=[2,12288,1572864,1572864] contiguous=1 nbytes=1572864 values=[0] stats=fnv64:bbbb count=1 sum=0 mean=0 max_abs=0\n",
        );

        let report = compare_tensor_traces(true, fused, direct);
        assert!(!report.matched);
        assert_eq!(report.compared_trace_count, 1);
        assert_eq!(report.mismatched_trace_count, 1);
        assert_eq!(report.mismatches[0].reason, "stats digest mismatch");
    }
}
