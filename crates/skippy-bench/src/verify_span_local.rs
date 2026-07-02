use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::{
    FlashAttentionType, RuntimeConfig, RuntimeLoadMode, SamplingConfig, StageModel, StageSession,
    parse_cache_type,
};

use crate::cli::VerifySpanLocalArgs;

#[derive(Debug, Serialize)]
struct TimingStats {
    count: usize,
    total_us: u128,
    avg_us: f64,
    min_us: u128,
    p50_us: u128,
    p95_us: u128,
    max_us: u128,
}

#[derive(Debug, Serialize)]
struct TimingShape {
    first_half: TimingStats,
    second_half: TimingStats,
    second_half_avg_vs_first_half_avg: f64,
    first_sample_us: u128,
    last_sample_us: u128,
    samples_us: Vec<u128>,
}

#[derive(Debug, Serialize)]
struct VerifySpanLocalReport {
    mode: &'static str,
    model_path: PathBuf,
    layer_end: u32,
    split_layer: Option<u32>,
    ctx_size: u32,
    n_gpu_layers: i32,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    cache_type_k: String,
    cache_type_v: String,
    prompt_token_count: usize,
    verify_tokens: Vec<i32>,
    warmup: usize,
    iterations: usize,
    batched_width2: TimingStats,
    serial_two_decode_mtp_n1: TimingStats,
    split_inprocess_width2: Option<SplitInprocessReport>,
    batched_avg_vs_serial_avg: f64,
    batched_token_per_sec: f64,
    serial_token_per_sec: f64,
    first_batched_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct SplitInprocessReport {
    split_layer: u32,
    boundary_payload_bytes: usize,
    serial_boundary_payload_bytes: usize,
    total: TimingStats,
    stage0: TimingStats,
    stage1: TimingStats,
    serial_total: TimingStats,
    serial_stage0: TimingStats,
    serial_stage1: TimingStats,
    total_token_per_sec: f64,
    serial_total_token_per_sec: f64,
    total_avg_vs_full_batched_avg: f64,
    total_avg_vs_serial_total_avg: f64,
    diagnostics: SplitTimingDiagnostics,
    first_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct SplitTimingDiagnostics {
    batched_total: TimingShape,
    batched_stage0: TimingShape,
    batched_stage1: TimingShape,
    serial_total: TimingShape,
    serial_stage0: TimingShape,
    serial_stage1: TimingShape,
}

pub fn verify_span_local(args: VerifySpanLocalArgs) -> Result<()> {
    validate_args(&args)?;
    let output = args.output.clone();
    let full = run_full_model_samples(&args)?;
    let split = match args.split_layer {
        Some(split_layer) => Some(run_split_inprocess_samples(
            &args,
            split_layer,
            &full.tokens,
            &full.verify_tokens,
            full.samples.batched_avg_us()?,
        )?),
        None => None,
    };
    let report = build_report(args, full, split)?;
    let encoded = serde_json::to_vec_pretty(&report)?;

    if let Some(path) = output {
        fs::write(&path, &encoded)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn validate_args(args: &VerifySpanLocalArgs) -> Result<()> {
    if args.layer_end == 0 {
        bail!("layer_end must be greater than zero");
    }
    if args.iterations == 0 {
        bail!("iterations must be greater than zero");
    }
    if let Some(split_layer) = args.split_layer
        && (split_layer == 0 || split_layer >= args.layer_end)
    {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    Ok(())
}

fn full_runtime_config(args: &VerifySpanLocalArgs) -> Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
        mlock: false,
        selected_backend_device: None,
        cache_type_k: parse_cache_type(&args.cache_type_k)?,
        cache_type_v: parse_cache_type(&args.cache_type_v)?,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
    })
}

struct FullModelSamples {
    tokens: Vec<i32>,
    verify_tokens: Vec<i32>,
    samples: SampleSet,
}

fn run_full_model_samples(args: &VerifySpanLocalArgs) -> Result<FullModelSamples> {
    let config = full_runtime_config(args)?;
    let model = StageModel::open(&args.model_path, &config)
        .with_context(|| format!("failed to open {}", args.model_path.display()))?;
    let tokens = model
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    if tokens.is_empty() {
        bail!("prompt produced no tokens");
    }

    let mut session = model.create_session().context("failed to create session")?;
    session
        .prefill_chunked(&tokens)
        .context("failed to prefill prompt")?;
    let base_token_count = session.token_count();
    let verify_tokens = choose_verify_tokens(
        &mut session,
        base_token_count,
        &tokens,
        &args.prompt,
        &config,
    )?;
    let samples = run_samples(
        &mut session,
        base_token_count,
        &verify_tokens,
        args.warmup,
        args.iterations,
    )?;
    Ok(FullModelSamples {
        tokens,
        verify_tokens,
        samples,
    })
}

fn choose_verify_tokens(
    session: &mut StageSession,
    base_token_count: u64,
    prompt_tokens: &[i32],
    prompt: &str,
    config: &RuntimeConfig,
) -> Result<Vec<i32>> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before choosing verify tokens")?;
    let current = *prompt_tokens
        .first()
        .context("prompt produced no token for verify-token seed")?;
    let (_after_current, native_mtp, _frame) = session
        .decode_step_frame_sampled_mtp(current, Some(&SamplingConfig::default()), None, 0, 1)
        .with_context(|| {
            format!(
                "failed to get native MTP draft from {} after prompt {:?}",
                model_description(config),
                prompt
            )
        })?;
    let Some(draft) = native_mtp else {
        bail!("model did not produce a native MTP n=1 draft token");
    };
    session
        .trim_session(base_token_count)
        .context("failed to trim session after choosing verify tokens")?;
    let draft_token = draft
        .token_ids
        .first()
        .copied()
        .context("native MTP draft did not include a token")?;
    Ok(vec![current, draft_token])
}

fn run_samples(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    warmup: usize,
    iterations: usize,
) -> Result<SampleSet> {
    let total = warmup
        .checked_add(iterations)
        .context("sample count overflow")?;
    let mut batched = Vec::with_capacity(iterations);
    let mut serial = Vec::with_capacity(iterations);
    let mut first_batched_prediction = None;
    let mut first_serial_prediction = None;

    {
        let mut targets = SampleRecordTargets {
            batched: &mut batched,
            serial: &mut serial,
            first_batched_prediction: &mut first_batched_prediction,
            first_serial_prediction: &mut first_serial_prediction,
        };
        for index in 0..total {
            let record = index >= warmup;
            if index.is_multiple_of(2) {
                measure_batched_then_serial(
                    session,
                    base_token_count,
                    verify_tokens,
                    record,
                    &mut targets,
                )?;
            } else {
                measure_serial_then_batched(
                    session,
                    base_token_count,
                    verify_tokens,
                    record,
                    &mut targets,
                )?;
            }
        }
    }

    Ok(SampleSet {
        batched,
        serial,
        first_batched_prediction: first_batched_prediction.unwrap_or_default(),
        first_serial_prediction: first_serial_prediction.unwrap_or_default(),
    })
}

fn measure_batched_then_serial(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    record: bool,
    targets: &mut SampleRecordTargets<'_>,
) -> Result<()> {
    let (batched_duration, batched_prediction) =
        measure_batched(session, base_token_count, verify_tokens)?;
    let (serial_duration, serial_prediction) =
        measure_serial(session, base_token_count, verify_tokens)?;
    record_sample(
        record,
        (batched_duration, batched_prediction),
        (serial_duration, serial_prediction),
        targets,
    );
    Ok(())
}

fn measure_serial_then_batched(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    record: bool,
    targets: &mut SampleRecordTargets<'_>,
) -> Result<()> {
    let (serial_duration, serial_prediction) =
        measure_serial(session, base_token_count, verify_tokens)?;
    let (batched_duration, batched_prediction) =
        measure_batched(session, base_token_count, verify_tokens)?;
    record_sample(
        record,
        (batched_duration, batched_prediction),
        (serial_duration, serial_prediction),
        targets,
    );
    Ok(())
}

struct SampleRecordTargets<'a> {
    batched: &'a mut Vec<Duration>,
    serial: &'a mut Vec<Duration>,
    first_batched_prediction: &'a mut Option<Vec<i32>>,
    first_serial_prediction: &'a mut Option<Vec<i32>>,
}

fn record_sample(
    record: bool,
    batched_sample: (Duration, Vec<i32>),
    serial_sample: (Duration, Vec<i32>),
    targets: &mut SampleRecordTargets<'_>,
) {
    if !record {
        return;
    }
    targets.batched.push(batched_sample.0);
    targets.serial.push(serial_sample.0);
    targets
        .first_batched_prediction
        .get_or_insert(batched_sample.1);
    targets
        .first_serial_prediction
        .get_or_insert(serial_sample.1);
}

fn measure_batched(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
) -> Result<(Duration, Vec<i32>)> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before batched verify")?;
    let start = Instant::now();
    let prediction = session
        .verify_tokens_frame_sampled(verify_tokens, Some(&SamplingConfig::default()), None, 0)
        .context("batched width-2 VerifySpan failed")?
        .0;
    Ok((start.elapsed(), prediction))
}

fn measure_serial(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
) -> Result<(Duration, Vec<i32>)> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before serial verify")?;
    let start = Instant::now();
    let prediction = serial_decode_mtp_n1(session, verify_tokens)?;
    Ok((start.elapsed(), prediction))
}

fn run_split_inprocess_samples(
    args: &VerifySpanLocalArgs,
    split_layer: u32,
    tokens: &[i32],
    verify_tokens: &[i32],
    full_batched_avg_us: f64,
) -> Result<SplitInprocessReport> {
    let (stage0_config, stage1_config) = split_runtime_configs(args, split_layer)?;
    let stage0 = StageModel::open(&args.model_path, &stage0_config)
        .context("failed to open in-process split stage 0")?;
    let stage1 = StageModel::open(&args.model_path, &stage1_config)
        .context("failed to open in-process split stage 1")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create in-process split stage 0 session")?;
    let mut session1 = stage1
        .create_session()
        .context("failed to create in-process split stage 1 session")?;
    prefill_split_sessions(&mut session0, &mut session1, tokens)?;
    let base0 = session0.token_count();
    let base1 = session1.token_count();
    let samples = run_split_samples(
        &mut session0,
        &mut session1,
        base0,
        base1,
        verify_tokens,
        args.warmup,
        args.iterations,
    )?;
    split_report(split_layer, samples, full_batched_avg_us)
}

fn split_runtime_configs(
    args: &VerifySpanLocalArgs,
    split_layer: u32,
) -> Result<(RuntimeConfig, RuntimeConfig)> {
    let cache_type_k = parse_cache_type(&args.cache_type_k)?;
    let cache_type_v = parse_cache_type(&args.cache_type_v)?;
    let stage0 = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: split_layer,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
        mlock: false,
        selected_backend_device: None,
        cache_type_k,
        cache_type_v,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: true,
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage1 = RuntimeConfig {
        stage_index: 1,
        layer_start: split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
        mlock: false,
        selected_backend_device: None,
        cache_type_k,
        cache_type_v,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: false,
        include_output: true,
        filter_tensors_on_load: true,
    };
    Ok((stage0, stage1))
}

fn prefill_split_sessions(
    session0: &mut StageSession,
    session1: &mut StageSession,
    tokens: &[i32],
) -> Result<()> {
    let (_stage0_prediction, boundary) = session0
        .prefill_chunk_frame_sampled(tokens, Some(&SamplingConfig::default()), None, 0)
        .context("in-process split stage 0 failed to prefill")?;
    if boundary.payload.is_empty() {
        bail!("in-process split stage 0 produced an empty prefill activation frame");
    }
    session1
        .prefill_chunk_frame_sampled(tokens, Some(&SamplingConfig::default()), Some(&boundary), 0)
        .context("in-process split stage 1 failed to prefill")?;
    Ok(())
}

fn run_split_samples(
    session0: &mut StageSession,
    session1: &mut StageSession,
    base0: u64,
    base1: u64,
    verify_tokens: &[i32],
    warmup: usize,
    iterations: usize,
) -> Result<SplitSampleSet> {
    let total = warmup
        .checked_add(iterations)
        .context("split sample count overflow")?;
    let mut total_samples = Vec::with_capacity(iterations);
    let mut stage0_samples = Vec::with_capacity(iterations);
    let mut stage1_samples = Vec::with_capacity(iterations);
    let mut serial_total_samples = Vec::with_capacity(iterations);
    let mut serial_stage0_samples = Vec::with_capacity(iterations);
    let mut serial_stage1_samples = Vec::with_capacity(iterations);
    let mut boundary_payload_bytes = 0usize;
    let mut serial_boundary_payload_bytes = 0usize;
    let mut first_prediction = None;
    let mut first_serial_prediction = None;

    for index in 0..total {
        let (batched, serial) = if index.is_multiple_of(2) {
            (
                measure_split_batched(session0, session1, base0, base1, verify_tokens)?,
                measure_split_serial(session0, session1, base0, base1, verify_tokens)?,
            )
        } else {
            let serial = measure_split_serial(session0, session1, base0, base1, verify_tokens)?;
            let batched = measure_split_batched(session0, session1, base0, base1, verify_tokens)?;
            (batched, serial)
        };
        if index >= warmup {
            total_samples.push(batched.total);
            stage0_samples.push(batched.stage0);
            stage1_samples.push(batched.stage1);
            serial_total_samples.push(serial.total);
            serial_stage0_samples.push(serial.stage0);
            serial_stage1_samples.push(serial.stage1);
            boundary_payload_bytes = batched.boundary_payload_bytes;
            serial_boundary_payload_bytes = serial.boundary_payload_bytes;
            first_prediction.get_or_insert(batched.prediction);
            first_serial_prediction.get_or_insert(serial.prediction);
        }
    }

    Ok(SplitSampleSet {
        total: total_samples,
        stage0: stage0_samples,
        stage1: stage1_samples,
        serial_total: serial_total_samples,
        serial_stage0: serial_stage0_samples,
        serial_stage1: serial_stage1_samples,
        boundary_payload_bytes,
        serial_boundary_payload_bytes,
        first_prediction: first_prediction.unwrap_or_default(),
        first_serial_prediction: first_serial_prediction.unwrap_or_default(),
    })
}

fn measure_split_batched(
    session0: &mut StageSession,
    session1: &mut StageSession,
    base0: u64,
    base1: u64,
    verify_tokens: &[i32],
) -> Result<SplitSample> {
    session0
        .trim_session(base0)
        .context("failed to trim split stage 0 before verify")?;
    session1
        .trim_session(base1)
        .context("failed to trim split stage 1 before verify")?;

    let total_start = Instant::now();
    let stage0_start = Instant::now();
    let (_stage0_prediction, boundary) = session0
        .verify_tokens_frame_sampled(verify_tokens, Some(&SamplingConfig::default()), None, 0)
        .context("in-process split stage 0 VerifySpan failed")?;
    let stage0 = stage0_start.elapsed();
    let boundary_payload_bytes = boundary.payload.len();
    if boundary_payload_bytes == 0 {
        bail!("in-process split stage 0 produced an empty VerifySpan activation frame");
    }

    let stage1_start = Instant::now();
    let prediction = session1
        .verify_tokens_frame_sampled(
            verify_tokens,
            Some(&SamplingConfig::default()),
            Some(&boundary),
            0,
        )
        .context("in-process split stage 1 VerifySpan failed")?
        .0;
    let stage1 = stage1_start.elapsed();
    Ok(SplitSample {
        total: total_start.elapsed(),
        stage0,
        stage1,
        boundary_payload_bytes,
        prediction,
    })
}

fn measure_split_serial(
    session0: &mut StageSession,
    session1: &mut StageSession,
    base0: u64,
    base1: u64,
    verify_tokens: &[i32],
) -> Result<SplitSample> {
    session0
        .trim_session(base0)
        .context("failed to trim split stage 0 before serial verify")?;
    session1
        .trim_session(base1)
        .context("failed to trim split stage 1 before serial verify")?;

    let total_start = Instant::now();
    let mut stage0_total = Duration::ZERO;
    let mut stage1_total = Duration::ZERO;
    let mut boundary_payload_bytes = 0usize;
    let mut prediction = Vec::with_capacity(verify_tokens.len() + 3);
    let mut last_draft = None;

    for token_id in verify_tokens {
        let stage0_start = Instant::now();
        let (_stage0_prediction, _stage0_draft, boundary) = session0
            .decode_step_frame_sampled_mtp(*token_id, Some(&SamplingConfig::default()), None, 0, 1)
            .context("in-process split stage 0 serial decode failed")?;
        stage0_total += stage0_start.elapsed();
        if boundary.payload.is_empty() {
            bail!("in-process split stage 0 produced an empty serial activation frame");
        }
        boundary_payload_bytes += boundary.payload.len();

        let stage1_start = Instant::now();
        let (predicted, native_mtp, _output) = session1
            .decode_step_frame_sampled_mtp(
                *token_id,
                Some(&SamplingConfig::default()),
                Some(&boundary),
                0,
                1,
            )
            .context("in-process split stage 1 serial decode failed")?;
        stage1_total += stage1_start.elapsed();
        if predicted >= 0 {
            prediction.push(predicted);
        }
        last_draft = native_mtp;
    }

    if let Some(draft) = last_draft {
        if let Some(token) = draft.token_ids.first().copied() {
            prediction.push(token);
        }
        prediction.push(i32::try_from(draft.proposal_compute_us.max(0)).unwrap_or(i32::MAX));
    }

    Ok(SplitSample {
        total: total_start.elapsed(),
        stage0: stage0_total,
        stage1: stage1_total,
        boundary_payload_bytes,
        prediction,
    })
}

fn serial_decode_mtp_n1(session: &mut StageSession, verify_tokens: &[i32]) -> Result<Vec<i32>> {
    let mut predicted_tokens = Vec::with_capacity(verify_tokens.len() + 3);
    let mut last_draft = None;
    for token_id in verify_tokens {
        let (predicted, native_mtp, _frame) = session
            .decode_step_frame_sampled_mtp(*token_id, Some(&SamplingConfig::default()), None, 0, 1)
            .context("serial native MTP n=1 decode failed")?;
        if predicted >= 0 {
            predicted_tokens.push(predicted);
        }
        last_draft = native_mtp;
    }
    if let Some(draft) = last_draft {
        if let Some(token) = draft.token_ids.first().copied() {
            predicted_tokens.push(token);
        }
        predicted_tokens.push(i32::try_from(draft.proposal_compute_us.max(0)).unwrap_or(i32::MAX));
    }
    Ok(predicted_tokens)
}

#[derive(Debug)]
struct SplitSample {
    total: Duration,
    stage0: Duration,
    stage1: Duration,
    boundary_payload_bytes: usize,
    prediction: Vec<i32>,
}

#[derive(Debug)]
struct SplitSampleSet {
    total: Vec<Duration>,
    stage0: Vec<Duration>,
    stage1: Vec<Duration>,
    serial_total: Vec<Duration>,
    serial_stage0: Vec<Duration>,
    serial_stage1: Vec<Duration>,
    boundary_payload_bytes: usize,
    serial_boundary_payload_bytes: usize,
    first_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

fn split_report(
    split_layer: u32,
    samples: SplitSampleSet,
    full_batched_avg_us: f64,
) -> Result<SplitInprocessReport> {
    let total = timing_stats(&samples.total)?;
    let serial_total = timing_stats(&samples.serial_total)?;
    let total_avg = total.avg_us;
    let serial_total_avg = serial_total.avg_us;
    Ok(SplitInprocessReport {
        split_layer,
        boundary_payload_bytes: samples.boundary_payload_bytes,
        serial_boundary_payload_bytes: samples.serial_boundary_payload_bytes,
        stage0: timing_stats(&samples.stage0)?,
        stage1: timing_stats(&samples.stage1)?,
        serial_stage0: timing_stats(&samples.serial_stage0)?,
        serial_stage1: timing_stats(&samples.serial_stage1)?,
        total,
        serial_total,
        total_token_per_sec: verified_tokens_per_sec(total_avg, 2),
        serial_total_token_per_sec: verified_tokens_per_sec(serial_total_avg, 2),
        total_avg_vs_full_batched_avg: total_avg / full_batched_avg_us,
        total_avg_vs_serial_total_avg: total_avg / serial_total_avg,
        diagnostics: split_timing_diagnostics(&samples)?,
        first_prediction: samples.first_prediction,
        first_serial_prediction: samples.first_serial_prediction,
    })
}

fn split_timing_diagnostics(samples: &SplitSampleSet) -> Result<SplitTimingDiagnostics> {
    Ok(SplitTimingDiagnostics {
        batched_total: timing_shape(&samples.total)?,
        batched_stage0: timing_shape(&samples.stage0)?,
        batched_stage1: timing_shape(&samples.stage1)?,
        serial_total: timing_shape(&samples.serial_total)?,
        serial_stage0: timing_shape(&samples.serial_stage0)?,
        serial_stage1: timing_shape(&samples.serial_stage1)?,
    })
}

#[derive(Debug)]
struct SampleSet {
    batched: Vec<Duration>,
    serial: Vec<Duration>,
    first_batched_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

impl SampleSet {
    fn batched_avg_us(&self) -> Result<f64> {
        Ok(timing_stats(&self.batched)?.avg_us)
    }
}

fn build_report(
    args: VerifySpanLocalArgs,
    full: FullModelSamples,
    split_inprocess_width2: Option<SplitInprocessReport>,
) -> Result<VerifySpanLocalReport> {
    let batched_width2 = timing_stats(&full.samples.batched)?;
    let serial_two_decode_mtp_n1 = timing_stats(&full.samples.serial)?;
    let batched_avg = batched_width2.avg_us;
    let serial_avg = serial_two_decode_mtp_n1.avg_us;
    Ok(VerifySpanLocalReport {
        mode: "verify-span-local",
        model_path: args.model_path,
        layer_end: args.layer_end,
        split_layer: args.split_layer,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        prompt_token_count: full.tokens.len(),
        verify_tokens: full.verify_tokens,
        warmup: args.warmup,
        iterations: args.iterations,
        batched_width2,
        serial_two_decode_mtp_n1,
        split_inprocess_width2,
        batched_avg_vs_serial_avg: batched_avg / serial_avg,
        batched_token_per_sec: verified_tokens_per_sec(batched_avg, 2),
        serial_token_per_sec: verified_tokens_per_sec(serial_avg, 2),
        first_batched_prediction: full.samples.first_batched_prediction,
        first_serial_prediction: full.samples.first_serial_prediction,
    })
}

fn timing_stats(samples: &[Duration]) -> Result<TimingStats> {
    if samples.is_empty() {
        bail!("cannot summarize empty timing samples");
    }
    let mut micros = samples.iter().map(Duration::as_micros).collect::<Vec<_>>();
    micros.sort_unstable();
    let total_us = micros.iter().sum::<u128>();
    let avg_us = total_us as f64 / micros.len() as f64;
    Ok(TimingStats {
        count: micros.len(),
        total_us,
        avg_us,
        min_us: micros[0],
        p50_us: percentile(&micros, 0.50),
        p95_us: percentile(&micros, 0.95),
        max_us: *micros.last().context("missing max timing sample")?,
    })
}

fn timing_shape(samples: &[Duration]) -> Result<TimingShape> {
    if samples.is_empty() {
        bail!("cannot summarize empty timing shape");
    }
    let split_at = (samples.len() / 2).max(1);
    let (first_half, second_half) = samples.split_at(split_at);
    let second_half = if second_half.is_empty() {
        first_half
    } else {
        second_half
    };
    let first_half_stats = timing_stats(first_half)?;
    let second_half_stats = timing_stats(second_half)?;
    let samples_us = samples.iter().map(Duration::as_micros).collect::<Vec<_>>();
    Ok(TimingShape {
        second_half_avg_vs_first_half_avg: second_half_stats.avg_us / first_half_stats.avg_us,
        first_half: first_half_stats,
        second_half: second_half_stats,
        first_sample_us: samples_us[0],
        last_sample_us: *samples_us.last().context("missing timing shape sample")?,
        samples_us,
    })
}

fn percentile(sorted_micros: &[u128], percentile: f64) -> u128 {
    let last_index = sorted_micros.len().saturating_sub(1);
    let index = (last_index as f64 * percentile).round() as usize;
    sorted_micros[index.min(last_index)]
}

fn verified_tokens_per_sec(avg_us: f64, token_count: usize) -> f64 {
    token_count as f64 * 1_000_000.0 / avg_us
}

fn model_description(config: &RuntimeConfig) -> String {
    format!(
        "layers={}..{} ctx={} n_gpu_layers={}",
        config.layer_start, config.layer_end, config.ctx_size, config.n_gpu_layers
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{percentile, timing_shape, timing_stats, verified_tokens_per_sec};

    #[test]
    fn timing_stats_sorts_and_summarizes_microseconds() {
        let stats = timing_stats(&[
            Duration::from_micros(30),
            Duration::from_micros(10),
            Duration::from_micros(20),
        ])
        .unwrap();

        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_us, 60);
        assert_eq!(stats.min_us, 10);
        assert_eq!(stats.p50_us, 20);
        assert_eq!(stats.max_us, 30);
    }

    #[test]
    fn percentile_clamps_to_last_sample() {
        assert_eq!(percentile(&[10, 20, 30], 1.0), 30);
    }

    #[test]
    fn token_rate_uses_verified_token_count() {
        assert_eq!(verified_tokens_per_sec(20_000.0, 2), 100.0);
    }

    #[test]
    fn timing_shape_reports_half_drift_in_sample_order() {
        let shape = timing_shape(&[
            Duration::from_micros(10),
            Duration::from_micros(20),
            Duration::from_micros(30),
            Duration::from_micros(50),
        ])
        .unwrap();

        assert_eq!(shape.first_sample_us, 10);
        assert_eq!(shape.last_sample_us, 50);
        assert_eq!(shape.samples_us, vec![10, 20, 30, 50]);
        assert_eq!(shape.first_half.avg_us, 15.0);
        assert_eq!(shape.second_half.avg_us, 40.0);
        assert_eq!(shape.second_half_avg_vs_first_half_avg, 40.0 / 15.0);
    }
}
