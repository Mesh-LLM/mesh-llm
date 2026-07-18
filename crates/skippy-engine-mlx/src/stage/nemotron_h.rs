//! Stateless Nemotron-H MoE blocks behind the shared stage-engine contract.

use std::{
    io::Write,
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use safemlx::{
    Array, Device, DeviceType, Dtype, Stream,
    memory::{active_memory, cache_memory, peak_memory, reset_peak_memory},
    module::{Module, ModuleParameters, ModuleParametersExt},
};
use safemlx_lm::{
    models::nemotron_h::{BlockInput, LayerBlockType, TransformerBlock, get_nemotron_h_model_args},
    weights::{StrictLoadConfig, StrictLoadReport, load_safetensors_dir_strict},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_engine::{
    StageActivation, StageEngine, StageEngineInfo, StageExecutionKind, StageExecutionOutput,
    StageExecutionRequest,
};
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
    encode_f32_activation_payload, recv_ready, recv_reply, write_stage_message,
};
use skippy_server::engine_transport::{EngineStageServerOptions, serve_stage_engine_until};

use crate::derived::{
    nemotron_h_validation_values, validate_nemotron_h_moe_stage_output,
    validate_nemotron_h_moe_stage_output_for_tokens,
};

use super::{MlxComputeDtype, MlxStageEngine, MlxStageEngineConfig, array_activation};

/// Tolerance-aware direct-block versus shared-stage-contract evidence.
#[derive(Clone, Debug, Serialize)]
pub struct MlxNemotronHStageValidationReport {
    pub model_dir: PathBuf,
    pub layer: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub output_is_finite: bool,
    pub direct_output_f32_sha256: String,
    pub stage_output_f32_sha256: String,
    pub output_within_tolerance: bool,
    pub cross_session_stable: bool,
    pub session_reset_stable: bool,
    pub executions_compared: usize,
    pub max_abs_diff: f32,
    pub max_relative_diff_for_reference_magnitude_above_atol: f32,
    pub cross_session_max_abs_diff: f32,
    pub cross_session_max_relative_diff_for_reference_magnitude_above_atol: f32,
    pub reset_max_abs_diff: f32,
    pub reset_max_relative_diff_for_reference_magnitude_above_atol: f32,
    pub comparison_atol: f32,
    pub comparison_rtol: f32,
    pub mlx_active_memory_bytes: usize,
    pub mlx_cache_memory_bytes: usize,
    pub mlx_peak_memory_bytes: usize,
}

/// Evidence from an intentionally unnecessary two-stage binary-wire chain.
#[derive(Clone, Debug, Serialize)]
pub struct MlxNemotronHWireValidationReport {
    pub model_dir: PathBuf,
    pub layer: usize,
    pub token_count: usize,
    pub wire_dtype: String,
    pub input_shape: Vec<usize>,
    pub captured_shape: Vec<usize>,
    pub output_is_finite: bool,
    pub output_within_tolerance: bool,
    pub max_abs_diff: f32,
    pub max_relative_diff_for_reference_magnitude_above_atol: f32,
    pub comparison_atol: f32,
    pub comparison_rtol: f32,
    pub predicted_sentinel: i32,
    pub forwarded_kind: String,
    pub forwarded_session_id: u64,
    pub downstream_reset_session_id: u64,
    pub mlx_active_memory_bytes: usize,
    pub mlx_cache_memory_bytes: usize,
    pub mlx_peak_memory_bytes: usize,
}

/// Compares direct safemlx execution with execution through `StageEngine`.
pub fn validate_nemotron_h_stage_engine(
    model_dir: impl AsRef<Path>,
    layer: usize,
) -> Result<MlxNemotronHStageValidationReport> {
    let model_dir = model_dir.as_ref();
    const ATOL: f32 = 1.0e-4;
    const RTOL: f32 = 1.0e-4;
    let (direct, direct_values) = validate_nemotron_h_moe_stage_output(model_dir, layer)?;
    reset_peak_memory()?;
    let engine = MlxStageEngine::spawn(MlxStageEngineConfig {
        model_dir: model_dir.to_path_buf(),
        model_id: "nemotron-h-stage-validation".to_string(),
        stage_index: 1,
        layer_start: u32::try_from(layer)?,
        layer_end: u32::try_from(layer.checked_add(1).context("layer index overflow")?)?,
        compute_dtype: MlxComputeDtype::Bf16,
        weight_quantization: None,
        ctx_size: Some(1),
    })?;
    let width = usize::try_from(engine.info().activation_width)?;
    let values = nemotron_h_validation_values(i32::try_from(width)?, 1)?;
    let first = execute_validation_input(&engine, 1, width, &values)?;
    let second_session = execute_validation_input(&engine, 2, width, &values)?;
    engine.reset_session(2)?;
    engine.reset_session(1)?;
    let after_reset = execute_validation_input(&engine, 1, width, &values)?;
    let outputs = [first, second_session, after_reset];
    let output_is_finite = outputs
        .iter()
        .flat_map(StageActivation::values)
        .all(f32::is_finite);
    ensure!(
        output_is_finite,
        "Nemotron-H stage-engine output contains non-finite values"
    );
    let stage_output_f32_sha256 = bytes_sha256(&outputs[0].f32_le_bytes);
    let mut comparison = OutputComparison::accumulator();
    for output in &outputs {
        comparison.include(compare_outputs(
            &direct_values,
            &output.values(),
            ATOL,
            RTOL,
        )?);
    }
    let cross_session_comparison =
        compare_outputs(&outputs[0].values(), &outputs[1].values(), ATOL, RTOL)?;
    let cross_session_stable = cross_session_comparison.all_close;
    ensure!(
        cross_session_stable,
        "Nemotron-H output changed across session IDs: max_abs={} max_relative_for_reference_magnitude_above_atol={} atol={ATOL} rtol={RTOL}",
        cross_session_comparison.max_abs,
        cross_session_comparison.max_relative_for_reference_magnitude_above_atol,
    );
    let reset_comparison = compare_outputs(&outputs[0].values(), &outputs[2].values(), ATOL, RTOL)?;
    let session_reset_stable = reset_comparison.all_close;
    ensure!(
        session_reset_stable,
        "Nemotron-H output changed after session reset: max_abs={} max_relative_for_reference_magnitude_above_atol={} atol={ATOL} rtol={RTOL}",
        reset_comparison.max_abs,
        reset_comparison.max_relative_for_reference_magnitude_above_atol,
    );
    let output_within_tolerance = comparison.all_close;
    ensure!(
        output_within_tolerance,
        "Nemotron-H stage-engine output differs from direct block execution: max_abs={} max_relative_for_reference_magnitude_above_atol={} atol={ATOL} rtol={RTOL}",
        comparison.max_abs,
        comparison.max_relative_for_reference_magnitude_above_atol,
    );
    Ok(MlxNemotronHStageValidationReport {
        model_dir: model_dir.to_path_buf(),
        layer,
        input_shape: vec![1, 1, width],
        output_shape: vec![1, outputs[0].token_count, outputs[0].width],
        output_is_finite,
        direct_output_f32_sha256: direct.output_f32_sha256,
        stage_output_f32_sha256,
        output_within_tolerance,
        cross_session_stable,
        session_reset_stable,
        executions_compared: outputs.len(),
        max_abs_diff: comparison.max_abs,
        max_relative_diff_for_reference_magnitude_above_atol: comparison
            .max_relative_for_reference_magnitude_above_atol,
        cross_session_max_abs_diff: cross_session_comparison.max_abs,
        cross_session_max_relative_diff_for_reference_magnitude_above_atol:
            cross_session_comparison.max_relative_for_reference_magnitude_above_atol,
        reset_max_abs_diff: reset_comparison.max_abs,
        reset_max_relative_diff_for_reference_magnitude_above_atol: reset_comparison
            .max_relative_for_reference_magnitude_above_atol,
        comparison_atol: ATOL,
        comparison_rtol: RTOL,
        mlx_active_memory_bytes: active_memory()?,
        mlx_cache_memory_bytes: cache_memory()?,
        mlx_peak_memory_bytes: peak_memory()?,
    })
}

/// Sends one deterministic residual through the real binary stage wire and a
/// synthetic final capture stage, then compares the captured residual with a
/// direct safemlx block execution.
pub fn validate_nemotron_h_binary_wire(
    model_dir: impl AsRef<Path>,
    layer: usize,
    wire_dtype: WireActivationDType,
) -> Result<MlxNemotronHWireValidationReport> {
    validate_nemotron_h_binary_wire_tokens(model_dir, layer, wire_dtype, 1)
}

/// Runs [`validate_nemotron_h_binary_wire`] with a configurable prefill size.
pub fn validate_nemotron_h_binary_wire_tokens(
    model_dir: impl AsRef<Path>,
    layer: usize,
    wire_dtype: WireActivationDType,
    token_count: usize,
) -> Result<MlxNemotronHWireValidationReport> {
    const PREDICTED_SENTINEL: i32 = 424_242;
    let model_dir = model_dir.as_ref();
    ensure!(token_count > 0, "binary wire validation needs tokens");
    let token_count_u32 = u32::try_from(token_count).context("token count exceeds u32")?;
    let (atol, rtol) = wire_tolerances(wire_dtype)?;
    let (_direct, direct_values) =
        validate_nemotron_h_moe_stage_output_for_tokens(model_dir, layer, token_count)?;
    reset_peak_memory()?;
    let engine = Arc::new(MlxStageEngine::spawn(MlxStageEngineConfig {
        model_dir: model_dir.to_path_buf(),
        model_id: "nemotron-h-wire-validation".to_string(),
        stage_index: 1,
        layer_start: u32::try_from(layer)?,
        layer_end: u32::try_from(layer.checked_add(1).context("layer index overflow")?)?,
        compute_dtype: MlxComputeDtype::Bf16,
        weight_quantization: None,
        ctx_size: Some(token_count_u32),
    })?);
    let width = usize::try_from(engine.info().activation_width)?;
    let values = nemotron_h_validation_values(i32::try_from(width)?, token_count)?;
    let (captured_tx, captured_rx) = mpsc::channel();
    let (reset_tx, reset_rx) = mpsc::channel();
    let capture = Arc::new(CaptureStageEngine::new(
        engine.info(),
        PREDICTED_SENTINEL,
        token_count,
        captured_tx,
        reset_tx,
    )?);
    let (capture_server, capture_addr, capture_ready) = WireServer::spawn_ready(
        capture,
        EngineStageServerOptions {
            bind_addr: "127.0.0.1:0".parse()?,
            downstream_addr: None,
            wire_dtype,
        },
    )?;
    drop(capture_ready);
    let (stage_server, _stage_addr, mut client) = WireServer::spawn_ready(
        engine,
        EngineStageServerOptions {
            bind_addr: "127.0.0.1:0".parse()?,
            downstream_addr: Some(capture_addr),
            wire_dtype,
        },
    )?;
    let input = StageActivation::from_values(token_count, width, &values)?;
    let message = wire_validation_message(&input, wire_dtype)?;
    write_stage_message(&mut client, &message, wire_dtype)?;
    client.flush().ok();
    let reply = recv_reply(&mut client)?;
    ensure!(
        reply.kind == WireReplyKind::PredictedToken && reply.predicted == PREDICTED_SENTINEL,
        "binary wire chain returned the wrong final reply"
    );
    let captured = captured_rx
        .recv_timeout(Duration::from_secs(5))
        .context("capture stage did not receive the Nemotron-H residual")?;
    let stop = StageWireMessage::stop_with_identity(wire_dtype, 1, 1);
    write_stage_message(&mut client, &stop, wire_dtype)?;
    client.flush().ok();
    ensure!(
        recv_reply(&mut client)?.kind == WireReplyKind::Ack,
        "binary wire chain stop did not return ACK"
    );
    let downstream_reset_session_id = reset_rx
        .recv_timeout(Duration::from_secs(5))
        .context("capture stage did not receive the forwarded session reset")?;
    ensure!(
        downstream_reset_session_id == 1,
        "capture stage reset the wrong session"
    );
    let mlx_active_memory_bytes = active_memory()?;
    let mlx_cache_memory_bytes = cache_memory()?;
    let mlx_peak_memory_bytes = peak_memory()?;
    drop(client);
    stage_server.stop()?;
    capture_server.stop()?;

    let captured_values = captured.values();
    let output_is_finite = captured_values.iter().copied().all(f32::is_finite);
    ensure!(output_is_finite, "binary wire output is not finite");
    let comparison = compare_outputs(&direct_values, &captured_values, atol, rtol)?;
    ensure!(
        comparison.all_close,
        "binary wire output differs from direct execution: dtype={} max_abs={} max_relative_for_reference_magnitude_above_atol={} atol={atol} rtol={rtol}",
        wire_dtype_label(wire_dtype)?,
        comparison.max_abs,
        comparison.max_relative_for_reference_magnitude_above_atol,
    );
    Ok(MlxNemotronHWireValidationReport {
        model_dir: model_dir.to_path_buf(),
        layer,
        token_count,
        wire_dtype: wire_dtype_label(wire_dtype)?.to_string(),
        input_shape: vec![1, token_count, width],
        captured_shape: vec![1, captured.token_count, captured.width],
        output_is_finite,
        output_within_tolerance: comparison.all_close,
        max_abs_diff: comparison.max_abs,
        max_relative_diff_for_reference_magnitude_above_atol: comparison
            .max_relative_for_reference_magnitude_above_atol,
        comparison_atol: atol,
        comparison_rtol: rtol,
        predicted_sentinel: PREDICTED_SENTINEL,
        forwarded_kind: "prefill_final".to_string(),
        forwarded_session_id: 1,
        downstream_reset_session_id,
        mlx_active_memory_bytes,
        mlx_cache_memory_bytes,
        mlx_peak_memory_bytes,
    })
}

fn wire_tolerances(wire_dtype: WireActivationDType) -> Result<(f32, f32)> {
    match wire_dtype {
        WireActivationDType::F32 => Ok((1.0e-4, 1.0e-4)),
        WireActivationDType::F16 => Ok((5.0e-4, 1.0e-3)),
        other => bail!("Nemotron-H binary-wire validation does not support {other:?}"),
    }
}

fn wire_dtype_label(wire_dtype: WireActivationDType) -> Result<&'static str> {
    match wire_dtype {
        WireActivationDType::F32 => Ok("f32"),
        WireActivationDType::F16 => Ok("f16"),
        other => bail!("Nemotron-H binary-wire validation does not support {other:?}"),
    }
}

fn wire_validation_message(
    input: &StageActivation,
    wire_dtype: WireActivationDType,
) -> Result<StageWireMessage> {
    let kind = WireMessageKind::PrefillFinalEmbd;
    let token_count = i32::try_from(input.token_count).context("token count exceeds i32")?;
    let mut state = StageStateHeader::new(kind, wire_dtype);
    state.current_token = 0;
    state.prompt_token_count = token_count;
    state.source_stage_index = 0;
    Ok(StageWireMessage {
        kind,
        pos_start: 0,
        token_count,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![0; input.token_count],
        positions: (0..token_count).collect(),
        activation: encode_f32_activation_payload(
            wire_dtype,
            token_count,
            i32::try_from(input.width)?,
            &input.f32_le_bytes,
        )?,
        raw_bytes: Vec::new(),
    })
}

fn execute_validation_input(
    engine: &MlxStageEngine,
    session_id: u64,
    width: usize,
    values: &[f32],
) -> Result<StageActivation> {
    engine
        .execute(StageExecutionRequest {
            session_id,
            kind: StageExecutionKind::Prefill,
            token_ids: vec![0],
            positions: vec![0],
            input: Some(StageActivation::from_values(1, width, values)?),
            sampling: None,
        })?
        .activation
        .context("Nemotron-H internal stage returned no activation")
}

struct CaptureStageEngine {
    info: StageEngineInfo,
    predicted_sentinel: i32,
    expected_token_ids: Vec<i32>,
    expected_positions: Vec<i32>,
    captured: mpsc::Sender<StageActivation>,
    reset: mpsc::Sender<u64>,
}

impl CaptureStageEngine {
    fn new(
        source: &StageEngineInfo,
        predicted_sentinel: i32,
        token_count: usize,
        captured: mpsc::Sender<StageActivation>,
        reset: mpsc::Sender<u64>,
    ) -> Result<Self> {
        let stage_index = source
            .stage_index
            .checked_add(1)
            .context("capture stage index overflow")?;
        let total_layers = source
            .layer_end
            .checked_add(1)
            .context("capture layer index overflow")?;
        let info = StageEngineInfo {
            engine: "capture".to_string(),
            model_id: source.model_id.clone(),
            stage_index,
            layer_start: source.layer_end,
            layer_end: total_layers,
            total_layers,
            activation_width: source.activation_width,
        };
        info.validate()?;
        let token_count = i32::try_from(token_count).context("token count exceeds i32")?;
        Ok(Self {
            info,
            predicted_sentinel,
            expected_token_ids: vec![0; usize::try_from(token_count)?],
            expected_positions: (0..token_count).collect(),
            captured,
            reset,
        })
    }
}

impl StageEngine for CaptureStageEngine {
    fn info(&self) -> &StageEngineInfo {
        &self.info
    }

    fn execute(&self, request: StageExecutionRequest) -> Result<StageExecutionOutput> {
        ensure!(
            request.kind == StageExecutionKind::PrefillFinal,
            "capture stage received the wrong execution kind"
        );
        ensure!(
            request.session_id == 1,
            "capture stage received the wrong session"
        );
        ensure!(
            request.token_ids == self.expected_token_ids,
            "capture stage received the wrong token sideband"
        );
        ensure!(
            request.positions == self.expected_positions,
            "capture stage received the wrong position sideband"
        );
        let input = request
            .input
            .context("capture stage requires a residual activation")?;
        self.captured
            .send(input)
            .map_err(|_| anyhow!("wire validation capture receiver was dropped"))?;
        Ok(StageExecutionOutput {
            activation: None,
            predicted_tokens: vec![self.predicted_sentinel],
        })
    }

    fn reset_session(&self, session_id: u64) -> Result<()> {
        self.reset
            .send(session_id)
            .map_err(|_| anyhow!("wire validation reset receiver was dropped"))
    }
}

struct WireServer {
    shutdown: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<Result<()>>>,
}

impl WireServer {
    fn spawn(engine: Arc<dyn StageEngine>, options: EngineStageServerOptions) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = Arc::clone(&shutdown);
        let join =
            thread::spawn(move || serve_stage_engine_until(engine, options, thread_shutdown));
        Self {
            shutdown,
            join: Some(join),
        }
    }

    fn spawn_ready(
        engine: Arc<dyn StageEngine>,
        options: EngineStageServerOptions,
    ) -> Result<(Self, SocketAddr, TcpStream)> {
        const BIND_ATTEMPTS: usize = 3;
        let mut last_error = None;
        for _ in 0..BIND_ATTEMPTS {
            let addr = reserve_loopback_addr()?;
            let mut attempt_options = options.clone();
            attempt_options.bind_addr = addr;
            let server = Self::spawn(Arc::clone(&engine), attempt_options);
            match connect_ready(addr) {
                Ok(client) => return Ok((server, addr, client)),
                Err(connect_error) => {
                    let server_error = server.stop().err();
                    last_error = Some(match server_error {
                        Some(server_error) => anyhow!(
                            "connect wire stage at {addr}: {connect_error:#}; server failed: {server_error:#}"
                        ),
                        None => connect_error,
                    });
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("wire stage did not start")))
            .context("start binary wire validation server")
    }

    fn stop(mut self) -> Result<()> {
        self.finish()
    }

    fn finish(&mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        let Some(join) = self.join.take() else {
            return Ok(());
        };
        join.join()
            .map_err(|_| anyhow!("binary wire validation server panicked"))?
    }
}

impl Drop for WireServer {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn reserve_loopback_addr() -> Result<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn connect_ready(addr: SocketAddr) -> Result<TcpStream> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let error = match TcpStream::connect(addr) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream
                    .set_read_timeout(Some(Duration::from_millis(250)))
                    .ok();
                match recv_ready(&mut stream) {
                    Ok(()) => {
                        stream.set_read_timeout(None).ok();
                        return Ok(stream);
                    }
                    Err(error) => anyhow!(error).context("receive wire ready handshake"),
                }
            }
            Err(error) => anyhow!(error).context("connect TCP socket"),
        };
        if Instant::now() >= deadline {
            return Err(error).with_context(|| format!("connect wire stage at {addr}"));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

struct OutputComparison {
    all_close: bool,
    max_abs: f32,
    max_relative_for_reference_magnitude_above_atol: f32,
}

impl OutputComparison {
    const fn accumulator() -> Self {
        Self {
            all_close: true,
            max_abs: 0.0,
            max_relative_for_reference_magnitude_above_atol: 0.0,
        }
    }

    fn include(&mut self, next: Self) {
        self.all_close = self.all_close && next.all_close;
        self.max_abs = self.max_abs.max(next.max_abs);
        self.max_relative_for_reference_magnitude_above_atol = self
            .max_relative_for_reference_magnitude_above_atol
            .max(next.max_relative_for_reference_magnitude_above_atol);
    }
}

fn compare_outputs(
    direct: &[f32],
    staged: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<OutputComparison> {
    ensure!(
        direct.len() == staged.len(),
        "direct and staged output lengths differ"
    );
    let mut comparison = OutputComparison::accumulator();
    for (&expected, &actual) in direct.iter().zip(staged) {
        let abs = (expected - actual).abs();
        let relative = if expected.abs() > atol {
            abs / expected.abs()
        } else {
            0.0
        };
        comparison.max_abs = comparison.max_abs.max(abs);
        comparison.max_relative_for_reference_magnitude_above_atol = comparison
            .max_relative_for_reference_magnitude_above_atol
            .max(relative);
        comparison.all_close = comparison.all_close && abs <= atol + rtol * expected.abs();
    }
    Ok(comparison)
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) struct NemotronHMoeStage {
    block: TransformerBlock,
    stream: Stream,
    compute_dtype: Dtype,
    ctx_size: Option<usize>,
    info: StageEngineInfo,
}

impl NemotronHMoeStage {
    pub(super) fn load(config: MlxStageEngineConfig) -> Result<Self> {
        ensure!(
            config.compute_dtype == MlxComputeDtype::Bf16,
            "Nemotron-H staged execution is currently validated only with BF16 compute"
        );
        ensure!(
            config.weight_quantization.is_none(),
            "Nemotron-H stages must be loaded from an already-derived checkpoint"
        );
        ensure!(
            config.layer_end == config.layer_start.saturating_add(1),
            "Nemotron-H staged execution currently requires exactly one layer"
        );
        let args = get_nemotron_h_model_args(&config.model_dir)?;
        let layer = usize::try_from(config.layer_start)?;
        ensure!(
            args.layer_block_types()?.get(layer) == Some(&LayerBlockType::Moe),
            "Nemotron-H staged execution currently supports only stateless MoE layers"
        );
        let info = StageEngineInfo {
            engine: "mlx".to_string(),
            model_id: config.model_id,
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            total_layers: u32::try_from(args.num_hidden_layers)?,
            activation_width: u32::try_from(args.hidden_size)?,
        };
        info.validate()?;
        ensure!(
            !info.is_first() && !info.is_final(),
            "Nemotron-H MoE proof stages must be internal residual stages"
        );

        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut block = TransformerBlock::new(&args, layer, &stream)?;
        let load_config =
            StrictLoadConfig::default().strip_prefix(format!("model.layers.{layer}."));
        let mut load_report = StrictLoadReport::default();
        load_safetensors_dir_strict(
            &mut block,
            &config.model_dir,
            &weights_stream,
            &load_config,
            &mut load_report,
        )?;
        load_report.finish(&block, &load_config)?;
        block.copy_to_stream(&stream)?;
        stream.synchronize()?;
        tracing::info!(
            model = %info.model_id,
            stage = info.stage_index,
            layer = info.layer_start,
            tensors = block.parameters().flatten().len(),
            weight_quantization = "checkpoint",
            "MLX Nemotron-H MoE stage loaded",
        );
        Ok(Self {
            block,
            stream,
            compute_dtype: config.compute_dtype.mlx(),
            ctx_size: config.ctx_size.map(usize::try_from).transpose()?,
            info,
        })
    }

    pub(super) const fn info(&self) -> &StageEngineInfo {
        &self.info
    }

    pub(super) fn execute(
        &mut self,
        request: StageExecutionRequest,
    ) -> Result<StageExecutionOutput> {
        if request.kind == StageExecutionKind::Verify {
            bail!("MLX Nemotron-H stage verification is not implemented yet");
        }
        if request
            .sampling
            .as_ref()
            .is_some_and(|sampling| sampling.enabled())
        {
            bail!("MLX staged execution currently supports greedy sampling only");
        }
        ensure!(!request.token_ids.is_empty(), "stage request has no tokens");
        let input = request
            .input
            .as_ref()
            .context("Nemotron-H MoE stage requires residual input")?;
        ensure!(
            input.token_count == request.token_ids.len(),
            "input activation token count does not match token sideband"
        );
        ensure!(
            input.width == self.info.activation_width as usize,
            "input activation width mismatch"
        );
        if let Some(ctx_size) = self.ctx_size {
            ensure!(
                input.token_count <= ctx_size,
                "MLX stage context limit {ctx_size} exceeded by {} tokens",
                input.token_count
            );
        }

        let hidden = Array::from_slice(
            &input.values(),
            &[
                1,
                i32::try_from(input.token_count)?,
                i32::try_from(input.width)?,
            ],
        )
        .as_dtype(self.compute_dtype, &self.stream)?;
        let output = self.block.forward(
            BlockInput {
                x: &hidden,
                mask: None,
                cache: None,
            },
            &self.stream,
        )?;
        Ok(StageExecutionOutput {
            activation: Some(array_activation(&output, &self.stream)?),
            predicted_tokens: Vec::new(),
        })
    }

    pub(super) fn reset_session(&mut self, _session_id: u64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_comparison_accepts_roundoff_and_rejects_drift() {
        let close = compare_outputs(
            &[1.0, -0.5, 0.0],
            &[1.000_000_1, -0.5, 1.0e-5],
            1.0e-4,
            1.0e-4,
        )
        .unwrap();
        assert!(close.all_close);
        assert!(close.max_abs <= 1.0e-5);
        assert!(close.max_relative_for_reference_magnitude_above_atol < 1.0e-6);

        let drift = compare_outputs(&[1.0, -0.5], &[1.01, -0.5], 1.0e-4, 1.0e-4).unwrap();
        assert!(!drift.all_close);
        assert!(drift.max_abs > 0.009);
    }
}
