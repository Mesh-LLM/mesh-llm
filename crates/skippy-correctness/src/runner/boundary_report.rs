use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use sha2::{Digest, Sha256};
use skippy_runtime::{
    ActivationDesc, GGML_TYPE_F16, MtpSource, RuntimeActivationDType, RuntimeActivationLayout,
    RuntimeConfig, StageModel, TokenSignal,
};

use crate::{
    cli::{BoundaryReportArgs, RuntimeArgs},
    report::{
        BaselineReport, BoundaryDTypeReport, BoundaryDTypeScanSplit, BoundaryTensorReport,
        FrameDescReport, TokenSignalReport,
    },
};

use super::{
    native_mtp::emit_report,
    single_step::run_full_model_decode,
    stage_execution::{
        FullModelResult, PackageStageSpec, ensure_matches, parse_split_list, runtime_flash_attn,
        runtime_load_mode, runtime_model_identity, stage_model_resolution, status,
    },
};

/// Logit-level parity tolerance for the final-stage token signal. Token ids
/// must match exactly; margin and entropy are compared within this absolute
/// tolerance to absorb backend scheduling noise across graph partitions.
const SIGNAL_ABS_TOLERANCE: f32 = 1e-3;

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

const GGML_TYPE_F32_I32: i32 = 0;
const GGML_TYPE_F16_I32: i32 = 1;
const GGML_TYPE_BF16_I32: i32 = 30;

pub fn boundary_report(args: BoundaryReportArgs) -> Result<()> {
    let splits = parse_split_list(&args.splits)?;
    if splits.is_empty() {
        bail!("no splits requested");
    }
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let baseline_signal = token_signal_report(&baseline.token_signal);

    let mut results = Vec::with_capacity(splits.len());
    for split_layer in &splits {
        let split_layer = *split_layer;
        if split_layer == 0 || split_layer >= args.runtime.layer_end {
            bail!(
                "split layer {split_layer} must be greater than zero and less than layer_end {}",
                args.runtime.layer_end
            );
        }
        results.push(probe_split(&args.runtime, &model_identity, &baseline, split_layer));
    }

    let mismatch_count = results
        .iter()
        .filter(|result| {
            !result.desc_matches_tensor || result.predicted_token_matches != Some(true)
        })
        .count();
    // Signal drift (margin/entropy vs the one-graph baseline) is reported but
    // not gated: staged graphs legitimately reassociate floating-point ops,
    // and the tolerance for that drift is not yet characterized per family.
    let signal_mismatch_count = results
        .iter()
        .filter(|result| result.signal_matches != Some(true))
        .count();
    let error_count = results
        .iter()
        .filter(|result| result.error.is_some())
        .count();
    let f32_boundary_count = results
        .iter()
        .filter(|result| {
            result
                .boundary_tensor
                .is_some_and(|tensor| tensor.ggml_type == GGML_TYPE_F32_I32)
        })
        .count();
    let non_f32_boundary_count = results.len() - f32_boundary_count;
    let matches = mismatch_count == 0;
    let report = BoundaryDTypeReport {
        mode: "boundary-dtype",
        status: status(matches),
        model_identity,
        layer_end: args.runtime.layer_end,
        prompt: args.runtime.prompt.clone(),
        splits,
        split_count: results.len(),
        f32_boundary_count,
        non_f32_boundary_count,
        mismatch_count,
        signal_mismatch_count,
        error_count,
        matches,
        baseline: BaselineReport {
            token_id: baseline.token_id,
            predicted_token: baseline.predicted_token,
            second_predicted_token: baseline.second_predicted_token,
        },
        baseline_signal,
        results,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(matches, args.allow_mismatch)?;
    Ok(())
}

/// Probes one split cut in-process: stage 0 (embeddings..split) decodes one
/// token, its native boundary tensor type is compared against the activation
/// frame descriptor, and the frame is transported into stage 1 (split..end)
/// whose final token signal is compared against the full-model baseline.
/// Probe failures are recorded per split instead of aborting the scan.
fn probe_split(
    runtime: &RuntimeArgs,
    model_identity: &ModelIdentity,
    baseline: &FullModelResult,
    split_layer: u32,
) -> BoundaryDTypeScanSplit {
    match run_split_probe(runtime, model_identity, baseline, split_layer) {
        Ok(split) => split,
        Err(error) => BoundaryDTypeScanSplit {
            split_layer,
            layer_start: 0,
            layer_end: runtime.layer_end,
            boundary_tensor: None,
            frame_desc: None,
            boundary_payload_sha256: None,
            desc_matches_tensor: false,
            predicted_token: None,
            signal: None,
            predicted_token_matches: None,
            signal_matches: None,
            error: Some(format!("{error:#}")),
        },
    }
}

fn run_split_probe(
    runtime: &RuntimeArgs,
    model_identity: &ModelIdentity,
    baseline: &FullModelResult,
    split_layer: u32,
) -> Result<BoundaryDTypeScanSplit> {
    let stage0_spec = PackageStageSpec {
        topology_id: "correctness-boundary-dtype",
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: split_layer,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_spec = PackageStageSpec {
        topology_id: "correctness-boundary-dtype",
        stage_id: "stage-1",
        stage_index: 1,
        layer_start: split_layer,
        layer_end: runtime.layer_end,
        include_embeddings: false,
        include_output: true,
    };
    let stage0_path = stage_model_resolution(
        &runtime.model,
        runtime.stage_model.as_ref(),
        runtime.stage_load_mode,
        model_identity,
        stage0_spec,
    )?
    .path;
    let stage1_path = stage_model_resolution(
        &runtime.model,
        runtime.stage_model.as_ref(),
        runtime.stage_load_mode,
        model_identity,
        stage1_spec,
    )?
    .path;

    let stage0 = StageModel::open(
        &stage0_path,
        &stage_config(runtime, 0, 0, split_layer, true, false),
    )
    .with_context(|| format!("failed to open stage 0 for split {split_layer}"))?;
    let tokens = stage0
        .tokenize(&runtime.prompt, true)
        .with_context(|| format!("failed to tokenize prompt for split {split_layer}"))?;
    let token_id = *tokens
        .first()
        .with_context(|| format!("prompt produced no tokens for split {split_layer}"))?;
    let mut session0 = stage0
        .create_session()
        .with_context(|| format!("failed to create stage 0 session for split {split_layer}"))?;
    let (_predicted, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .with_context(|| format!("stage 0 failed to decode for split {split_layer}"))?;
    let boundary_tensor = session0
        .boundary_tensor_info()
        .with_context(|| format!("failed to read boundary tensor info for split {split_layer}"))?
        .map(|info| BoundaryTensorReport {
            ggml_type: info.ggml_type,
            ggml_type_name: info.ggml_type_name(),
            ne: info.ne,
            element_size: info.element_size,
        });
    let frame_desc = FrameDescReport {
        dtype: activation_dtype_name(boundary.desc.dtype),
        dtype_value: boundary.desc.dtype as i32,
        layout: activation_layout_name(boundary.desc.layout),
        token_count: boundary.desc.token_count,
        payload_bytes: boundary.desc.payload_bytes,
    };
    let boundary_payload_sha256 = Some(hex_sha256(&boundary.payload));

    let stage1 = StageModel::open(
        &stage1_path,
        &stage_config(runtime, 1, split_layer, runtime.layer_end, false, true),
    )
    .with_context(|| format!("failed to open stage 1 for split {split_layer}"))?;
    let mut session1 = stage1
        .create_session()
        .with_context(|| format!("failed to create stage 1 session for split {split_layer}"))?;
    let (predicted, _final_frame) = session1
        .decode_step_frame(token_id, Some(&boundary), 0)
        .with_context(|| format!("stage 1 failed to decode boundary for split {split_layer}"))?;
    let signal = token_signal_report(
        &session1
            .last_token_signal()
            .with_context(|| format!("failed to read stage 1 token signal for split {split_layer}"))?,
    );

    let desc_matches_tensor = boundary_matches(&boundary.desc, boundary_tensor.as_ref());
    let predicted_token_matches = Some(predicted == baseline.predicted_token);
    let signal_matches = Some(signals_within_tolerance(&baseline.token_signal, &signal));
    Ok(BoundaryDTypeScanSplit {
        split_layer,
        layer_start: 0,
        layer_end: runtime.layer_end,
        boundary_tensor,
        frame_desc: Some(frame_desc),
        boundary_payload_sha256,
        desc_matches_tensor,
        predicted_token: Some(predicted),
        signal: Some(signal),
        predicted_token_matches,
        signal_matches,
        error: None,
    })
}

fn stage_config(
    runtime: &RuntimeArgs,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> RuntimeConfig {
    RuntimeConfig {
        stage_index,
        layer_start,
        layer_end,
        ctx_size: runtime.ctx_size,
        lane_count: 1,
        n_batch: runtime.n_batch,
        n_ubatch: runtime.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: runtime.n_gpu_layers,
        mmap: None,
        mlock: false,
        selected_backend_device: None,
        load_mode: runtime_load_mode(runtime.stage_load_mode),
        projector_path: None,
        include_embeddings,
        include_output,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: true,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: runtime_flash_attn(runtime.flash_attn),
    }
}

/// The wire descriptor must describe the native boundary tensor exactly:
/// same element type and a payload that covers the full tensor extent.
fn boundary_matches(desc: &ActivationDesc, tensor: Option<&BoundaryTensorReport>) -> bool {
    let Some(tensor) = tensor else {
        return false;
    };
    let desc_type = match desc.dtype {
        RuntimeActivationDType::F32 => GGML_TYPE_F32_I32,
        RuntimeActivationDType::F16 => GGML_TYPE_F16_I32,
        RuntimeActivationDType::Bf16 => GGML_TYPE_BF16_I32,
        RuntimeActivationDType::Unknown => -1,
    };
    if desc_type != tensor.ggml_type {
        return false;
    }
    let elements: i64 = tensor.ne.iter().product();
    let tensor_bytes = u64::try_from(elements.saturating_mul(i64::from(tensor.element_size)))
        .unwrap_or(u64::MAX);
    desc.payload_bytes == tensor_bytes
}

fn signals_within_tolerance(baseline: &TokenSignal, probe: &TokenSignalReport) -> bool {
    baseline.top_token == probe.top_token
        && baseline.second_token == probe.second_token
        && (baseline.margin - probe.margin).abs() <= SIGNAL_ABS_TOLERANCE
        && (baseline.entropy - probe.entropy).abs() <= SIGNAL_ABS_TOLERANCE
}

fn token_signal_report(signal: &TokenSignal) -> TokenSignalReport {
    TokenSignalReport {
        entropy: signal.entropy,
        top_logprob: signal.top_logprob,
        second_logprob: signal.second_logprob,
        margin: signal.margin,
        top_token: signal.top_token,
        second_token: signal.second_token,
    }
}

fn activation_dtype_name(dtype: RuntimeActivationDType) -> &'static str {
    match dtype {
        RuntimeActivationDType::F32 => "f32",
        RuntimeActivationDType::F16 => "f16",
        RuntimeActivationDType::Bf16 => "bf16",
        RuntimeActivationDType::Unknown => "unknown",
    }
}

fn activation_layout_name(layout: RuntimeActivationLayout) -> &'static str {
    match layout {
        RuntimeActivationLayout::Opaque => "opaque",
        RuntimeActivationLayout::TokenMajor => "token-major",
    }
}
