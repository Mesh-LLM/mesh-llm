//! Wired-memory preflight for the in-process concurrent stage pair.
//!
//! The boundary-report lane opens stage 0 and stage 1 as two concurrent llama
//! contexts in one process. On Apple Silicon their mmap'd MTL0 buffers are
//! wired (unpageable) while resident, so the pair's GPU demand is roughly
//! additive: stage-0 mapped weights + stage-1 mapped weights + per-context KV
//! and compute buffers. When that demand crosses the available budget, Metal
//! command buffers fail with `kIOGPUCommandBufferCallbackErrorOutOfMemory` at
//! execution time - which, before patch 0036(a), was silently swallowed
//! (garbage predictions with error_count = 0; see
//! RESEARCH/GLM45_AIR_METAL_OOM_ROOT_CAUSE_2026_08_29.md).
//!
//! This module estimates the pair's demand from the GGUF layer layout and
//! refuses to open the stages when it cannot fit the available budget,
//! turning a silent-corruption failure mode into a per-split, human-readable
//! error.
//!
//! Calibration note: the estimator's constants (per-context overhead,
//! conservative side-tensor placement) are deliberately conservative; the
//! decision threshold that separates the measured failing cuts (ngl99,
//! splits 30..33) from the passing configuration (ngl40, 45/45) comes from
//! the device's working-set budget, not from a hard-coded model of any one
//! machine. Hardware validation lives in the studio54 matrix run.

use std::path::Path;

use anyhow::{Context, Result, bail};

use model_artifact::gguf::{GgufLayerByteProfile, scan_gguf_layer_byte_profile};

/// Per-context overhead added to each stage's GPU weight bytes: KV cache and
/// compute buffers. Measured 76 + 75 MiB on glm45-air studio54 runs at ctx 32;
/// 512 MiB leaves margin for larger ctx/ubatch configurations and FA variants.
pub const PER_CONTEXT_OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// One stage's GPU residency as the lane configures it.
#[derive(Clone, Debug)]
pub struct StageResidency {
    /// Weight bytes this stage maps on the GPU (wired while resident):
    /// repeating layers in the GPU window plus the side tensors it keeps.
    pub gpu_weight_bytes: u64,
}

impl StageResidency {
    /// Estimated wired bytes for this stage's context.
    pub fn resident_bytes(&self) -> u64 {
        self.gpu_weight_bytes.saturating_add(PER_CONTEXT_OVERHEAD_BYTES)
    }
}

/// Compute one stage's GPU residency from the layer profile.
///
/// `i_gpu_start` mirrors llama.cpp's placement rule: with `n_gpu_layers` the
/// LAST `n_gpu_layers` of the full model go on the GPU, so a stage's GPU
/// layers are the intersection `[max(layer_start, i_gpu_start), layer_end)`.
/// Side tensors the stage keeps (embeddings / output head) are counted as
/// GPU-resident whenever included - an overestimate that only makes the
/// preflight more conservative (they follow layer-0 / last-layer placement).
pub fn stage_residency(
    profile: &GgufLayerByteProfile,
    layer_start: usize,
    layer_end: usize,
    i_gpu_start: usize,
    include_embeddings: bool,
    include_output: bool,
) -> StageResidency {
    let mut gpu_weight_bytes = 0u64;
    if include_embeddings {
        gpu_weight_bytes = gpu_weight_bytes.saturating_add(profile.input_side_bytes);
    }
    if include_output {
        gpu_weight_bytes = gpu_weight_bytes.saturating_add(profile.output_side_bytes);
    }
    let gpu_from = layer_start.max(i_gpu_start);
    for (layer, layer_bytes) in profile.layer_bytes.iter().enumerate() {
        if layer < layer_start || layer >= layer_end {
            continue;
        }
        if layer >= gpu_from {
            gpu_weight_bytes = gpu_weight_bytes.saturating_add(*layer_bytes);
        }
    }
    StageResidency { gpu_weight_bytes }
}

/// Decision of [`stage_pair_fits`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreflightDecision {
    /// The pair fits; carry the numbers for reporting.
    Fits { resident_bytes: u64, budget_bytes: u64 },
    /// The pair cannot fit; carry the numbers for the error text.
    Exceeds { resident_bytes: u64, budget_bytes: u64 },
}

/// Pure arithmetic: does the concurrent stage pair fit `budget_bytes`?
pub fn stage_pair_fits(
    stage0: &StageResidency,
    stage1: &StageResidency,
    budget_bytes: u64,
) -> PreflightDecision {
    let resident = stage0
        .resident_bytes()
        .saturating_add(stage1.resident_bytes());
    if resident <= budget_bytes {
        PreflightDecision::Fits { resident_bytes: resident, budget_bytes }
    } else {
        PreflightDecision::Exceeds { resident_bytes: resident, budget_bytes }
    }
}

/// macOS `iogpu.wired_limit_mb`: 0 (or unreadable) means the default budget
/// (the Metal device's recommended working-set size); a positive value is the
/// operator's wired limit in MiB and takes precedence.
///
/// Deliberately NOT `currentFreeMemory`: the Metal free reading does not
/// reliably account for file-backed wired residency (the stage mappings) and
/// residency-set keep-alive holds freed pages wired for minutes
/// (ggml-metal-device.m), so a free-based budget is both stale and blind to
/// the exact demand this preflight tracks. The working-set ceiling plus the
/// lane's own additive demand is the stable, honest comparison; co-resident
/// load is the runner's idle-gate responsibility.
pub fn wired_budget_bytes(memory_total: u64, wired_limit_mb: u64) -> u64 {
    if wired_limit_mb > 0 {
        wired_limit_mb.saturating_mul(1024 * 1024)
    } else {
        memory_total
    }
}

/// Read the effective wired limit. Precedence: the `SKIPPY_WIRED_LIMIT_MB`
/// env override (testing/matrix knob, no root needed), then macOS
/// `iogpu.wired_limit_mb` via sysctl. None/0/unreadable means the default
/// working-set budget.
#[cfg(target_os = "macos")]
pub fn read_iogpu_wired_limit_mb() -> Option<u64> {
    if let Some(value) = std::env::var_os("SKIPPY_WIRED_LIMIT_MB") {
        return value.to_str().and_then(|text| text.trim().parse::<u64>().ok());
    }
    let output = std::process::Command::new("sysctl")
        .args(["-n", "iogpu.wired_limit_mb"])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.trim().parse::<u64>().ok()
}

#[cfg(not(target_os = "macos"))]
pub fn read_iogpu_wired_limit_mb() -> Option<u64> {
    None
}

/// Query the Metal device's memory budget inputs: the device's total
/// (recommended working set) memory and the operator's wired limit. Returns
/// None when no GPU device is visible (CPU-only run) or the platform is not
/// macOS - callers skip the preflight then.
pub fn metal_device_budget() -> Result<Option<(u64, u64)>> {
    let devices = skippy_runtime::backend_devices().context("failed to enumerate backend devices")?;
    let Some(gpu) = devices
        .iter()
        .find(|device| device.device_type == skippy_runtime::BackendDeviceType::Gpu)
        .or_else(|| {
            devices
                .iter()
                .find(|device| device.device_type == skippy_runtime::BackendDeviceType::IntegratedGpu)
        })
    else {
        return Ok(None);
    };
    Ok(Some((gpu.memory_total, read_iogpu_wired_limit_mb().unwrap_or(0))))
}

/// Full preflight for one split of the boundary lane: estimate both stages'
/// wired demand from the model's GGUF layer profile and the configured
/// `n_gpu_layers`, and refuse to open the pair when it cannot fit the budget.
///
/// The budget is the device's working-set ceiling (`memory_total`, i.e.
/// recommendedMaxWorkingSetSize) clamped by the operator's
/// `iogpu.wired_limit_mb` when set. Failures of the
/// estimation itself (unreadable GGUF) also refuse the split - on this lane a
/// false "fits" produces silent numeric corruption, which is the exact
/// failure mode this preflight exists to prevent.
pub fn preflight_stage_pair(
    model_path: &Path,
    layer_end: usize,
    split_layer: usize,
    n_gpu_layers: u64,
    memory_total: u64,
    wired_limit_mb: u64,
) -> Result<()> {
    if split_layer == 0 || split_layer >= layer_end {
        bail!("split layer {split_layer} out of range 1..{layer_end}");
    }
    let profile = scan_gguf_layer_byte_profile(model_path).with_context(|| {
        format!("wired preflight: cannot read GGUF layer profile of {model_path:?}")
    })?;
    // llama.cpp places the LAST n_gpu_layers layers on the GPU; with the
    // stage filter each stage keeps the intersection of its range with that
    // window (src/llama-model.cpp, stage_gpu_start/stage_gpu_end).
    let i_gpu_start = layer_end.saturating_sub(n_gpu_layers.clamp(0, u64::try_from(layer_end).unwrap_or(0)) as usize);
    let stage0 = stage_residency(&profile, 0, split_layer, i_gpu_start, true, false);
    let stage1 = stage_residency(&profile, split_layer, layer_end, i_gpu_start, false, true);
    let budget = wired_budget_bytes(memory_total, wired_limit_mb);
    match stage_pair_fits(&stage0, &stage1, budget) {
        PreflightDecision::Fits { .. } => Ok(()),
        PreflightDecision::Exceeds { resident_bytes, budget_bytes } => bail!(
            "wired-memory preflight: concurrent stage pair for split {split_layer} needs \
             ~{resident_bytes} bytes but only {budget_bytes} bytes are available \
             (memory_total={memory_total}, iogpu.wired_limit_mb={wired_limit_mb}, \
             n_gpu_layers={n_gpu_layers}); reduce --n-gpu-layers or raise the wired \
             limit - running anyway produces silent GPU OOM corruption",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    /// glm45-air-shaped profile: 46 repeating layers, first 6 dense/smaller,
    /// the rest expert-heavy (MoE weights dominate), plus token embeddings
    /// and the output head. Round numbers at model scale (~70 GiB total) -
    /// the shape matters, not the exact sizes.
    fn glm45_air_profile() -> GgufLayerByteProfile {
        let mut layer_bytes = vec![0u64; 46];
        for (i, layer) in layer_bytes.iter_mut().enumerate() {
            *layer = if i < 6 { 80 * MIB } else { 1_540 * MIB };
        }
        GgufLayerByteProfile {
            layer_bytes,
            input_side_bytes: 2 * GIB,
            output_side_bytes: 700 * MIB,
        }
    }

    #[test]
    fn window_intersection_matches_llama_placement() {
        let profile = glm45_air_profile();
        // ngl40 of 47 total layers: GPU window starts at layer 7.
        let stage0 = stage_residency(&profile, 0, 30, 7, true, false);
        // GPU weights: embeddings (conservatively) + layers 7..30 (all
        // expert-size; the 80 MiB dense layers are 0..5, below the window).
        let expected0 = 2 * GIB + 23 * 1_540 * MIB;
        assert_eq!(stage0.gpu_weight_bytes, expected0);

        let stage1 = stage_residency(&profile, 30, 46, 7, false, true);
        // Layer 30..46 all >= 7: fully on GPU, plus the output head.
        assert_eq!(stage1.gpu_weight_bytes, 16 * 1_540 * MIB + 700 * MIB);
    }

    #[test]
    fn full_offload_puts_every_layer_on_gpu() {
        let profile = glm45_air_profile();
        let i_gpu_start = 0; // ngl >= layer count
        let stage0 = stage_residency(&profile, 0, 30, i_gpu_start, true, false);
        let stage1 = stage_residency(&profile, 30, 46, i_gpu_start, false, true);
        let total0: u64 = 2 * GIB + 6 * 80 * MIB + 24 * 1_540 * MIB;
        let total1: u64 = 16 * 1_540 * MIB + 700 * MIB;
        assert_eq!(stage0.gpu_weight_bytes, total0);
        assert_eq!(stage1.gpu_weight_bytes, total1);
    }

    #[test]
    fn overhead_added_per_stage() {
        let profile = glm45_air_profile();
        let stage = stage_residency(&profile, 0, 46, 0, true, true);
        assert_eq!(
            stage.resident_bytes(),
            stage.gpu_weight_bytes + PER_CONTEXT_OVERHEAD_BYTES
        );
    }

    #[test]
    fn decision_flips_across_budget() {
        let profile = glm45_air_profile();
        let i_gpu_start = 0;
        let s0 = stage_residency(&profile, 0, 30, i_gpu_start, true, false);
        let s1 = stage_residency(&profile, 30, 46, i_gpu_start, false, true);
        let resident = s0.resident_bytes() + s1.resident_bytes();

        assert_eq!(
            stage_pair_fits(&s0, &s1, resident),
            PreflightDecision::Fits { resident_bytes: resident, budget_bytes: resident }
        );
        assert_eq!(
            stage_pair_fits(&s0, &s1, resident - 1),
            PreflightDecision::Exceeds { resident_bytes: resident, budget_bytes: resident - 1 }
        );
    }

    #[test]
    fn partial_offload_fits_where_full_offload_exceeds() {
        let profile = glm45_air_profile();
        let full0 = stage_residency(&profile, 0, 30, 0, true, false);
        let full1 = stage_residency(&profile, 30, 46, 0, false, true);
        let part0 = stage_residency(&profile, 0, 30, 7, true, false);
        let part1 = stage_residency(&profile, 30, 46, 7, false, true);

        let full_total = full0.resident_bytes() + full1.resident_bytes();
        let part_total = part0.resident_bytes() + part1.resident_bytes();
        // Partial offload strictly reduces wired demand...
        assert!(part_total < full_total);
        // ...so there is a budget where the decision flips: the glm45-air
        // dose-response mechanism (ngl99 fails, ngl40 passes).
        let budget = (part_total + full_total) / 2;
        assert!(matches!(
            stage_pair_fits(&full0, &full1, budget),
            PreflightDecision::Exceeds { .. }
        ));
        assert!(matches!(
            stage_pair_fits(&part0, &part1, budget),
            PreflightDecision::Fits { .. }
        ));
    }

    #[test]
    fn wired_limit_overrides_free_memory() {
        // limit set: budget is exactly the limit, regardless of free memory
        assert_eq!(wired_budget_bytes(80 * GIB, 20 * 1024), 20 * 1024 * MIB);
        // no limit: budget is the device's working-set size
        assert_eq!(wired_budget_bytes(80 * GIB, 0), 80 * GIB);
    }
}
