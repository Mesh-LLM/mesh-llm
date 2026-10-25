//! Native stage memory budget used by placement and post-plan validation.
//!
//! A single calculation owns the KV compute reserve, lane scaling and fixed
//! node headroom so callers cannot approve a cut that the planner would reject.

// The 10% reserve keeps the OS and frontend outside the stage's advertised
// budget. The 1 GiB floor covers context-independent compute buffers observed
// on split hosts; tiny nodes cap that floor at half their budget. KV's separate
// 100/85 charge below covers buffers that grow with context length.
const RUNTIME_NODE_HEADROOM_NUMERATOR: u64 = 1;
const RUNTIME_NODE_HEADROOM_DENOMINATOR: u64 = 10;
const RUNTIME_NODE_HEADROOM_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;

const KV_COMPUTE_RESERVE_NUMERATOR: u128 = 100;
const KV_COMPUTE_RESERVE_DENOMINATOR: u128 = 85;

pub(super) fn candidate_bytes_per_layer(
    weight_per_layer: u64,
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
) -> Option<u64> {
    // Each lane needs a full context allocation in either KV layout.
    let kv_bytes = u128::from(kv_per_layer)
        .checked_mul(u128::from(context_length))?
        .checked_mul(parallel_lanes as u128)?;
    // Charge KV at 100/85 so 15% of the node's post-weight space is held back
    // for llama.cpp compute-graph buffers/scratch (mirrors the single-node
    // context planner's `usable_kv_cache_budget`). This scales the reserve with
    // context length, matching how compute buffers grow with `n_ctx`.
    let kv_with_compute_reserve = kv_bytes
        .checked_mul(KV_COMPUTE_RESERVE_NUMERATOR)?
        .div_ceil(KV_COMPUTE_RESERVE_DENOMINATOR);
    let total = u128::from(weight_per_layer).checked_add(kv_with_compute_reserve)?;
    total.try_into().ok()
}

pub(super) fn layer_required_bytes(
    layer_weights: &[u64],
    recurrent_bytes_by_layer: &[u64],
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
) -> Option<Vec<u64>> {
    layer_weights
        .iter()
        .zip(recurrent_bytes_by_layer.iter().copied())
        .map(|(weight, recurrent_bytes)| {
            candidate_bytes_per_layer(*weight, kv_per_layer, context_length, parallel_lanes)
                .and_then(|base| {
                    recurrent_bytes
                        .checked_mul(parallel_lanes as u64)
                        .and_then(|recurrent| base.checked_add(recurrent))
                })
        })
        .collect()
}

/// Reserve context-independent process and runtime memory on each node.
pub fn default_runtime_headroom_bytes(vram_bytes: u64) -> u64 {
    let proportional = vram_bytes
        .saturating_mul(RUNTIME_NODE_HEADROOM_NUMERATOR)
        .div_ceil(RUNTIME_NODE_HEADROOM_DENOMINATOR);
    proportional
        .max(RUNTIME_NODE_HEADROOM_FLOOR_BYTES.min(vram_bytes / 2))
        .min(vram_bytes)
}

/// The planner's per-layer estimate, saturated for diagnostic reporting.
pub fn diagnostic_candidate_bytes_per_layer(
    weight_per_layer: u64,
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
) -> u64 {
    candidate_bytes_per_layer(
        weight_per_layer,
        kv_per_layer,
        context_length,
        parallel_lanes,
    )
    .unwrap_or(u64::MAX)
}

/// Recheck a stage after its boundaries change, using the planner's exact
/// per-layer accounting. Inputs must cover every model layer.
pub fn required_stage_bytes(
    layer_weights: &[u64],
    recurrent_bytes_by_layer: &[u64],
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
    layer_start: u32,
    layer_end: u32,
) -> Option<u64> {
    let costs = layer_required_bytes(
        layer_weights,
        recurrent_bytes_by_layer,
        kv_per_layer,
        context_length,
        parallel_lanes,
    )?;
    let start = usize::try_from(layer_start).ok()?;
    let end = usize::try_from(layer_end).ok()?;
    costs
        .get(start..end)?
        .iter()
        .try_fold(0u64, |sum, cost| sum.checked_add(*cost))
}

/// Reprice the weights held by a stage after its layer boundaries move.
/// Without complete per-layer detail, use the released even-share fallback.
pub fn repriced_stage_weight_bytes(
    layer_weights: &[u64],
    model_weight_bytes: u64,
    layer_count: u32,
    layer_start: u32,
    layer_end: u32,
) -> u64 {
    if layer_weights.len() == layer_count as usize {
        let start = (layer_start as usize).min(layer_weights.len());
        let end = (layer_end as usize).min(layer_weights.len());
        return layer_weights
            .get(start..end)
            .unwrap_or(&[])
            .iter()
            .fold(0u64, |sum, weight| sum.saturating_add(*weight));
    }
    let per_layer = model_weight_bytes / u64::from(layer_count.max(1));
    u64::from(layer_end.saturating_sub(layer_start)).saturating_mul(per_layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_recheck_charges_kv_for_each_lane() {
        let weights = [100, 200];
        let recurrent = [3, 5];
        let single = required_stage_bytes(&weights, &recurrent, 85, 10, 1, 0, 2).unwrap();
        let four_lanes = required_stage_bytes(&weights, &recurrent, 85, 10, 4, 0, 2).unwrap();
        assert_eq!(single, 2_308);
        assert_eq!(four_lanes, 8_332);
        assert!(four_lanes > single);
    }

    #[test]
    fn headroom_keeps_small_nodes_eligible() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(default_runtime_headroom_bytes(0), 0);
        assert_eq!(default_runtime_headroom_bytes(gib), gib / 2);
        assert_eq!(default_runtime_headroom_bytes(80 * gib), 8 * gib);
    }

    #[test]
    fn moved_cut_reprices_exact_or_even_weights() {
        assert_eq!(
            repriced_stage_weight_bytes(&[50, 100, 200], 350, 3, 1, 3),
            300
        );
        assert_eq!(repriced_stage_weight_bytes(&[], 300, 3, 1, 3), 200);
    }
}
