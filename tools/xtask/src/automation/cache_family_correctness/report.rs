//! Admit the current product state-handoff report without claiming live model proof.
use super::{admission::Receipt, catalog::Topology};
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::Value;
#[derive(Deserialize)]
struct Report {
    mode: String,
    status: String,
    matches: bool,
    predicted_token_matches: bool,
    cache_hit_matches: bool,
    model_identity: Value,
    state_payload_kind: String,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    requested_prefix_token_count: u32,
    benchmark_prompt_token_count: u32,
    benchmark_prompt_text: String,
    activation_width: u32,
    cache_hit_repeats: u32,
    cache_hit_import_ms: Vec<f64>,
    cache_hit_decode_ms: Vec<f64>,
    recompute_total_ms: f64,
    cache_hit_total_ms: f64,
}
pub(super) fn accept(value: &Value, receipt: &Receipt, topology: Topology) -> DynResult<()> {
    let r: Report = serde_json::from_value(value.clone())?;
    let (start, end, index) = receipt.admitted.range(topology, receipt.layers)?;
    let payload = super::catalog::family(&receipt.admitted.case_key)?.1;
    if r.mode != "state-handoff"
        || r.status != "pass"
        || !r.matches
        || !r.predicted_token_matches
        || !r.cache_hit_matches
        || r.model_identity["model_id"] != receipt.admitted.model_id
        || r.state_payload_kind != payload
        || (r.layer_start, r.layer_end, r.stage_index) != (start, end, index)
        || r.activation_width != receipt.activation_width
        || r.requested_prefix_token_count != receipt.admitted.prefix_tokens
        || r.benchmark_prompt_token_count != receipt.admitted.prefix_tokens.saturating_add(1)
        || r.benchmark_prompt_text.is_empty()
        || r.benchmark_prompt_text.len() > 65536
        || r.cache_hit_repeats != receipt.admitted.cache_hit_repeats
        || r.cache_hit_import_ms.len() != r.cache_hit_repeats as usize
        || r.cache_hit_decode_ms.len() != r.cache_hit_repeats as usize
        || r.cache_hit_import_ms
            .iter()
            .chain(&r.cache_hit_decode_ms)
            .chain([&r.recompute_total_ms, &r.cache_hit_total_ms])
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("cache product report failed correctness/request/timing correlation".into());
    }
    let mut total = 0.0_f64;
    for (import, decode) in r.cache_hit_import_ms.iter().zip(&r.cache_hit_decode_ms) {
        let pair = import + decode;
        total += pair;
        if !pair.is_finite() || !total.is_finite() {
            return Err("cache computed timing overflow".into());
        }
    }
    let observed = total / f64::from(r.cache_hit_repeats);
    if !observed.is_finite() {
        return Err("cache computed mean is not finite".into());
    }
    if (observed - r.cache_hit_total_ms).abs() > 1e-8 * observed.abs().max(1.0) {
        return Err("cache mean timing disagrees with observed samples".into());
    }
    Ok(())
}
