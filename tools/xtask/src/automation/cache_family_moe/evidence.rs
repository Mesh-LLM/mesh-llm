//! Actual inspector and state-handoff fields; absent sequence IDs stay unmeasured.
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;
const MARKERS: &[&str] = &[
    "_exps",
    "_shexp",
    ".expert",
    "_expert",
    "ffn_gate_inp",
    "expert_gate",
    "exp_probs",
];
#[derive(Deserialize)]
struct Inspection {
    tensor_count: usize,
    tensors: Vec<Tensor>,
}
#[derive(Deserialize)]
struct Tensor {
    name: String,
    layer_index: Option<u32>,
    byte_size: u64,
}
pub(super) fn experts(value: &Value, start: u32, end: u32) -> DynResult<Value> {
    let report: Inspection = serde_json::from_value(value.clone())?;
    if report.tensor_count != report.tensors.len() || report.tensors.len() > 100000 || start >= end
    {
        return Err("invalid MoE inspector count/range".into());
    }
    let mut names = BTreeSet::new();
    let mut layers = BTreeSet::new();
    let mut bytes = 0_u64;
    let mut count = 0_u32;
    let mut sample = Vec::new();
    for tensor in report.tensors {
        if tensor.name.is_empty()
            || tensor.name.len() > 4096
            || tensor.name.chars().any(char::is_control)
            || !names.insert(tensor.name.clone())
        {
            return Err("invalid/duplicate inspected tensor name".into());
        }
        let Some(layer) = tensor.layer_index else {
            continue;
        };
        if layer < start || layer >= end || !MARKERS.iter().any(|m| tensor.name.contains(m)) {
            continue;
        }
        bytes = bytes
            .checked_add(tensor.byte_size)
            .ok_or("MoE expert byte total overflow")?;
        count = count.checked_add(1).ok_or("MoE tensor count overflow")?;
        layers.insert(layer);
        if sample.len() < 8 {
            sample.push(tensor.name);
        }
    }
    Ok(
        json!({"expert_tensor_count":count,"expert_tensor_bytes":bytes,"expert_layer_count":layers.len(),"expert_layers":layers,"sample_expert_tensors":sample}),
    )
}
pub(super) fn row(case: &Value, topology: &Value, report: &Value, expert: &Value) -> Value {
    let pass = report["status"] == "pass"
        && expert["expert_tensor_count"]
            .as_u64()
            .is_some_and(|n| n > 0)
        && report["suffix_prefill_matches"] != false;
    json!({"family":case["family"],"model_id":report["model_identity"]["model_id"],"payload":report["state_payload_kind"],"topology":topology,"layer_start":report["layer_start"],"layer_end":report["layer_end"],"status":if pass{"pass"}else{"fail"},"correctness_status":report["status"],"native_seq_remapped":null,"source_native_seq_id":null,"restore_native_seq_id":null,"sequence_observation":"current_product_report_does_not_expose_sequence_IDs","prompt_tokens":report["prompt_token_count"],"suffix_prefill_matches":report["suffix_prefill_matches"],"cache_hit_matches":report["cache_hit_matches"],"resident_state_bytes":report["resident_state_bytes"],"cache_storage_bytes":report["cache_storage_bytes"],"serialized_payload_bytes":report["state_bytes"],"borrowed_resident_hits":report["borrowed_resident_hits"],"expert":expert,"scope":"expert_tensor_presence_plus_owned_cache_correctness_not_expert_route_trace_or_remap_attestation"})
}
