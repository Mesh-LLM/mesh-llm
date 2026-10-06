//! Typed smoke table; missing native sequence IDs remain explicit.
use serde_json::Value;
fn cell(value: &Value) -> String {
    match value {
        Value::String(s) => s.replace('|', "\\|").replace(['\r', '\n'], " "),
        Value::Null => "n/a".into(),
        _ => value.to_string(),
    }
}
pub(super) fn markdown(receipt: &Value) -> String {
    let mut out = format!(
        "# MoE expert cache smoke\n\nStatus: {}.\n\nExpert tensor presence and owned correctness only; sequence IDs unmeasured.\n\n| Family | Model ref | Topology | Layers | Status | Seq remap | Suffix match | Hits | Expert layers | Expert tensors | Expert bytes | Resident bytes | Cache storage bytes | Serialized bytes |\n|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|\n",
        cell(&receipt["status"])
    );
    if let Some(cases) = receipt["cases"].as_array() {
        for case in cases {
            if let Some(rows) = case["observation"]["rows"].as_array() {
                for row in rows {
                    let range = serde_json::json!(format!(
                        "{}..{}",
                        cell(&row["layer_start"]),
                        cell(&row["layer_end"])
                    ));
                    let values = [
                        &row["family"],
                        &row["model_id"],
                        &row["topology"],
                        &range,
                        &row["status"],
                        &row["native_seq_remapped"],
                        &row["suffix_prefill_matches"],
                        &row["cache_hit_matches"],
                        &row["expert"]["expert_layers"],
                        &row["expert"]["expert_tensor_count"],
                        &row["expert"]["expert_tensor_bytes"],
                        &row["resident_state_bytes"],
                        &row["cache_storage_bytes"],
                        &row["serialized_payload_bytes"],
                    ];
                    out.push_str(&format!(
                        "| {} |\n",
                        values.into_iter().map(cell).collect::<Vec<_>>().join(" | ")
                    ));
                }
            }
        }
    }
    out
}
