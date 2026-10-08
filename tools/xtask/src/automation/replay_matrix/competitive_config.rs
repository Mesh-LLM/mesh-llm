//! Admission of competitive matrix inputs; no backend import or model execution.
use crate::command::DynResult;
use serde_json::Value;
use std::collections::BTreeSet;
pub(super) const LADDER: [u64; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
pub(super) fn positive(value: &Value, field: &str) -> DynResult<u64> {
    value
        .as_u64()
        .filter(|number| *number > 0)
        .ok_or_else(|| format!("{field} must be a positive nonboolean integer").into())
}
fn hex(value: &Value, width: usize, field: &str) -> DynResult<()> {
    if !value.as_str().is_some_and(|text| {
        text.len() == width
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err(format!("{field} must be a lowercase {width}-digit immutable identity").into());
    }
    Ok(())
}
fn text<'a>(value: &'a Value, field: &str) -> DynResult<&'a str> {
    value
        .as_str()
        .filter(|text| !text.is_empty() && !text.chars().any(char::is_control))
        .ok_or_else(|| format!("{field} must be nonempty text").into())
}
fn optional_backends(model: &Value) -> DynResult<()> {
    if let Some(support) = model.get("comparison_support") {
        for (arm, support) in support
            .as_object()
            .ok_or("comparison_support must be an object")?
        {
            if !["vllm", "sglang"].contains(&arm.as_str()) {
                return Err("unknown comparison backend".into());
            }
            let available = support["available"]
                .as_bool()
                .ok_or("comparison support requires available boolean")?;
            if !available {
                text(&support["reason"], "comparison exclusion reason")?;
            }
        }
    }
    if let Some(inputs) = model.get("comparison_inputs") {
        for (arm, input) in inputs
            .as_object()
            .ok_or("comparison_inputs must be an object")?
        {
            if !["vllm", "sglang"].contains(&arm.as_str()) {
                return Err("unknown comparison input backend".into());
            }
            text(&input["repo"], "comparison input repo")?;
            hex(&input["revision"], 40, "comparison input revision")?;
            hex(&input["sha256"], 64, "comparison input digest")?;
            hex(
                &input["tensor_equivalence_sha256"],
                64,
                "comparison tensor equivalence",
            )?;
        }
    }
    Ok(())
}
fn model(model: &Value) -> DynResult<&str> {
    let key = text(&model["key"], "model key")?;
    for field in ["family", "repo", "filename", "model_id", "cache_payload"] {
        text(&model[field], field)?;
    }
    hex(&model["revision"], 40, "model revision")?;
    for field in ["sha256", "tokenizer_sha256"] {
        hex(&model[field], 64, field)?;
    }
    text(&model["vllm_hf_config"]["repo"], "HF config repo")?;
    hex(
        &model["vllm_hf_config"]["revision"],
        40,
        "HF config revision",
    )?;
    hex(&model["vllm_hf_config"]["sha256"], 64, "HF config digest")?;
    for field in ["layer_end", "synthetic_context_size"] {
        positive(&model[field], field)?;
    }
    for field in ["thoughtworks_context_size", "thoughtworks_active_lanes"] {
        if let Some(value) = model.get(field) {
            positive(value, field)?;
        }
    }
    if let Some(capacity) = model.get("vllm_capacity") {
        for field in ["reference_tokens", "reference_blocks"] {
            positive(&capacity[field], field)?;
        }
    }
    optional_backends(model)?;
    Ok(key)
}
pub(in crate::automation::replay_matrix) fn admit(document: &Value) -> DynResult<()> {
    if document["schema_version"].as_u64() != Some(1) {
        return Err("competitive config schema_version must be 1".into());
    }
    hex(
        &document["baseline"]["llama_cpp_revision"],
        40,
        "baseline revision",
    )?;
    text(
        &document["baseline"]["llama_benchy_version"],
        "benchy version",
    )?;
    if document["concurrency"] != serde_json::json!(LADDER) {
        return Err("competitive concurrency ladder differs from reviewed contract".into());
    }
    let models = document["models"]
        .as_array()
        .filter(|rows| !rows.is_empty())
        .ok_or("competitive config requires models")?;
    let mut keys = BTreeSet::new();
    for row in models {
        if !keys.insert(model(row)?) {
            return Err("duplicate competitive model key".into());
        }
    }
    let synthetic = &document["synthetic"];
    positive(&synthetic["prompt_tokens"], "synthetic prompt tokens")?;
    for tokens in synthetic["output_tokens"]
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or("synthetic output ladder must not be empty")?
    {
        positive(tokens, "synthetic output tokens")?;
    }
    let trace = &document["thoughtworks"];
    for field in [
        "context_size",
        "active_lanes",
        "minimum_prompts",
        "output_tokens",
    ] {
        positive(&trace[field], field)?;
    }
    hex(&trace["dataset"]["revision"], 40, "dataset revision")?;
    hex(&trace["dataset"]["sha256"], 64, "dataset digest")?;
    let selection = &trace["selection"];
    hex(&selection["manifest_sha256"], 64, "prompt manifest digest")?;
    let families = positive(&selection["families"], "prompt families")?;
    let repeats = positive(&selection["requests_per_family"], "prompt repetitions")?;
    if selection["rows"]
        .as_array()
        .map(Vec::len)
        .and_then(|count| u64::try_from(count).ok())
        != Some(families)
    {
        return Err("selection must pin one row per family".into());
    }
    let available = families
        .checked_mul(repeats)
        .ok_or("prompt count overflow")?;
    if available < 256 || available % 256 != 0 {
        return Err("prompt selection must contain complete c256 waves".into());
    }
    Ok(())
}
