use super::contract::Input;
use crate::{automation::hf_certify::admission::Artifact, command::DynResult};
use serde_json::{Value, json};
use std::path::Path;
fn fields(value: &Value, keys: &[&str]) -> DynResult<()> {
    let object = value.as_object().ok_or("native compose report object")?;
    if object.len() != keys.len() || keys.iter().any(|k| !object.contains_key(*k)) {
        return Err("native compose report fields refused".into());
    }
    Ok(())
}
pub(super) fn compose(input: &Input, root: &Path, value: &Value) -> DynResult<()> {
    fields(
        value,
        &[
            "target_shard",
            "mtp_gguf",
            "output",
            "metadata_shard",
            "target_tensors",
            "mtp_tensors",
            "appended_bytes",
            "block_count",
        ],
    )?;
    if value["target_shard"]
        != serde_json::to_value(&input.target_parts[input.expected_parts - 1].path)?
        || value["mtp_gguf"] != serde_json::to_value(&input.mtp_gguf.path)?
        || value["output"]
            != serde_json::to_value(root.join(input.remote_name(input.expected_parts - 1)))?
        || value["metadata_shard"] != serde_json::to_value(root.join(input.remote_name(0)))?
        || value["block_count"] != input.mtp_block + 1
        || ["target_tensors", "mtp_tensors", "appended_bytes"]
            .iter()
            .any(|k| value[*k].as_u64().is_none_or(|n| n == 0))
    {
        return Err("native composition path/block/tensor report correlation refused".into());
    }
    Ok(())
}
pub(super) fn model_parts(input: &Input, root: &Path) -> Vec<std::path::PathBuf> {
    input
        .target_parts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i == 0 || i == input.expected_parts - 1 {
                root.join(input.remote_name(i))
            } else {
                p.path.clone()
            }
        })
        .collect()
}
pub(super) fn validation(input: &Input, root: &Path, value: &Value) -> DynResult<()> {
    fields(
        value,
        &[
            "projector",
            "model_parts",
            "mtp_draft",
            "layer_count",
            "mtp_layer_count",
            "ctx_size",
            "native_mtp_multimodal_feature",
            "session_created",
        ],
    )?;
    if !value["projector"].is_null()
        || value["model_parts"] != serde_json::to_value(model_parts(input, root))?
        || value["mtp_draft"] != serde_json::to_value(&input.mtp_gguf.path)?
        || value["layer_count"] != input.mtp_block + 1
        || value["mtp_layer_count"] != 1
        || value["ctx_size"] != 64
        || value["native_mtp_multimodal_feature"] != true
        || value["session_created"] != true
    {
        return Err(
            "no-projector MTP attach report identity/feature/session correlation refused".into(),
        );
    }
    Ok(())
}
pub(super) fn publication(input: &Input, outputs: &[Artifact]) -> Value {
    let entries=input.target_parts.iter().enumerate().map(|(i,part)|{
        let artifact=if i==0 {&outputs[0]} else if i==input.expected_parts-1 {&outputs[1]} else {part};
        json!({"local":artifact.path,"sha256":artifact.sha256,"path_in_repo":input.remote_name(i),"source_kind":if i==0{"patched-first-metadata"}else if i==input.expected_parts-1{"composed-last-tensors"}else{"untouched-pinned-middle"}})
    }).collect::<Vec<_>>();
    json!({"schema_version":1,"status":"PLAN_ONLY_NOT_PUBLISHED","repo_id":input.composite_repo,"repo_type":"model","create_private":false,"entries":entries,"credential_names":["HF_TOKEN"],"custody":"all local entries observed after successful native attachment; no remote commit or token authority established","jobs":{"status":"NOT_ADMITTED","reason":"local paths are not immutable HF volume locators; require separately selected image digest, source revisions, mounted binary/input and bounded CPU flavor/cost before using existing model-package::jobs::JobSpec","execution":null},"upload_execution":null})
}
