//! Existing hash/model/tree owners execute in an owned bounded admission child.
use super::catalog::Input;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
#[derive(Deserialize, Serialize)]
pub(super) struct Receipt {
    pub request_sha256: String,
    pub admitted: Input,
    pub layers: u32,
    pub activation_width: u32,
    pub model_identity: Value,
}
pub(super) fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
pub(super) fn run(input: &Path, output: &Path) -> DynResult<()> {
    if !input.is_absolute() || !output.is_absolute() {
        return Err("cache admission paths must be absolute".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("cache admission output must be fresh".into()),
    }
    let bytes = crate::automation::receipt_files::bounded(input, 1024 * 1024)?;
    let mut admitted: Input = serde_json::from_slice(&bytes)?;
    admitted.validate()?;
    for (path, expected) in [
        (&mut admitted.correctness, &admitted.correctness_sha256),
        (&mut admitted.stage_server, &admitted.stage_server_sha256),
    ] {
        *path = path.canonicalize()?;
        if !std::fs::symlink_metadata(&*path)?.is_file() {
            return Err("cache pinned inputs must be regular".into());
        }
        if crate::product::digest::file_sha256(path).map_err(|e| e.error)? != *expected {
            return Err("cache input digest mismatch".into());
        }
    }
    admitted.native_build = admitted.native_build.canonicalize()?;
    crate::automation::native_artifact_identity::verify(
        &admitted.native_build,
        &admitted.native_build_sha256,
    )?;
    crate::automation::cache_family_profile::observe(&mut admitted.toolkit_directories)?;
    let (model, layers, activation_width) = if admitted.artifact.is_some() {
        if admitted.case_key == "deepseek3" {
            admitted.model = admitted.model.canonicalize()?;
        } else {
            admitted.model = admitted
                .model
                .parent()
                .ok_or("shard parent")?
                .canonicalize()?
                .join(admitted.model.file_name().ok_or("primary name")?);
        }
        let model = super::artifact::inspect(&mut admitted)?;
        let layers = model["dimensions"]["layer_count"]
            .as_u64()
            .ok_or("artifact layers")?
            .try_into()?;
        let width = model["dimensions"]["activation_width"]
            .as_u64()
            .ok_or("artifact width")?
            .try_into()?;
        (model, layers, width)
    } else {
        admitted.model = admitted.model.canonicalize()?;
        if !std::fs::symlink_metadata(&admitted.model)?.is_file() {
            return Err("model must be regular".into());
        }
        crate::automation::replay_matrix::model_preflight::require_single_file(&admitted.model)?;
        let model = crate::automation::replay_matrix::model_preflight::verify(
            &admitted.model,
            &admitted.model_sha256,
            u64::from(admitted.ctx_size),
        )?;
        let shape = crate::automation::replay_matrix::model_preflight::dimensions::inspect(
            &admitted.model,
        )?
        .ok_or("cache model dimensions absent")?;
        (
            serde_json::to_value(model)?,
            shape.block_count.try_into()?,
            shape.activation_width.try_into()?,
        )
    };
    let receipt = Receipt {
        request_sha256: hash(&bytes),
        layers,
        activation_width,
        model_identity: model,
        admitted,
    };
    if receipt.layers < 3 || receipt.activation_width == 0 {
        return Err("invalid cache model dimensions".into());
    }
    crate::automation::receipt_files::fresh(output, &serde_json::to_vec(&receipt)?)
}
