//! Blocking file/GGUF/tree observation is confined to a parent-supervised child.
use super::contract::Input;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
#[derive(Deserialize, Serialize)]
pub(super) struct Receipt {
    pub schema_version: u64,
    pub request_sha256: String,
    pub admitted: Input,
    pub model_identity: Value,
}
pub(super) fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
pub(super) fn run(input: &Path, output: &Path) -> DynResult<()> {
    let bytes = crate::automation::receipt_files::bounded(input, 1024 * 1024)?;
    let mut admitted: Input = serde_json::from_slice(&bytes)?;
    admitted.validate()?;
    admitted.binary = admitted.binary.canonicalize()?;
    if !std::fs::symlink_metadata(&admitted.binary)?.is_file() {
        return Err("cache cell binary must be canonical regular file".into());
    }
    if crate::product::digest::file_sha256(&admitted.binary).map_err(|e| e.error)?
        != admitted.binary_sha256
    {
        return Err("cache cell binary SHA mismatch".into());
    }
    admitted.native_build = admitted.native_build.canonicalize()?;
    crate::automation::native_artifact_identity::verify(
        &admitted.native_build,
        &admitted.native_build_sha256,
    )?;
    crate::automation::cache_family_profile::observe(&mut admitted.toolkit_directories)?;
    let model = if let Some(mut artifact) = admitted.artifact.clone() {
        admitted.model = admitted
            .model
            .parent()
            .ok_or("shard parent")?
            .canonicalize()?
            .join(admitted.model.file_name().ok_or("primary name")?);
        let mut profile = crate::automation::cache_family_correctness::artifact::Inspection {
            case_key: "minimax_m27".into(),
            model: admitted.model.clone(),
            model_sha256: admitted.model_sha256.clone(),
            native_build: admitted.native_build.clone(),
            ctx_size: admitted.ctx_size,
            cell_seconds: admitted.execution_timeout_secs,
            settings: admitted.environment.clone(),
            model_id: admitted.model_id.clone(),
        };
        let receipt = crate::automation::cache_family_correctness::artifact::inspect_profile(
            &mut profile,
            &mut artifact,
        )?;
        admitted.artifact = Some(artifact);
        receipt
    } else {
        admitted.model = admitted.model.canonicalize()?;
        if !std::fs::symlink_metadata(&admitted.model)?.is_file() {
            return Err("cache model must be regular".into());
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
        .ok_or("cache cell model dimensions absent")?;
        if shape.block_count != u64::from(admitted.layer_end) {
            return Err("cache cell full model layer range mismatch".into());
        }
        serde_json::to_value(model)?
    };
    let receipt = Receipt {
        schema_version: 1,
        request_sha256: hash(&bytes),
        admitted,
        model_identity: model,
    };
    crate::automation::receipt_files::fresh(output, &serde_json::to_vec(&receipt)?)
}
