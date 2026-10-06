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
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(input, 1024 * 1024)?;
    let mut admitted: Input = serde_json::from_slice(&bytes)?;
    admitted.validate()?;
    for (path, expected) in [
        (&mut admitted.correctness, &admitted.correctness_sha256),
        (&mut admitted.stage_server, &admitted.stage_server_sha256),
        (&mut admitted.model, &admitted.model_sha256),
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
    crate::automation::waiting_prefix::native_identity::verify(
        &admitted.native_build,
        &admitted.native_build_sha256,
    )?;
    crate::automation::replay_matrix::model_preflight::require_single_file(&admitted.model)?;
    let model = crate::automation::replay_matrix::model_preflight::verify(
        &admitted.model,
        &admitted.model_sha256,
        u64::from(admitted.ctx_size),
    )?;
    let shape =
        crate::automation::replay_matrix::model_preflight::dimensions::inspect(&admitted.model)?
            .ok_or("cache model dimensions absent")?;
    let receipt = Receipt {
        request_sha256: hash(&bytes),
        layers: shape.block_count.try_into()?,
        activation_width: shape.activation_width.try_into()?,
        model_identity: serde_json::to_value(model)?,
        admitted,
    };
    if receipt.layers < 3 || receipt.activation_width == 0 {
        return Err("invalid cache model dimensions".into());
    }
    crate::automation::waiting_prefix::adaptive_identity::fresh(
        output,
        &serde_json::to_vec(&receipt)?,
    )
}
