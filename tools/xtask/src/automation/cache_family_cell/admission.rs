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
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(input, 1024 * 1024)?;
    let mut admitted: Input = serde_json::from_slice(&bytes)?;
    admitted.validate()?;
    for (path, expected) in [
        (&mut admitted.binary, &admitted.binary_sha256),
        (&mut admitted.model, &admitted.model_sha256),
    ] {
        *path = path.canonicalize()?;
        if !std::fs::symlink_metadata(&*path)?.is_file() {
            return Err("cache cell artifact must be canonical regular file".into());
        }
        if crate::product::digest::file_sha256(path).map_err(|e| e.error)? != *expected {
            return Err("cache cell artifact SHA mismatch".into());
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
            .ok_or("cache cell model dimensions absent")?;
    if shape.block_count != u64::from(admitted.layer_end) {
        return Err("cache cell full model layer range mismatch".into());
    }
    let receipt = Receipt {
        schema_version: 1,
        request_sha256: hash(&bytes),
        admitted,
        model_identity: serde_json::to_value(model)?,
    };
    crate::automation::waiting_prefix::adaptive_identity::fresh(
        output,
        &serde_json::to_vec(&receipt)?,
    )
}
