//! Explicit, pinned four-model KV validation; never an automatic CI hardware gate.
mod contract;
mod evidence;
mod execution;
mod jy_plan;
mod preparation;
#[cfg(test)]
mod tests;
use crate::command::DynResult;
use std::path::{Path, PathBuf};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "cargo xtool automation kv-validation {{pin|prepare|run|jy-plan}} --input ABS_JSON --output ABS_FRESH_PATH"
        );
        return Ok(());
    }
    let [verb, input_flag, input, output_flag, output] = args else {
        return Err("KV validation requires verb --input ABS_JSON --output ABS_FRESH_PATH".into());
    };
    if input_flag != "--input" || output_flag != "--output" {
        return Err("invalid KV validation arguments".into());
    }
    let output = PathBuf::from(output);
    contract::absolute(&output)?;
    if verb == "run-env" {
        if input != "KV_VALIDATION_REQUEST_JSON" {
            return Err("KV worker requires its dedicated request environment".into());
        }
        let bytes = std::env::var(input)?;
        if bytes.len() > 1024 * 1024 {
            return Err("KV worker request exceeds1MiB".into());
        }
        return execution::run(serde_json::from_str(&bytes)?, &output);
    }
    let input = PathBuf::from(input);
    contract::absolute(&input)?;
    let bytes = super::receipt_files::bounded(&input, 1024 * 1024)?;
    match verb.as_str() {
        "pin" => preparation::pin(serde_json::from_slice(&bytes)?, &output),
        "prepare" => preparation::run(serde_json::from_slice(&bytes)?, &output),
        "run" => execution::run(serde_json::from_slice(&bytes)?, &output),
        "jy-plan" => jy_plan::run(serde_json::from_slice(&bytes)?, &output),
        _ => Err("unknown KV validation verb".into()),
    }
}
fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
fn publish(path: &Path, value: &impl serde::Serialize) -> DynResult<()> {
    super::receipt_files::fresh(path, &serde_json::to_vec_pretty(value)?)
}
fn verify_bundle(
    path: &Path,
    expected: &str,
    backend: contract::Backend,
    native_source_commit: &str,
) -> DynResult<()> {
    super::native_artifact_identity::verify(path, expected)?;
    let manifest: serde_json::Value = serde_json::from_slice(&super::receipt_files::bounded(
        &path.join("manifest.json"),
        1024 * 1024,
    )?)?;
    verify_manifest(&manifest, backend, native_source_commit)
}
fn verify_manifest(
    manifest: &serde_json::Value,
    backend: contract::Backend,
    native_source_commit: &str,
) -> DynResult<()> {
    if manifest["runtime"]["backend"]["kind"] != backend.kind()
        || manifest["build"]["backend"] != backend.kind()
        || manifest["build"]["llama_patched_sha"] != native_source_commit
        || manifest["runtime"]["libraries"]
            .as_array()
            .is_none_or(Vec::is_empty)
    {
        return Err("KV runtime requires matching source and exact backend leaf manifest".into());
    }
    Ok(())
}
