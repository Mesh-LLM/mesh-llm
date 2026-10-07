//! Direct current-battery cache admission reuses the controller's source-plan gate.
use super::{admission, contract::CachePolicy};
use crate::automation::replay_matrix::model_preflight::{dimensions, tensor_layouts};
use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use std::{fs, path::Path};

pub(crate) fn admit(root: &Path, manifest: &Path, plan: &Path, cache: &Path) -> DynResult<()> {
    admit_with_descriptors(root, manifest, plan, cache, false)
}

pub(crate) fn admit_descriptors(
    root: &Path,
    manifest: &Path,
    plan: &Path,
    cache: &Path,
) -> DynResult<()> {
    admit_with_descriptors(root, manifest, plan, cache, true)
}

fn admit_with_descriptors(
    root: &Path,
    manifest: &Path,
    plan: &Path,
    cache: &Path,
    descriptors: bool,
) -> DynResult<()> {
    let root = root.canonicalize()?;
    let before = fs::read(plan)?;
    let manifest_before = fs::read(manifest)?;
    // Derive the supplied filter and shard count, rather than imposing 256 shards.
    crate::ci_plan::family::verify_controller_plan(&root, manifest, plan)?;
    // Explicit caller cache authority: never fall back to the controller's HOME.
    let cache = cache.canonicalize()?;
    if descriptors {
        let constants = root.join(".deps/llama.cpp/gguf-py/gguf/constants.py");
        let (captured, layouts) = tensor_layouts::read(&constants)?;
        admission::verify_descriptors(&before, cache, &layouts)?;
        if fs::read(constants)? != captured {
            return Err("prepared GGML layout data changed during admission".into());
        }
    } else {
        admission::verify(&before, &CachePolicy::GgufMetadata { root: cache })?;
    }
    if fs::read(plan)? != before || fs::read(manifest)? != manifest_before {
        return Err("battery cache policy inputs changed during admission".into());
    }
    Ok(())
}

pub(crate) fn inspect(path: &Path) -> DynResult<()> {
    let found = dimensions::inspect(path)?
        .ok_or("GGUF has no positive block_count and embedding_length metadata")?;
    let value = serde_json::json!({
        "architecture":found.architecture,
        "layer_count":found.block_count,
        "activation_width":found.activation_width,
        "mtp_layers":found.mtp_layers,
    });
    CheckReport::success(format!("{}\n", serde_json::to_string(&value)?)).emit()
}
