//! Current battery plan admission. Native execution remains owned by the battery.
#[path = "family_battery_environment.rs"]
mod environment;

use crate::command::DynResult;
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Deserialize)]
struct PlanShards {
    shards: Vec<Shard>,
}

#[derive(Deserialize)]
struct Shard {
    shard_index: u64,
}

fn admit(root: &Path, manifest: &Path, plan: &Path, requested: &str) -> DynResult<()> {
    let root = root.canonicalize()?;
    let manifest_before = fs::read(manifest)?;
    let plan_before = fs::read(plan)?;
    // This verifier derives the supplied selection and shard count. It does not
    // force the controller's 256-shard planning request on a direct battery.
    crate::ci_plan::family::verify_controller_plan(&root, manifest, plan)?;
    let shards: PlanShards = serde_json::from_slice(&plan_before)?;
    if !requested.is_empty() {
        if !requested.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("shard index must contain decimal digits".into());
        }
        let index: u64 = requested.parse()?;
        if !shards.shards.iter().any(|shard| shard.shard_index == index) {
            return Err(format!("policy plan has no shard index {index}").into());
        }
    }
    if fs::read(manifest)? != manifest_before || fs::read(plan)? != plan_before {
        return Err("battery policy inputs changed during admission".into());
    }
    Ok(())
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let [mode, root, manifest, plan, cache] = args
        && mode == "--cache"
    {
        return crate::automation::canary_source_plan::battery_cache::admit(
            Path::new(root),
            Path::new(manifest),
            Path::new(plan),
            Path::new(cache),
        );
    }
    if let [mode, model] = args
        && mode == "--inspect-gguf"
    {
        return crate::automation::canary_source_plan::battery_cache::inspect(Path::new(model));
    }
    if let [mode, artifact, models, minimum, output] = args
        && mode == "--environment"
    {
        return environment::run(
            Path::new(artifact),
            Path::new(models),
            minimum,
            Path::new(output),
        );
    }
    let [root, manifest, plan, shard] = args else {
        return Err(
            "usage: automation family-battery-policy ROOT MANIFEST PLAN SHARD_INDEX_OR_EMPTY"
                .into(),
        );
    };
    admit(Path::new(root), Path::new(manifest), Path::new(plan), shard)
}

#[cfg(test)]
#[path = "family_battery_policy_tests.rs"]
mod tests;
