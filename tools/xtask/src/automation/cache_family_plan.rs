//! Source-owned cache-family catalog/profile planning; never starts a runtime.
#[path = "cache_family_plan/contract.rs"]
mod contract;
#[path = "cache_family_plan/planning.rs"]
mod planning;
#[cfg(test)]
#[path = "cache_family_plan/tests.rs"]
mod tests;
use crate::command::DynResult;
use std::path::Path;
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "cargo xtool automation cache-family-plan --input ABS_JSON --output ABS_FRESH_JSON"
        );
        return Ok(());
    }
    let [input_flag, input, output_flag, output] = args else {
        return Err("cache-family-plan requires --input ABS --output ABS".into());
    };
    if input_flag != "--input"
        || output_flag != "--output"
        || !Path::new(input).is_absolute()
        || !Path::new(output).is_absolute()
    {
        return Err("cache-family-plan requires absolute ordered input/output flags".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("cache-family-plan output must be fresh".into()),
    }
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(
        Path::new(input),
        1024 * 1024,
    )?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    let plan = planning::plan(&input)?;
    let bytes = serde_json::to_vec_pretty(&plan)?;
    if bytes.len() > 32 * 1024 * 1024 {
        return Err("cache plan output exceeds32MiB".into());
    }
    crate::automation::waiting_prefix::adaptive_identity::fresh(Path::new(output), &bytes)?;
    Ok(())
}

/// Reuse the catalog/profile planner with an already bounded local request.
pub(in crate::automation) fn plan_value(value: serde_json::Value) -> DynResult<serde_json::Value> {
    planning::plan(&serde_json::from_value::<contract::Input>(value)?)
}
