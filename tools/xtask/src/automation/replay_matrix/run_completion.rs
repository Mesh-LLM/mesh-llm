use super::run_workload::Input;
use crate::command::DynResult;
use std::path::Path;

pub(super) fn complete(
    root: Option<&Path>,
    input: &Input,
    document: &mut serde_json::Value,
    run_path: &Path,
    mut passed: bool,
) -> DynResult<()> {
    let results = input.output.join("results.json");
    crate::command::write_json_file(&results, &document["results"])?;
    let rows = input.output.join("summary/comparison.json");
    super::run_transport::invoke(root, "pooled-rows", &results, &rows)?;
    let gate_input = input.output.join("acceptance-input.json");
    let gate_output = input.output.join("summary/gates.json");
    let comparison: serde_json::Value = serde_json::from_slice(&std::fs::read(&rows)?)?;
    crate::command::write_json_file(
        &gate_input,
        &serde_json::json!({"rows":comparison,
        "prompt_token_range":input.prompt_token_range,"min_cache_pct":input.min_cache_pct,
        "require_output_match":input.require_output_match,"max_ttft_regression_pct":input.max_ttft_regression_pct}),
    )?;
    let acceptance =
        super::run_transport::invoke(root, "acceptance-gates", &gate_input, &gate_output);
    document["gates"] = serde_json::from_slice(&std::fs::read(&gate_output)?)?;
    passed &= acceptance.is_ok();
    if !passed {
        document["gates"]["passed"] = false.into();
    }
    document["gates"]["session_acceptance_failures"] = serde_json::json!(
        document["results"]
            .as_array()
            .ok_or("invalid results")?
            .iter()
            .flat_map(|result| result["cells"].as_array().into_iter().flatten())
            .filter(|cell| cell["acceptance"]["passed"] != true)
            .map(|cell| cell["acceptance"].clone())
            .collect::<Vec<_>>()
    );
    document["completed_at"] = crate::ci_operations::ci_metrics_time::isoformat(
        crate::ci_operations::ci_metrics_time::Instant::now(),
    )
    .into();
    super::run_snapshot::write(run_path, document)?;
    super::report::write(&input.output, &serde_json::from_value(document.clone())?)?;
    if passed {
        Ok(())
    } else {
        Err("run session acceptance failed; complete evidence retained".into())
    }
}
