use super::arm_pass::Input;
use crate::command::DynResult;
use std::path::Path;

pub(super) fn retain(
    input: &Input,
    paths: (&Path, &Path, &Path),
    levels: &[usize],
    execution: &DynResult<()>,
) -> DynResult<bool> {
    let (result_path, log, warmup_summary) = paths;
    let mut cells = Vec::new();
    for level in levels {
        let path = input.output.join(format!("c-{level}.json"));
        if path.try_exists()? {
            cells.push(load(&path)?);
        }
    }
    let warmup = if warmup_summary.try_exists()? {
        load(warmup_summary)?
    } else {
        serde_json::Value::Null
    };
    let passed = execution.is_ok() && cells.len() == levels.len();
    let acceptance_failed = cells
        .iter()
        .any(|cell| cell["acceptance"]["passed"] == false)
        && super::pass_lifecycle::acceptance_exit(&input.output)?;
    let mut result = serde_json::json!({"label":input.label,"ref":input.reference,"commit":input.commit,"pass":input.pass,
        "model_id":warmup["model_id"],"server_log":log,"warmup":warmup,"cells":cells,"passed":passed,
        "acceptance_failed":acceptance_failed});
    if let Some(build) = &input.external {
        result["engine"] = serde_json::to_value(build.engine)?;
        result["version"] = build.version.clone().into();
        result["version_sha256"] = build.version_sha256.clone().into();
        result["provenance"] = serde_json::to_value(&build.provenance)?;
        let path = log.with_extension("command.json");
        if path.try_exists()? {
            result["command"] = load(&path)?;
        }
    }
    if input.qualification.is_none() {
        result["context_evidence"] = super::run_qualification::not_requested();
    }
    result["artifact_sha256"] =
        serde_json::to_value(super::pass_identity::capture(&input.output)?)?;
    if let Err(error) = execution {
        result["error"] = error.to_string().into();
    }
    crate::command::write_json_file(result_path, &result)?;
    Ok(passed)
}

fn load(path: &Path) -> DynResult<serde_json::Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
