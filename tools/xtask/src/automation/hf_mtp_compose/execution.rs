use super::{contract::Input, identity, reports};
use crate::{
    automation::hf_certify::{admission, execution as process_owner},
    command::DynResult,
    process::Cancellation,
};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
fn text(path: &Path) -> DynResult<String> {
    Ok(path.to_str().ok_or("compose path Unicode")?.into())
}
fn compose_args(input: &Input, root: &Path) -> DynResult<Vec<String>> {
    Ok(vec![
        "compose-mtp".into(),
        "--target-shard".into(),
        text(&input.target_parts[input.expected_parts - 1].path)?,
        "--mtp-gguf".into(),
        text(&input.mtp_gguf.path)?,
        "--output".into(),
        text(&root.join(input.remote_name(input.expected_parts - 1)))?,
        "--mtp-block".into(),
        input.mtp_block.to_string(),
        "--metadata-shard".into(),
        text(&input.target_parts[0].path)?,
        "--metadata-output".into(),
        text(&root.join(input.remote_name(0)))?,
        "--set-kv".into(),
        "nemotron_h_moe.nextn_predict_layers=1".into(),
        "--json".into(),
    ])
}
fn validation_args(input: &Input, root: &Path) -> DynResult<Vec<String>> {
    let mut args = vec!["validate-mtp-attach".into()];
    for part in reports::model_parts(input, root) {
        args.extend(["--model".into(), text(&part)?]);
    }
    args.extend([
        "--mtp-draft".into(),
        text(&input.mtp_gguf.path)?,
        "--layer-count".into(),
        (input.mtp_block + 1).to_string(),
        "--mtp-layer-count".into(),
        "1".into(),
        "--ctx-size".into(),
        "64".into(),
        "--json".into(),
    ]);
    Ok(args)
}
fn invoke(
    input: &Input,
    args: Vec<String>,
    root: &Path,
    label: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    let raw = process_owner::run_process(&input.binary.path, args, root, label, deadline, cancel)?;
    admission::publish(
        &root.join(format!("{label}-process.json")),
        &json!({"diagnostic":format!("{:?}",raw.process),"outcome":format!("{:?}",raw.process.outcome),"exit_code":raw.process.status.as_ref().and_then(std::process::ExitStatus::code),"stdout_suppressed_lines":raw.process.stdout.suppressed_lines,"stderr_suppressed_lines":raw.process.stderr.suppressed_lines}),
    )?;
    if !process_owner::clean(&raw) {
        return Err("compose child incomplete/nonzero/cleanup/capture refusal".into());
    }
    Ok(serde_json::from_slice(
        raw.stdout
            .as_ref()
            .ok_or("native compose stdout")?
            .as_bytes(),
    )?)
}
pub(super) fn execute(
    input: &Input,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    report: &mut Value,
) -> DynResult<Value> {
    let before = identity::observe(input, root, "before", deadline, cancel)?;
    report["admitted"] = serde_json::to_value(&before.admitted)?;
    let admitted = &before.admitted;
    let composed = invoke(
        admitted,
        compose_args(admitted, root)?,
        root,
        "compose",
        deadline,
        cancel,
    )?;
    reports::compose(admitted, root, &composed)?;
    report["compose_report"] = composed;
    let outputs = identity::observe(admitted, root, "composed", deadline, cancel)?;
    if outputs.admitted != *admitted {
        return Err("compose input drift before attachment".into());
    }
    report["output_observations_before_validation"] = serde_json::to_value(&outputs.outputs)?;
    let validation = invoke(
        admitted,
        validation_args(admitted, root)?,
        root,
        "validation",
        deadline,
        cancel,
    )?;
    reports::validation(admitted, root, &validation)?;
    report["validation_report"] = validation;
    let after = identity::observe(admitted, root, "after", deadline, cancel)?;
    if after.admitted != *admitted || after.outputs != outputs.outputs {
        return Err("compose inputs/output changed during validation".into());
    }
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err("compose final cancelled/deadline".into());
    }
    let publication = reports::publication(admitted, &after.outputs);
    report["source_unchanged"] = Value::Bool(true);
    report["outputs"] = serde_json::to_value(after.outputs)?;
    Ok(publication)
}
