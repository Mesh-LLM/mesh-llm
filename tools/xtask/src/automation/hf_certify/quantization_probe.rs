//! Two finite actual supplied-tool windows, consumed status and golden byte custody.
#[path = "quantization_probe/contract.rs"]
mod contract;
#[path = "quantization_probe/final_publication.rs"]
mod final_publication;
#[cfg(test)]
#[path = "quantization_probe/tests.rs"]
mod tests;
use super::{admission, bootstrap, execution, publication};
use crate::automation::command_interrupt::Interrupt;
use crate::{command::DynResult, process::Cancellation};
use contract::Input;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
fn check(until: Instant, cancel: &Cancellation) -> DynResult<()> {
    if Instant::now() >= until || cancel.is_cancelled() {
        return Err("quant probe shared deadline/cancellation".into());
    }
    Ok(())
}
fn pin(a: &admission::Artifact, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    if a.path.canonicalize()? != a.path {
        return Err("probe pins must be canonical regular files".into());
    }
    if bootstrap::execution::observe(&a.path, until, cancel)? != a.sha256 {
        return Err("quant probe immutable byte pin refused".into());
    }
    check(until, cancel)
}
fn custody(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    for a in [
        &input.tool,
        &input.tool_source,
        &input.runtime,
        &input.loader,
        &input.manifest,
        &input.tensor_recipe,
    ]
    .into_iter()
    .chain(input.source_parts.iter())
    {
        pin(a, until, cancel)?;
    }
    check(until, cancel)
}
fn gguf(a: &admission::Artifact) -> DynResult<()> {
    if std::fs::symlink_metadata(&a.path)?.len() > 64 * 1048576 {
        return Err("finite GGUF probe bound".into());
    }
    let bytes = admission::read(&a.path, 64 * 1048576)?;
    if bytes.len() < 24
        || bytes[..4] != *b"GGUF"
        || u32::from_le_bytes(bytes[4..8].try_into()?) != 3
        || admission::digest(&bytes) != a.sha256
    {
        return Err("finite supplied GGUF version/golden/source bytes refused".into());
    }
    Ok(())
}
fn manifest(input: &Input) -> DynResult<()> {
    let bytes = admission::read(&input.manifest.path, 1048576)?;
    if admission::digest(&bytes) != input.manifest.sha256 {
        return Err("bounded manifest pin differs".into());
    }
    let v: Value = serde_json::from_slice(&bytes)?;
    if v["schema_version"] != 1
        || v["kind"] != "QUANTIZE_GGUF"
        || v["expected_splits"] != 2
        || v["window_size"] != 1
        || v["source"] != input.source_root.to_string_lossy().as_ref()
        || v["target"] != input.target_root.to_string_lossy().as_ref()
        || v["source_prefix"] != input.source_prefix
        || v["target_prefix"] != input.target_prefix
        || v["output_basename"] != input.basename
        || v["tensor_type_file"] != input.tensor_recipe.path.to_string_lossy().as_ref()
        || !v["quant"].as_str().is_some_and(|s| !s.is_empty())
    {
        return Err("finite quantizer manifest/source/recipe/window correlation refused".into());
    }
    Ok(())
}
fn observed(
    input: &Input,
    args: Vec<String>,
    root: &Path,
    label: &str,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<Value> {
    check(until, cancel)?;
    let p = execution::run_process(&input.tool.path, args, root, label, until, cancel)?;
    evidence[label] = publication::process_observation(&p);
    if !execution::clean(&p) {
        return Err("finite supplied quantizer child incomplete; observations retained".into());
    }
    check(until, cancel)?;
    Ok(serde_json::from_slice(
        p.stdout.as_ref().ok_or("probe stdout absent")?.as_bytes(),
    )?)
}
fn status(
    input: &Input,
    completed: usize,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    let label = format!("status-{completed}");
    let v = observed(
        input,
        vec![
            "status".into(),
            "--manifest".into(),
            input.manifest.path.to_string_lossy().into(),
            "--json".into(),
        ],
        root,
        &label,
        until,
        cancel,
        evidence,
    )?;
    if v["expected_splits"] != 2
        || v["completed_count"] != completed
        || v["missing_count"] != (2 - completed)
        || v["complete"] != (completed == 2)
    {
        return Err("finite quantizer consumed status/window progress refused".into());
    }
    evidence[format!("progress-{completed}")] = v;
    Ok(())
}
fn prepare(input: &Input, root: &Path, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    input.validate()?;
    if root.canonicalize()? != root || !std::fs::symlink_metadata(root)?.is_dir() {
        return Err("probe evidence requires actual canonical private directory".into());
    }
    check(until, cancel)?;
    custody(input, until, cancel)?;
    manifest(input)?;
    for target in [&input.target_root, &input.work_root] {
        if std::fs::symlink_metadata(target).is_ok() {
            return Err("finite probe workspace must be fresh".into());
        }
        let parent = target
            .parent()
            .ok_or("probe target parent")?
            .canonicalize()?;
        if parent != target.parent().ok_or("probe parent")? {
            return Err("probe workspace canonical parent required".into());
        }
    }
    if !contract::disjoint(root, &input.source_root)
        || !contract::disjoint(root, &input.target_root)
        || !contract::disjoint(root, &input.work_root)
    {
        return Err("probe evidence workspace overlaps source/output".into());
    }
    let source_directory = input.source_root.join(&input.source_prefix);
    if source_directory.canonicalize()? != source_directory {
        return Err("source prefix must be canonical".into());
    }
    let mut actual = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&source_directory)? {
        check(until, cancel)?;
        actual.insert(entry?.path());
        if actual.len() > 2 {
            return Err("finite source prefix exceeds closed roster".into());
        }
    }
    if actual != input.source_parts.iter().map(|a| a.path.clone()).collect() {
        return Err("finite source prefix full roster refused".into());
    }
    for a in &input.source_parts {
        gguf(a)?;
    }
    check(until, cancel)
}
fn verify(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    let verified = observed(
        input,
        vec![
            "verify-job".into(),
            "--manifest".into(),
            input.manifest.path.to_string_lossy().into(),
            "--llama-load".into(),
            "--llama-cli".into(),
            input.loader.path.to_string_lossy().into(),
            "--check-tensors".into(),
            "--json".into(),
        ],
        root,
        "verify",
        until,
        cancel,
        evidence,
    )?;
    let a = &verified["artifact"];
    if a["complete"] != true
        || a["expected_splits"] != 2
        || a["completed_count"] != 2
        || a["root"] != input.target_root.to_string_lossy().as_ref()
        || a["prefix"] != input.target_prefix
        || a["basename"] != input.basename
        || verified["llama_load"]["success"] != true
        || verified["llama_load"]["status_code"] != 0
        || verified["llama_load"]["llama_cli"] != input.loader.path.to_string_lossy().as_ref()
        || verified["llama_load"]["model"]
            != input.golden_outputs[0].path.to_string_lossy().as_ref()
    {
        return Err("finite golden GGUF verification/load profile refused".into());
    }
    evidence["verification"] = verified;
    check(until, cancel)
}
/// Component observations only: caller must hold its own Interrupt until final admission.
/// No Job, publication, cleanup or tool-profile qualification is authorized by this return.
fn execute(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    prepare(input, root, until, cancel)?;
    for i in 1..=2 {
        check(until, cancel)?;
        let label = format!("window-{i}");
        let p = execution::run_process(
            &input.tool.path,
            input.quant_args(),
            root,
            &label,
            until,
            cancel,
        )?;
        evidence[&label] = publication::process_observation(&p);
        if !execution::clean(&p) {
            return Err("supplied quantizer finite window failed; no fallback".into());
        }
        check(until, cancel)?;
        status(input, i, root, until, cancel, evidence)?;
        for a in &input.golden_outputs[..i] {
            gguf(a)?;
            pin(a, until, cancel)?;
        }
        if i == 1 && std::fs::symlink_metadata(&input.golden_outputs[1].path).is_ok() {
            return Err("quantizer did not honor one-window residency profile".into());
        }
        custody(input, until, cancel)?;
    }
    verify(input, root, until, cancel, evidence)?;

    for a in &input.golden_outputs {
        pin(a, until, cancel)?;
    }
    custody(input, until, cancel)?;
    evidence["tool_profile_qualified"] = json!(false);
    evidence["remote_mutation_performed"] = json!(false);
    check(until, cancel)
}

/// Explicit finite probe only; raw observations are not a durable admission receipt.
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, path, b, output] = args else {
        return Err("quantizer-window-probe --input FILE --output-directory FRESH".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("quantizer probe closed flags".into());
    }
    let input: Input = serde_json::from_slice(&admission::read(Path::new(path), 1048576)?)?;
    input.validate()?;
    let root = std::path::absolute(output)?;
    let root = root
        .parent()
        .ok_or("probe output parent")?
        .canonicalize()?
        .join(root.file_name().ok_or("probe output leaf")?);
    if [&input.source_root, &input.target_root, &input.work_root]
        .iter()
        .any(|p| !contract::disjoint(&root, p))
    {
        return Err("probe output must be disjoint from source/work/target".into());
    }
    if [
        &input.tool,
        &input.tool_source,
        &input.runtime,
        &input.loader,
        &input.manifest,
        &input.tensor_recipe,
    ]
    .into_iter()
    .chain(input.source_parts.iter())
    .any(|a| a.path.starts_with(&root))
    {
        return Err("probe output cannot own immutable input pins".into());
    }
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + std::time::Duration::from_secs(input.timeout_seconds);
    let mut evidence = json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&input)?),
        "status":"OBSERVATIONS_ONLY","tool_profile_qualified":false,"remote_mutation_performed":false,
        "completed_job":false,"workflow_allowance_seconds":input.workflow.job_allowance_seconds()});
    let result = execute(&input, &root, until, &cancel, &mut evidence);
    final_publication::finish(
        &mut evidence,
        &root.join("observations.json"),
        until,
        interrupt,
        result,
        &mut || Ok(()),
    )
}
