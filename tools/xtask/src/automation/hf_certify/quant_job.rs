//! Sequential whole-roster quantization; existing owners perform every native/HF effect.
#[cfg(all(test, unix))]
#[path = "quant_job/chain_tests.rs"]
pub(in crate::automation::hf_certify) mod chain_tests;
#[path = "quant_job/contract.rs"]
pub(in crate::automation::hf_certify) mod contract;
#[path = "quant_job/evidence.rs"]
mod evidence;
#[path = "quant_job/final_commit.rs"]
mod final_commit;
#[path = "quant_job/package.rs"]
mod package;
#[cfg(test)]
#[path = "quant_job/tests.rs"]
mod tests;
use super::{admission, bootstrap, execution, publication, quantization_window as window};
use crate::{command::DynResult, process::Cancellation};
use contract::Input;
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn check(until: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= until {
        return Err("quant job inherited deadline/cancellation refused".into());
    }
    Ok(())
}
fn seconds(until: Instant, cancel: &Cancellation) -> DynResult<u64> {
    check(until, cancel)?;
    let n = until
        .saturating_duration_since(Instant::now())
        .as_secs()
        .min(86400);
    if n <= 3 {
        return Err("quant job cleanup margin exhausted".into());
    }
    Ok(n - 3)
}
fn child(
    binary: &admission::Artifact,
    args: Vec<String>,
    root: &Path,
    label: &str,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<crate::process::RawProcessReport> {
    window::pin(binary, until, cancel)?;
    let p = execution::run_process(&binary.path, args, root, label, until, cancel)?;
    evidence[label] = publication::process_observation(&p);
    window::pin(binary, until, cancel)?;
    if !execution::clean(&p) {
        return Err("quant job owned child refused; observations retained".into());
    }
    check(until, cancel)?;
    Ok(p)
}
fn observed(
    binary: &admission::Artifact,
    args: Vec<String>,
    root: &Path,
    label: &str,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<Value> {
    let p = child(binary, args, root, label, until, cancel, evidence)?;
    Ok(serde_json::from_slice(
        p.stdout.as_ref().ok_or("quant stdout absent")?.as_bytes(),
    )?)
}
fn source_custody(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    for a in input.window_template.pins() {
        window::pin(a, until, cancel)?;
    }
    Ok(())
}
fn initial(input: &Input) -> DynResult<Value> {
    Ok(
        json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(input)?),
        "status":"FAILED","error":null,"completed_job":false,"tool_profile_qualified":false,
        "workflow_qualified":false,"full_roster_verified":false,"final_commit":null,
        "verify_job":null,"package":null,"windows":[],"workflow_allowance_seconds":input.workflow.allowance()}),
    )
}
/// Outer delivery owns Interrupt, final durable publication and export; this function never renews its deadline.
pub(in crate::automation::hf_certify) fn execute(
    input: &Input,
    root: &Path,
    inherited: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    *evidence = initial(input)?;
    if let Err(error) = input.validate() {
        evidence["error"] = json!("quant request admission refused");
        return Err(error);
    }
    let until = inherited.min(Instant::now() + Duration::from_secs(input.timeout_seconds));
    let result = execute_inner(input, root, until, cancel, evidence);
    if result.is_err() || check(until, cancel).is_err() {
        evidence["completed_job"] = json!(false);
        evidence["status"] = json!("FAILED");
        evidence["error"] =
            json!("quant job phase or terminal refused; inspect retained observations");
        return result.and(Err("quant job incomplete; observations retained".into()));
    }
    if serde_json::to_vec(evidence)?.len() > 1048576 {
        evidence["error"] = json!("quant terminal evidence bound exceeded");
        return Err("quant terminal evidence bound exceeded".into());
    }
    check(until, cancel)?;
    evidence["completed_job"] = json!(true);
    evidence["status"] = json!(if input.package.is_some() {
        "QUANTIZATION_PACKAGED"
    } else {
        "QUANTIZATION_PUBLISHED"
    });
    Ok(())
}
fn execute_inner(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    input.validate()?;
    check(until, cancel)?;
    if root.canonicalize()? != root
        || !std::fs::symlink_metadata(root)?.is_dir()
        || [
            &input.window_template.source_root,
            &input.window_template.target_root,
            &input.window_template.work_root,
        ]
        .iter()
        .any(|p| !window::contract::disjoint(root, p))
        || input
            .window_template
            .pins()
            .iter()
            .any(|a| a.path.starts_with(root))
        || input.loader.path.starts_with(root)
        || input.window_template.credential_file.starts_with(root)
    {
        return Err("quant job evidence overlaps immutable/mutable input".into());
    }
    window::pin(&input.loader, until, cancel)?;
    source_custody(input, until, cancel)?;
    if let Some(p) = &input.package {
        for a in [&p.writer, &p.writer_source, &p.generation_defaults] {
            if a.path.starts_with(root) {
                return Err("package immutable input overlaps evidence".into());
            }
            window::pin(a, until, cancel)?;
        }
    }
    evidence::repository(
        input,
        &input.window_template.target_repo,
        window::helper::Context {
            root,
            until,
            cancel,
            evidence,
            label: "quant-repository",
        },
    )?;
    let mut roster = windows(input, root, until, cancel, evidence, window::execute)?;
    source_custody(input, until, cancel)?;
    let artifact_root =
        final_commit::publish_and_verify(input, root, until, cancel, &mut roster, evidence)?;
    final_commit::native_verify(input, &artifact_root, root, until, cancel, evidence)?;
    if let Some(p) = &input.package {
        package::execute(input, p, &artifact_root, root, until, cancel, evidence)?;
    }
    source_custody(input, until, cancel)?;
    check(until, cancel)
}
fn windows<F>(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
    mut run: F,
) -> DynResult<Vec<Value>>
where
    F: FnMut(&window::contract::Input, &Path, Instant, &Cancellation, &mut Value) -> DynResult<()>,
{
    let mut roster = Vec::new();
    for ordinal in 1..=input.window_template.expected_splits {
        check(until, cancel)?;
        let phase_root = root.join(format!("window-{ordinal:05}"));
        std::fs::create_dir(&phase_root)?;
        let request = input.window(ordinal)?;
        let mut row = json!({"ordinal":ordinal,"window_uploaded":false,"completed_job":false});
        let phase_until = until.min(Instant::now() + Duration::from_secs(request.timeout_seconds));
        let result = run(&request, &phase_root, phase_until, cancel, &mut row);
        let observed = evidence::retain(&phase_root, "window", &row)?;
        evidence["windows"].as_array_mut().ok_or("quant windows absent")?.push(json!({
            "ordinal":ordinal,"window_uploaded":row["window_uploaded"],"error":row["error"],"observation":observed}));
        result?;
        final_commit::window_artifacts(&request, &phase_root, &row, &mut roster)?;
    }
    Ok(roster)
}
