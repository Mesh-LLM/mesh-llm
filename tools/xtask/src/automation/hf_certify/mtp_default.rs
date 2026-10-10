//! Immutable checkpoint staging, native composition, then one ordered public model commit.
#[path = "mtp_default/contract.rs"]
pub(super) mod contract;
#[path = "mtp_default/delivery.rs"]
pub(super) mod delivery;
#[path = "mtp_default/publication.rs"]
mod publication;
#[path = "mtp_default/staging.rs"]
mod staging;
use super::{admission, bootstrap};
use crate::{
    automation::{command_interrupt::Interrupt, hf_mtp_compose::raw_conversion},
    command::DynResult,
    process::Cancellation,
};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn check(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err("compose-default shared deadline/cancellation".into());
    }
    Ok(())
}
fn execute(
    input: &contract::Input,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<Value> {
    let work_deadline = deadline
        .checked_sub(Duration::from_secs(input.publication_reserve_seconds))
        .ok_or("publication reserve")?;
    check(work_deadline, cancel)?;
    let stage = root.join("staging");
    std::fs::create_dir(&stage)?;
    let staged = staging::execute(input, &stage, work_deadline, cancel, evidence)?;
    let build = root.join("bootstrap");
    std::fs::create_dir(&build)?;
    let mut rows = Vec::new();
    let result =
        bootstrap::execution::execute(&input.bootstrap, &build, work_deadline, cancel, &mut rows);
    evidence["bootstrap_phases"] = json!(rows);
    let observed = result?;
    evidence["observed_bootstrap"] = serde_json::to_value(&observed)?;
    let template = raw_conversion::Template {
        checkpoint_directory: serde_json::from_value(staged["checkpoint_directory"].clone())?,
        checkpoint_files: serde_json::from_value(staged["checkpoint_files"].clone())?,
        tokenizer_profile: serde_json::from_value(staged["tokenizer_profile"].clone())?,
        target_parts: input.target_parts.clone(),
        target_basename: input.target_basename.clone(),
        composite_basename: input.composite_basename.clone(),
        expected_parts: input.target_parts.len(),
        mtp_block: input.mtp_block,
        composite_repo: input.composite_repo.clone(),
    };
    let conversion = root.join("composition");
    std::fs::create_dir(&conversion)?;
    let plan = template.execute(
        &conversion,
        &raw_conversion::Context {
            binary: &observed.binary,
            mesh_revision: &observed.mesh_commit,
            deadline: work_deadline,
            cancellation: cancel,
        },
        &mut evidence["native_composition"],
    )?;
    check(work_deadline, cancel)?;
    let publish = root.join("publisher");
    std::fs::create_dir(&publish)?;
    publication::execute(input, &publish, &plan, deadline, cancel, evidence)?;
    check(deadline, cancel)?;
    Ok(plan)
}
/// Execute the same full composition phases under the caller's inherited deadline/cancellation.
pub(super) fn execute_inherited(
    bytes: &[u8],
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<(Value, DynResult<()>)> {
    let input: contract::Input = serde_json::from_slice(bytes)?;
    input.validate()?;
    if !input.dry_run && !cfg!(target_os = "linux") {
        return Err("composition Jobs execution is Linux-only".into());
    }
    std::fs::create_dir(root)?;
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"error":null,"staging_process":null,"staging_receipt":null,"bootstrap_phases":[],"observed_bootstrap":null,"native_composition":{},"repository_process":null,"repository_receipt":null,"ordered_publication":null,"real_family_qualified":false,"declared_image_measured":false});
    let result = if input.dry_run {
        check(deadline, cancel).map(|()| {
            evidence["status"] = json!("DRY_RUN_NOT_EXECUTED");
        })
    } else {
        let result = execute(&input, root, deadline, cancel, &mut evidence);
        finalize(result, Ok(()), deadline, cancel, &mut evidence)
    };
    Ok((evidence, result))
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, path, b, out] = args else {
        return Err("compose-default --input FILE --output-directory FRESH".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("compose-default closed flags".into());
    }
    let bytes = admission::read(Path::new(path), 1048576)?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    if !input.dry_run && !cfg!(target_os = "linux") {
        return Err(
            "compose-default observed bootstrap is Linux-only; no acquisition launched".into(),
        );
    }
    let requested = std::path::absolute(out)?;
    let parent = requested.parent().ok_or("output parent")?.canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("output leaf")?);
    std::fs::create_dir(&root)?;
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"request_transport_sha256":admission::digest(&bytes),"error":null,"staging_process":null,"staging_receipt":null,"bootstrap_phases":[],"observed_bootstrap":null,"native_composition":{},"repository_process":null,"repository_receipt":null,"ordered_publication":null,"real_family_qualified":false,"declared_image_measured":false});
    if input.dry_run {
        evidence["status"] = json!("DRY_RUN_NOT_EXECUTED");
        evidence["phase_plan"] = json!([
            "immutable-checkpoint-tokenizer-stitch",
            "owned-native-bootstrap",
            "BF16-native-conversion-verify",
            "first-metadata-last-tensors-attach",
            "public-repository-provision",
            "ordered-parent-bound-publication"
        ]);
        admission::publish(&root.join("report.json"), &evidence)?;
        return Ok(());
    }
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.overall_seconds);
    let result = execute(&input, &root, deadline, &cancel, &mut evidence).and_then(|plan| {
        if admission::read(Path::new(path), 1048576)? != bytes {
            return Err("compose-default original request drift".into());
        }
        Ok(plan)
    });
    let finish = interrupt.finish().map_err(|e| e.to_string().into());
    let result = finalize(result, finish, deadline, &cancel, &mut evidence);
    admission::publish(&root.join("report.json"), &evidence)?;
    result
}
fn finalize(
    result: DynResult<Value>,
    finish: DynResult<()>,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    match (result, finish) {
        (Ok(plan), Ok(())) if check(deadline, cancel).is_ok() => {
            evidence["status"] = json!("COMPOSED_PUBLISHED");
            evidence["publication_plan"] = plan;
            Ok(())
        }
        (r, f) => {
            evidence["error"] = json!(format!(
                "{}{}",
                r.err()
                    .map_or("terminal compose admission failed".into(), |e| e
                        .to_string()),
                f.err().map_or(String::new(), |e| format!("; {e}"))
            ));
            Err("compose-default failed; partial phases and mutation evidence retained".into())
        }
    }
}
#[cfg(test)]
#[path = "mtp_default/tests.rs"]
mod tests;
