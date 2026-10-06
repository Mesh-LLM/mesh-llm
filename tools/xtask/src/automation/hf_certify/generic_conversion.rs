//! Original split/spool/status/card composition; supplied native tools remain declared prerequisites.
#[path = "generic_conversion/artifact_workspace.rs"]
mod artifact_workspace;
#[path = "generic_conversion/contract.rs"]
mod contract;
#[path = "generic_conversion/delivery.rs"]
mod delivery;
pub(super) fn artifact_workspace_worker(args: &[String]) -> DynResult<()> {
    artifact_workspace::worker(args)
}
#[path = "generic_conversion/operator.rs"]
mod operator;
pub(super) fn delivery(args: &[String]) -> DynResult<()> {
    delivery::run(args)
}
pub(super) fn operator(args: &[String]) -> DynResult<()> {
    operator::run(args)
}
#[path = "generic_conversion/execution.rs"]
mod execution;
#[path = "generic_conversion/helper_contract.rs"]
mod helper_contract;
#[path = "generic_conversion/identity.rs"]
mod identity;
#[path = "generic_conversion/publication.rs"]
mod publication;
#[path = "generic_conversion/receipt.rs"]
mod receipt;
#[path = "generic_conversion/workspace.rs"]
mod workspace;
use crate::{
    automation::{command_interrupt::Interrupt, hf_certify::admission},
    command::DynResult,
    process::Cancellation,
};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(super) fn guard(until: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= until {
        return Err("generic conversion terminal cancellation/deadline".into());
    }
    Ok(())
}
pub(super) fn identity_worker(args: &[String]) -> DynResult<()> {
    identity::worker(args)
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, path, b, output] = args else {
        return Err(
            "hf-certify generic-conversion --input FILE --output-directory FRESH_DIRECTORY".into(),
        );
    };
    if a != "--input" || b != "--output-directory" {
        return Err("generic conversion closed flags".into());
    }
    let input: contract::Input =
        serde_json::from_slice(&admission::read(Path::new(path), 8 * 1048576)?)?;
    input.validate()?;
    let root = std::path::absolute(output)?;
    let root = root
        .parent()
        .ok_or("generic evidence parent")?
        .canonicalize()?
        .join(root.file_name().ok_or("generic evidence leaf")?);
    let work = input.work_directory.canonicalize()?;
    if work != input.work_directory || root.starts_with(&work) || work.starts_with(&root) {
        return Err("generic evidence/work custody overlap".into());
    }
    if !input.upload_only {
        let source = input.source.canonicalize()?;
        if source != input.source
            || work.starts_with(&source)
            || source.starts_with(&work)
            || root.starts_with(&source)
            || source.starts_with(&root)
        {
            return Err("generic source/work/evidence custody overlap".into());
        }
    }
    workspace::admit(&input)?;
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + Duration::from_secs(input.timeout_seconds);
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"pre_identity":null,"convert_process":null,"verify_process":null,"verification":null,"artifact_roster":null,"local_completed":false,"dry_run_completed":false,"publication_completed":false,"repository":null,"model_publication":null,"error":null,"scope":"supplied static native binary and complete local source/output bytes; no build or hosted/model qualification"});
    let result = execution::execute(&input, &root, until, &cancel, &mut evidence).and_then(|()| {
        if input.publish_confirmed {
            publication::execute(&input, &root, until, &cancel, &mut evidence)
        } else {
            Ok(())
        }
    });
    let finished = interrupt.finish();
    let terminal = guard(until, &cancel);
    let complete = result.is_ok() && finished.is_ok() && terminal.is_ok();
    if complete {
        evidence["status"] = json!(if input.dry_run {
            "DRY_RUN_COMPLETED"
        } else if input.publish_confirmed {
            "PUBLISHED"
        } else {
            "LOCAL_ARTIFACT_READY"
        });
    } else {
        evidence["error"] = json!(
            result
                .err()
                .map(|e| e.to_string())
                .into_iter()
                .chain(finished.err().map(|e| e.to_string()))
                .chain(terminal.err().map(|e| e.to_string()))
                .collect::<Vec<_>>()
                .join("; ")
        );
        evidence["local_completed"] = json!(false);
        evidence["dry_run_completed"] = json!(false);
        evidence["publication_completed"] = json!(false);
    }
    admission::publish(&root.join("generic-conversion.json"), &evidence)?;
    if complete {
        Ok(())
    } else {
        Err("generic conversion incomplete; native work and partial evidence retained".into())
    }
}
#[cfg(test)]
#[path = "generic_conversion/tests.rs"]
mod tests;
