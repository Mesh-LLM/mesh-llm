//! Supplied quantizer Jobs under one deadline and the existing durable JSON exporter.
use super::{admission, bootstrap, job_worker::receipt_export, quant_job};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
#[path = "quant_job_delivery/contract.rs"]
mod contract;
#[path = "quant_job_delivery/final_publication.rs"]
mod final_publication;
use contract::Input;
fn guard(until: Instant, cancel: &Cancellation) -> DynResult<()> {
    if Instant::now() >= until || cancel.is_cancelled() {
        return Err("quant Jobs terminal deadline/cancel refused".into());
    }
    Ok(())
}
fn runner(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    guard(until, cancel)?;
    let actual = std::env::current_exe()?.canonicalize()?;
    if input.runner.path.canonicalize()? != actual
        || bootstrap::execution::observe_runner(&actual, until, cancel)? != input.runner.sha256
    {
        return Err("quant Jobs self-runner custody refused".into());
    }
    guard(until, cancel)
}
fn phase(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    credential: &Path,
    evidence: &mut Value,
) -> DynResult<()> {
    runner(input, until, cancel)?;
    let remaining = until
        .checked_duration_since(Instant::now())
        .map(|d| d.as_secs())
        .filter(|s| *s >= 5)
        .ok_or("quant Jobs remaining phase allowance")?;
    let mut operator = input.operator.clone();
    operator["timeout_seconds"] = json!(remaining);
    operator["window_template"]["credential_file"] = json!(credential);
    let operator: quant_job::contract::Input = serde_json::from_value(operator)?;
    operator.validate()?;
    evidence["operator_request_sha256"] = json!(admission::digest(&serde_json::to_vec(&operator)?));
    let operator_root = root.join("operator");
    std::fs::create_dir(&operator_root)?;
    let result = quant_job::execute(
        &operator,
        &operator_root,
        until,
        cancel,
        &mut evidence["operator"],
    );
    result?;
    let expected = if input.workflow == "quantization" {
        "QUANTIZATION_PUBLISHED"
    } else {
        "QUANTIZATION_PACKAGED"
    };
    let op = &evidence["operator"];
    if op["request_sha256"] != evidence["operator_request_sha256"]
        || op["status"] != expected
        || op["error"] != Value::Null
        || op["completed_job"] != true
        || op["full_roster_verified"] != true
        || op["verify_job"]["completed"] != true
        || op["final_commit"]
            .as_str()
            .is_none_or(|s| !bootstrap::contract::hex(s, 40))
        || input.workflow == "quantization-and-package"
            && (op["package"]["completed"] != true
                || op["package"]["final_commit"]
                    .as_str()
                    .is_none_or(|s| !bootstrap::contract::hex(s, 40)))
    {
        return Err("quant coordinator returned incomplete artifact/package proof".into());
    }
    runner(input, until, cancel)?;
    guard(until, cancel)
}
pub(in crate::automation::hf_certify) fn run(args: &[String]) -> DynResult<()> {
    let [flag, input, out_flag, output] = args else {
        return Err("quant-job-worker --input FILE|--input-environment MESH_HF_JOB_INPUT|--mounted-input-environment MESH_HF_JOB_REQUEST --output-directory FRESH".into());
    };
    if out_flag != "--output-directory" {
        return Err("quant worker closed flags".into());
    }
    let bytes = contract::transport(flag, input)?;
    let input: Input =
        serde_json::from_slice(&bytes).map_err(|_| "quant worker typed input refused")?;
    input.validate()?;
    let root = contract::root(output, &input)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + Duration::from_secs(input.timeout_secs);
    let native_until = until - Duration::from_secs(input.receipt_export.export_budget_secs);
    let mut evidence = json!({"schema_version":1,"workflow":input.workflow,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"transport_input_sha256":admission::digest(&bytes),"operator_request_sha256":null,"operator":null,"error":null,"image_observed":false,"cost_observed":false});
    let credential = contract::credential(&root);
    let result = match credential.as_ref() {
        Ok(file) => phase(
            &input,
            &root,
            native_until,
            &cancel,
            file.path(),
            &mut evidence,
        ),
        Err(_) => Err("quant Jobs credential admission refused".into()),
    };
    let candidate = result.is_ok() && guard(native_until, &cancel).is_ok();
    if candidate {
        evidence["status"] = json!(if input.workflow == "quantization" {
            "QUANTIZATION_PUBLISHED"
        } else {
            "QUANTIZATION_PACKAGED"
        });
    } else {
        evidence["error"] = json!("quant coordinator incomplete; bounded observed rows retained");
    }
    let path = root.join("native-job.json");
    let mut native = final_publication::OwnedReceipt::new(path.clone());
    native.write(&evidence)?;
    let mut export = json!({"completed":false});
    let exported = receipt_export::execute(
        &input.receipt_export,
        &path,
        &root,
        until,
        &cancel,
        credential.as_ref().ok().map(|f| f.path()),
        &mut export,
    );
    let cleaned = credential.map_or(Ok(()), tempfile::NamedTempFile::close);
    let candidate = candidate && exported.is_ok() && cleaned.is_ok();
    let mut delivery = json!({"schema_version":1,"workflow":input.workflow,"status":"FAILED","request_sha256":evidence["request_sha256"],"native_completed":candidate,"export":export,"locator":exported.ok(),"image_observed":false,"cost_observed":false});
    final_publication::finish(
        &root,
        &mut evidence,
        &mut native,
        &mut delivery,
        candidate,
        until,
        interrupt,
    )
}

#[cfg(all(test, unix))]
#[path = "quant_job_delivery/tests.rs"]
mod tests;
