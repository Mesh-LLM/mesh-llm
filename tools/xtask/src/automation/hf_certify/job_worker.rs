//! Native certification and separately typed supplied/raw composition with pinned durable JSON evidence.
#[path = "job_worker/composition.rs"]
mod composition;
#[path = "job_worker/contract.rs"]
mod contract;
#[path = "job_worker/native_composition.rs"]
mod native_composition;
#[path = "job_worker/operator.rs"]
mod operator;
#[path = "job_worker/receipt_export.rs"]
mod receipt_export;
use super::{acquisition, admission, bootstrap};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};

fn check(deadline: Instant, cancellation: &Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        return Err("native job cancelled".into());
    }
    if Instant::now() >= deadline {
        return Err("native job shared deadline expired".into());
    }
    Ok(())
}
fn runner(
    identity: &admission::Artifact,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    check(deadline, cancel)?;
    let actual = std::env::current_exe()?.canonicalize()?;
    if actual != identity.path.canonicalize()?
        || bootstrap::execution::observe(&actual, deadline, cancel)? != identity.sha256
    {
        return Err("native job runner identity mismatch".into());
    }
    check(deadline, cancel)
}
fn execute(
    input: &contract::Input,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    input.validate()?;
    runner(&input.runner, deadline, cancel)?;
    let observed = bootstrap_phase(&input.bootstrap, root, deadline, cancel, evidence)?;
    check(deadline, cancel)?;
    let (phase_deadline, phase_secs) = certification_window(deadline, cancel)?;
    let request = acquisition::Request {
        schema_version: 1,
        certification: input.certification.bind(&observed, phase_secs)?,
        projector: input.projector.clone(),
    };
    request.validate()?;
    evidence["acquisition_request_sha256"] =
        json!(admission::digest(&serde_json::to_vec(&request)?));
    let acquisition_root = root.join("acquisition");
    std::fs::create_dir(&acquisition_root)?;
    acquisition::execute(
        &request,
        &acquisition_root,
        phase_deadline,
        cancel,
        &mut evidence["acquisition"],
    )?;
    runner(&input.runner, phase_deadline, cancel)?;
    check(phase_deadline, cancel)
}
fn bootstrap_phase(
    input: &bootstrap::contract::Input,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<bootstrap::execution::ObservedBootstrap> {
    let bootstrap_root = root.join("bootstrap");
    std::fs::create_dir(&bootstrap_root)?;
    let mut phases = Vec::new();
    let observed =
        bootstrap::execution::execute(input, &bootstrap_root, deadline, cancel, &mut phases);
    evidence["bootstrap"]["phases"] = json!(phases);
    let observed = observed?;
    evidence["bootstrap"]["observed"] = json!(&observed);
    evidence["bootstrap"]["status"] = json!("BOOTSTRAP_COMPLETED");
    Ok(observed)
}
fn certification_window(deadline: Instant, cancel: &Cancellation) -> DynResult<(Instant, u64)> {
    check(deadline, cancel)?;
    let phase_deadline = deadline.min(Instant::now() + Duration::from_secs(3600));
    let phase_secs = phase_deadline
        .checked_duration_since(Instant::now())
        .map(|d| d.as_secs())
        .filter(|s| *s >= 5)
        .ok_or("native job has no legal certification allowance after bootstrap")?;
    // Existing G1 child owner subtracts its3s cleanup reserve from this same absolute deadline.
    Ok((phase_deadline, phase_secs))
}
fn publication_credential(
    root: &Path,
    value: Option<std::ffi::OsString>,
) -> DynResult<tempfile::NamedTempFile> {
    let value = value.ok_or("explicit publication secret absent")?;
    let value = value.to_str().ok_or("publication secret Unicode refused")?;
    if value.is_empty()
        || value.len() > 8192
        || value
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        return Err("publication secret grammar/bound refused".into());
    }
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    use std::io::Write as _;
    file.write_all(value.as_bytes())?;
    file.as_file_mut().sync_all()?;
    Ok(file)
}
fn environment_input(value: Option<std::ffi::OsString>) -> DynResult<Vec<u8>> {
    let value = value.ok_or("native job input environment missing")?;
    let value = value
        .to_str()
        .ok_or("native job input environment must be UTF-8")?;
    if value.is_empty() || value.len() > 65536 {
        return Err("native job input environment byte bound".into());
    }
    Ok(value.as_bytes().to_vec())
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    if let [verb, rest @ ..] = args {
        match verb.as_str() {
            "operator" => return operator::run(rest),
            "operator-identity" => return operator::identity(rest),
            _ => (),
        }
    }
    let [a, input_path, b, output] = args else {
        return Err("hf-certify job-worker --input FILE --output-directory FRESH_DIRECTORY".into());
    };
    if b != "--output-directory" {
        return Err("native job closed flags".into());
    }
    let bytes = match a.as_str() {
        "--input" => admission::read(Path::new(input_path), 262144)?,
        "--input-environment" if input_path == "MESH_HF_JOB_INPUT" => {
            environment_input(std::env::var_os("MESH_HF_JOB_INPUT"))?
        }
        _ => return Err("native job closed input transport refused".into()),
    };
    let input: contract::JobInput =
        serde_json::from_slice(&bytes).map_err(|_| "native job typed input refused")?;
    input.validate()?;
    let requested = std::path::absolute(output)?;
    let parent = requested
        .parent()
        .ok_or("native job output parent")?
        .canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("native job output leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs());
    let mut evidence = json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&input)?),"transport_input_sha256":admission::digest(&bytes),"status":"FAILED","bootstrap":{"status":"FAILED","phases":[]},"acquisition_request_sha256":null,"acquisition":{"certification":{}},"error":null,"image_observed":false,"rate_or_cost_observed":false,"publication_completed":false,"host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH}});
    let credential = if input
        .receipt_export()
        .is_some_and(|e| e.credential_environment)
    {
        Some(publication_credential(
            &root,
            std::env::var_os("MESH_HF_PUBLICATION_TOKEN"),
        )?)
    } else {
        None
    };
    let native_deadline = input.receipt_export().map_or(deadline, |e| {
        deadline - Duration::from_secs(e.export_budget_secs)
    });
    evidence["receipt_export_reserved_secs"] =
        json!(input.receipt_export().map(|e| e.export_budget_secs));
    let result = match &input {
        contract::JobInput::Certification(i) => {
            execute(i, &root, native_deadline, &cancel, &mut evidence).map(|_| None)
        }
        contract::JobInput::Composition(i) => {
            composition::execute(i, &root, native_deadline, &cancel, &mut evidence).map(Some)
        }
        contract::JobInput::Native(i) => {
            native_composition::execute(i, &root, native_deadline, &cancel, &mut evidence).map(Some)
        }
    };
    let pending_plan = result.as_ref().ok().and_then(|p| p.as_ref()).cloned();
    let native_complete = result.is_ok() && check(native_deadline, &cancel).is_ok();
    evidence["status"] = json!(if native_complete {
        input.native_status()
    } else {
        "FAILED"
    });
    if !native_complete {
        evidence["error"] = json!("native job phase ownership refused; partial evidence retained");
    }
    let native_path = root.join("native-job.json");
    admission::publish(&native_path, &evidence)?;
    let mut export_evidence = json!({"completed":false});
    let exported = match input.receipt_export() {
        Some(config) => receipt_export::execute(
            config,
            &native_path,
            &root,
            deadline,
            &cancel,
            credential.as_ref().map(|f| f.path()),
            &mut export_evidence,
        )
        .map(Some),
        None => Ok(None),
    };
    let credential_cleanup = credential.map_or(Ok(()), tempfile::NamedTempFile::close);
    let finished = interrupt.finish();
    let mut complete = native_complete
        && exported.is_ok()
        && credential_cleanup.is_ok()
        && finished.is_ok()
        && check(deadline, &cancel).is_ok();
    let mut plan_owned = false;
    if complete && let Some(mut plan) = pending_plan {
        plan["request_sha256"] = evidence["request_sha256"].clone();
        plan["required_receipt"] = json!({"path":"native-job.json","status":"COMPOSED","request_sha256":evidence["request_sha256"],"sha256":admission::digest(&admission::read(&native_path,1024*1024)?),"terminal_delivery_path":"native-job-delivery.json","consumer_rule":"require matching successful terminal delivery plus all entry pins; plan alone is not remote publication authority"});
        plan_owned = admission::publish(&root.join("composition-plan.json"), &plan).is_ok();
        complete = plan_owned && check(deadline, &cancel).is_ok();
        if !complete && plan_owned {
            std::fs::remove_file(root.join("composition-plan.json"))?;
            plan_owned = false;
        }
    }
    let mut locator = exported.ok().flatten();
    if let Some(value) = locator.as_mut() {
        value.delivery_complete = complete;
    }
    let delivery = json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&input)?),"status":if complete{if locator.is_some(){"DELIVERED"}else{match &input{contract::JobInput::Certification(_)=>"LOCAL_CERTIFIED",contract::JobInput::Composition(_)|contract::JobInput::Native(_)=>"LOCAL_COMPOSED"}}}else{"FAILED"},"native_receipt_sha256":admission::digest(&admission::read(&native_path,1024*1024)?),"export":export_evidence,"locator":locator,"image_observed":false,"rate_or_cost_observed":false});
    if let Err(error) = admission::publish(&root.join("native-job-delivery.json"), &delivery) {
        if plan_owned {
            std::fs::remove_file(root.join("composition-plan.json"))?;
        }
        return Err(error);
    }
    if let Some(value) = locator {
        println!("MESH_NATIVE_DELIVERY {}", serde_json::to_string(&value)?);
    }
    if complete {
        Ok(())
    } else {
        Err("native job or durable receipt export failed; partial evidence retained".into())
    }
}
#[cfg(test)]
#[path = "job_worker/tests.rs"]
mod tests;
