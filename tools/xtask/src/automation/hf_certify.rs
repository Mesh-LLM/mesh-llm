//! Local certification, pinned acquisition/bootstrap and isolated publisher children. Job-worker can export correlated JSON evidence; local composition accepts supplied GGUF or explicitly pinned Nemotron checkpoint inputs.
mod acquisition;
pub(in crate::automation) mod admission;
mod bootstrap;
pub(in crate::automation) mod execution;
#[path = "hf_certify/generic_conversion.rs"]
mod generic_conversion;
#[path = "hf_certify/job_request_transport.rs"]
mod job_request_transport;
#[path = "hf_certify/job_worker.rs"]
mod job_worker;
#[path = "hf_certify/mtp_default.rs"]
mod mtp_default;
#[path = "hf_certify/publication.rs"]
mod publication;
#[path = "hf_certify/quantization_probe.rs"]
mod quantization_probe;
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let [verb, rest @ ..] = args
        && verb == "quantizer-window-probe"
    {
        return quantization_probe::run(rest);
    }
    if let Some((verb, rest)) = args.split_first()
        && verb == "composition-job-worker"
    {
        return mtp_default::delivery::run(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "artifact-workspace-worker"
    {
        return generic_conversion::artifact_workspace_worker(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "generic-job-worker"
    {
        return generic_conversion::delivery(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "generic-job"
    {
        return generic_conversion::operator(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "compose-default"
    {
        return mtp_default::run(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "generic-conversion"
    {
        return generic_conversion::run(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "generic-identity-worker"
    {
        return generic_conversion::identity_worker(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "publication-child"
    {
        return publication::run(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "job-worker"
    {
        return job_worker::run(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "bootstrap"
    {
        return bootstrap::run(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "acquire"
    {
        return acquisition::run(rest);
    }
    if let [verb, rest @ ..] = args
        && verb == "identity-worker"
    {
        return admission::worker(rest);
    }
    if args == ["--help"] {
        println!(
            "automation hf-certify --input FILE --output-directory FRESH_DIRECTORY; local standalone CPU product validation; job-worker operator selects mounted model-root/model-pattern/expected-parts and chains existing certification; job-worker chains bootstrap/acquisition/certification or distinct supplied-converted/native-nemotron composition with optional pinned receipt export; generic-conversion --input FILE --output-directory FRESH_DIRECTORY preserves supplied native split/spool/status/card and explicit complete-folder publication; publication-child --input FILE --output-directory FRESH_DIRECTORY supervises a supplied GGUF publisher; generic-job --input prepared-manifest --output-directory fresh-evidence observes G3 build and accepts original conversion flags; no hosted conversion qualification; compose-default --input FILE --output-directory FRESH chains immutable checkpoint staging/bootstrap/native attach/confirmed ordered publication, or dry-run"
        );
        return Ok(());
    }
    let [a, path, b, output] = args else {
        return Err("hf-certify --input FILE --output-directory FRESH_DIRECTORY".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("hf-certify closed flags".into());
    }
    let input: admission::Input =
        serde_json::from_slice(&admission::read(Path::new(path), 262144)?)?;
    input.validate()?;
    let requested = std::path::absolute(output)?;
    let parent = requested.parent().ok_or("output parent")?.canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("output leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"admitted":null,"native_report":null,"source_unchanged":false,"error":null,"profile_scope":"supplied standalone static CPU profile; real native loader/ABI acceptance belongs to product validation","host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH}});
    let result = execution::execute(&input, &root, deadline, &cancel, &mut evidence);
    let interruption = interrupt.finish();
    match (result, interruption) {
        (Ok(()), Ok(())) => evidence["status"] = json!("PASS"),
        (result, signal) => {
            evidence["error"] = json!(format!(
                "{}{}",
                result.err().map_or(String::new(), |e| e.to_string()),
                signal.err().map_or(String::new(), |e| format!("; {e}"))
            ))
        }
    }
    admission::publish(&root.join("report.json"), &evidence)?;
    println!("{}", root.join("report.json").display());
    if evidence["status"] == "PASS" {
        Ok(())
    } else {
        Err("HF certification failed; owned evidence retained".into())
    }
}

#[cfg(test)]
#[path = "hf_certify/tests.rs"]
mod tests;
