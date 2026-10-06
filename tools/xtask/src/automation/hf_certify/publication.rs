//! Pinned isolated publisher subprocess. Transport admission is not public HF qualification.
#[path = "publication/contract.rs"]
mod contract;
#[path = "publication/receipt.rs"]
mod receipt;
use super::{admission, bootstrap, execution};
use crate::{
    command::DynResult,
    process::{self, Cancellation},
};
pub(super) use contract::Request;
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn check(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err("publication shared deadline/cancellation refused".into());
    }
    Ok(())
}
fn pins(request: &Request, deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    for pin in [&request.helper, &request.helper_source] {
        check(deadline, cancel)?;
        if bootstrap::execution::observe(&pin.path, deadline, cancel)? != pin.sha256 {
            return Err("publication helper/source identity mismatch".into());
        }
    }
    check(deadline, cancel)
}
pub(super) fn process_observation(report: &process::RawProcessReport) -> Value {
    let p = &report.process;
    let stream = |s: &process::StreamReport, raw: &Option<process::RawBytes>| {
        json!({"bytes_seen":s.bytes_seen,
        "raw_bytes_captured":raw.as_ref().map(|v|v.as_bytes().len()),"line_capture_complete":s.line_capture_complete,
        "truncated":s.truncated,"oversized_lines":s.oversized_lines,"suppressed_lines":s.suppressed_lines})
    };
    json!({"outcome":format!("{:?}",p.outcome),"exit_code":p.status.and_then(|s|s.code()),"failure_present":p.failure.is_some(),
        "cleanup":{"complete":p.cleanup.complete,"forced":p.cleanup.forced,"graceful_signal_failed":p.cleanup.graceful_signal_failed,
            "failure_present":p.cleanup.failure.is_some()},"stdout":stream(&p.stdout,&report.stdout),"stderr":stream(&p.stderr,&report.stderr)})
}
fn observed(
    path: &Path,
    input: &contract::PublisherInput,
    hash: &str,
    progress: bool,
) -> DynResult<Option<(receipt::Receipt, Value)>> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("publication receipt metadata refused".into()),
        Ok(metadata) if !metadata.is_file() => {
            return Err("publication receipt regular file refused".into());
        }
        _ => (),
    }
    let receipt: receipt::Receipt = serde_json::from_slice(&admission::read(path, 512 * 1024)?)?;
    let projection = receipt.observe(input, hash, progress)?;
    Ok(Some((receipt, projection)))
}
pub(super) fn execute(
    request: &Request,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    *evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":null,"input_timeout_ms":null,
        "helper_source_scope":"supplied source-file identity; no binary build attribution","pre_identity":false,"post_identity":false,
        "process":null,"partial_progress":null,"final_receipt":null,"progress_error":false,"final_error":false,"error":null});
    let result = execute_owned(request, root, deadline, cancel, evidence);
    let terminal = check(deadline, cancel);
    if result.is_ok() && terminal.is_ok() {
        evidence["status"] = json!("PUBLISHED");
        Ok(())
    } else {
        evidence["error"] =
            json!("publication child or terminal custody refused; partial observations retained");
        Err("publication child failed; partial observations retained".into())
    }
}
fn execute_owned(
    request: &Request,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    request.validate()?;
    if !cfg!(unix) {
        return Err("publisher helper adapter requires Unix safe FD ownership".into());
    }
    check(deadline, cancel)?;
    if !root.is_absolute()
        || !std::fs::symlink_metadata(root)?.is_dir()
        || std::fs::read_dir(root)?.next().is_some()
    {
        return Err(
            "publication owning phase directory must be absolute regular empty directory".into(),
        );
    }
    let root = root.canonicalize()?;
    pins(request, deadline, cancel)?;
    evidence["pre_identity"] = json!(true);
    let allowance = deadline
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .filter(|d| !d.is_zero())
        .ok_or("publication cleanup reserve leaves no execution allowance")?;
    let mut input = request.input.clone();
    input.execution_timeout_ms = input
        .execution_timeout_ms
        .min(u64::try_from(allowance.as_millis())?);
    if input.execution_timeout_ms == 0 {
        return Err("publication submillisecond allowance refused".into());
    }
    let hash = admission::digest(&serde_json::to_vec(&input)?);
    evidence["request_sha256"] = json!(&hash);
    evidence["input_timeout_ms"] = json!(input.execution_timeout_ms);
    let path = root.join("publisher-input.json");
    admission::publish(&path, &input)?;
    let helper_output = root.join("helper-output");
    let args = vec![
        "publish".into(),
        "--input".into(),
        path.to_str().ok_or("publisher input Unicode")?.into(),
        "--output-directory".into(),
        helper_output
            .to_str()
            .ok_or("publisher output Unicode")?
            .into(),
    ];
    let report = execution::run_process(
        &request.helper.path,
        args,
        &root,
        "publisher",
        deadline,
        cancel,
    )?;
    evidence["process"] = process_observation(&report);
    if !std::fs::symlink_metadata(&helper_output).is_ok_and(|metadata| metadata.is_dir()) {
        evidence["progress_error"] = json!(true);
        evidence["final_error"] = json!(true);
        return Err("publication child output directory refused; no receipt followed".into());
    }
    let progress = observed(&helper_output.join("progress.json"), &input, &hash, true);
    match progress {
        Ok(Some((_, value))) => evidence["partial_progress"] = value,
        Ok(None) => (),
        Err(_) => evidence["progress_error"] = json!(true),
    }
    let final_receipt = observed(
        &helper_output.join("publication.json"),
        &input,
        &hash,
        false,
    );
    let accepted = match final_receipt {
        Ok(Some((receipt, value))) => {
            evidence["final_receipt"] = value;
            receipt.accepted()
        }
        _ => {
            evidence["final_error"] = json!(true);
            false
        }
    };
    let post = pins(request, deadline, cancel);
    evidence["post_identity"] = json!(post.is_ok());
    if !execution::clean(&report)
        || report.process.stdout.oversized_lines != 0
        || report.process.stderr.oversized_lines != 0
        || !accepted
        || post.is_err()
        || evidence["progress_error"] == true
    {
        return Err("publication child receipt/process/identity admission refused".into());
    }
    check(deadline, cancel)
}
/// Closed local frontend for root registration; the Jobs caller can use execute directly.
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [flag, input_path, outflag, output] = args else {
        return Err(
            "hf-certify publication-child --input FILE --output-directory FRESH_ABSOLUTE_DIRECTORY"
                .into(),
        );
    };
    if flag != "--input" || outflag != "--output-directory" {
        return Err("publication child closed flags".into());
    }
    let request: Request =
        serde_json::from_slice(&admission::read(Path::new(input_path), 512 * 1024)?)?;
    request.validate()?;
    let output = Path::new(output);
    if !output.is_absolute() {
        return Err("publication child output must be absolute".into());
    }
    let parent = output
        .parent()
        .ok_or("publication output parent")?
        .canonicalize()?;
    let root = parent.join(output.file_name().ok_or("publication output name")?);
    std::fs::create_dir(&root)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now()
        .checked_add(
            Duration::from_millis(request.input.execution_timeout_ms)
                .checked_add(Duration::from_secs(3))
                .ok_or("publication timeout overflow")?,
        )
        .ok_or("publication deadline overflow")?;
    let mut evidence = json!({});
    let result = execute(&request, &root, deadline, &cancel, &mut evidence);
    let finished = interrupt.finish();
    let terminal = check(deadline, &cancel);
    let complete = result.is_ok() && finished.is_ok() && terminal.is_ok();
    if !complete {
        evidence["status"] = json!("FAILED");
        evidence["error"] = json!(
            "publication frontend or terminal custody refused; partial observations retained"
        );
    }
    admission::publish(&root.join("publication-child.json"), &evidence)?;
    if complete {
        Ok(())
    } else {
        Err("publication child frontend failed; partial observations retained".into())
    }
}
#[cfg(test)]
#[path = "publication/tests.rs"]
mod tests;
