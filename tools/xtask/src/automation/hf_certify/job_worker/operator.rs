//! Mounted layout discovery feeding the existing certification worker, never a second engine.
use super::{
    super::{admission, execution},
    contract,
};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    mode: admission::Mode,
    projector: admission::Artifact,
    layer_count: u32,
    mtp_layer_count: Option<u32>,
    ctx_size: u32,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Worker {
    schema_version: u32,
    workflow: contract::Workflow,
    timeout_secs: u64,
    runner: admission::Artifact,
    bootstrap: super::super::bootstrap::contract::Input,
    certification: Profile,
    projector: super::super::acquisition::Projector,
    #[serde(default)]
    receipt_export: Option<super::receipt_export::Config>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    model_root: PathBuf,
    model_pattern: String,
    expected_parts: usize,
    mtp_draft: Option<PathBuf>,
    worker: Worker,
}
impl Input {
    fn worker(
        &self,
        parts: Vec<admission::Artifact>,
        draft: Option<admission::Artifact>,
    ) -> DynResult<contract::Input> {
        let mut value = serde_json::to_value(&self.worker)?;
        let count = parts.len();
        value["certification"]["target_parts"] = json!(parts);
        value["certification"]["expected_parts"] = json!(count);
        value["certification"]["mtp_draft"] = json!(draft);
        let worker: contract::Input = serde_json::from_value(value)?;
        worker.validate()?;
        Ok(worker)
    }
    fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !(30..=86400).contains(&self.worker.timeout_secs)
            || self
                .worker
                .receipt_export
                .as_ref()
                .is_some_and(|e| e.credential_environment)
        {
            return Err("operator schema/budget/credential transport refused".into());
        }
        // Projector-only historically ignores the target layout and draft arguments.
        if self.worker.certification.mode == admission::Mode::ProjectorOnly {
            self.worker(Vec::new(), None)?;
            return Ok(());
        }
        if self.model_pattern.is_empty()
            || self.model_pattern.len() > 512
            || self.model_pattern.chars().any(char::is_control)
            || Path::new(&self.model_pattern)
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            || !(1..=1024).contains(&self.expected_parts)
            || self.mtp_draft.is_none()
        {
            return Err("operator relative pattern/nonempty target roster/draft refused".into());
        }
        glob::Pattern::new(&self.model_pattern)?;
        // The structural pin is internal only. Actual mounted bytes replace it before worker launch.
        let root = std::path::absolute(&self.model_root)?;
        let parts = (0..self.expected_parts)
            .map(|n| admission::Artifact {
                path: root.join(format!("__operator_placeholder_{n:04}.gguf")),
                sha256: "0".repeat(64),
            })
            .collect();
        let draft = admission::Artifact {
            path: root.join("__operator_draft.gguf"),
            sha256: "0".repeat(64),
        };
        self.worker(parts, Some(draft))?;
        Ok(())
    }
}
fn check(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err("operator cancelled/deadline".into());
    }
    Ok(())
}
fn select(input: &Input) -> DynResult<contract::Input> {
    input.validate()?;
    if input.worker.certification.mode == admission::Mode::ProjectorOnly {
        return input.worker(Vec::new(), None);
    }
    let root = input.model_root.canonicalize()?;
    if !std::fs::metadata(&root)?.is_dir() {
        return Err("operator model root directory required".into());
    }
    // Escape literal root metacharacters; pattern remains the operator's original glob.
    let pattern = format!(
        "{}/{}",
        glob::Pattern::escape(root.to_str().ok_or("operator Unicode root")?),
        input.model_pattern
    );
    let matches = glob::glob_with(
        &pattern,
        glob::MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: false,
        },
    )?;
    let mut paths = Vec::new();
    for item in matches {
        if paths.len() >= 1024 {
            return Err("operator roster exceeds bound".into());
        }
        paths.push(item?);
    }
    paths.sort();
    if paths.len() != input.expected_parts {
        return Err("operator expected target parts differs from matched roster".into());
    }
    let parts = paths
        .iter()
        .map(|p| admission::observe(p, true))
        .collect::<DynResult<Vec<_>>>()?;
    // Preserve sorted logical glob order through canonical identity admission; aliases may reverse canonical names.
    let draft = admission::observe(
        input.mtp_draft.as_ref().ok_or("operator draft absent")?,
        true,
    )?;
    input.worker(parts, Some(draft))
}
pub(super) fn identity(args: &[String]) -> DynResult<()> {
    let [a, source, b, output] = args else {
        return Err("operator identity closed flags".into());
    };
    if a != "--input" || b != "--output" {
        return Err("operator identity closed flags".into());
    }
    let input: Input = serde_json::from_slice(&admission::read(Path::new(source), 262144)?)?;
    admission::publish(Path::new(output), &select(&input)?)
}
fn selection(root: &Path, deadline: Instant, cancel: &Cancellation) -> DynResult<contract::Input> {
    let source = root.join("selection-input.json");
    let output = root.join("worker-input.json");
    let args = vec![
        "automation".into(),
        "hf-certify".into(),
        "job-worker".into(),
        "operator-identity".into(),
        "--input".into(),
        source.to_str().ok_or("source Unicode")?.into(),
        "--output".into(),
        output.to_str().ok_or("output Unicode")?.into(),
    ];
    let report = execution::run_process(
        &std::env::current_exe()?,
        args,
        root,
        "selection",
        deadline,
        cancel,
    )?;
    if !execution::clean(&report) {
        return Err("operator selection refusal; owned logs retained".into());
    }
    check(deadline, cancel)?;
    Ok(serde_json::from_slice(&admission::read(&output, 262144)?)?)
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, source, b, output] = args else {
        return Err("operator --input FILE --output-directory FRESH_DIRECTORY".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("operator closed flags".into());
    }
    let bytes = admission::read(Path::new(source), 262144)?;
    let mut input: Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    if input.worker.certification.mode == admission::Mode::MtpAttach {
        input.model_root = std::path::absolute(&input.model_root)?;
        input.mtp_draft = input
            .mtp_draft
            .as_ref()
            .map(std::path::absolute)
            .transpose()?;
    }
    let requested = std::path::absolute(output)?;
    let root = requested
        .parent()
        .ok_or("operator parent")?
        .canonicalize()?
        .join(requested.file_name().ok_or("operator leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    // Selection and the nested worker share this allowance; each child reserves owned shutdown time.
    let deadline = Instant::now() + Duration::from_secs(input.worker.timeout_secs);
    admission::publish(&root.join("selection-input.json"), &input)?;
    let result = execute(&root, deadline, &cancel);
    let finished = interrupt.finish();
    let terminal = check(deadline, &cancel);
    let complete = result.is_ok() && finished.is_ok() && terminal.is_ok();
    admission::publish(
        &root.join("operator.json"),
        &json!({"schema_version":1,"request_sha256":admission::digest(&bytes),"status":if complete{"CERTIFIED"}else{"FAILED"},"worker_delivery":"worker/native-job-delivery.json","error":result.err().map(|e|e.to_string()),"terminal_error":terminal.err().map(|e|e.to_string()),"signal_finish_failed":finished.is_err(),"custody":"observed supplied mounted roster; no hosted/model acceptance inferred"}),
    )?;
    if complete {
        println!("{}", root.join("certification-report.json").display());
        Ok(())
    } else {
        Err("operator certification failed; partial evidence retained".into())
    }
}
fn execute(root: &Path, deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    let mut worker = selection(root, deadline, cancel)?;
    let remaining=deadline.checked_duration_since(Instant::now()).and_then(|d|d.checked_sub(Duration::from_secs(3))).map(|d|d.as_secs()).filter(|s|*s>=30).ok_or("operator has no worker allowance after selection/cleanup reserve; increase timeout_secs")?;
    worker.timeout_secs = remaining;
    worker.bootstrap.timeout_seconds = remaining;
    worker.validate()?;
    let bounded = root.join("bounded-worker-input.json");
    admission::publish(&bounded, &worker)?;
    let args = vec![
        "automation".into(),
        "hf-certify".into(),
        "job-worker".into(),
        "--input".into(),
        bounded.to_str().ok_or("input Unicode")?.into(),
        "--output-directory".into(),
        root.join("worker").to_str().ok_or("worker Unicode")?.into(),
    ];
    let report = execution::run_process(
        &std::env::current_exe()?,
        args,
        root,
        "worker",
        deadline,
        cancel,
    )?;
    if !execution::clean(&report) {
        return Err("operator worker refusal; partial observations retained".into());
    }
    let delivery: serde_json::Value = serde_json::from_slice(&admission::read(
        &root.join("worker/native-job-delivery.json"),
        1048576,
    )?)?;
    if !["LOCAL_CERTIFIED", "DELIVERED"]
        .iter()
        .any(|s| delivery["status"] == *s)
        || delivery["request_sha256"]
            != admission::digest(&serde_json::to_vec(&contract::JobInput::Certification(
                worker,
            ))?)
    {
        return Err("operator terminal worker delivery mismatch".into());
    }
    let native_bytes = admission::read(&root.join("worker/native-job.json"), 1048576)?;
    let native: serde_json::Value = serde_json::from_slice(&native_bytes)?;
    if delivery["native_receipt_sha256"] != admission::digest(&native_bytes)
        || native["status"] != "CERTIFIED"
        || !native["error"].is_null()
    {
        return Err("operator native receipt/delivery correlation refused".into());
    }
    check(deadline, cancel)?;
    let report = &native["acquisition"]["certification"]["native_report"];
    if !report.is_object() {
        return Err("operator native report missing".into());
    }
    admission::publish(&root.join("certification-report.json"), report)?;
    check(deadline, cancel)
}
