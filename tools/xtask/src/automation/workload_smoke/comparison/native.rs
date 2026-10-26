use crate::{
    automation::command_interrupt::Interrupt,
    command::DynResult,
    process::{self, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value},
};
use serde::Deserialize;
use std::{path::Path, time::Duration};

pub(super) fn normalized(text: &str) -> DynResult<String> {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return Err("encoder-decoder oracle response has no text".into());
    }
    Ok(text)
}
#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<Text>,
}
#[derive(Deserialize)]
struct Text {
    text: String,
}
pub(super) fn compare(candidate: &[u8], reference: &str) -> DynResult<String> {
    let response: CompletionResponse = serde_json::from_slice(candidate)?;
    let [choice] = response.choices.as_slice() else {
        return Err("encoder-decoder oracle response has invalid choices".into());
    };
    let candidate = normalized(&choice.text)?;
    let reference = normalized(reference)?;
    if candidate != reference {
        return Err(format!("encoder-decoder text differs from monolithic reference: candidate={candidate:?}, reference={reference:?}").into());
    }
    Ok(format!("identical normalized text={candidate:?}"))
}

pub(super) fn completion(executable: &Path, model: &Path) -> DynResult<String> {
    let executable = std::fs::canonicalize(executable)?;
    if !executable.is_file() || !model.is_file() {
        return Err("native completion requires executable and model files".into());
    }
    let arguments = [
        "-m",
        model.to_str().ok_or("model path is not Unicode")?,
        "-p",
        super::super::ENCODER_DECODER_PROMPT,
        "-n",
        "32",
        "-c",
        "0",
        "-b",
        "2048",
        "-ub",
        "2048",
        "-ngl",
        "0",
        "-s",
        "1",
        "--temp",
        "0",
        "--no-repack",
        "--no-display-prompt",
        "--simple-io",
    ]
    .into_iter()
    .map(|value| Value::Public(value.into()))
    .collect();
    let spec = ProcessSpec {
        executable,
        arguments,
        cwd: std::env::current_dir()?,
        environment: std::env::vars_os()
            .map(|(key, value)| (key, Value::Public(value)))
            .collect(),
    };
    let limits = Limits {
        execution: Duration::from_secs(240),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let interrupt = Interrupt::install()?;
    let result = process::supervise_raw(
        &spec,
        &limits,
        &interrupt.cancellation(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(4 * 1024 * 1024),
            stderr: None,
        },
    );
    interrupt.finish()?;
    let report = result?;
    if report.process.failure.is_some()
        || !report.process.cleanup.complete
        || !report.process.success()
    {
        return Err(format!(
            "monolithic encoder-decoder completion failed: {:?}",
            report.process
        )
        .into());
    }
    let raw = report
        .stdout
        .ok_or("native completion missing raw stdout")?;
    let text = std::str::from_utf8(raw.as_bytes())?.trim();
    // Only the terminal runner marker is metadata. Internal occurrences remain model text.
    let text = text.strip_suffix("[end of text]").unwrap_or(text).trim();
    if text.is_empty() {
        return Err("monolithic encoder-decoder completion produced no text".into());
    }
    Ok(text.to_owned())
}
