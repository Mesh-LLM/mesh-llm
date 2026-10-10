mod args;
mod generation;
mod invocation;
mod publication;
mod scope;
use super::command_interrupt::Interrupt;
use super::rewriter_patch::{encode_mail_patch, shards::encode_family_shards};
use crate::command::DynResult;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawBytes, RawCaptureOptions, Readiness,
    Value,
};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;
pub(crate) const USAGE: &str = "cargo xtool automation native-generator --source-root <dir> --git <absolute-executable> --output <patch> --max-diff-bytes <positive-integer> [--diff-base <revision>] [--timeout <seconds>] [--shard-output-dir <dir> --family-source-map <json> --family-manifest <json>]";
#[derive(Debug, thiserror::Error)]
enum Error {
    #[error(transparent)]
    Interrupt(#[from] super::command_interrupt::Reason),
    #[error("{phase} failed: {report:?}")]
    Command {
        phase: &'static str,
        report: Box<process::ProcessReport>,
    },
    #[error("source tree must be clean before generation")]
    Dirty,
    #[error("no model sources found under source root")]
    Sources,
    #[error("invalid UTF-8 in git identity or status")]
    Utf8(#[from] std::str::Utf8Error),
    #[error(transparent)]
    Report(#[from] super::rewriter_report::generator::Error),
    #[error("invalid native generator arguments: {0}")]
    Arguments(&'static str),
    #[error(transparent)]
    Process(#[from] process::Failure),
    #[error(
        "git diff failed: outcome={outcome:?}, exit={exit:?}, cleanup_complete={cleanup_complete}"
    )]
    Git {
        outcome: process::Outcome,
        exit: Option<i32>,
        cleanup_complete: bool,
    },
    #[error("git diff returned no complete raw stdout")]
    MissingPayload,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Patch(#[from] super::rewriter_patch::PatchError),
    #[error(transparent)]
    Shards(#[from] super::rewriter_patch::shards::ShardError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("generator staging entropy unavailable")]
    Entropy,
    #[error("JSON input exceeds 16 MiB")]
    JsonLimit,
}
struct GitDiff {
    executable: PathBuf,
    source_root: PathBuf,
    base: String,
    environment: BTreeMap<OsString, Value>,
    max_bytes: std::num::NonZeroUsize,
    timeout: Duration,
}
fn capture_diff(input: &GitDiff, cancellation: &Cancellation) -> Result<RawBytes, Error> {
    let arguments = [
        "diff",
        "--no-ext-diff",
        "--binary",
        "--full-index",
        &input.base,
        "--",
        "src/models",
    ]
    .into_iter()
    .map(|value| Value::Public(value.into()))
    .collect();
    let spec = ProcessSpec {
        executable: input.executable.clone(),
        arguments,
        cwd: input.source_root.clone(),
        environment: input
            .environment
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    Value::Public(value) => Value::Public(value.clone()),
                    Value::Secret(value) => Value::Secret(value.clone()),
                };
                (key.clone(), value)
            })
            .collect(),
    };
    let limits = Limits {
        execution: input.timeout,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        cancellation,
        RawCaptureOptions {
            stdout: Some(input.max_bytes),
            stderr: None,
        },
    )?;
    if !report.process.success() {
        if let Some(error) = report.process.failure {
            return Err(Error::Process(error));
        }
        return Err(Error::Git {
            outcome: report.process.outcome,
            exit: report.process.status.and_then(|status| status.code()),
            cleanup_complete: report.process.cleanup.complete,
        });
    }
    report.stdout.ok_or(Error::MissingPayload)
}
pub(crate) fn run(arguments: &[String]) -> DynResult<()> {
    if let [verb, rest @ ..] = arguments
        && verb == "contracts"
    {
        return super::native_contracts::run(rest);
    }
    if let [verb, rest @ ..] = arguments
        && verb == "generate"
    {
        return generation::run(rest);
    }
    if arguments == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let options = args::parse(arguments)?;
    let result = scope::run(|interrupt| {
        let diff = capture_diff(&options.git, &interrupt.cancellation())?;
        interrupt.check()?;
        publish(diff.as_bytes(), &options)
    });
    match result {
        Ok(()) => Ok(()),
        Err(scope::Failure::Operation(error)) => Err(error.into()),
        Err(error @ scope::Failure::Finalization { .. }) => Err(error.into()),
    }
}
fn publish(diff: &[u8], options: &args::Options) -> Result<(), Error> {
    publish_count(diff, options).map(|_| ())
}
fn publish_count(diff: &[u8], options: &args::Options) -> Result<usize, Error> {
    let mail = encode_mail_patch("skippy: generate model-family stage controls", diff)?;
    std::fs::create_dir_all(parent(&options.output))?;
    std::fs::write(&options.output, &mail.bytes)?;
    if let Some(shards) = &options.shards {
        let map = publication::read_json(&shards.map)?;
        let manifest = publication::read_json(&shards.manifest)?;
        let encoded = encode_family_shards(diff, &map, &manifest)?;
        publication::publish(&shards.output, &encoded)?;
        return Ok(encoded.shards.len());
    }
    Ok(0)
}
fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}
#[cfg(test)]
#[path = "native_generator/tests.rs"]
mod tests;
