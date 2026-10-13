use super::capture::NativeSwift;
use super::error::Error;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Copy)]
pub(super) enum Operation {
    Update,
    Verify,
}

impl Operation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Update => "update",
            Self::Verify => "verify",
        }
    }
}

pub(super) struct Options<'a> {
    pub(super) operation: Operation,
    pub(super) tag: &'a str,
    pub(super) artifact: &'a str,
    pub(super) manifest: &'a str,
    pub(super) executable: PathBuf,
    pub(super) timeout: Duration,
    pub(super) max_bytes: NonZeroUsize,
}

impl Options<'_> {
    pub(super) fn native(&self) -> Result<NativeSwift, Error> {
        Ok(NativeSwift {
            executable: self.executable.clone(),
            cwd: std::env::current_dir()?,
            artifact: self.artifact.into(),
            timeout: self.timeout,
            max_bytes: self.max_bytes,
        })
    }
}

pub(super) fn parse(args: &[String]) -> Result<Options<'_>, &'static str> {
    let [operation, tag, artifact, manifest, executable, timeout, cap] = args else {
        return Err("expected seven arguments");
    };
    let operation = match operation.as_str() {
        "update" => Operation::Update,
        "verify" => Operation::Verify,
        _ => return Err("expected update or verify"),
    };
    let executable = PathBuf::from(executable);
    if !executable.is_absolute() {
        return Err("Swift executable must be absolute");
    }
    let seconds = timeout
        .parse::<u64>()
        .ok()
        .filter(|seconds| (1..=3600).contains(seconds))
        .ok_or("timeout must be 1..3600 seconds")?;
    let max_bytes = cap
        .parse::<usize>()
        .ok()
        .filter(|bytes| (1..=1048576).contains(bytes))
        .and_then(NonZeroUsize::new)
        .ok_or("max output must be 1..1048576 bytes per stream")?;
    Ok(Options {
        operation,
        tag,
        artifact,
        manifest,
        executable,
        timeout: Duration::from_secs(seconds),
        max_bytes,
    })
}
