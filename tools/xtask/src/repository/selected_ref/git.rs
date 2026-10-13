use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
pub(super) struct Transaction<'a> {
    root: &'a Path,
    executable: PathBuf,
    deadline: Instant,
    cancellation: Cancellation,
}
impl<'a> Transaction<'a> {
    pub(super) fn new(
        root: &'a Path,
        budget: Duration,
        cancellation: Cancellation,
    ) -> DynResult<Self> {
        let executable =
            std::env::split_paths(&std::env::var_os("PATH").ok_or("Git PATH missing")?)
                .map(|path| path.join("git"))
                .find(|path| path.is_file())
                .ok_or("Git executable missing")?;
        Ok(Self {
            root,
            executable: std::path::absolute(executable)?,
            deadline: Instant::now() + budget,
            cancellation,
        })
    }
    pub(super) fn checked(&self, arguments: &[&str]) -> DynResult<String> {
        if self.cancellation.is_cancelled() {
            return Err("selected-ref transaction cancelled".into());
        }
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or("selected-ref transaction exceeded its budget")?;
        let mut environment = BTreeMap::from([
            ("GIT_MASTER".into(), Value::Public("1".into())),
            ("GIT_TERMINAL_PROMPT".into(), Value::Public("0".into())),
        ]);
        for key in ["PATH", "HOME", "SYSTEMROOT"] {
            if let Some(value) = std::env::var_os(key) {
                environment.insert(key.into(), Value::Public(value));
            }
        }
        let mut argv = vec![
            Value::Public("-C".into()),
            Value::Public(self.root.as_os_str().into()),
        ];
        argv.extend(
            arguments
                .iter()
                .map(|argument| Value::Public((*argument).into())),
        );
        let spec = ProcessSpec {
            executable: self.executable.clone(),
            arguments: argv,
            cwd: self.root.into(),
            environment,
        };
        let limits = Limits {
            execution: remaining,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let output = process::supervise_raw(
            &spec,
            &limits,
            &self.cancellation,
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(1024 * 1024),
                stderr: None,
            },
        )?;
        if !admitted_completion(&output.process) {
            return Err("Git failed or exceeded selected-ref process/output bounds".into());
        }
        let raw = output.stdout.ok_or("Git stdout protocol capture missing")?;
        let text = String::from_utf8(raw.as_bytes().to_vec())?;
        // Git textual protocol line terminators, not interpreter-specific string stripping.
        Ok(text.trim_end_matches(['\r', '\n']).to_owned())
    }
}

fn admitted_completion(output: &process::ProcessReport) -> bool {
    output.outcome == process::Outcome::Exited
        && output.success()
        && !output.stdout.truncated
        && !output.stderr.truncated
}

#[cfg(test)]
#[path = "git_tests.rs"]
mod tests;
