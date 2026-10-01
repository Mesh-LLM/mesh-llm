use super::Error;
use crate::{
    command_interrupt::Interrupt,
    process::{self, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

pub(crate) struct Git {
    executable: PathBuf,
    environment: BTreeMap<OsString, OsString>,
}
impl Git {
    pub(crate) fn new(
        executable: PathBuf,
        mut environment: BTreeMap<OsString, OsString>,
    ) -> Result<Self, Error> {
        if !executable.is_absolute() {
            return Err(Error::Input("Git executable must be absolute"));
        }
        environment.insert("GIT_MASTER".into(), "1".into());
        Ok(Self {
            executable,
            environment,
        })
    }

    pub(super) fn list(&self, workspace: &Path, interrupt: &Interrupt) -> Result<Vec<u8>, Error> {
        self.invoke(
            workspace,
            &[
                "worktree".into(),
                "list".into(),
                "--porcelain".into(),
                "-z".into(),
            ],
            interrupt,
        )
    }

    pub(super) fn remove(
        &self,
        workspace: &Path,
        path: &Path,
        interrupt: &Interrupt,
    ) -> Result<(), Error> {
        self.invoke(
            workspace,
            &[
                "worktree".into(),
                "remove".into(),
                "--force".into(),
                path.as_os_str().to_owned(),
            ],
            interrupt,
        )?;
        Ok(())
    }

    fn invoke(
        &self,
        workspace: &Path,
        args: &[OsString],
        interrupt: &Interrupt,
    ) -> Result<Vec<u8>, Error> {
        interrupt.check()?;
        let mut arguments = vec![
            Value::Public("-C".into()),
            Value::Public(workspace.as_os_str().to_owned()),
        ];
        arguments.extend(args.iter().cloned().map(Value::Public));
        let spec = ProcessSpec {
            executable: self.executable.clone(),
            arguments,
            cwd: workspace.to_owned(),
            environment: self
                .environment
                .iter()
                .map(|(key, value)| (key.clone(), Value::Public(value.clone())))
                .collect(),
        };
        let limits = Limits {
            execution: Duration::from_secs(120),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(5),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise_raw(
            &spec,
            &limits,
            &interrupt.cancellation(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(16 * 1024 * 1024),
                stderr: None,
            },
        )?;
        if !report.process.success() {
            return Err(Error::Git(Box::new(report.process)));
        }
        interrupt.check()?;
        report
            .stdout
            .map(|bytes| bytes.as_bytes().to_vec())
            .ok_or(Error::Input("Git stdout capture missing"))
    }
}
