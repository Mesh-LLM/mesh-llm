use super::Error;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

pub(crate) struct NativeLipo {
    pub(crate) executable: PathBuf,
    pub(crate) cwd: PathBuf,
    pub(crate) environment: BTreeMap<OsString, OsString>,
    pub(crate) timeout: Duration,
    pub(crate) max_bytes: NonZeroUsize,
}

impl NativeLipo {
    pub(super) fn inspect(
        &self,
        binary: &Path,
        cancellation: &Cancellation,
    ) -> Result<BTreeSet<String>, Error> {
        let binary = std::path::absolute(binary).map_err(|error| Error::io(binary, error))?;
        let spec = ProcessSpec {
            executable: self.executable.clone(),
            cwd: self.cwd.clone(),
            arguments: vec![
                Value::Public("-archs".into()),
                Value::Public(binary.as_os_str().to_owned()),
            ],
            environment: self
                .environment
                .iter()
                .map(|(key, value)| (key.clone(), Value::Public(value.clone())))
                .collect(),
        };
        let limits = Limits {
            execution: self.timeout,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 0,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise_raw(
            &spec,
            &limits,
            cancellation,
            RawCaptureOptions {
                stdout: Some(self.max_bytes),
                stderr: Some(self.max_bytes),
            },
        )
        .map_err(|source| Error::Spawn {
            binary: binary.clone(),
            source,
        })?;
        if !report.process.success() || report.stdout.is_none() || report.stderr.is_none() {
            return Err(Error::Native {
                binary,
                report: Box::new(report),
            });
        }
        let stdout = match report.stdout {
            Some(stdout) => stdout,
            None => {
                return Err(Error::Native {
                    binary,
                    report: Box::new(report),
                });
            }
        };
        let text = std::str::from_utf8(stdout.as_bytes())?;
        let architectures: BTreeSet<String> = text
            .split(python_whitespace)
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect();
        if architectures.is_empty() {
            return Err(Error::Contract(format!(
                "lipo reported no architectures for XCFramework binary: {}",
                binary.display()
            )));
        }
        Ok(architectures)
    }
}

fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}
