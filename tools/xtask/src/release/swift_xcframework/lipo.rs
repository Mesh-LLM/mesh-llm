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
        architectures(text)
    }
}

fn architectures(text: &str) -> Result<BTreeSet<String>, Error> {
    let names: BTreeSet<String> = text.split_ascii_whitespace().map(str::to_owned).collect();
    if names.is_empty() {
        return Err(Error::Contract("lipo reported no architectures".into()));
    }
    if names.iter().any(|name| {
        !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    }) {
        return Err(Error::Contract(
            "lipo reported a malformed architecture name".into(),
        ));
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::architectures;

    #[test]
    fn native_architecture_names_accept_ascii_spacing_and_deduplicate() {
        let actual = architectures("arm64\tx86_64\r\narm64 ").unwrap();
        assert_eq!(
            actual,
            ["arm64", "x86_64"].into_iter().map(str::to_owned).collect()
        );
    }

    #[test]
    fn native_architecture_names_reject_control_and_unicode_separators() {
        for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{a0}', '\u{2003}'] {
            assert!(architectures(&format!("arm64{separator}x86_64")).is_err());
        }
    }

    #[test]
    fn native_architecture_names_reject_empty_or_malformed_output_without_echoing_payload() {
        for text in [
            "",
            " \t\r\n",
            "arm64;private-payload",
            "arm64/private-payload",
        ] {
            let error = architectures(text).unwrap_err().to_string();
            assert!(!error.contains("private-payload"));
        }
    }
}
