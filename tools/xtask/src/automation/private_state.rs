use crate::process::{OutputFiles, Value};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(test)]
#[path = "../../tests/migration_lifecycle/state.rs"]
mod tests;

#[cfg(windows)]
#[path = "private_state/windows_privacy.rs"]
mod windows_privacy;

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("private state entropy unavailable")]
    EntropyUnavailable,
    #[cfg(windows)]
    #[error("{0}")]
    Invalid(&'static str),
    #[error("{operation} failed ({kind:?}, OS code {code:?})")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        code: Option<i32>,
    },
}

impl Error {
    fn io(operation: &'static str, error: io::Error) -> Self {
        Self::Io {
            operation,
            kind: error.kind(),
            code: error.raw_os_error(),
        }
    }
}

#[derive(Debug)]
pub(super) enum FinishError<E> {
    Prior(E),
    Deletion {
        kind: io::ErrorKind,
        code: Option<i32>,
        preceding: Option<Box<E>>,
    },
}

pub(super) struct PrivateState {
    root: PathBuf,
    closed: bool,
}

impl PrivateState {
    pub(super) fn create(parent: &Path, prefix: &'static str) -> Result<Self, Error> {
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).map_err(|_| Error::EntropyUnavailable)?;
        let root = parent.join(format!("{prefix}.{}", hex::encode(random)));
        private_directory(&root)?;
        Ok(Self {
            root,
            closed: false,
        })
    }

    pub(super) fn prepare(&self) -> Result<(), Error> {
        for name in [
            "home",
            "cache",
            "config",
            "xdg-runtime",
            "runtime-cache",
            "runtime",
            "tmp",
        ] {
            private_directory(&self.root.join(name))?;
        }
        Ok(())
    }

    pub(super) fn environment(&self, native_root: &Path) -> BTreeMap<OsString, Value> {
        let mut environment: BTreeMap<_, _> = host_environment().collect();
        for (key, relative) in [
            ("HOME", "home"),
            ("USERPROFILE", "home"),
            ("APPDATA", "config"),
            ("LOCALAPPDATA", "cache"),
            ("MESH_LLM_CONFIG", "config.toml"),
            ("MESH_LLM_RUNTIME_ROOT", "runtime"),
            ("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", "runtime-cache"),
            ("XDG_CACHE_HOME", "cache"),
            ("XDG_CONFIG_HOME", "config"),
            ("XDG_RUNTIME_DIR", "xdg-runtime"),
            ("TMPDIR", "tmp"),
            ("TEMP", "tmp"),
            ("TMP", "tmp"),
        ] {
            environment.insert(key.into(), Value::Public(self.root.join(relative).into()));
        }
        environment.insert(
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR".into(),
            Value::Public(native_root.into()),
        );
        environment
    }

    pub(super) fn output_files(&self) -> OutputFiles {
        OutputFiles {
            stdout: Some(self.root.join("stdout.log")),
            stderr: Some(self.root.join("stderr.log")),
        }
    }

    pub(super) fn model_fit(&self, sizes: Option<(u32, u32)>) -> Result<(), Error> {
        if let Some((batch, ubatch)) = sizes {
            fs::write(
                self.root.join("config.toml"),
                format!(
                    "version = 1\n\n[defaults.model_fit]\nbatch = {batch}\nubatch = {ubatch}\n"
                ),
            )
            .map_err(|error| Error::io("write model fit", error))?;
        }
        Ok(())
    }

    pub(super) fn finish<T, E>(mut self, result: Result<T, E>) -> Result<T, FinishError<E>> {
        self.closed = true;
        match fs::remove_dir_all(&self.root) {
            Ok(()) => result.map_err(FinishError::Prior),
            Err(error) => Err(FinishError::Deletion {
                kind: error.kind(),
                code: error.raw_os_error(),
                preceding: result.err().map(Box::new),
            }),
        }
    }
}

impl Drop for PrivateState {
    fn drop(&mut self) {
        if !self.closed
            && let Err(error) = fs::remove_dir_all(&self.root)
        {
            eprintln!(
                "client readiness {}",
                Error::io("unwinding state deletion", error)
            );
        }
    }
}

#[cfg(unix)]
fn private_directory(path: &Path) -> Result<(), Error> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    builder
        .create(path)
        .map_err(|error| Error::io("create private directory", error))
}

#[cfg(windows)]
fn private_directory(path: &Path) -> Result<(), Error> {
    windows_privacy::create(path)
}

fn host_environment() -> impl Iterator<Item = (OsString, Value)> {
    [
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    ]
    .into_iter()
    .filter_map(|key| {
        std::env::var_os(key)
            .filter(|value| !value.is_empty())
            .map(|value| (key.into(), Value::Secret(value)))
    })
}
