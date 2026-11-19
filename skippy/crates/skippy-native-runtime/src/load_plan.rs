use crate::InstalledNativeRuntime;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeLoadPlan {
    pub release_version: String,
    pub native_runtime_id: String,
    pub root: PathBuf,
    pub libraries: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_backend_dir: Option<PathBuf>,
}

impl InstalledNativeRuntime {
    pub fn load_plan(&self) -> Result<NativeRuntimeLoadPlan> {
        let libraries = self
            .manifest
            .runtime
            .libraries
            .iter()
            .map(|path| self.path.join(path))
            .collect::<Vec<_>>();
        if libraries.is_empty() {
            bail!(
                "native runtime {} does not declare loadable libraries",
                self.native_runtime_id
            );
        }
        for library in &libraries {
            if !library.is_file() {
                bail!("native runtime library is missing: {}", library.display());
            }
        }
        let cpu_backend_dir = declared_cpu_backend_dir(
            &self.path,
            self.manifest.runtime.files.keys().map(String::as_str),
        )?;
        Ok(NativeRuntimeLoadPlan {
            release_version: self.release_version.clone(),
            native_runtime_id: self.native_runtime_id.clone(),
            root: self.path.clone(),
            libraries,
            cpu_backend_dir,
        })
    }
}

fn declared_cpu_backend_dir<'a>(
    runtime_root: &Path,
    files: impl Iterator<Item = &'a str>,
) -> Result<Option<PathBuf>> {
    let mut directories = BTreeSet::new();
    for file in files {
        let path = Path::new(file);
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with("libggml-cpu-") && name.contains(".so") {
            directories.insert(
                path.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf(),
            );
        }
    }
    match directories.len() {
        0 => Ok(None),
        1 => Ok(directories
            .into_iter()
            .next()
            .map(|directory| runtime_root.join(directory))),
        _ => bail!("native runtime CPU backend variants span multiple directories"),
    }
}
