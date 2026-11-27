use crate::InstalledNativeRuntime;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeLoadPlan {
    pub release_version: String,
    pub native_runtime_id: String,
    pub root: PathBuf,
    pub libraries: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpu_backend_modules: Vec<PathBuf>,
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
        let cpu_backend_modules = declared_cpu_backend_modules(
            &self.path,
            self.manifest.runtime.files.keys().map(String::as_str),
        );
        for module in &cpu_backend_modules {
            if !module.is_file() {
                bail!(
                    "native runtime CPU backend module is missing: {}",
                    module.display()
                );
            }
        }
        Ok(NativeRuntimeLoadPlan {
            release_version: self.release_version.clone(),
            native_runtime_id: self.native_runtime_id.clone(),
            root: self.path.clone(),
            libraries,
            cpu_backend_modules,
        })
    }
}

fn declared_cpu_backend_modules<'a>(
    runtime_root: &Path,
    files: impl Iterator<Item = &'a str>,
) -> Vec<PathBuf> {
    files
        .filter_map(|file| {
            let path = Path::new(file);
            let name = path.file_name()?.to_str()?;
            (name.starts_with("libggml-cpu-") && name.contains(".so"))
                .then(|| runtime_root.join(path))
        })
        .collect()
}
