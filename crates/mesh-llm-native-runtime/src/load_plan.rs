use crate::InstalledNativeRuntime;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeLoadPlan {
    pub mesh_version: String,
    pub native_runtime_id: String,
    pub root: PathBuf,
    pub libraries: Vec<PathBuf>,
}

impl InstalledNativeRuntime {
    pub fn load_plan(&self) -> Result<NativeRuntimeLoadPlan> {
        let libraries = self
            .manifest
            .artifact
            .library_paths
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
        Ok(NativeRuntimeLoadPlan {
            mesh_version: self.mesh_version.clone(),
            native_runtime_id: self.native_runtime_id.clone(),
            root: self.path.clone(),
            libraries,
        })
    }
}
