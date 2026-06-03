use crate::NativeRuntimeBackend;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub const NATIVE_RUNTIME_MANIFEST_FILE: &str = "manifest.json";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimePlatform {
    pub os: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeArtifact {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_version: Option<String>,
    pub skippy_abi: String,
    pub platform: NativeRuntimePlatform,
    pub backend: NativeRuntimeBackend,
    #[serde(default)]
    pub rank: i64,
    pub libraries: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeManifest {
    pub runtime: NativeRuntimeArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeReleaseManifest {
    pub mesh_version: String,
    pub skippy_abi: String,
    #[serde(default)]
    pub artifacts: Vec<NativeRuntimeArtifact>,
}

impl NativeRuntimeArtifact {
    pub fn native_runtime_id(&self) -> &str {
        &self.id
    }

    pub fn mesh_version_or<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.mesh_version.as_deref().unwrap_or(fallback)
    }
}

impl NativeRuntimeManifest {
    pub fn read_from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(NATIVE_RUNTIME_MANIFEST_FILE);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read native runtime manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse native runtime manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn write_to_dir(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create native runtime dir {}", dir.display()))?;
        let path = dir.join(NATIVE_RUNTIME_MANIFEST_FILE);
        let text = serde_json::to_string_pretty(self)?;
        fs::write(&path, format!("{text}\n"))
            .with_context(|| format!("write native runtime manifest {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        validate_artifact(&self.runtime)
    }
}

impl NativeRuntimeReleaseManifest {
    pub fn read_from_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read native runtime release manifest {}", path.display()))?;
        Self::from_json_str(&text)
            .with_context(|| format!("parse native runtime release manifest {}", path.display()))
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let manifest: Self =
            serde_json::from_str(text).context("parse native runtime release manifest")?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.mesh_version.trim().is_empty() {
            bail!("native runtime release manifest mesh_version is empty");
        }
        if self.skippy_abi.trim().is_empty() {
            bail!("native runtime release manifest skippy_abi is empty");
        }
        for artifact in &self.artifacts {
            validate_artifact(artifact)?;
            if artifact.skippy_abi != self.skippy_abi {
                bail!(
                    "native runtime artifact {} has skippy_abi {}, expected {}",
                    artifact.id,
                    artifact.skippy_abi,
                    self.skippy_abi
                );
            }
        }
        Ok(())
    }
}

fn validate_artifact(artifact: &NativeRuntimeArtifact) -> Result<()> {
    if artifact.id.trim().is_empty() {
        bail!("native runtime artifact id is empty");
    }
    if artifact.skippy_abi.trim().is_empty() {
        bail!(
            "native runtime artifact {} skippy_abi is empty",
            artifact.id
        );
    }
    if artifact.platform.os.trim().is_empty() || artifact.platform.arch.trim().is_empty() {
        bail!(
            "native runtime artifact {} must declare platform os and arch",
            artifact.id
        );
    }
    if artifact.libraries.is_empty() {
        bail!(
            "native runtime artifact {} must declare at least one library",
            artifact.id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NativeRuntimeBackend;

    #[test]
    fn reads_native_runtime_manifest_shape() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join(NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "runtime": {
    "id": "meshllm-runtime-linux-x86_64-cuda12",
    "mesh_version": "0.68.0",
    "skippy_abi": "0.1.25",
    "platform": {
      "os": "linux",
      "arch": "x86_64",
      "target": "x86_64-unknown-linux-gnu"
    },
    "backend": {
      "kind": "cuda",
      "cuda": {
        "toolkit_major": 12,
        "gpu_arches": ["sm_90"]
      }
    },
    "rank": 650,
    "libraries": ["lib/libllama.so"]
  }
}"#,
        )
        .unwrap();

        let manifest = NativeRuntimeManifest::read_from_dir(temp.path()).unwrap();

        assert_eq!(manifest.runtime.id, "meshllm-runtime-linux-x86_64-cuda12");
        assert_eq!(manifest.runtime.skippy_abi, "0.1.25");
        assert_eq!(manifest.runtime.backend.kind.as_str(), "cuda");
    }

    #[test]
    fn reads_release_manifest() {
        let manifest = NativeRuntimeReleaseManifest::from_json_str(
            r#"{
  "mesh_version": "0.68.0",
  "skippy_abi": "0.1.25",
  "artifacts": [
    {
      "id": "meshllm-runtime-linux-x86_64-cpu",
      "mesh_version": "0.68.0",
      "skippy_abi": "0.1.25",
      "platform": { "os": "linux", "arch": "x86_64" },
      "backend": { "kind": "cpu" },
      "rank": 100,
      "libraries": ["lib/libllama.so"]
    }
  ]
}"#,
        )
        .unwrap();

        assert_eq!(manifest.artifacts.len(), 1);
        assert_eq!(manifest.artifacts[0].backend, NativeRuntimeBackend::cpu());
    }
}
