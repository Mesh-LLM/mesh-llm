use crate::NativeRuntimeFlavor;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::Path};

pub const NATIVE_RUNTIME_MANIFEST_FILE: &str = "manifest.json";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeRequirement {
    #[serde(default)]
    pub gpu_name_contains: Vec<String>,
    #[serde(default)]
    pub min_cuda_compute_capability: Option<u32>,
    #[serde(default)]
    pub min_driver_version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeArtifact {
    #[serde(alias = "artifact_id")]
    pub native_runtime_id: String,
    #[serde(alias = "sdk_version")]
    pub mesh_version: String,
    #[serde(default)]
    pub target_triple: Option<String>,
    pub os: String,
    pub arch: String,
    pub flavor: NativeRuntimeFlavor,
    #[serde(default)]
    pub priority: i64,
    #[serde(default)]
    pub skippy_abi_version: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub library_paths: Vec<String>,
    #[serde(default)]
    pub requirements: Vec<NativeRuntimeRequirement>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeManifest {
    pub artifact: NativeRuntimeArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeReleaseManifest {
    pub mesh_version: String,
    #[serde(default)]
    pub artifacts: Vec<NativeRuntimeArtifact>,
}

impl NativeRuntimeManifest {
    pub fn read_from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(NATIVE_RUNTIME_MANIFEST_FILE);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read native runtime manifest {}", path.display()))?;
        let mut value: Value = serde_json::from_str(&text)
            .with_context(|| format!("parse native runtime manifest {}", path.display()))?;
        if let Some(artifact) = value.get_mut("artifact") {
            normalize_legacy_aliases(artifact)?;
            let artifact = serde_json::from_value(artifact.take())
                .with_context(|| format!("parse native runtime artifact {}", path.display()))?;
            return Ok(Self { artifact });
        }
        normalize_legacy_aliases(&mut value)?;
        let artifact = serde_json::from_value(value)
            .with_context(|| format!("parse native runtime artifact {}", path.display()))?;
        Ok(Self { artifact })
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
        validate_artifact(&self.artifact)
    }
}

fn normalize_legacy_aliases(value: &mut Value) -> Result<()> {
    let Some(object) = value.as_object_mut() else {
        return Ok(());
    };
    normalize_alias_pair(object, "native_runtime_id", "artifact_id")?;
    normalize_alias_pair(object, "mesh_version", "sdk_version")?;
    Ok(())
}

fn normalize_alias_pair(
    object: &mut serde_json::Map<String, Value>,
    canonical: &str,
    alias: &str,
) -> Result<()> {
    match (object.get(canonical), object.get(alias)) {
        (Some(left), Some(right)) if left != right => {
            bail!("{canonical} and {alias} disagree in native runtime manifest")
        }
        (Some(_), Some(_)) => {
            object.remove(alias);
        }
        (None, Some(_)) => {
            if let Some(value) = object.remove(alias) {
                object.insert(canonical.to_string(), value);
            }
        }
        _ => {}
    }
    Ok(())
}

impl NativeRuntimeReleaseManifest {
    pub fn read_from_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read native runtime release manifest {}", path.display()))?;
        let mut value: Value = serde_json::from_str(&text)
            .with_context(|| format!("parse native runtime release manifest {}", path.display()))?;
        if let Some(artifacts) = value.get_mut("artifacts").and_then(Value::as_array_mut) {
            for artifact in artifacts {
                normalize_legacy_aliases(artifact)?;
            }
        }
        let manifest: Self = serde_json::from_value(value)
            .with_context(|| format!("parse native runtime release manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.mesh_version.trim().is_empty() {
            bail!("native runtime release manifest mesh_version is empty");
        }
        for artifact in &self.artifacts {
            validate_artifact(artifact)?;
            if artifact.mesh_version != self.mesh_version {
                bail!(
                    "native runtime artifact {} has mesh_version {}, expected {}",
                    artifact.native_runtime_id,
                    artifact.mesh_version,
                    self.mesh_version
                );
            }
        }
        Ok(())
    }
}

fn validate_artifact(artifact: &NativeRuntimeArtifact) -> Result<()> {
    if artifact.native_runtime_id.trim().is_empty() {
        bail!("native runtime artifact id is empty");
    }
    if artifact.mesh_version.trim().is_empty() {
        bail!(
            "native runtime artifact {} mesh_version is empty",
            artifact.native_runtime_id
        );
    }
    if artifact.os.trim().is_empty() || artifact.arch.trim().is_empty() {
        bail!(
            "native runtime artifact {} must declare os and arch",
            artifact.native_runtime_id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_direct_sdk_manifest_shape() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join(NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "artifact_id": "meshllm-native-linux-x86_64-cpu",
  "sdk_version": "0.68.0",
  "target_triple": "x86_64-unknown-linux-gnu",
  "os": "linux",
  "arch": "x86_64",
  "flavor": "cpu",
  "library_paths": ["lib/libmeshllm_ffi.so"]
}"#,
        )
        .unwrap();

        let manifest = NativeRuntimeManifest::read_from_dir(temp.path()).unwrap();

        assert_eq!(
            manifest.artifact.native_runtime_id,
            "meshllm-native-linux-x86_64-cpu"
        );
        assert_eq!(manifest.artifact.mesh_version, "0.68.0");
    }

    #[test]
    fn reads_manifests_with_canonical_and_legacy_aliases() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join(NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "artifact_id": "meshllm-native-runtime-linux-x86_64-cpu",
  "native_runtime_id": "meshllm-native-runtime-linux-x86_64-cpu",
  "sdk_version": "0.68.0",
  "mesh_version": "0.68.0",
  "target_triple": "x86_64-unknown-linux-gnu",
  "os": "linux",
  "arch": "x86_64",
  "flavor": "cpu",
  "library_paths": ["lib/libllama.so"]
}"#,
        )
        .unwrap();

        let manifest = NativeRuntimeManifest::read_from_dir(temp.path()).unwrap();

        assert_eq!(
            manifest.artifact.native_runtime_id,
            "meshllm-native-runtime-linux-x86_64-cpu"
        );
        assert_eq!(manifest.artifact.mesh_version, "0.68.0");
    }

    #[test]
    fn reads_release_manifest_with_canonical_and_legacy_aliases() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("native-runtimes.json");
        fs::write(
            &path,
            r#"{
  "mesh_version": "0.68.0",
  "artifacts": [
    {
      "artifact_id": "meshllm-native-runtime-linux-x86_64-cpu",
      "native_runtime_id": "meshllm-native-runtime-linux-x86_64-cpu",
      "mesh_version": "0.68.0",
      "target_triple": "x86_64-unknown-linux-gnu",
      "os": "linux",
      "arch": "x86_64",
      "flavor": "cpu",
      "library_paths": ["lib/libllama.so"]
    }
  ]
}"#,
        )
        .unwrap();

        let manifest = NativeRuntimeReleaseManifest::read_from_path(&path).unwrap();

        assert_eq!(manifest.artifacts.len(), 1);
        assert_eq!(
            manifest.artifacts[0].native_runtime_id,
            "meshllm-native-runtime-linux-x86_64-cpu"
        );
    }

    #[test]
    fn preserves_unknown_flavor_as_string() {
        let artifact = NativeRuntimeArtifact {
            native_runtime_id: "meshllm-native-linux-x86_64-cuda-sm120".to_string(),
            mesh_version: "0.68.0".to_string(),
            target_triple: Some("x86_64-unknown-linux-gnu".to_string()),
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            flavor: NativeRuntimeFlavor::Other("cuda-sm120".to_string()),
            priority: 0,
            skippy_abi_version: None,
            url: None,
            sha256: None,
            signature: None,
            library_paths: vec!["lib/libmeshllm_ffi.so".to_string()],
            requirements: Vec::new(),
        };

        let text = serde_json::to_string(&artifact).unwrap();
        let reparsed = serde_json::from_str::<NativeRuntimeArtifact>(&text).unwrap();

        assert_eq!(reparsed.flavor, artifact.flavor);
        assert!(text.contains("\"cuda-sm120\""));
    }
}
