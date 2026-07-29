use crate::NativeRuntimeBackend;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
};

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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tools: BTreeMap<String, String>,
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
        manifest.verify_contents(dir)?;
        Ok(manifest)
    }

    pub fn write_to_dir(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create native runtime dir {}", dir.display()))?;
        let mut manifest = self.clone();
        if manifest.runtime.files.is_empty() {
            for library in &manifest.runtime.libraries {
                let path = checked_runtime_path(dir, library)?;
                manifest
                    .runtime
                    .files
                    .insert(library.clone(), sha256_file(&path)?);
            }
        }
        manifest.validate()?;
        let path = dir.join(NATIVE_RUNTIME_MANIFEST_FILE);
        let text = serde_json::to_string_pretty(&manifest)?;
        fs::write(&path, format!("{text}\n"))
            .with_context(|| format!("write native runtime manifest {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        validate_artifact(&self.runtime)
    }

    fn verify_contents(&self, dir: &Path) -> Result<()> {
        if self.runtime.files.is_empty() {
            bail!(
                "native runtime artifact {} does not declare file checksums",
                self.runtime.id
            );
        }
        for library in &self.runtime.libraries {
            if !self.runtime.files.contains_key(library) {
                bail!(
                    "native runtime artifact {} library {} is missing a file checksum",
                    self.runtime.id,
                    library
                );
            }
        }
        verify_file_checksums(dir, "file", &self.runtime.files)?;
        verify_file_checksums(dir, "tool", &self.runtime.tools)
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
    for (path, checksum) in artifact.files.iter().chain(&artifact.tools) {
        validate_runtime_path(path)?;
        normalize_sha256(checksum).with_context(|| {
            format!(
                "native runtime artifact {} has invalid checksum for {}",
                artifact.id, path
            )
        })?;
    }
    Ok(())
}

fn verify_file_checksums(
    root: &Path,
    kind: &str,
    checksums: &BTreeMap<String, String>,
) -> Result<()> {
    for (relative, expected) in checksums {
        let path = checked_runtime_path(root, relative)?;
        let actual = sha256_file(&path)?;
        let expected = normalize_sha256(expected)?;
        if actual != expected {
            bail!(
                "native runtime {kind} checksum mismatch for {relative}: expected {expected}, got {actual}"
            );
        }
    }
    Ok(())
}

fn checked_runtime_path(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_runtime_path(relative)?;
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalize native runtime root {}", root.display()))?;
    let path = root.join(relative);
    let path = path
        .canonicalize()
        .with_context(|| format!("canonicalize native runtime file {}", path.display()))?;
    if !path.starts_with(&root) {
        bail!("native runtime path escapes its bundle: {relative}");
    }
    if !path.is_file() {
        bail!("native runtime path is not a file: {}", path.display());
    }
    Ok(path)
}

fn validate_runtime_path(relative: &str) -> Result<()> {
    let path = Path::new(relative);
    if relative.trim().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("native runtime path must be a safe relative file path: {relative}");
    }
    Ok(())
}

fn normalize_sha256(value: &str) -> Result<String> {
    let value = value
        .trim()
        .strip_prefix("sha256:")
        .unwrap_or(value.trim())
        .to_ascii_lowercase();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("expected a 64-character SHA-256 digest");
    }
    Ok(value)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open native runtime file {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("read native runtime file {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NativeRuntimeBackend;

    #[test]
    fn reads_native_runtime_manifest_shape() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("lib/libllama.so");
        fs::create_dir_all(library.parent().unwrap()).unwrap();
        fs::write(&library, b"native runtime").unwrap();
        let checksum = sha256_file(&library).unwrap();
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
    "libraries": ["lib/libllama.so"],
    "files": {"lib/libllama.so": "__CHECKSUM__"},
    "tools": {}
  }
}"#
            .replace("__CHECKSUM__", &checksum),
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

    #[test]
    fn rejects_tampered_runtime_file() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("lib/libllama.so");
        fs::create_dir_all(library.parent().unwrap()).unwrap();
        fs::write(&library, b"native runtime").unwrap();
        let manifest = NativeRuntimeManifest {
            runtime: NativeRuntimeArtifact {
                id: "meshllm-runtime-linux-x86_64-cpu".to_string(),
                mesh_version: Some("0.68.0".to_string()),
                skippy_abi: "0.1.25".to_string(),
                platform: NativeRuntimePlatform {
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    target: None,
                },
                backend: NativeRuntimeBackend::cpu(),
                rank: 0,
                libraries: vec!["lib/libllama.so".to_string()],
                files: Default::default(),
                tools: Default::default(),
                url: None,
                sha256: None,
                signature: None,
            },
        };
        manifest.write_to_dir(temp.path()).unwrap();
        fs::write(&library, b"tampered runtime").unwrap();

        let error = NativeRuntimeManifest::read_from_dir(temp.path()).unwrap_err();

        assert!(error.to_string().contains("checksum mismatch"));
        assert!(error.to_string().contains("lib/libllama.so"));
    }

    #[test]
    fn rejects_runtime_manifest_without_file_checksums() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("lib")).unwrap();
        fs::write(temp.path().join("lib/libllama.so"), b"native runtime").unwrap();
        fs::write(
            temp.path().join(NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "runtime": {
    "id": "meshllm-runtime-linux-x86_64-cpu",
    "mesh_version": "0.68.0",
    "skippy_abi": "0.1.25",
    "platform": {"os": "linux", "arch": "x86_64"},
    "backend": {"kind": "cpu"},
    "libraries": ["lib/libllama.so"]
  }
}"#,
        )
        .unwrap();

        let error = NativeRuntimeManifest::read_from_dir(temp.path()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not declare file checksums")
        );
    }

    #[test]
    fn rejects_runtime_checksum_path_traversal() {
        let artifact = NativeRuntimeArtifact {
            id: "meshllm-runtime-linux-x86_64-cpu".to_string(),
            mesh_version: Some("0.68.0".to_string()),
            skippy_abi: "0.1.25".to_string(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                target: None,
            },
            backend: NativeRuntimeBackend::cpu(),
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            files: BTreeMap::from([("../outside".to_string(), "0".repeat(64))]),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        };

        let error = validate_artifact(&artifact).unwrap_err();

        assert!(error.to_string().contains("safe relative file path"));
    }
}
