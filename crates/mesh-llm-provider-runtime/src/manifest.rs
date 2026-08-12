use anyhow::{Context, Result, bail};
use semver::Version;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub const PROVIDER_RUNTIME_SCHEMA_VERSION: u32 = 1;
pub const PROVIDER_RUNTIME_MANIFEST_FILE: &str = "provider-runtime.json";
pub const PROVIDER_RUNTIME_RELEASE_MANIFEST_FILE: &str = "provider-runtimes.json";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimePlatform {
    pub os: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_os_version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeModel {
    pub id: String,
    pub kind: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeSignature {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team_identifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing_identifier: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entitlements: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notarized: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeArtifact {
    pub id: String,
    pub version: String,
    pub provider_kind: String,
    pub protocol_version: String,
    pub platform: ProviderRuntimePlatform,
    pub entrypoint: String,
    pub models: Vec<ProviderRuntimeModel>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub features: BTreeSet<String>,
    pub files: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub build: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<ProviderRuntimeSignature>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeManifest {
    pub schema_version: u32,
    pub runtime: ProviderRuntimeArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeReleaseManifest {
    pub schema_version: u32,
    #[serde(default)]
    pub artifacts: Vec<ProviderRuntimeArtifact>,
}

impl ProviderRuntimeArtifact {
    pub fn validate(&self) -> Result<()> {
        validate_coordinate("provider runtime id", &self.id)?;
        Version::parse(&self.version)
            .with_context(|| format!("provider runtime {} has invalid version", self.id))?;
        validate_coordinate("provider kind", &self.provider_kind)?;
        validate_protocol_version(&self.protocol_version)?;
        self.platform.validate()?;
        validate_runtime_path(&self.entrypoint)?;
        if self.models.is_empty() {
            bail!(
                "provider runtime {} must declare at least one model",
                self.id
            );
        }
        let mut model_ids = BTreeSet::new();
        for model in &self.models {
            validate_model(model)?;
            if !model_ids.insert(&model.id) {
                bail!("provider runtime {} repeats model {}", self.id, model.id);
            }
        }
        for feature in &self.features {
            validate_coordinate("provider runtime feature", feature)?;
        }
        if self.files.is_empty() {
            bail!("provider runtime {} must declare checked files", self.id);
        }
        if !self.files.contains_key(&self.entrypoint) {
            bail!(
                "provider runtime {} entrypoint {} is missing from files",
                self.id,
                self.entrypoint
            );
        }
        for (path, checksum) in &self.files {
            validate_runtime_path(path)?;
            normalize_sha256(checksum).with_context(|| {
                format!(
                    "provider runtime {} has invalid checksum for {path}",
                    self.id
                )
            })?;
        }
        if self.url.is_some() && self.archive_sha256.is_none() {
            bail!(
                "downloadable provider runtime {} is missing archive_sha256",
                self.id
            );
        }
        if let Some(checksum) = &self.archive_sha256 {
            normalize_sha256(checksum).with_context(|| {
                format!("provider runtime {} has invalid archive checksum", self.id)
            })?;
        }
        Ok(())
    }

    pub fn payload_matches(&self, other: &Self) -> bool {
        let mut left = self.clone();
        let mut right = other.clone();
        left.url = None;
        left.archive_sha256 = None;
        right.url = None;
        right.archive_sha256 = None;
        left == right
    }

    pub fn platform_key(&self) -> String {
        format!("{}-{}", self.platform.os, self.platform.arch)
    }
}

impl ProviderRuntimePlatform {
    fn validate(&self) -> Result<()> {
        validate_coordinate("provider runtime platform os", &self.os)?;
        validate_coordinate("provider runtime platform arch", &self.arch)?;
        if let Some(target) = &self.target {
            validate_coordinate("provider runtime platform target", target)?;
        }
        if let Some(version) = &self.minimum_os_version {
            parse_dotted_version(version).with_context(|| {
                format!("invalid minimum OS version for {}/{}", self.os, self.arch)
            })?;
        }
        Ok(())
    }
}

impl ProviderRuntimeManifest {
    pub fn read_from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(PROVIDER_RUNTIME_MANIFEST_FILE);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read provider runtime manifest {}", path.display()))?;
        let manifest = Self::from_json_str(&text)
            .with_context(|| format!("parse provider runtime manifest {}", path.display()))?;
        manifest.verify_contents(dir)?;
        Ok(manifest)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(text)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn write_to_dir(&self, dir: &Path) -> Result<()> {
        self.validate()?;
        fs::create_dir_all(dir)
            .with_context(|| format!("create provider runtime directory {}", dir.display()))?;
        let path = dir.join(PROVIDER_RUNTIME_MANIFEST_FILE);
        let text = serde_json::to_string_pretty(self)?;
        fs::write(&path, format!("{text}\n"))
            .with_context(|| format!("write provider runtime manifest {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_version(self.schema_version)?;
        self.runtime.validate()
    }

    pub fn verify_contents(&self, dir: &Path) -> Result<()> {
        self.validate()?;
        let root = dir
            .canonicalize()
            .with_context(|| format!("canonicalize provider runtime root {}", dir.display()))?;
        for (relative, expected) in &self.runtime.files {
            let path = checked_runtime_file(&root, relative)?;
            let actual = sha256_file(&path)?;
            let expected = normalize_sha256(expected)?;
            if actual != expected {
                bail!(
                    "provider runtime checksum mismatch for {relative}: expected {expected}, got {actual}"
                );
            }
        }
        verify_entrypoint_executable(&root.join(&self.runtime.entrypoint))
    }
}

impl ProviderRuntimeReleaseManifest {
    pub fn read_from_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| {
            format!("read provider runtime release manifest {}", path.display())
        })?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(text)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_version(self.schema_version)?;
        let mut coordinates = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            let coordinate = (
                &artifact.id,
                &artifact.version,
                &artifact.platform.os,
                &artifact.platform.arch,
            );
            if !coordinates.insert(coordinate) {
                bail!(
                    "provider runtime release manifest repeats {} {} for {}/{}",
                    artifact.id,
                    artifact.version,
                    artifact.platform.os,
                    artifact.platform.arch
                );
            }
        }
        Ok(())
    }
}

pub(crate) fn normalize_sha256(value: &str) -> Result<String> {
    let trimmed = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    let digest = trimmed
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if digest.len() == 64 && digest.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Ok(digest)
    } else {
        bail!("invalid SHA-256 digest: {value}")
    }
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("open provider runtime file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash provider runtime file {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn validate_runtime_path(relative: &str) -> Result<()> {
    let path = Path::new(relative);
    if relative.trim().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("provider runtime path must be a safe relative path: {relative}");
    }
    Ok(())
}

pub(crate) fn parse_dotted_version(value: &str) -> Result<Vec<u64>> {
    let components = value
        .split('.')
        .map(|part| {
            part.parse::<u64>()
                .with_context(|| format!("invalid version component {part}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if components.is_empty() {
        bail!("version must not be empty");
    }
    Ok(components)
}

fn validate_schema_version(version: u32) -> Result<()> {
    if version != PROVIDER_RUNTIME_SCHEMA_VERSION {
        bail!(
            "unsupported provider runtime schema version {version}; expected {PROVIDER_RUNTIME_SCHEMA_VERSION}"
        );
    }
    Ok(())
}

fn validate_coordinate(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
    {
        bail!("{label} contains unsupported characters: {value}");
    }
    Ok(())
}

fn validate_protocol_version(value: &str) -> Result<()> {
    let components = parse_dotted_version(value)?;
    if components.len() != 2 {
        bail!("provider runtime protocol version must use major.minor: {value}");
    }
    Ok(())
}

fn validate_model(model: &ProviderRuntimeModel) -> Result<()> {
    if model.id.trim().is_empty()
        || model.id.trim() != model.id
        || model.id.len() > 256
        || model.id.chars().any(char::is_control)
    {
        bail!("provider runtime model id is invalid: {}", model.id);
    }
    validate_coordinate("provider runtime model kind", &model.kind)
}

fn checked_runtime_file(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_runtime_path(relative)?;
    let path = root.join(relative);
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("inspect provider runtime file {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("provider runtime file must not be a symlink: {relative}");
    }
    if !metadata.is_file() {
        bail!("provider runtime path is not a file: {relative}");
    }
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize provider runtime file {}", path.display()))?;
    if !canonical.starts_with(root) {
        bail!("provider runtime path escapes its bundle: {relative}");
    }
    Ok(canonical)
}

#[cfg(unix)]
fn verify_entrypoint_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = fs::metadata(path)?.permissions().mode();
    if mode & 0o111 == 0 {
        bail!(
            "provider runtime entrypoint is not executable: {}",
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn verify_entrypoint_executable(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_artifact(checksum: String) -> ProviderRuntimeArtifact {
        ProviderRuntimeArtifact {
            id: "meshllm-apple-runtime-darwin-arm64".to_string(),
            version: "0.1.0".to_string(),
            provider_kind: "apple".to_string(),
            protocol_version: "0.1".to_string(),
            platform: ProviderRuntimePlatform {
                os: "macos".to_string(),
                arch: "arm64".to_string(),
                target: Some("aarch64-apple-darwin".to_string()),
                minimum_os_version: Some("27.0".to_string()),
            },
            entrypoint: "bin/mesh-apple-runtime".to_string(),
            models: vec![ProviderRuntimeModel {
                id: "apple/system".to_string(),
                kind: "system".to_string(),
            }],
            features: BTreeSet::from(["streaming".to_string()]),
            files: BTreeMap::from([("bin/mesh-apple-runtime".to_string(), checksum)]),
            build: BTreeMap::new(),
            signature: None,
            url: None,
            archive_sha256: None,
        }
    }

    fn fixture_bundle() -> (tempfile::TempDir, ProviderRuntimeManifest) {
        let temp = tempfile::tempdir().unwrap();
        let binary = temp.path().join("bin/mesh-apple-runtime");
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, b"provider runtime").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let manifest = ProviderRuntimeManifest {
            schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
            runtime: fixture_artifact(sha256_file(&binary).unwrap()),
        };
        manifest.write_to_dir(temp.path()).unwrap();
        (temp, manifest)
    }

    #[test]
    fn verifies_a_complete_bundle() {
        let (temp, expected) = fixture_bundle();
        let actual = ProviderRuntimeManifest::read_from_dir(temp.path()).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_unsafe_entrypoint_paths() {
        let mut artifact = fixture_artifact("a".repeat(64));
        artifact.entrypoint = "../mesh-apple-runtime".to_string();
        let error = artifact.validate().unwrap_err();
        assert!(
            error.to_string().contains("safe relative path"),
            "{error:?}"
        );
    }

    #[test]
    fn rejects_tampered_files() {
        let (temp, _) = fixture_bundle();
        fs::write(temp.path().join("bin/mesh-apple-runtime"), b"tampered").unwrap();
        let error = ProviderRuntimeManifest::read_from_dir(temp.path()).unwrap_err();
        assert!(error.to_string().contains("checksum mismatch"), "{error:?}");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_payloads() {
        use std::os::unix::fs::symlink;
        let (temp, manifest) = fixture_bundle();
        let external = temp.path().join("external");
        fs::write(&external, b"provider runtime").unwrap();
        let binary = temp.path().join("bin/mesh-apple-runtime");
        fs::remove_file(&binary).unwrap();
        symlink(&external, &binary).unwrap();
        manifest.write_to_dir(temp.path()).unwrap();
        let error = ProviderRuntimeManifest::read_from_dir(temp.path()).unwrap_err();
        assert!(
            error.to_string().contains("must not be a symlink"),
            "{error:?}"
        );
    }
}
