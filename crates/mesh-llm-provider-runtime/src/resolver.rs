use crate::{
    InstalledProviderRuntime, ProviderRuntimeArtifact, ProviderRuntimeCache,
    ProviderRuntimeManifest, ProviderRuntimeReleaseManifest,
};
use anyhow::{Result, bail};
use semver::Version;
use serde::{Deserialize, Serialize};
#[cfg(target_os = "macos")]
use std::process::Command;
use std::{cmp::Ordering, path::PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeHost {
    pub os: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os_version: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderRuntimeSource {
    Bundle { path: PathBuf },
    Installed { path: PathBuf },
    Download { url: String, sha256: String },
    Missing,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRuntimeResolution {
    pub selected: ProviderRuntimeArtifact,
    pub source: ProviderRuntimeSource,
}

#[derive(Clone, Debug)]
pub struct ProviderRuntimeResolver {
    host: ProviderRuntimeHost,
    release_manifest: ProviderRuntimeReleaseManifest,
    cache: ProviderRuntimeCache,
    bundle_dirs: Vec<PathBuf>,
}

#[derive(Clone)]
struct Candidate {
    artifact: ProviderRuntimeArtifact,
    source: ProviderRuntimeSource,
}

impl ProviderRuntimeHost {
    pub fn current() -> Self {
        Self {
            os: canonical_os(std::env::consts::OS).to_string(),
            arch: canonical_arch(std::env::consts::ARCH).to_string(),
            os_version: current_os_version(),
        }
    }
}

impl ProviderRuntimeResolver {
    pub fn new(
        host: ProviderRuntimeHost,
        release_manifest: ProviderRuntimeReleaseManifest,
        cache: ProviderRuntimeCache,
    ) -> Self {
        Self {
            host,
            release_manifest,
            cache,
            bundle_dirs: Vec::new(),
        }
    }

    pub fn with_bundle_dirs(mut self, bundle_dirs: Vec<PathBuf>) -> Self {
        self.bundle_dirs = bundle_dirs;
        self
    }

    pub fn resolve(&self, request: &ProviderRuntimeRequest) -> Result<ProviderRuntimeResolution> {
        if request.artifact_id.is_none()
            && request.provider_kind.is_none()
            && request.model_id.is_none()
        {
            bail!("provider runtime resolution requires an artifact, provider, or model selector");
        }
        let mut candidates = self.collect_candidates()?;
        candidates.retain(|candidate| self.matches(&candidate.artifact, request));
        candidates.sort_by(compare_candidates);
        let Some(selected) = candidates.into_iter().next() else {
            bail!(
                "no compatible executable provider runtime found for {}/{}",
                self.host.os,
                self.host.arch
            );
        };
        Ok(ProviderRuntimeResolution {
            selected: selected.artifact,
            source: selected.source,
        })
    }

    fn collect_candidates(&self) -> Result<Vec<Candidate>> {
        let mut candidates = Vec::new();
        for path in &self.bundle_dirs {
            let manifest = ProviderRuntimeManifest::read_from_dir(path)?;
            candidates.push(Candidate {
                artifact: manifest.runtime,
                source: ProviderRuntimeSource::Bundle { path: path.clone() },
            });
        }
        for installed in self.cache.list()? {
            candidates.push(installed_candidate(installed));
        }
        for artifact in &self.release_manifest.artifacts {
            let source = match (&artifact.url, &artifact.archive_sha256) {
                (Some(url), Some(sha256)) => ProviderRuntimeSource::Download {
                    url: url.clone(),
                    sha256: sha256.clone(),
                },
                _ => ProviderRuntimeSource::Missing,
            };
            candidates.push(Candidate {
                artifact: artifact.clone(),
                source,
            });
        }
        Ok(candidates)
    }

    fn matches(
        &self,
        artifact: &ProviderRuntimeArtifact,
        request: &ProviderRuntimeRequest,
    ) -> bool {
        canonical_os(&artifact.platform.os) == canonical_os(&self.host.os)
            && canonical_arch(&artifact.platform.arch) == canonical_arch(&self.host.arch)
            && minimum_os_matches(artifact, &self.host)
            && request
                .artifact_id
                .as_ref()
                .is_none_or(|expected| expected == &artifact.id)
            && request
                .provider_kind
                .as_ref()
                .is_none_or(|expected| expected == &artifact.provider_kind)
            && request
                .protocol_version
                .as_ref()
                .is_none_or(|expected| expected == &artifact.protocol_version)
            && request
                .model_id
                .as_ref()
                .is_none_or(|expected| artifact.models.iter().any(|model| &model.id == expected))
    }
}

fn installed_candidate(installed: InstalledProviderRuntime) -> Candidate {
    Candidate {
        artifact: installed.manifest.runtime,
        source: ProviderRuntimeSource::Installed {
            path: installed.path,
        },
    }
}

fn compare_candidates(left: &Candidate, right: &Candidate) -> Ordering {
    let left_version = Version::parse(&left.artifact.version).expect("validated runtime version");
    let right_version = Version::parse(&right.artifact.version).expect("validated runtime version");
    right_version
        .cmp(&left_version)
        .then_with(|| source_rank(&left.source).cmp(&source_rank(&right.source)))
        .then_with(|| left.artifact.id.cmp(&right.artifact.id))
}

fn source_rank(source: &ProviderRuntimeSource) -> u8 {
    match source {
        ProviderRuntimeSource::Bundle { .. } => 0,
        ProviderRuntimeSource::Installed { .. } => 1,
        ProviderRuntimeSource::Download { .. } => 2,
        ProviderRuntimeSource::Missing => 3,
    }
}

fn minimum_os_matches(artifact: &ProviderRuntimeArtifact, host: &ProviderRuntimeHost) -> bool {
    let Some(minimum) = &artifact.platform.minimum_os_version else {
        return true;
    };
    let Some(host_version) = &host.os_version else {
        return false;
    };
    dotted_version_at_least(host_version, minimum)
}

fn dotted_version_at_least(actual: &str, minimum: &str) -> bool {
    let Ok(mut actual) = crate::manifest::parse_dotted_version(actual) else {
        return false;
    };
    let Ok(mut minimum) = crate::manifest::parse_dotted_version(minimum) else {
        return false;
    };
    let width = actual.len().max(minimum.len());
    actual.resize(width, 0);
    minimum.resize(width, 0);
    actual >= minimum
}

fn canonical_os(value: &str) -> &str {
    match value {
        "darwin" => "macos",
        other => other,
    }
}

fn canonical_arch(value: &str) -> &str {
    match value {
        "aarch64" => "arm64",
        "amd64" => "x86_64",
        other => other,
    }
}

#[cfg(target_os = "macos")]
fn current_os_version() -> Option<String> {
    let output = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(not(target_os = "macos"))]
fn current_os_version() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PROVIDER_RUNTIME_SCHEMA_VERSION, ProviderRuntimeModel, ProviderRuntimePlatform};
    use std::collections::{BTreeMap, BTreeSet};

    fn artifact(version: &str, url: Option<&str>) -> ProviderRuntimeArtifact {
        ProviderRuntimeArtifact {
            id: "apple-runtime".to_string(),
            version: version.to_string(),
            provider_kind: "apple".to_string(),
            protocol_version: "0.1".to_string(),
            platform: ProviderRuntimePlatform {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                target: None,
                minimum_os_version: Some("27.0".to_string()),
            },
            entrypoint: "bin/provider".to_string(),
            models: vec![ProviderRuntimeModel {
                id: "apple/system".to_string(),
                kind: "system".to_string(),
            }],
            features: BTreeSet::new(),
            files: BTreeMap::from([("bin/provider".to_string(), "a".repeat(64))]),
            build: BTreeMap::new(),
            signature: None,
            url: url.map(str::to_string),
            archive_sha256: url.map(|_| "b".repeat(64)),
        }
    }

    #[test]
    fn selects_the_newest_compatible_download() {
        let temp = tempfile::tempdir().unwrap();
        let resolver = ProviderRuntimeResolver::new(
            ProviderRuntimeHost {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                os_version: Some("27.1".to_string()),
            },
            ProviderRuntimeReleaseManifest {
                schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
                artifacts: vec![
                    artifact("0.1.0", Some("https://example.invalid/old.zip")),
                    artifact("0.2.0", Some("https://example.invalid/new.zip")),
                ],
            },
            ProviderRuntimeCache::new(temp.path().join("cache")),
        );

        let resolution = resolver
            .resolve(&ProviderRuntimeRequest {
                provider_kind: Some("apple".to_string()),
                model_id: Some("apple/system".to_string()),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(resolution.selected.version, "0.2.0");
        assert!(matches!(
            resolution.source,
            ProviderRuntimeSource::Download { .. }
        ));
    }

    #[test]
    fn rejects_a_runtime_above_the_host_os_version() {
        let temp = tempfile::tempdir().unwrap();
        let resolver = ProviderRuntimeResolver::new(
            ProviderRuntimeHost {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                os_version: Some("26.6".to_string()),
            },
            ProviderRuntimeReleaseManifest {
                schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
                artifacts: vec![artifact("0.1.0", None)],
            },
            ProviderRuntimeCache::new(temp.path().join("cache")),
        );

        assert!(
            resolver
                .resolve(&ProviderRuntimeRequest {
                    provider_kind: Some("apple".to_string()),
                    ..Default::default()
                })
                .is_err()
        );
    }
}
