use crate::{
    HostRuntimeProfile, NativeRuntimeArtifact, NativeRuntimeCache, NativeRuntimeFlavor,
    NativeRuntimeReleaseManifest,
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, path::PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSelection {
    Recommended,
    Flavor(NativeRuntimeFlavor),
    Id(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRejection {
    MeshVersionMismatch { expected: String, actual: String },
    OsMismatch { expected: String, actual: String },
    ArchMismatch { expected: String, actual: String },
    TargetTripleMismatch { expected: String, actual: String },
    FlavorNotSupported { flavor: NativeRuntimeFlavor },
    GpuRequirementNotMet { required_name_contains: String },
    SelectionMismatch { selection: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub artifact: NativeRuntimeArtifact,
    pub compatible: bool,
    pub rank: i64,
    #[serde(default)]
    pub rejection_reasons: Vec<CandidateRejection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeSource {
    Installed { path: PathBuf },
    Bundle { path: PathBuf },
    Download { url: String },
    Missing,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeResolution {
    pub selected: NativeRuntimeArtifact,
    pub source: NativeRuntimeSource,
    #[serde(default)]
    pub evaluated: Vec<CandidateEvaluation>,
}

pub struct NativeRuntimeResolver {
    mesh_version: String,
    profile: HostRuntimeProfile,
    release_manifest: NativeRuntimeReleaseManifest,
    cache: NativeRuntimeCache,
    bundle_dirs: Vec<PathBuf>,
}

impl RuntimeSelection {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::Recommended);
        };
        let value = value.trim();
        if value.is_empty() {
            return Ok(Self::Recommended);
        }
        if value.starts_with("meshllm-native-") || value.starts_with("mesh-llm-native-") {
            return Ok(Self::Id(value.to_string()));
        }
        Ok(Self::Flavor(value.parse()?))
    }
}

impl NativeRuntimeResolver {
    pub fn new(
        mesh_version: impl Into<String>,
        profile: HostRuntimeProfile,
        release_manifest: NativeRuntimeReleaseManifest,
        cache: NativeRuntimeCache,
    ) -> Self {
        Self {
            mesh_version: mesh_version.into(),
            profile,
            release_manifest,
            cache,
            bundle_dirs: Vec::new(),
        }
    }

    pub fn with_bundle_dirs(mut self, bundle_dirs: Vec<PathBuf>) -> Self {
        self.bundle_dirs = bundle_dirs;
        self
    }

    pub fn resolve(&self, selection: &RuntimeSelection) -> Result<NativeRuntimeResolution> {
        let evaluated = evaluate_candidates(
            &self.release_manifest.artifacts,
            &self.profile,
            &self.mesh_version,
            selection,
        );
        let Some(selected) = best_candidate(&evaluated) else {
            bail!(
                "no compatible native runtime found for {} on {}/{}",
                self.mesh_version,
                self.profile.os,
                self.profile.arch
            );
        };
        Ok(NativeRuntimeResolution {
            source: self.source_for_artifact(&selected.artifact)?,
            selected: selected.artifact.clone(),
            evaluated,
        })
    }

    fn source_for_artifact(&self, artifact: &NativeRuntimeArtifact) -> Result<NativeRuntimeSource> {
        let installed = self
            .cache
            .find_installed(&artifact.mesh_version, &artifact.native_runtime_id)?;
        if let Some(installed) = installed {
            return Ok(NativeRuntimeSource::Installed {
                path: installed.path,
            });
        }
        for dir in &self.bundle_dirs {
            let Ok(manifest) = crate::NativeRuntimeManifest::read_from_dir(dir) else {
                continue;
            };
            if manifest.artifact.native_runtime_id == artifact.native_runtime_id
                && manifest.artifact.mesh_version == artifact.mesh_version
            {
                return Ok(NativeRuntimeSource::Bundle { path: dir.clone() });
            }
        }
        Ok(artifact
            .url
            .as_ref()
            .map(|url| NativeRuntimeSource::Download { url: url.clone() })
            .unwrap_or(NativeRuntimeSource::Missing))
    }
}

pub fn select_native_runtime(
    release_manifest: &NativeRuntimeReleaseManifest,
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    selection: &RuntimeSelection,
) -> Option<CandidateEvaluation> {
    let evaluated = evaluate_candidates(
        &release_manifest.artifacts,
        profile,
        mesh_version,
        selection,
    );
    best_candidate(&evaluated).cloned()
}

fn evaluate_candidates(
    artifacts: &[NativeRuntimeArtifact],
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    selection: &RuntimeSelection,
) -> Vec<CandidateEvaluation> {
    artifacts
        .iter()
        .map(|artifact| evaluate_artifact(artifact, profile, mesh_version, selection))
        .collect()
}

fn evaluate_artifact(
    artifact: &NativeRuntimeArtifact,
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    selection: &RuntimeSelection,
) -> CandidateEvaluation {
    let mut reasons = Vec::new();
    if artifact.mesh_version != mesh_version {
        reasons.push(CandidateRejection::MeshVersionMismatch {
            expected: mesh_version.to_string(),
            actual: artifact.mesh_version.clone(),
        });
    }
    if artifact.os != profile.os {
        reasons.push(CandidateRejection::OsMismatch {
            expected: profile.os.clone(),
            actual: artifact.os.clone(),
        });
    }
    if artifact.arch != profile.arch {
        reasons.push(CandidateRejection::ArchMismatch {
            expected: profile.arch.clone(),
            actual: artifact.arch.clone(),
        });
    }
    if let (Some(expected), Some(actual)) = (&artifact.target_triple, &profile.target_triple)
        && expected != actual
    {
        reasons.push(CandidateRejection::TargetTripleMismatch {
            expected: expected.clone(),
            actual: actual.clone(),
        });
    }
    if !profile.supports_flavor(&artifact.flavor) {
        reasons.push(CandidateRejection::FlavorNotSupported {
            flavor: artifact.flavor.clone(),
        });
    }
    for requirement in &artifact.requirements {
        for required_name_contains in &requirement.gpu_name_contains {
            if !profile.has_gpu_name_matching(required_name_contains) {
                reasons.push(CandidateRejection::GpuRequirementNotMet {
                    required_name_contains: required_name_contains.clone(),
                });
            }
        }
    }
    if let Some(reason) = selection_mismatch(selection, artifact) {
        reasons.push(reason);
    }
    CandidateEvaluation {
        artifact: artifact.clone(),
        compatible: reasons.is_empty(),
        rank: artifact.priority + artifact.flavor.default_rank(),
        rejection_reasons: reasons,
    }
}

fn selection_mismatch(
    selection: &RuntimeSelection,
    artifact: &NativeRuntimeArtifact,
) -> Option<CandidateRejection> {
    match selection {
        RuntimeSelection::Recommended => None,
        RuntimeSelection::Flavor(flavor) if flavor == &artifact.flavor => None,
        RuntimeSelection::Id(id) if id == &artifact.native_runtime_id => None,
        RuntimeSelection::Flavor(flavor) => Some(CandidateRejection::SelectionMismatch {
            selection: flavor.to_string(),
        }),
        RuntimeSelection::Id(id) => Some(CandidateRejection::SelectionMismatch {
            selection: id.clone(),
        }),
    }
}

fn best_candidate(evaluated: &[CandidateEvaluation]) -> Option<&CandidateEvaluation> {
    evaluated
        .iter()
        .filter(|candidate| candidate.compatible)
        .max_by(compare_candidates)
}

fn compare_candidates(left: &&CandidateEvaluation, right: &&CandidateEvaluation) -> Ordering {
    left.rank.cmp(&right.rank).then_with(|| {
        right
            .artifact
            .native_runtime_id
            .cmp(&left.artifact.native_runtime_id)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HostRuntimeProfile, NativeRuntimeArtifact, NativeRuntimeFlavor};
    use std::collections::BTreeSet;

    fn artifact(id: &str, flavor: NativeRuntimeFlavor) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            native_runtime_id: id.to_string(),
            mesh_version: "0.68.0".to_string(),
            target_triple: None,
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            flavor,
            priority: 0,
            skippy_abi_version: None,
            url: None,
            sha256: None,
            signature: None,
            library_paths: Vec::new(),
            requirements: Vec::new(),
        }
    }

    fn profile() -> HostRuntimeProfile {
        HostRuntimeProfile {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            target_triple: None,
            available_flavors: BTreeSet::from([
                NativeRuntimeFlavor::Cpu,
                NativeRuntimeFlavor::Cuda,
            ]),
            gpus: Vec::new(),
        }
    }

    #[test]
    fn recommended_prefers_accelerated_runtime() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            artifacts: vec![
                artifact("meshllm-native-linux-x86_64-cpu", NativeRuntimeFlavor::Cpu),
                artifact(
                    "meshllm-native-linux-x86_64-cuda",
                    NativeRuntimeFlavor::Cuda,
                ),
            ],
        };
        let selected = select_native_runtime(
            &manifest,
            &profile(),
            "0.68.0",
            &RuntimeSelection::Recommended,
        )
        .unwrap();
        assert_eq!(
            selected.artifact.native_runtime_id,
            "meshllm-native-linux-x86_64-cuda"
        );
    }

    #[test]
    fn exact_version_mismatch_rejects_candidate() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.67.0".to_string(),
            artifacts: vec![NativeRuntimeArtifact {
                mesh_version: "0.67.0".to_string(),
                ..artifact(
                    "meshllm-native-linux-x86_64-cuda",
                    NativeRuntimeFlavor::Cuda,
                )
            }],
        };
        assert!(
            select_native_runtime(
                &manifest,
                &profile(),
                "0.68.0",
                &RuntimeSelection::Recommended
            )
            .is_none()
        );
    }
}
