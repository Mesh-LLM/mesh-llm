use super::{
    NativeRuntimeArtifact, NativeRuntimeManifest, NativeRuntimePlatform,
    NativeRuntimeReleaseManifest,
};
use crate::NativeRuntimeBackend;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const GENERATION: u32 = 2;

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum Manifest {
    Current(CurrentManifest),
    Legacy(LegacyManifest),
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CurrentManifest {
    schema_version: u32,
    runtime: NativeRuntimeArtifact,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LegacyManifest {
    runtime: LegacyArtifact,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum ReleaseManifest {
    Current(CurrentReleaseManifest),
    Legacy(LegacyReleaseManifest),
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CurrentReleaseManifest {
    schema_version: u32,
    release_version: String,
    skippy_abi: String,
    #[serde(default)]
    artifacts: Vec<NativeRuntimeArtifact>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LegacyReleaseManifest {
    mesh_version: String,
    skippy_abi: String,
    #[serde(default)]
    artifacts: Vec<LegacyArtifact>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LegacyArtifact {
    id: String,
    mesh_version: String,
    skippy_abi: String,
    platform: NativeRuntimePlatform,
    backend: NativeRuntimeBackend,
    #[serde(default)]
    rank: i64,
    libraries: Vec<String>,
    #[serde(default)]
    files: BTreeMap<String, String>,
    #[serde(default)]
    tools: BTreeMap<String, String>,
    url: Option<String>,
    sha256: Option<String>,
    signature: Option<String>,
}

impl From<LegacyArtifact> for NativeRuntimeArtifact {
    fn from(value: LegacyArtifact) -> Self {
        Self {
            id: value.id,
            release_version: Some(value.mesh_version),
            skippy_abi: value.skippy_abi,
            platform: value.platform,
            backend: value.backend,
            rank: value.rank,
            libraries: value.libraries,
            files: value.files,
            tools: value.tools,
            url: value.url,
            sha256: value.sha256,
            signature: value.signature,
        }
    }
}

fn check_generation(actual: u32) -> Result<(), String> {
    if actual == GENERATION {
        Ok(())
    } else {
        Err(format!(
            "unsupported native runtime manifest schema_version {actual}; expected {GENERATION} or the unversioned Mesh format"
        ))
    }
}

impl TryFrom<Manifest> for NativeRuntimeManifest {
    type Error = String;

    fn try_from(value: Manifest) -> Result<Self, Self::Error> {
        let runtime = match value {
            Manifest::Current(value) => {
                check_generation(value.schema_version)?;
                value.runtime
            }
            Manifest::Legacy(value) => value.runtime.into(),
        };
        Ok(Self { runtime })
    }
}

impl From<NativeRuntimeManifest> for Manifest {
    fn from(value: NativeRuntimeManifest) -> Self {
        Self::Current(CurrentManifest {
            schema_version: GENERATION,
            runtime: value.runtime,
        })
    }
}

impl TryFrom<ReleaseManifest> for NativeRuntimeReleaseManifest {
    type Error = String;

    fn try_from(value: ReleaseManifest) -> Result<Self, Self::Error> {
        match value {
            ReleaseManifest::Current(value) => {
                check_generation(value.schema_version)?;
                Ok(Self {
                    release_version: value.release_version,
                    skippy_abi: value.skippy_abi,
                    artifacts: value.artifacts,
                })
            }
            ReleaseManifest::Legacy(value) => Ok(Self {
                release_version: value.mesh_version,
                skippy_abi: value.skippy_abi,
                artifacts: value.artifacts.into_iter().map(Into::into).collect(),
            }),
        }
    }
}

impl From<NativeRuntimeReleaseManifest> for ReleaseManifest {
    fn from(value: NativeRuntimeReleaseManifest) -> Self {
        Self::Current(CurrentReleaseManifest {
            schema_version: GENERATION,
            release_version: value.release_version,
            skippy_abi: value.skippy_abi,
            artifacts: value.artifacts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn readers_accept_schema_two_and_reject_mixed_or_invalid_versions() {
        let artifact = json!({
            "id": "runtime-a", "release_version": "1.2.3", "skippy_abi": "0.1.57",
            "platform": {"os": "macos", "arch": "aarch64"},
            "backend": {"kind": "cpu"}, "libraries": ["lib/runtime.dylib"]
        });
        let current = json!({"schema_version": 2, "runtime": artifact});
        let parsed: NativeRuntimeManifest = serde_json::from_value(current.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap()["schema_version"], 2);
        let catalog = json!({"schema_version": 2, "release_version": "1.2.3",
            "skippy_abi": "0.1.57", "artifacts": [artifact]});
        let parsed: NativeRuntimeReleaseManifest = serde_json::from_value(catalog.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(parsed).unwrap()["release_version"],
            "1.2.3"
        );
        for generation in [
            None,
            Some(Value::Null),
            Some(json!(1)),
            Some(json!(3)),
            Some(json!("2")),
        ] {
            let mut runtime = current.clone();
            let mut release = catalog.clone();
            for value in [&mut runtime, &mut release] {
                match &generation {
                    Some(generation) => value["schema_version"] = generation.clone(),
                    None => {
                        value.as_object_mut().unwrap().remove("schema_version");
                    }
                }
            }
            assert!(serde_json::from_value::<NativeRuntimeManifest>(runtime).is_err());
            assert!(serde_json::from_value::<NativeRuntimeReleaseManifest>(release).is_err());
        }
        let mut wrong_spelling = current;
        let fields = wrong_spelling["runtime"].as_object_mut().unwrap();
        let release = fields.remove("release_version").unwrap();
        fields.insert("mesh_version".into(), release);
        assert!(serde_json::from_value::<NativeRuntimeManifest>(wrong_spelling).is_err());
    }

    #[test]
    fn released_v078_catalog_maps_to_internal_release_version() {
        // Published asset: https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.78.0/native-runtimes.json
        let manifest = NativeRuntimeReleaseManifest::from_json_str(include_str!(
            "../../tests/fixtures/native-runtimes-v0.78.0.json"
        ))
        .unwrap();
        assert_eq!(manifest.release_version, "0.78.0");
        assert_eq!(manifest.skippy_abi, "0.1.66");
        assert_eq!(manifest.artifacts.len(), 13);
        assert!(
            manifest
                .artifacts
                .iter()
                .all(|artifact| { artifact.release_version.as_deref() == Some("0.78.0") })
        );
    }

    #[test]
    fn legacy_runtime_requires_mesh_version_and_rejects_mixed_formats() {
        let legacy = json!({"runtime": {
            "id": "runtime-a", "mesh_version": "0.78.0", "skippy_abi": "0.1.66",
            "platform": {"os": "linux", "arch": "x86_64"},
            "backend": {"kind": "cpu"}, "libraries": ["lib/runtime.so"]
        }});
        let parsed: NativeRuntimeManifest = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(parsed.runtime.release_version.as_deref(), Some("0.78.0"));
        let mut missing = legacy.clone();
        missing["runtime"]
            .as_object_mut()
            .unwrap()
            .remove("mesh_version");
        assert!(serde_json::from_value::<NativeRuntimeManifest>(missing).is_err());
        let mut mixed = legacy.clone();
        mixed["runtime"]["release_version"] = json!("0.78.0");
        assert!(serde_json::from_value::<NativeRuntimeManifest>(mixed).is_err());
        let mut schema = legacy;
        schema["schema_version"] = json!(2);
        assert!(serde_json::from_value::<NativeRuntimeManifest>(schema).is_err());
    }
}
