use super::{NativeRuntimeArtifact, NativeRuntimeManifest, NativeRuntimeReleaseManifest};
use serde::{Deserialize, Serialize};

const GENERATION: u32 = 2;

#[derive(Deserialize, Serialize)]
pub(super) struct Manifest {
    schema_version: u32,
    runtime: NativeRuntimeArtifact,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReleaseManifest {
    schema_version: u32,
    release_version: String,
    skippy_abi: String,
    #[serde(default)]
    artifacts: Vec<NativeRuntimeArtifact>,
}

fn check_generation(actual: u32) -> Result<(), String> {
    if actual == GENERATION {
        Ok(())
    } else {
        Err(format!(
            "unsupported native runtime manifest schema_version {actual}; expected {GENERATION}; legacy caches require explicit import"
        ))
    }
}

impl TryFrom<Manifest> for NativeRuntimeManifest {
    type Error = String;

    fn try_from(value: Manifest) -> Result<Self, Self::Error> {
        check_generation(value.schema_version)?;
        Ok(Self {
            runtime: value.runtime,
        })
    }
}

impl From<NativeRuntimeManifest> for Manifest {
    fn from(value: NativeRuntimeManifest) -> Self {
        Self {
            schema_version: GENERATION,
            runtime: value.runtime,
        }
    }
}

impl TryFrom<ReleaseManifest> for NativeRuntimeReleaseManifest {
    type Error = String;

    fn try_from(value: ReleaseManifest) -> Result<Self, Self::Error> {
        check_generation(value.schema_version)?;
        Ok(Self {
            release_version: value.release_version,
            skippy_abi: value.skippy_abi,
            artifacts: value.artifacts,
        })
    }
}

impl From<NativeRuntimeReleaseManifest> for ReleaseManifest {
    fn from(value: NativeRuntimeReleaseManifest) -> Self {
        Self {
            schema_version: GENERATION,
            release_version: value.release_version,
            skippy_abi: value.skippy_abi,
            artifacts: value.artifacts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn normal_readers_require_the_current_generation_and_release_spelling() {
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
}
