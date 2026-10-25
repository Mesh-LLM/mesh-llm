//! Convert trusted, supervised worker receipts into runner identity types.
use super::{
    identity_worker::{Evidence, Input},
    manifest_output, plan,
};
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) struct Metadata {
    pub sides: [plan::Side; 2],
    pub binaries: [manifest_output::Binary; 2],
    pub host: manifest_output::Host,
    pub model: std::path::PathBuf,
    pub source_model_sha256: String,
    pub model_metadata: Value,
    pub runtime_roots: [std::path::PathBuf; 2],
    /// The caller must validate runtime packages with the owning verifier before launch.
    pub runtime_packages_verified: bool,
}
fn digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
pub(super) fn bind(
    input: &Input,
    sides: &[plan::Side; 2],
    evidence: Evidence,
    versions: [Option<String>; 2],
) -> DynResult<Metadata> {
    if input.schema_version != 1
        || evidence.schema_version != 1
        || evidence.model_metadata.sha256 != evidence.model.sha256
        || !digest(&evidence.model.sha256)
        || !evidence.model.path.is_absolute()
        || evidence.model.bytes == 0
        || evidence.model_metadata.native_context_tokens < input.minimum_context_tokens
    {
        return Err("metadata receipt schema, model digest or context identity mismatch".into());
    }
    for (index, side) in sides.iter().enumerate() {
        let binary = &evidence.binaries[index];
        if side.binary != input.binaries[index]
            || !binary.file.path.is_absolute()
            || !digest(&binary.file.sha256)
            || binary.file.bytes == 0
            || binary
                .file
                .path
                .parent()
                .map(|parent| parent.join("native-runtimes"))
                != Some(binary.adjacent_runtime_root.clone())
            || binary.runtime_validation != "pending_owning_native_package_and_loader_policy"
        {
            return Err("metadata receipt side identity or adjacent runtime root mismatch".into());
        }
    }
    let [first, second] = evidence.binaries;
    let [first_version, second_version] = versions;
    let system = match std::env::consts::OS {
        "macos" => "Darwin",
        "linux" => "Linux",
        other => other,
    };
    let machine = match (system, std::env::consts::ARCH) {
        ("Darwin", "aarch64") => "arm64",
        (_, arch) => arch,
    };
    let mut first_side = sides[0].clone();
    first_side.binary = first.file.path.clone();
    let mut second_side = sides[1].clone();
    second_side.binary = second.file.path.clone();
    Ok(Metadata {
        sides: [first_side, second_side],
        binaries: [
            manifest_output::Binary {
                path: first.file.path,
                sha256: first.file.sha256,
                version: first_version,
            },
            manifest_output::Binary {
                path: second.file.path,
                sha256: second.file.sha256,
                version: second_version,
            },
        ],
        host: manifest_output::Host::classify(system.into(), machine.into()),
        model: evidence.model.path,
        source_model_sha256: evidence.model.sha256,
        model_metadata: json!(evidence.model_metadata),
        runtime_roots: [first.adjacent_runtime_root, second.adjacent_runtime_root],
        runtime_packages_verified: false,
    })
}
#[cfg(test)]
mod tests {
    use super::super::identity_worker::{BinaryIdentity, FileIdentity, ModelMetadata};
    use super::*;
    use std::path::PathBuf;
    fn value() -> (Input, [plan::Side; 2], Evidence) {
        let root = tempfile::tempdir().unwrap().path().to_owned();
        let path = root.join("binary");
        let model = root.join("model.gguf");
        let input = Input {
            schema_version: 1,
            binaries: [path.clone(), path.clone()],
            model: model.clone(),
            minimum_context_tokens: 512,
        };
        let sides = plan::sides(
            path.clone(),
            None,
            &[plan::Mode::Production, plan::Mode::EventDisabled],
        )
        .unwrap();
        let binary = || BinaryIdentity {
            file: FileIdentity {
                path: path.clone(),
                sha256: "a".repeat(64),
                bytes: 10,
            },
            adjacent_runtime_root: root.join("native-runtimes"),
            runtime_validation: "pending_owning_native_package_and_loader_policy".into(),
        };
        let evidence = Evidence {
            schema_version: 1,
            binaries: [binary(), binary()],
            model: FileIdentity {
                path: model,
                sha256: "b".repeat(64),
                bytes: 30,
            },
            model_metadata: ModelMetadata {
                sha256: "b".repeat(64),
                architecture: "fixture".into(),
                native_context_tokens: 1024,
            },
        };
        (input, sides, evidence)
    }
    #[test]
    fn receipt_binding_returns_runner_types_without_claiming_runtime_validation() {
        let (input, sides, evidence) = value();
        let result = bind(&input, &sides, evidence, [Some("v1".into()), None]).unwrap();
        assert_eq!(result.binaries[0].version.as_deref(), Some("v1"));
        assert_eq!(result.sides[1].mode, plan::Mode::EventDisabled);
        assert_eq!(result.binaries[0].path, result.sides[0].binary);
        assert_eq!(result.source_model_sha256, "b".repeat(64));
        assert!(!result.runtime_packages_verified);
    }
    #[test]
    fn receipt_binding_refuses_invalid_model_or_side_correlation() {
        let (input, sides, mut evidence) = value();
        evidence.model_metadata.sha256 = "c".repeat(64);
        assert!(bind(&input, &sides, evidence, [None, None]).is_err());
        let (mut input, sides, evidence) = value();
        input.binaries[0] = PathBuf::from("/different/binary");
        assert!(bind(&input, &sides, evidence, [None, None]).is_err());
    }
    #[test]
    fn receipt_binding_refuses_escaping_runtime_or_fabricated_validation_status() {
        let (input, sides, mut evidence) = value();
        evidence.binaries[0].adjacent_runtime_root = PathBuf::from("/outside/root");
        assert!(bind(&input, &sides, evidence, [None, None]).is_err());
        let (input, sides, mut evidence) = value();
        evidence.binaries[0].runtime_validation = "verified".into();
        assert!(bind(&input, &sides, evidence, [None, None]).is_err());
    }
}
