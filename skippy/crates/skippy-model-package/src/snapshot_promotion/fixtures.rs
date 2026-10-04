use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, PackageManifest, SourceFile, SourceModel,
    TensorCatalog,
};
use std::collections::BTreeMap;

pub(super) fn manifest(path: &str, byte_size: u64, sha256: String) -> Vec<u8> {
    let mut manifest = PackageManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        package_id: String::new(),
        model_id: "fixture/model".into(),
        source_model: SourceModel {
            sha256: "b".repeat(64),
            metadata_artifact_id: "metadata".into(),
            repo: Some("fixture/model".into()),
            revision: Some("b".repeat(40)),
            primary_file: Some("model.gguf".into()),
            canonical_ref: None,
            distribution_id: None,
            files: vec![SourceFile {
                path: "model.gguf".into(),
                byte_size: 256,
                sha256: "b".repeat(64),
            }],
        },
        format: "gguf".into(),
        layer_count: 1,
        model_metadata: BTreeMap::new(),
        artifact_catalog: ArtifactCatalog {
            entries: vec![Artifact {
                id: "metadata".into(),
                path: path.into(),
                byte_size,
                sha256,
            }],
        },
        tensor_catalog: TensorCatalog {
            entries: Vec::new(),
        },
        sidecars: Vec::new(),
        publisher_metadata: Vec::new(),
        publisher_defaults: None,
        generation: None,
        native_abi_version: "7".into(),
        generator_version: "fixture".into(),
        created_at_unix_secs: 0,
    };
    manifest.package_id = manifest.computed_package_id().unwrap();
    manifest.validate_root().unwrap();
    serde_json::to_vec(&manifest).unwrap()
}
