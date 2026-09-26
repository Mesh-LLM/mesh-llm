use super::*;
use skippy_package_format::{ValidationCode, ValidationErrors};

fn fixture(root: &Path, producer_abi: &str) -> PackageManifestV2 {
    let mut manifest = write_test_package_v2_fixture(
        root,
        "fixture/model",
        &[("payload", "shared/payload.gguf", "blk.0.weight")],
    )
    .unwrap();
    manifest.native_abi_version = producer_abi.to_string();
    manifest.package_id = manifest.computed_package_id().unwrap();
    write_manifest(root, &manifest);
    manifest
}

fn write_manifest(root: &Path, manifest: &PackageManifestV2) {
    std::fs::write(
        root.join(PACKAGE_V2_MANIFEST),
        serde_json::to_vec(manifest).unwrap(),
    )
    .unwrap();
}

fn identities(root: &Path) -> [Result<SkippyPackageIdentity>; 2] {
    [
        identity_from_package_v2(root),
        identity_from_package_v2_metadata("hf://fixture/model@revision", &root.to_string_lossy()),
    ]
}

#[test]
fn package_admission_accepts_different_producer_abis() {
    // Published catalog versions plus a future producer: neither older nor
    // newer native function signatures imply a different on-disk format.
    for producer_abi in ["0.1.53", "0.1.54", "0.1.57", "99.0.0"] {
        let root = tempfile::tempdir().unwrap();
        let manifest = fixture(root.path(), producer_abi);
        for result in identities(root.path()) {
            let identity = result.unwrap();
            assert_eq!(identity.layer_count, 2);
            assert_eq!(identity.tensor_count, 1);
            assert_eq!(identity.source_model_sha256, manifest.source_model.sha256);
        }
        let stored: PackageManifestV2 =
            serde_json::from_slice(&std::fs::read(root.path().join(PACKAGE_V2_MANIFEST)).unwrap())
                .unwrap();
        assert_eq!(stored.native_abi_version, producer_abi);
        assert_eq!(stored.package_id, manifest.package_id);
    }
}

#[test]
fn producer_abi_remains_bound_to_package_identity() {
    let root = tempfile::tempdir().unwrap();
    let mut manifest = fixture(root.path(), "0.1.54");
    manifest.native_abi_version = "0.1.53".to_string();
    write_manifest(root.path(), &manifest);
    for result in identities(root.path()) {
        let error = result.unwrap_err();
        let validation = error.downcast_ref::<ValidationErrors>().unwrap();
        assert!(validation.issues().iter().any(|issue| {
            issue.code == ValidationCode::PackageIdentityMismatch && issue.path == "package_id"
        }));
    }
}

#[test]
fn different_producer_abi_does_not_bypass_metadata_integrity() {
    let root = tempfile::tempdir().unwrap();
    let mut manifest = fixture(root.path(), "99.0.0");
    let metadata = manifest
        .artifact_catalog
        .entries
        .iter_mut()
        .find(|artifact| artifact.id == "metadata")
        .unwrap();
    metadata.sha256 = "0".repeat(64);
    manifest.package_id = manifest.computed_package_id().unwrap();
    write_manifest(root.path(), &manifest);
    for result in identities(root.path()) {
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("SHA-256 differs"), "{error}");
    }
}
