use std::{fs, path::Path, process::Command};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

#[test]
fn console_verifier_when_assets_are_present_and_then_missing() {
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("assets")).unwrap();
    fs::write(directory.path().join("assets/main.js"), "export {};").unwrap();
    fs::write(
        directory.path().join("index.html"),
        "<script type=\"module\" src=\"/assets/main.js\"></script>",
    )
    .unwrap();
    let manifest = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["prepared-input", "sdk-console-manifest"])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(manifest.status.success());
    let verify = || {
        Command::new("bash")
            .arg(root().join("scripts/verify-sdk-console-assets.sh"))
            .arg(directory.path())
            .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
            .output()
            .unwrap()
    };
    assert!(verify().status.success());
    fs::remove_file(directory.path().join("assets/main.js")).unwrap();
    assert!(!verify().status.success());
}

#[test]
fn crate_metadata_when_field_is_selected_or_unsupported() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = directory.path().join("manifest.json");
    fs::write(&manifest, r#"{"artifact_id":"meshllm-native-linux-x86_64-cpu","sdk_version":"1.2.3","platform":"linux-x86_64","flavor":"cpu","target_triple":"x86_64-unknown-linux-gnu","backend":"cpu"}"#).unwrap();
    let select = |field| {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["prepared-input", "native-sdk-crate-field"])
            .arg(&manifest)
            .arg(field)
            .output()
            .unwrap()
    };
    let version = select("sdk_version");
    assert!(version.status.success());
    assert_eq!(version.stdout, b"1.2.3\n");
    assert!(!select("arbitrary").status.success());
}

#[cfg(target_os = "macos")]
#[test]
fn privacy_wrapper_when_real_template_is_valid() {
    let result = Command::new("bash")
        .arg(root().join("scripts/verify-swift-privacy-manifest.sh"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn native_sdk_wrapper_when_verified_library_is_packaged() {
    use sha2::{Digest, Sha256};
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("meshllm-native-linux-x86_64-cpu");
    fs::create_dir_all(artifact.join("lib")).unwrap();
    let library = b"inert native SDK library fixture";
    fs::write(artifact.join("lib/libmesh.so"), library).unwrap();
    let manifest = serde_json::json!({
        "schema_version": 1,
        "artifact_id": "meshllm-native-linux-x86_64-cpu",
        "native_runtime_id": "meshllm-native-linux-x86_64-cpu",
        "sdk_version": "1.2.3",
        "mesh_version": "1.2.3",
        "target_triple": "x86_64-unknown-linux-gnu",
        "platform": "linux-x86_64", "os": "linux", "arch": "x86_64",
        "backend": "cpu", "flavor": "cpu", "cargo_profile": "release",
        "library": "lib/libmesh.so", "library_paths": ["lib/libmesh.so"],
        "library_sha256": hex::encode(Sha256::digest(library)),
        "requirements": [],
        "features": ["mesh-inference", "model-management", "local-serving", "chat", "responses"]
    });
    fs::write(
        artifact.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let result = Command::new("bash")
        .arg(root().join("scripts/verify-native-sdk-package.sh"))
        .arg(&artifact)
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
