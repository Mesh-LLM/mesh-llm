use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

pub(super) fn runtime_files() -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        (
            "manifest.json".into(),
            serde_json::to_vec(&json!({"runtime": {
                "id":"test-runtime", "mesh_version":"0.73.1", "skippy_abi":"0.1.0",
                "platform":{"os":"windows","arch":"x86_64","target":"x86_64-pc-windows-msvc"},
                "backend":{"kind":"cpu"}, "rank":0, "libraries":["lib/llama.dll"],
                "url":null,"sha256":null,"signature":null
            }}))
            .unwrap(),
        ),
        ("lib/llama.dll".into(), b"inert runtime fixture".to_vec()),
        ("Z-file".into(), b"uppercase".to_vec()),
        ("a-file".into(), b"lowercase".to_vec()),
    ])
}
fn tree_digest(files: &BTreeMap<String, Vec<u8>>) -> String {
    let mut hasher = Sha256::new();
    for (path, bytes) in files {
        hasher.update(u64::try_from(path.len()).unwrap().to_be_bytes());
        hasher.update(path.as_bytes());
        hasher.update(Sha256::digest(bytes));
    }
    hex::encode(hasher.finalize())
}
pub(super) fn product(host: &[u8], files: &BTreeMap<String, Vec<u8>>, tampered: bool) -> Value {
    json!({"schema_version":2,"contract":"mesh-llm-product-v2","mesh_version":"0.73.1","backend":"cpu",
        "host":{"path":"mesh-llm.exe","sha256":if tampered { "0".repeat(64) } else {hex::encode(Sha256::digest(host))}},
        "runtime":{"id":"test-runtime","path":"native-runtimes/test-runtime","sha256":tree_digest(files),
            "manifest_sha256":hex::encode(Sha256::digest(&files["manifest.json"]))}})
}
pub(super) fn write_bundle(root: &Path, host: &[u8], tampered: bool) {
    let bundle = root.join("mesh-bundle");
    fs::create_dir_all(&bundle).unwrap();
    fs::write(bundle.join("mesh-llm.exe"), host).unwrap();
    let files = runtime_files();
    for (path, bytes) in &files {
        let destination = bundle.join("native-runtimes/test-runtime").join(path);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::write(destination, bytes).unwrap();
    }
    fs::write(
        bundle.join("product-manifest.json"),
        serde_json::to_vec(&product(host, &files, tampered)).unwrap(),
    )
    .unwrap();
}
#[test]
fn portable_windows_installer_bundle_binds_all_files_and_preserves_case_order() {
    let root = tempfile::tempdir().unwrap();
    write_bundle(root.path(), b"portable inert host bytes", false);
    let runtime = root.path().join("mesh-bundle/native-runtimes/test-runtime");
    let digest = crate::product::digest::tree_sha256(&runtime)
        .map_err(|error| error.error)
        .unwrap();
    assert_eq!(digest, tree_digest(&runtime_files()));
    fs::write(runtime.join("Z-file"), b"changed uppercase").unwrap();
    assert_ne!(
        crate::product::digest::tree_sha256(&runtime)
            .map_err(|error| error.error)
            .unwrap(),
        digest
    );
    root.close()
        .expect("portable installer fixture deletion failed");
}
#[test]
fn portable_windows_installer_tamper_changes_only_claimed_host_identity() {
    let files = runtime_files();
    let healthy = product(b"inert host", &files, false);
    let mut tampered = product(b"inert host", &files, true);
    assert_ne!(healthy["host"]["sha256"], tampered["host"]["sha256"]);
    tampered["host"]["sha256"] = healthy["host"]["sha256"].clone();
    assert_eq!(tampered, healthy);
}
