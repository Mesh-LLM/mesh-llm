use std::fs;
use std::process::Command;

#[test]
fn inspect_local_gguf_does_not_prepare_download_caches() {
    let temp = tempfile::tempdir().unwrap();
    let model = temp.path().join("empty.gguf");
    // A valid GGUF with no metadata or tensors is enough to inspect its inventory.
    let mut gguf = b"GGUF".to_vec();
    gguf.extend_from_slice(&3_u32.to_le_bytes());
    gguf.extend_from_slice(&0_u64.to_le_bytes());
    gguf.extend_from_slice(&0_u64.to_le_bytes());
    gguf.resize(32, 0);
    fs::write(&model, gguf).unwrap();
    let cache = temp.path().join("unused-cache");
    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("inspect")
        .arg(&model)
        .env("HF_HOME", &cache)
        .env("HF_HUB_CACHE", cache.join("hub"))
        .env("HUGGINGFACE_HUB_CACHE", cache.join("hub"))
        .env("HF_XET_CACHE", cache.join("xet"))
        .env("MESH_LLM_DATA_DIR", cache.join("fallback"))
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let inventory: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(inventory["tensor_count"], 0);
    assert_eq!(inventory["tensors"], serde_json::json!([]));
    assert!(!cache.exists(), "local inspection created download caches");
}
