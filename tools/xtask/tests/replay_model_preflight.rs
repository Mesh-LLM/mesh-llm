use sha2::{Digest, Sha256};
use std::process::Command;

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
    bytes.extend(value.as_bytes());
}

#[test]
fn preflight_binds_digest_and_native_context_before_writing_identity() {
    let state = tempfile::tempdir().unwrap();
    let model = state.path().join("model.gguf");
    let output = state.path().join("identity.json");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, "fixture");
    string(&mut bytes, "fixture.context_length");
    bytes.extend(4_u32.to_le_bytes());
    bytes.extend(131072_u32.to_le_bytes());
    let digest = hex::encode(Sha256::digest(&bytes));
    std::fs::write(&model, bytes).unwrap();
    let invoke = |minimum: &str, output: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "model-preflight", "--model"])
            .arg(&model)
            .args([
                "--sha256",
                &digest,
                "--minimum-context-tokens",
                minimum,
                "--output",
            ])
            .arg(output)
            .output()
            .unwrap()
    };
    let result = invoke("131072", &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let identity: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(identity["sha256"], digest);
    assert_eq!(identity["architecture"], "fixture");
    assert_eq!(identity["native_context_tokens"], 131072);
    let denied = state.path().join("denied.json");
    assert!(!invoke("131073", &denied).status.success());
    assert!(!denied.exists());
    std::fs::write(&model, b"modified model").unwrap();
    assert!(!invoke("131072", &denied).status.success());
    assert!(!denied.exists());
}
