use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn direct_convert_native_preflight_resolves_auto_output_type() {
    let root = unique_temp_dir();
    let checkpoint = root.join("checkpoint");
    fs::create_dir_all(&checkpoint).unwrap();
    write_safetensor(
        &checkpoint.join("model.safetensors"),
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[2, 2],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        )],
    );

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-quantize"))
        .args(["convert", "--preflight-only", "--json"])
        .arg(&checkpoint)
        .output()
        .expect("skippy-quantize command should run");

    assert!(
        output.status.success(),
        "preflight should succeed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|err| panic!("parse preflight JSON: {err}\n{stdout}"));

    assert_eq!(report["backend_kind"], "native-rust");
    assert_eq!(report["backend_ready"], true);
    assert_eq!(report["expected_target_shards"], 1);
    assert_eq!(report["next_window"]["first_split"], 1);
    assert_eq!(report["next_window"]["last_split"], 1);
    let manifest_path = report["manifest_path"]
        .as_str()
        .expect("manifest_path should be a string");
    assert!(
        manifest_path.ends_with("/checkpoint/.checkpoint-bf16.bf16.skippy-convert.json"),
        "native auto output should resolve to bf16 manifest path, got {manifest_path}"
    );

    fs::remove_dir_all(root).unwrap();
}

fn unique_temp_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "skippy-convert-cli-test-{}-{nanos}-{counter}",
        std::process::id()
    ))
}

fn write_safetensor(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
    let mut offset = 0_u64;
    let mut entries = serde_json::Map::new();
    for (name, dtype, shape, bytes) in tensors {
        let end = offset + bytes.len() as u64;
        entries.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let header = serde_json::Value::Object(entries).to_string();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for (_, _, _, tensor_bytes) in tensors {
        bytes.extend_from_slice(tensor_bytes);
    }
    fs::write(path, bytes).unwrap();
}
