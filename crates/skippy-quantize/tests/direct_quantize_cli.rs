use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn direct_quantize_preflight_reports_requested_window_for_native_backends() {
    let root = unique_temp_dir();
    let source = root.join("source");
    let target = root.join("target");
    let source_prefix = source.join("BF16");
    let target_prefix = target.join("Q4");
    fs::create_dir_all(&source_prefix).unwrap();
    fs::create_dir_all(&target_prefix).unwrap();
    for index in 1..=3 {
        fs::write(
            source_prefix.join(format!("model-0000{index}-of-00003.gguf")),
            b"source shard",
        )
        .unwrap();
    }
    fs::write(
        target_prefix.join("model-q4-00002-of-00003.gguf"),
        b"already done",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-quantize"))
        .args([
            "quantize",
            "--backend",
            "llama-api",
            "--preflight-only",
            "--json",
            "--keep-split",
            "--first-split",
            "2",
            "--last-split",
            "3",
        ])
        .arg(source_prefix.join("model-00001-of-00003.gguf"))
        .arg(target_prefix.join("model-q4.gguf"))
        .arg("Q4_K")
        .output()
        .expect("skippy-quantize command should run");

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(
        stdout.contains(r#""requested_window""#),
        "preflight should report requested_window, got: {stdout}"
    );
    assert!(
        stdout.contains(r#""next_requested_window""#),
        "preflight should report next_requested_window, got: {stdout}"
    );
    assert!(
        stdout.contains(r#""first_split": 2"#) && stdout.contains(r#""last_split": 3"#),
        "preflight should include requested 2..3 window, got: {stdout}"
    );
    assert!(
        stdout.contains(r#""first_split": 3"#),
        "preflight should skip completed split 2 and report split 3 next, got: {stdout}"
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn direct_quantize_preflight_supports_current_directory_no_output_shape() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("model.gguf"), b"source shard").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-quantize"))
        .current_dir(&root)
        .args([
            "quantize",
            "--backend",
            "llama-api",
            "--preflight-only",
            "--json",
            "model.gguf",
            "Q4_K",
        ])
        .output()
        .expect("skippy-quantize command should run");

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|err| panic!("parse preflight JSON: {err}\n{stdout}"));

    assert_eq!(report["backend_kind"], "llama-api");
    assert_eq!(report["backend_ready"], true);
    assert_eq!(report["source_complete"], true);
    assert_eq!(report["expected_source_shards"], 1);
    assert_eq!(report["next_window"]["first_split"], 1);
    let manifest_path = report["manifest_path"]
        .as_str()
        .expect("manifest_path should be a string");
    assert!(
        manifest_path.ends_with(".ggml-model-Q4_K.Q4_K.skippy-quantize.json"),
        "no-output quantize shape should derive upstream-style sidecar, got {manifest_path}"
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn direct_quantize_preflight_accepts_base_quant_with_tensor_file() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("model.gguf"), b"source shard").unwrap();
    fs::write(root.join("tensor-types.txt"), b"blk.0.weight=Q8_0\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-quantize"))
        .current_dir(&root)
        .args([
            "quantize",
            "--backend",
            "llama-api",
            "--tensor-type-file",
            "tensor-types.txt",
            "--preflight-only",
            "--json",
            "model.gguf",
            "Q3_K_S",
        ])
        .output()
        .expect("skippy-quantize command should run");

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|err| panic!("parse preflight JSON: {err}\n{stdout}"));
    let manifest_path = report["manifest_path"]
        .as_str()
        .expect("manifest_path should be a string");
    assert!(
        manifest_path.ends_with(".ggml-model-Q3_K_S.Q3_K_S.skippy-quantize.json"),
        "base quant should drive output and sidecar names, got {manifest_path}"
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn direct_quantize_preflight_rejects_profile_quant_label() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("model.gguf"), b"source shard").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-quantize"))
        .current_dir(&root)
        .args([
            "quantize",
            "--backend",
            "llama-api",
            "--preflight-only",
            "--json",
            "model.gguf",
            "UD-Q3_K_S",
        ])
        .output()
        .expect("skippy-quantize command should run");

    assert!(
        !output.status.success(),
        "preflight should reject custom profile labels as quant modes"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("custom tensor-type recipes"),
        "error should explain profile labels are not quant modes, got: {stderr}"
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
        "skippy-quantize-cli-test-{}-{nanos}-{counter}",
        std::process::id()
    ))
}
