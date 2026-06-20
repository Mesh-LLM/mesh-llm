#![cfg(unix)]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn installed_shims_dispatch_by_argv0() {
    let root = unique_temp_dir();
    let bin_dir = root.join("bin");
    let checkpoint = root.join("checkpoint");
    fs::create_dir_all(&checkpoint).unwrap();
    fs::write(root.join("model.gguf"), b"source shard").unwrap();
    write_safetensor(
        &checkpoint.join("model.safetensors"),
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[2, 2],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        )],
    );

    let install = Command::new(env!("CARGO_BIN_EXE_skippy-quantize"))
        .args(["install-shims", "--dir"])
        .arg(&bin_dir)
        .args(["--binary", env!("CARGO_BIN_EXE_skippy-quantize")])
        .output()
        .expect("install-shims should run");
    assert!(
        install.status.success(),
        "install-shims should succeed: stderr={}",
        String::from_utf8_lossy(&install.stderr)
    );

    let convert_report = run_json_command(
        Command::new(bin_dir.join("hf_to_gguff.py"))
            .args(["--backend", "native-rust", "--preflight-only", "--json"])
            .arg(&checkpoint),
    );
    assert_eq!(convert_report["kind"], "CONVERT_HF");

    let quant_report = run_json_command(
        Command::new(bin_dir.join("llama-quantize"))
            .current_dir(&root)
            .args([
                "--llama-quantize",
                env!("CARGO_BIN_EXE_skippy-quantize"),
                "--preflight-only",
                "--json",
                "model.gguf",
                "Q4_K",
            ]),
    );
    assert_eq!(quant_report["kind"], "QUANTIZE_GGUF");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn hf_to_gguff_argv0_dispatches_to_direct_convert() {
    let root = unique_temp_dir();
    let bin_dir = root.join("bin");
    let checkpoint = root.join("checkpoint");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::create_dir_all(&checkpoint).unwrap();
    symlink(
        env!("CARGO_BIN_EXE_skippy-quantize"),
        bin_dir.join("hf_to_gguff.py"),
    )
    .unwrap();
    write_safetensor(
        &checkpoint.join("model.safetensors"),
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[2, 2],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        )],
    );

    let report = run_json_command(
        Command::new(bin_dir.join("hf_to_gguff.py"))
            .args(["--backend", "native-rust", "--preflight-only", "--json"])
            .arg(&checkpoint),
    );

    assert_eq!(report["kind"], "CONVERT_HF");
    assert_eq!(report["backend_kind"], "native-rust");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn llama_quantize_argv0_dispatches_to_direct_quantize() {
    let root = unique_temp_dir();
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::write(root.join("model.gguf"), b"source shard").unwrap();
    symlink(
        env!("CARGO_BIN_EXE_skippy-quantize"),
        bin_dir.join("llama-quantize"),
    )
    .unwrap();

    let report = run_json_command(
        Command::new(bin_dir.join("llama-quantize"))
            .current_dir(&root)
            .args([
                "--llama-quantize",
                env!("CARGO_BIN_EXE_skippy-quantize"),
                "--preflight-only",
                "--json",
                "model.gguf",
                "Q4_K",
            ]),
    );

    assert_eq!(report["kind"], "QUANTIZE_GGUF");
    assert_eq!(report["backend_kind"], "external-process");
    fs::remove_dir_all(root).unwrap();
}

fn unique_temp_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "skippy-argv0-shim-test-{}-{nanos}-{counter}",
        std::process::id()
    ))
}

fn run_json_command(command: &mut Command) -> serde_json::Value {
    let output = command.output().expect("command should run");
    assert!(
        output.status.success(),
        "command should succeed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    serde_json::from_str(&stdout).unwrap_or_else(|err| panic!("parse JSON: {err}\n{stdout}"))
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
