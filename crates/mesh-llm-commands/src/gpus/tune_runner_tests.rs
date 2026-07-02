use super::super::dispatch_gpu_command;
use super::*;
use mesh_llm_cli::{GpuCommand, benchmark::BenchmarkCommand};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    push_gguf_string(bytes, value);
}

fn write_valid_tune_fixture(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&8i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_u32_kv(&mut bytes, "llama.context_length", 8192);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    push_u32_kv(&mut bytes, "llama.attention.value_length", 128);
    let mut file = fs::File::create(&path).expect("test fixture should create GGUF file");
    file.write_all(&bytes)
        .expect("test fixture should write GGUF file");
    file.flush().expect("test fixture should flush GGUF file");
    path
}

fn write_config_with_models(config_path: &Path, models: &[PathBuf]) {
    let mut raw = String::from("version = 1\n");
    for model in models {
        raw.push_str(&format!("\n[[models]]\nmodel = \"{}\"\n", model.display()));
    }
    fs::write(config_path, raw).expect("fixture config should be written");
}

#[test]
fn gpu_tune_dispatches_to_runner() {
    let temp = tempdir().expect("tempdir should be created");
    let path = temp.path().join("sample.gguf");
    fs::write(&path, b"GGUF").expect("fixture GGUF should be written");
    let command = GpuCommand::Tune {
        model: Some(path.display().to_string()),
        models: Vec::new(),
        json: false,
        launch_args: false,
        apply: false,
        replace_existing: false,
    };

    let result = dispatch_gpu_command(None, false, Some(&command));

    assert!(
        result
            .expect_err("fake GGUF fixture should fail after runner planning")
            .to_string()
            .contains("could not read compact GGUF metadata"),
        "expected tune dispatch to reach the runner and fail in planning"
    );
}

#[test]
fn gpu_tune_apply_fails_when_no_target_has_writable_edits() {
    let temp = tempdir().expect("tempdir should be created");
    let path = write_valid_tune_fixture(temp.path(), "insufficient-fit.gguf");
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("fixture GGUF should be reopenable")
        .set_len(80 * 1024 * 1024 * 1024)
        .expect("fixture GGUF should support sparse resize");
    let config_path = temp.path().join("config.toml");
    fs::write(&config_path, "version = 1\n").expect("fixture config should be written");
    let command = GpuCommand::Tune {
        model: Some(path.display().to_string()),
        models: Vec::new(),
        json: false,
        launch_args: false,
        apply: true,
        replace_existing: false,
    };

    let result = dispatch_gpu_command(Some(&config_path), false, Some(&command));

    let error = result.expect_err("apply should fail when no target produced writable edits");
    let message = error.to_string();
    assert!(
        message.contains("gpu tune could not produce any safe config edits"),
        "expected explicit apply failure, got: {message}"
    );
    assert!(
        message.contains("model `") && message.contains("no safe startup plan fits"),
        "expected per-target insufficient-fit reason, got: {message}"
    );
    assert_eq!(
        fs::read_to_string(&config_path).expect("config should still be readable"),
        "version = 1\n",
        "failed apply should leave config unchanged"
    );
}

#[test]
fn gpu_tune_launch_args_is_read_only() {
    let temp = tempdir().expect("tempdir should be created");
    let path = write_valid_tune_fixture(temp.path(), "launch-args.gguf");
    let config_path = temp.path().join("config.toml");
    fs::write(&config_path, "version = 1\n").expect("fixture config should be written");
    let command = GpuCommand::Tune {
        model: Some(path.display().to_string()),
        models: Vec::new(),
        json: false,
        launch_args: true,
        apply: false,
        replace_existing: false,
    };
    let before = fs::read_to_string(&config_path).expect("config should be readable before run");
    let mut output = Vec::new();

    run_tune_command_with_writer(Some(&config_path), false, &command, &mut output)
        .expect("launch args review should succeed");

    let after = fs::read_to_string(&config_path).expect("config should be readable after run");
    assert_eq!(before, after, "launch-args must not mutate config");
    let rendered = String::from_utf8(output).expect("launch args output should be utf8");
    assert!(rendered.contains("mesh-llm serve --model"));
    assert!(rendered.contains("# effective config settings:"));
}

#[test]
fn gpu_tune_json_reports_per_model_errors_without_silent_failures() {
    let temp = tempdir().expect("tempdir should be created");
    let valid = write_valid_tune_fixture(temp.path(), "valid.gguf");
    let missing = temp.path().join("missing.gguf");
    let command = GpuCommand::Tune {
        model: None,
        models: vec![valid.display().to_string(), missing.display().to_string()],
        json: true,
        launch_args: false,
        apply: false,
        replace_existing: false,
    };
    let mut output = Vec::new();

    run_tune_command_with_writer(None, false, &command, &mut output)
        .expect("mixed review should still emit json output");

    let value: Value = serde_json::from_slice(&output).expect("json output should deserialize");
    assert_eq!(value["summary"]["total_targets"], Value::from(2));
    assert_eq!(value["summary"]["ready_targets"], Value::from(1));
    assert_eq!(value["summary"]["failed_targets"], Value::from(1));
    assert_eq!(value["targets"][0]["status"], Value::from("ready"));
    assert_eq!(value["targets"][1]["status"], Value::from("failed"));
    assert!(
        value["targets"][1]["reason"]
            .as_str()
            .expect("reason should be present")
            .contains("installed cache ref")
    );
}

#[test]
fn benchmark_tune_json_uses_benchmark_command_context() {
    let temp = tempdir().expect("tempdir should be created");
    let missing = temp.path().join("missing.gguf");
    let command = BenchmarkCommand::Tune {
        model: Some(missing.display().to_string()),
        models: Vec::new(),
        json: true,
        ctx_sizes: vec![4096],
        batch_sizes: vec![1024],
        ubatch_sizes: vec![256],
        mmap_values: Vec::new(),
        mlock_values: Vec::new(),
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        prompt: "hello".to_string(),
    };
    let mut output = Vec::new();

    let result = run_benchmark_tune_command_with_writer(None, &command, &mut output);

    let error = result.expect_err("missing target should fail after emitting json");
    assert!(
        error
            .to_string()
            .contains("gpu tune could not prepare any local targets"),
        "expected tune preparation failure, got: {error:#}"
    );
    let value: Value = serde_json::from_slice(&output).expect("json output should deserialize");
    assert_eq!(value["command"], Value::from("benchmark_tune"));
    assert_eq!(value["summary"]["failed_targets"], Value::from(1));
    assert!(
        value["benchmarks"]
            .as_array()
            .is_none_or(std::vec::Vec::is_empty),
        "missing target should not launch benchmark trials"
    );
}

#[test]
fn gpu_tune_uses_configured_models_when_no_explicit_targets_and_leaves_config_unchanged() {
    let temp = tempdir().expect("tempdir should be created");
    let first = write_valid_tune_fixture(temp.path(), "configured-a.gguf")
        .canonicalize()
        .expect("first fixture should canonicalize");
    let second = write_valid_tune_fixture(temp.path(), "configured-b.gguf")
        .canonicalize()
        .expect("second fixture should canonicalize");
    let config_path = temp.path().join("config.toml");
    write_config_with_models(&config_path, &[first, second]);
    let before = fs::read_to_string(&config_path).expect("config should be readable before run");
    let command = GpuCommand::Tune {
        model: None,
        models: Vec::new(),
        json: true,
        launch_args: false,
        apply: false,
        replace_existing: false,
    };
    let mut output = Vec::new();

    run_tune_command_with_writer(Some(&config_path), false, &command, &mut output)
        .expect("configured review should succeed");

    let report: Value = serde_json::from_slice(&output).expect("json output should deserialize");
    assert_eq!(report["summary"]["total_targets"], Value::from(2));
    assert_eq!(report["summary"]["ready_targets"], Value::from(2));
    assert_eq!(report["summary"]["failed_targets"], Value::from(0));
    assert_eq!(report["targets"][0]["selection"], Value::from("configured"));
    assert_eq!(report["targets"][1]["selection"], Value::from("configured"));
    assert_eq!(
        fs::read_to_string(&config_path).expect("config should be readable after run"),
        before,
        "review mode must not mutate config"
    );
}

#[test]
fn gpu_tune_explicit_model_limits_run_to_requested_target() {
    let temp = tempdir().expect("tempdir should be created");
    let first = write_valid_tune_fixture(temp.path(), "configured-a.gguf")
        .canonicalize()
        .expect("first fixture should canonicalize");
    let second = write_valid_tune_fixture(temp.path(), "configured-b.gguf")
        .canonicalize()
        .expect("second fixture should canonicalize");
    let config_path = temp.path().join("config.toml");
    write_config_with_models(&config_path, &[first, second.clone()]);
    let command = GpuCommand::Tune {
        model: Some(second.display().to_string()),
        models: Vec::new(),
        json: true,
        launch_args: false,
        apply: false,
        replace_existing: false,
    };
    let mut output = Vec::new();

    run_tune_command_with_writer(Some(&config_path), false, &command, &mut output)
        .expect("explicit review should succeed");

    let report: Value = serde_json::from_slice(&output).expect("json output should deserialize");
    assert_eq!(report["summary"]["total_targets"], Value::from(1));
    assert_eq!(
        report["targets"][0]["target"]["requested"],
        Value::from(second.display().to_string())
    );
    assert_eq!(
        report["targets"][0]["selection"],
        Value::from("explicit_configured")
    );
}

#[test]
fn gpu_tune_apply_reports_mixed_success_and_failure_and_writes_ready_target() {
    let temp = tempdir().expect("tempdir should be created");
    let valid = write_valid_tune_fixture(temp.path(), "apply-success.gguf")
        .canonicalize()
        .expect("fixture should canonicalize");
    let missing = temp.path().join("missing.gguf");
    let config_path = temp.path().join("config.toml");
    fs::write(&config_path, "version = 1\n").expect("fixture config should be written");
    let command = GpuCommand::Tune {
        model: None,
        models: vec![valid.display().to_string(), missing.display().to_string()],
        json: true,
        launch_args: false,
        apply: true,
        replace_existing: false,
    };
    let mut output = Vec::new();

    run_tune_command_with_writer(Some(&config_path), false, &command, &mut output)
        .expect("apply should succeed when at least one target writes");

    let report: Value = serde_json::from_slice(&output).expect("json output should deserialize");
    assert_eq!(report["summary"]["total_targets"], Value::from(2));
    assert_eq!(report["summary"]["written_targets"], Value::from(1));
    assert_eq!(report["summary"]["failed_targets"], Value::from(1));
    assert_eq!(report["targets"][0]["status"], Value::from("written"));
    assert_eq!(report["targets"][1]["status"], Value::from("failed"));
    let edited = fs::read_to_string(&config_path).expect("config should be readable after apply");
    assert!(edited.contains(&format!("model = \"{}\"", valid.display())));
    assert!(edited.contains("cache_type_k = \"q8_0\""));
}
