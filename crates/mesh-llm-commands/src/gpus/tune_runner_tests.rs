use super::*;
use mesh_llm_cli::benchmark::{BenchmarkCommand, BenchmarkTuneCommand};
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

#[test]
fn benchmark_tune_json_uses_benchmark_command_context() {
    let temp = tempdir().expect("tempdir should be created");
    let missing = temp.path().join("missing.gguf");
    let command = BenchmarkCommand::Tune(Box::new(BenchmarkTuneCommand {
        model: Some(missing.display().to_string()),
        models: Vec::new(),
        json: true,
        ctx_sizes: vec![4096],
        batch_sizes: vec![1024],
        ubatch_sizes: vec![256],
        apply: false,
        replace_existing: false,
        launch_args: false,
        mmap_values: Vec::new(),
        mlock_values: Vec::new(),
        flash_attention: Vec::new(),
        speculative_types: Vec::new(),
        no_speculative_tune: false,
        spec_draft_models: Vec::new(),
        spec_draft_max_tokens: Vec::new(),
        spec_draft_min_tokens: Vec::new(),
        spec_ngram_min: Vec::new(),
        spec_ngram_max: Vec::new(),
        spec_draft_acceptance_threshold: Vec::new(),
        spec_draft_split_probability: Vec::new(),
        throughput_tolerance_pct: 3.0,
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        debug_telemetry: false,
        prompt: "hello".to_string(),
    }));
    let mut output = Vec::new();

    let result = run_benchmark_tune_command_with_writer(None, &command, &mut output);

    let error = result.expect_err("missing target should fail after emitting json");
    assert!(
        error
            .to_string()
            .contains("benchmark tune could not prepare any local targets"),
        "expected benchmark preparation failure, got: {error:#}"
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
fn benchmark_tune_rejects_zero_only_candidate_values_before_running_trials() {
    let temp = tempdir().expect("tempdir should be created");
    let model = write_valid_tune_fixture(temp.path(), "zero-candidates.gguf");
    let command = BenchmarkCommand::Tune(Box::new(BenchmarkTuneCommand {
        model: Some(model.display().to_string()),
        models: Vec::new(),
        json: true,
        ctx_sizes: vec![0],
        batch_sizes: vec![1024],
        ubatch_sizes: vec![256],
        apply: false,
        replace_existing: false,
        launch_args: false,
        mmap_values: Vec::new(),
        mlock_values: Vec::new(),
        flash_attention: Vec::new(),
        speculative_types: Vec::new(),
        no_speculative_tune: false,
        spec_draft_models: Vec::new(),
        spec_draft_max_tokens: Vec::new(),
        spec_draft_min_tokens: Vec::new(),
        spec_ngram_min: Vec::new(),
        spec_ngram_max: Vec::new(),
        spec_draft_acceptance_threshold: Vec::new(),
        spec_draft_split_probability: Vec::new(),
        throughput_tolerance_pct: 10.0,
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        debug_telemetry: false,
        prompt: "hello".to_string(),
    }));
    let mut output = Vec::new();

    let result = run_benchmark_tune_command_with_writer(None, &command, &mut output);

    let error = result.expect_err("zero-only ctx sizes should be rejected");
    assert!(
        error
            .to_string()
            .contains("--ctx-sizes must include at least one positive value"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn benchmark_tune_allows_zero_speculative_draft_min_tokens() {
    let args = BenchmarkTuneArgs {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[],
        mlock_values: &[],
        flash_attention_values: &[],
        speculative_types: &[],
        no_speculative_tune: false,
        spec_draft_models: &[],
        spec_draft_max_tokens: &[3],
        spec_draft_min_tokens: &[0],
        spec_ngram_min: &[],
        spec_ngram_max: &[],
        spec_draft_acceptance_threshold: &[],
        spec_draft_split_probability: &[],
        throughput_tolerance_pct: 10.0,
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        debug_telemetry: false,
        prompt: "hello",
    };

    validate_benchmark_args(Some(&args)).expect("MTP min draft tokens may be zero");
}

#[test]
fn benchmark_tune_rejects_candidate_matrix_without_valid_batch_ubatch_pair() {
    let temp = tempdir().expect("tempdir should be created");
    let model = write_valid_tune_fixture(temp.path(), "invalid-batch-pair.gguf");
    let command = BenchmarkCommand::Tune(Box::new(BenchmarkTuneCommand {
        model: Some(model.display().to_string()),
        models: Vec::new(),
        json: true,
        ctx_sizes: vec![4096],
        batch_sizes: vec![512],
        ubatch_sizes: vec![1024],
        apply: false,
        replace_existing: false,
        launch_args: false,
        mmap_values: Vec::new(),
        mlock_values: Vec::new(),
        flash_attention: Vec::new(),
        speculative_types: Vec::new(),
        no_speculative_tune: false,
        spec_draft_models: Vec::new(),
        spec_draft_max_tokens: Vec::new(),
        spec_draft_min_tokens: Vec::new(),
        spec_ngram_min: Vec::new(),
        spec_ngram_max: Vec::new(),
        spec_draft_acceptance_threshold: Vec::new(),
        spec_draft_split_probability: Vec::new(),
        throughput_tolerance_pct: 10.0,
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        debug_telemetry: false,
        prompt: "hello".to_string(),
    }));
    let mut output = Vec::new();

    let result = run_benchmark_tune_command_with_writer(None, &command, &mut output);

    let error = result.expect_err("ubatch larger than every batch should be rejected");
    assert!(
        error
            .to_string()
            .contains("benchmark candidate matrix has no valid batch/ubatch pairs"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn benchmark_tune_rejects_out_of_range_probability_values() {
    let args = BenchmarkTuneArgs {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[],
        mlock_values: &[],
        flash_attention_values: &[],
        speculative_types: &[],
        no_speculative_tune: false,
        spec_draft_models: &[],
        spec_draft_max_tokens: &[],
        spec_draft_min_tokens: &[],
        spec_draft_acceptance_threshold: &[1.5],
        spec_draft_split_probability: &[],
        spec_ngram_min: &[],
        spec_ngram_max: &[],
        throughput_tolerance_pct: 10.0,
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        debug_telemetry: false,
        prompt: "hello",
    };

    let error = validate_benchmark_args(Some(&args))
        .expect_err("acceptance threshold > 1.0 should be rejected");
    assert!(
        error
            .to_string()
            .contains("values must be finite probabilities in [0.0, 1.0]"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn benchmark_tune_rejects_negative_probability_values() {
    let args = BenchmarkTuneArgs {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[],
        mlock_values: &[],
        flash_attention_values: &[],
        speculative_types: &[],
        no_speculative_tune: false,
        spec_draft_models: &[],
        spec_draft_max_tokens: &[],
        spec_draft_min_tokens: &[],
        spec_draft_acceptance_threshold: &[],
        spec_draft_split_probability: &[-0.1],
        spec_ngram_min: &[],
        spec_ngram_max: &[],
        throughput_tolerance_pct: 10.0,
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        debug_telemetry: false,
        prompt: "hello",
    };

    let error = validate_benchmark_args(Some(&args))
        .expect_err("negative split probability should be rejected");
    assert!(
        error
            .to_string()
            .contains("values must be finite probabilities in [0.0, 1.0]"),
        "unexpected error: {error:#}"
    );
}
