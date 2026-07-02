use super::{
    InvalidKvType, TuneGgufMetadataError, TuneTensorProfile, inspect_local_gguf_metadata,
    inspect_tune_target_metadata, validate_kv_cache_quant,
};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;

fn temp_file_path(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{unique}.gguf"))
}

fn write_bytes(prefix: &str, bytes: &[u8]) -> PathBuf {
    let path = temp_file_path(prefix);
    let mut file = fs::File::create(&path).expect("test fixture should create file");
    file.write_all(bytes)
        .expect("test fixture should write file");
    file.flush().expect("test fixture should flush file");
    path
}

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

fn push_tensor_info(bytes: &mut Vec<u8>, name: &str, offset: u64) {
    push_gguf_string(bytes, name);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&16u64.to_le_bytes());
    bytes.extend_from_slice(&GGUF_TYPE_UINT8.to_le_bytes());
    bytes.extend_from_slice(&offset.to_le_bytes());
}

fn align_offset(value: usize, alignment: usize) -> usize {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

fn write_valid_tune_fixture(include_tensors: bool) -> PathBuf {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&(if include_tensors { 2_i64 } else { 0_i64 }).to_le_bytes());
    bytes.extend_from_slice(&8i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_u32_kv(&mut bytes, "llama.context_length", 8192);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    push_u32_kv(&mut bytes, "llama.attention.value_length", 128);
    if include_tensors {
        push_tensor_info(&mut bytes, "blk.0.ffn_up_exps.weight", 0);
        push_tensor_info(&mut bytes, "blk.0.attn_q.weight", 64);
        let data_start = align_offset(bytes.len(), 32);
        bytes.resize(data_start + 96, 0);
    }
    write_bytes("mesh-llm-gpu-tune-valid", &bytes)
}

#[test]
fn gpu_tune_reads_compact_meta() {
    let path = write_valid_tune_fixture(true);

    let metadata = inspect_local_gguf_metadata("sample-model", &path)
        .expect("valid tune GGUF fixture should parse");

    assert_eq!(metadata.compact_meta.architecture, "llama");
    assert_eq!(metadata.compact_meta.context_length, 8192);
    assert_eq!(metadata.compact_meta.layer_count, 24);
    assert_eq!(metadata.compact_meta.effective_kv_head_count(), Some(8));
    assert_eq!(metadata.compact_meta.key_length, 128);
    assert_eq!(metadata.compact_meta.value_length, 128);
    match metadata.tensor_profile {
        TuneTensorProfile::Exact(profile) => {
            assert_eq!(profile.expert_tensor_bytes, 64);
            assert_eq!(profile.base_resident_bytes, 32);
        }
        TuneTensorProfile::DegradedFallback { .. } => {
            panic!("expected exact tensor profile")
        }
    }

    let _ = fs::remove_file(path);
}

#[test]
fn gpu_tune_reads_layer_package_metadata_from_shared_metadata_artifact() {
    let metadata_fixture = write_valid_tune_fixture(true);
    let package_dir = tempfile::tempdir().expect("package tempdir should be created");
    let package_metadata_path = package_dir.path().join("metadata.gguf");
    fs::copy(&metadata_fixture, &package_metadata_path)
        .expect("metadata fixture should be copied into package");
    fs::write(
        package_dir.path().join("model-package.json"),
        serde_json::json!({
            "source_model": {
                "files": [
                    {"path": "model-00001-of-00002.gguf", "size_bytes": 111},
                    {"path": "model-00002-of-00002.gguf", "size_bytes": 222}
                ]
            },
            "shared": {
                "metadata": {"path": "metadata.gguf", "artifact_bytes": 11},
                "embeddings": {"path": "embeddings.gguf", "artifact_bytes": 22},
                "output": {"path": "output.gguf", "artifact_bytes": 33}
            },
            "layers": [
                {"path": "layers/0.gguf", "artifact_bytes": 44}
            ]
        })
        .to_string(),
    )
    .expect("manifest should be written");

    let metadata = inspect_tune_target_metadata("package-model", package_dir.path())
        .expect("layer package metadata should parse through shared metadata GGUF");

    assert_eq!(metadata.compact_meta.architecture, "llama");
    assert_eq!(metadata.compact_meta.layer_count, 24);
    assert_eq!(metadata.model_bytes, 333);

    let _ = fs::remove_file(metadata_fixture);
}

#[test]
fn gpu_tune_reports_missing_required_metadata() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&1i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    let path = write_bytes("mesh-llm-gpu-tune-missing-meta", &bytes);

    let error = inspect_local_gguf_metadata("broken-model", &path)
        .expect_err("missing required metadata should fail");

    assert_eq!(
        error,
        TuneGgufMetadataError::MissingRequiredMetadata {
            model: "broken-model".to_string(),
            missing_fields: vec![
                "context_length",
                "layer_count",
                "kv_head_count",
                "key_length",
                "value_length",
            ],
        }
    );
    assert!(error.to_string().contains("model `broken-model`"));

    let _ = fs::remove_file(path);
}

#[test]
fn gpu_tune_degrades_safely_when_tensor_profile_is_unavailable() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&1i64.to_le_bytes());
    bytes.extend_from_slice(&8i64.to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_u32_kv(&mut bytes, "llama.context_length", 8192);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    push_u32_kv(&mut bytes, "llama.attention.value_length", 128);
    let path = write_bytes("mesh-llm-gpu-tune-broken-tensors", &bytes);

    let metadata = inspect_local_gguf_metadata("dense-model", &path)
        .expect("compact metadata should stay usable when tensor profile parsing fails");

    match metadata.tensor_profile {
        TuneTensorProfile::Exact(_) => panic!("expected degraded fallback"),
        TuneTensorProfile::DegradedFallback { model_bytes } => {
            assert_eq!(model_bytes, metadata.model_bytes);
            assert!(model_bytes > 0);
        }
    }

    let _ = fs::remove_file(path);
}

#[test]
fn gpu_tune_accepts_supported_kv_types() {
    let quant = validate_kv_cache_quant("sample-model", "q8_0", "q4_0")
        .expect("supported kv strings should parse");

    assert_eq!(
        quant,
        model_artifact::gguf::GgufKvCacheQuant::new(
            model_artifact::gguf::GgufKvCacheType::Q8_0,
            model_artifact::gguf::GgufKvCacheType::Q4_0,
        )
    );
}

#[test]
fn gpu_tune_rejects_unsupported_kv_types_with_model_name() {
    let error = validate_kv_cache_quant("bad-model", "q6_k", "q4_0")
        .expect_err("unsupported kv strings should fail");

    assert_eq!(
        error,
        TuneGgufMetadataError::UnsupportedKvTypes {
            model: "bad-model".to_string(),
            invalid_fields: vec![InvalidKvType {
                field_name: "cache_type_k",
                value: "q6_k".to_string(),
            }],
        }
    );
    assert!(error.to_string().contains("model `bad-model`"));
    assert!(error.to_string().contains("cache_type_k=`q6_k`"));
}
