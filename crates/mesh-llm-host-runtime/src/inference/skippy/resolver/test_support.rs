use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

use crate::inference::skippy::SkippyPackageIdentity;
use crate::plugin::MeshConfig;

pub(super) fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

pub(super) fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_gguf_string(bytes, value);
}

pub(super) fn temp_model_file() -> NamedTempFile {
    temp_model_file_with_tensor_names(&[], None)
}

pub(super) fn temp_model_file_with_tensor_names(
    tensor_names: &[&str],
    nextn_predict_layers: Option<u32>,
) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("temp model file");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&(tensor_names.len() as i64).to_le_bytes());
    bytes.extend_from_slice(&(8 + i64::from(nextn_predict_layers.is_some())).to_le_bytes());
    push_string_kv(&mut bytes, "general.architecture", "llama");
    push_string_kv(&mut bytes, "tokenizer.ggml.model", "gpt2");
    push_u32_kv(&mut bytes, "llama.context_length", 8192);
    push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
    push_u32_kv(&mut bytes, "llama.block_count", 24);
    push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
    push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
    push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
    if let Some(value) = nextn_predict_layers {
        push_u32_kv(&mut bytes, "llama.nextn_predict_layers", value);
    }
    for name in tensor_names {
        push_gguf_string(&mut bytes, name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
    }
    file.write_all(&bytes).expect("write fake gguf");
    file.flush().expect("flush fake gguf");
    file
}

pub(super) fn parse_config(toml: &str) -> MeshConfig {
    toml::from_str(toml).expect("config should parse")
}

pub(super) fn fake_package_identity(layer_count: u32) -> SkippyPackageIdentity {
    SkippyPackageIdentity {
        package_ref: "gguf:///models/qwen.gguf".to_string(),
        manifest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            .to_string(),
        source_model_path: PathBuf::from("/models/qwen.gguf"),
        source_model_sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
            .to_string(),
        source_model_bytes: 1234,
        source_files: Vec::new(),
        layer_count,
        activation_width: 4096,
        tensor_count: 100,
        generation: None,
    }
}

pub(super) fn fake_hf_package_identity(layer_count: u32) -> SkippyPackageIdentity {
    let mut package = fake_package_identity(layer_count);
    package.package_ref = "hf://meshllm/Qwen3-8B-Q4_K_M-layers".to_string();
    package
}
