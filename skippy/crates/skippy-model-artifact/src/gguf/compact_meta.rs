use super::{
    GgufCompactMeta, GgufHeader, GgufReader, GgufType, open_gguf_header, read_gguf_string,
    read_gguf_value_as_bool_list, read_gguf_value_as_f32, read_gguf_value_as_string_opt,
    read_gguf_value_as_u32, read_gguf_value_as_u32_list, read_u32, skip_gguf_value,
};
use std::io::{Seek, SeekFrom};
use std::path::Path;

/// Scan structural metadata from the declared architecture's KV namespace.
/// Reads only the header, never tensor data; malformed or missing architecture
/// metadata returns `None`. KV ordering cannot change namespace selection.
pub fn scan_gguf_compact_meta(path: &Path) -> Option<GgufCompactMeta> {
    let GgufHeader {
        file: mut reader,
        n_kv,
        ..
    } = open_gguf_header(path)?;
    let kv_start = reader.stream_position().ok()?;
    let architecture = read_architecture(&mut reader, n_kv)?;
    reader.seek(SeekFrom::Start(kv_start)).ok()?;
    let mut meta = GgufCompactMeta {
        architecture,
        ..GgufCompactMeta::default()
    };
    for _ in 0..n_kv {
        let key = read_gguf_string(&mut reader).ok()?;
        let value_type = GgufType::from_u32(read_u32(&mut reader).ok()?)?;
        match key.as_str() {
            "general.size_label" => {
                meta.parameter_size =
                    read_gguf_value_as_string_opt(&mut reader, value_type).ok()?;
            }
            "tokenizer.ggml.model" => {
                meta.tokenizer_model_name =
                    read_gguf_value_as_string_opt(&mut reader, value_type).ok()??;
            }
            _ => match key.split_once('.') {
                Some((namespace, field)) if namespace == meta.architecture => {
                    read_structural_field(&mut reader, value_type, field, &mut meta)?;
                }
                _ => skip_gguf_value(&mut reader, value_type).ok()?,
            },
        }
    }
    derive_attention_widths(&mut meta);
    Some(meta)
}

fn read_architecture(reader: &mut GgufReader, n_kv: usize) -> Option<String> {
    for _ in 0..n_kv {
        let key = read_gguf_string(reader).ok()?;
        let value_type = GgufType::from_u32(read_u32(reader).ok()?)?;
        if key == "general.architecture" {
            return read_gguf_value_as_string_opt(reader, value_type)
                .ok()?
                .filter(|architecture| !architecture.is_empty());
        }
        skip_gguf_value(reader, value_type).ok()?;
    }
    None
}

fn read_structural_field(
    reader: &mut GgufReader,
    value_type: GgufType,
    field: &str,
    meta: &mut GgufCompactMeta,
) -> Option<()> {
    let scalar = match field {
        "context_length" => Some(&mut meta.context_length),
        "max_len" => Some(&mut meta.max_len),
        "embedding_length" => Some(&mut meta.embedding_size),
        "attention.head_count" => Some(&mut meta.head_count),
        "block_count" => Some(&mut meta.layer_count),
        "feed_forward_length" => Some(&mut meta.feed_forward_length),
        "attention.key_length" => Some(&mut meta.key_length),
        "attention.key_length_swa" => Some(&mut meta.key_length_swa),
        "attention.value_length" => Some(&mut meta.value_length),
        "attention.value_length_swa" => Some(&mut meta.value_length_swa),
        "attention.kv_lora_rank" => Some(&mut meta.kv_lora_rank),
        "vocab_size" => Some(&mut meta.vocab_size),
        "expert_count" => Some(&mut meta.expert_count),
        "expert_used_count" => Some(&mut meta.expert_used_count),
        "nextn_predict_layers" => Some(&mut meta.nextn_predict_layers),
        "ssm.conv_kernel" => Some(&mut meta.ssm_conv_kernel),
        "ssm.inner_size" => Some(&mut meta.ssm_inner_size),
        "ssm.state_size" => Some(&mut meta.ssm_state_size),
        "ssm.group_count" => Some(&mut meta.ssm_group_count),
        "full_attention_interval" => Some(&mut meta.full_attention_interval),
        _ => None,
    };
    if let Some(destination) = scalar {
        if let Some(value) = read_gguf_value_as_u32(reader, value_type).ok()? {
            *destination = value;
        }
        return Some(());
    }
    match field {
        "attention.head_count_kv" => {
            if let Some(values) = read_gguf_value_as_u32_list(reader, value_type).ok()? {
                if values.len() == 1 {
                    meta.kv_head_count = values[0];
                } else {
                    meta.kv_head_counts = values;
                }
            }
        }
        "attention.recurrent_layers" => {
            if let Some(values) = read_gguf_value_as_bool_list(reader, value_type).ok()? {
                meta.recurrent_layers = values;
            }
        }
        "rope.scale" | "rope.freq_base" => {
            if let Some(value) = read_gguf_value_as_f32(reader, value_type).ok()? {
                if field == "rope.scale" {
                    meta.rope_scale = value;
                } else {
                    meta.rope_freq_base = value;
                }
            }
        }
        _ => skip_gguf_value(reader, value_type).ok()?,
    }
    Some(())
}

fn derive_attention_widths(meta: &mut GgufCompactMeta) {
    if meta.key_length == 0 && meta.head_count > 0 {
        meta.key_length = meta.embedding_size / meta.head_count;
    }
    if meta.value_length == 0 {
        meta.value_length = meta
            .effective_kv_head_count()
            .and_then(|heads| meta.embedding_size.checked_div(heads))
            .unwrap_or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique fixture path: the nanosecond clock can be coarse on virtualized
    /// runners, and parallel tests in this binary would otherwise collide on
    /// the same temp file name and delete each other's fixtures mid-scan.
    fn temp_fixture_path() -> std::path::PathBuf {
        static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("gguf-namespace-{nanos}-{sequence}.gguf"))
    }

    fn string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    }

    fn architecture(bytes: &mut Vec<u8>) {
        string(bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::String as u32).to_le_bytes());
        string(bytes, "future");
    }

    fn metadata(
        fields: &[(&str, u32)],
        architecture_first: Option<bool>,
    ) -> Option<GgufCompactMeta> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let count = fields.len() + usize::from(architecture_first.is_some());
        bytes.extend_from_slice(&(count as u64).to_le_bytes());
        if architecture_first == Some(true) {
            architecture(&mut bytes);
        }
        for (key, value) in fields {
            string(&mut bytes, key);
            bytes.extend_from_slice(&(GgufType::Uint32 as u32).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if architecture_first == Some(false) {
            architecture(&mut bytes);
        }
        let path = temp_fixture_path();
        std::fs::write(&path, bytes).unwrap();
        let result = scan_gguf_compact_meta(&path);
        std::fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn structural_fields_belong_to_declared_architecture_in_any_kv_order() {
        let mut fields = [
            ("future.block_count", 3),
            ("future.embedding_length", 128),
            ("future.context_length", 512),
            ("future.attention.head_count", 4),
            ("future.feed_forward_length", 256),
            ("future.attention.head_count_kv", 2),
            ("future.ssm.state_size", 16),
            ("future.posnet.block_count", 0),
            ("future.convnext.embedding_length", 0),
            ("future.aux.context_length", 0),
            ("future.attention.indexer.head_count", 8),
            ("future.aux.feed_forward_length", 0),
            ("clip.block_count", 99),
            ("clip.embedding_length", 99),
            ("clip.context_length", 99),
            ("clip.attention.head_count", 99),
            ("clip.feed_forward_length", 99),
            ("clip.attention.head_count_kv", 99),
            ("clip.ssm.state_size", 99),
            ("future.aux.attention.head_count_kv", 99),
        ];
        for reverse in [false, true] {
            if reverse {
                fields.reverse();
            }
            for architecture_first in [true, false] {
                let meta = metadata(&fields, Some(architecture_first)).unwrap();
                assert_eq!(meta.executable_layer_count(), Some(3));
                assert_eq!(meta.embedding_size, 128);
                assert_eq!(meta.context_length, 512);
                assert_eq!(meta.head_count, 4);
                assert_eq!(meta.feed_forward_length, 256);
                assert_eq!(meta.kv_head_count, 2);
                assert_eq!(meta.ssm_state_size, 16);
                assert_eq!(meta.key_length, 32);
                assert_eq!(meta.value_length, 64);
            }
        }
    }

    #[test]
    fn undeclared_architecture_does_not_admit_structural_metadata() {
        assert!(metadata(&[("future.block_count", 3)], None).is_none());
    }
}
