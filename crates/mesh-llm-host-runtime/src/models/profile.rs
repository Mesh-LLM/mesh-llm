use std::path::Path;
use std::sync::LazyLock;

use regex_lite::Regex;

pub(crate) fn served_model_metadata_for_model(
    model_name: &str,
) -> Option<crate::mesh::ServedModelMetadata> {
    let path = crate::models::find_model_path(model_name);
    served_model_metadata_for_path(model_name, &path)
}

pub(crate) fn served_model_metadata_for_path(
    model_name: &str,
    path: &Path,
) -> Option<crate::mesh::ServedModelMetadata> {
    let compact = path
        .exists()
        .then(|| crate::models::gguf::scan_gguf_compact_meta(path))
        .flatten();
    let metadata = match compact {
        Some(meta) => {
            // Sum across the whole shard set: `find_model_path` resolves a
            // split GGUF to its first part, and each shard's tensor-info table
            // holds only that shard's weights. Scanning one part reported
            // roughly `total / shard_count` — an ~80B 4-shard model advertised
            // 24.6B — which silently mis-ranked every size-based decision.
            let parameter_count = path
                .exists()
                .then(|| crate::models::gguf::scan_gguf_bundle_total_parameters(path))
                .flatten();
            let parameter_size =
                resolve_parameter_size(model_name, meta.parameter_size.clone(), parameter_count);
            let parameter_count_b = parameter_count.map(|total| total as f64 / 1e9);
            let kv_head_count = meta.effective_kv_head_count();
            crate::mesh::ServedModelMetadata {
                architecture: non_empty(meta.architecture),
                parameter_size,
                parameter_count_b,
                quant: path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .and_then(quant_from_text)
                    .or_else(|| quant_from_text(model_name)),
                native_context_length: non_zero(meta.context_length),
                tokenizer: non_empty(meta.tokenizer_model_name),
                layer_count: non_zero(meta.layer_count),
                embedding_size: non_zero(meta.embedding_size),
                head_count: non_zero(meta.head_count),
                kv_head_count,
                expert_count: non_zero(meta.expert_count),
                active_expert_count: non_zero(meta.expert_used_count),
            }
        }
        None => crate::mesh::ServedModelMetadata {
            parameter_size: resolve_parameter_size(model_name, None, None),
            // No GGUF to sum -> no authoritative size. Advertise none rather
            // than a name-guessed count (per i386 review); MoA treats a
            // sizeless model as the weakest.
            parameter_count_b: None,
            quant: quant_from_text(model_name),
            ..Default::default()
        },
    };
    (!metadata.is_empty()).then_some(metadata)
}

fn non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn non_zero(value: u32) -> Option<u32> {
    (value > 0).then_some(value)
}

fn quant_from_text(value: &str) -> Option<String> {
    let quant = crate::models::inventory::derive_quantization_type(value)
        .trim()
        .trim_end_matches(".gguf")
        .to_string();
    (!quant.is_empty()).then_some(quant)
}

/// Resolve a display label consistently: source metadata, verified tensor
/// count, then a guarded model-name fallback.
fn resolve_parameter_size(
    model_name: &str,
    source_size: Option<String>,
    parameter_count: Option<u64>,
) -> Option<String> {
    source_size
        .or_else(|| parameter_count.and_then(parameter_size_from_count))
        .or_else(|| parameter_size_from_text(model_name))
}

fn parameter_size_from_count(parameter_count: u64) -> Option<String> {
    if parameter_count == 0 {
        return None;
    }

    if parameter_count < 1_000_000_000 {
        let millions = parameter_count as f64 / 1e6;
        return Some(if millions >= 10.0 {
            format!("{millions:.0}M")
        } else {
            format!("{millions:.1}M")
        });
    }

    let billions = parameter_count as f64 / 1e9;
    let label = format!("{billions:.1}");
    Some(format!("{}B", label.strip_suffix(".0").unwrap_or(&label)))
}

fn parameter_size_from_text(text: &str) -> Option<String> {
    if text.starts_with("local-gguf/sha256-") {
        return None;
    }

    static MULTIPLIED_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)(?:^|[^a-z0-9.])(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)([bm])(?:$|[^a-z0-9.])")
            .unwrap()
    });
    static SIMPLE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)(?:^|[^a-z0-9.])(\d+(?:\.\d+)?)([bm])(?:$|[^a-z0-9.])").unwrap()
    });

    MULTIPLIED_RE
        .captures(text)
        .map(|captures| {
            format!(
                "{}x{}{}",
                &captures[1],
                &captures[2],
                captures[3].to_ascii_uppercase()
            )
        })
        .or_else(|| {
            SIMPLE_RE
                .captures(text)
                .map(|captures| format!("{}{}", &captures[1], captures[2].to_ascii_uppercase()))
        })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        parameter_size_from_count, parameter_size_from_text, resolve_parameter_size,
        served_model_metadata_for_path,
    };

    #[test]
    fn extracts_parameter_size_labels() {
        assert_eq!(
            parameter_size_from_text("Qwen3-32B-Q4_K_M").as_deref(),
            Some("32B")
        );
        assert_eq!(
            parameter_size_from_text("mixtral-8x7b").as_deref(),
            Some("8x7B")
        );
    }

    #[test]
    fn rejects_parameter_size_labels_embedded_in_synthetic_names() {
        assert_eq!(
            parameter_size_from_text("local-gguf/sha256-74a4da8c9fdbcd15bd1f6e06b796387d397b038d"),
            None
        );
        assert_eq!(parameter_size_from_text("model-dead8061beef"), None);
        assert_eq!(parameter_size_from_text("model-7bfoo"), None);
        assert_eq!(parameter_size_from_text("model-8x7bfoo"), None);
    }

    #[test]
    fn resolves_parameter_size_with_shared_source_precedence() {
        assert_eq!(
            resolve_parameter_size("model-7B", Some("6B".to_string()), Some(8_000_000_000))
                .as_deref(),
            Some("6B")
        );
        assert_eq!(
            resolve_parameter_size("model-7B", None, Some(8_000_000_000)).as_deref(),
            Some("8B")
        );
        assert_eq!(
            resolve_parameter_size("model-7B", None, None).as_deref(),
            Some("7B")
        );
    }

    #[test]
    fn formats_parameter_size_from_tensor_count() {
        assert_eq!(parameter_size_from_count(0), None);
        assert_eq!(
            parameter_size_from_count(494_000_000).as_deref(),
            Some("494M")
        );
        assert_eq!(
            parameter_size_from_count(1_235_000_000).as_deref(),
            Some("1.2B")
        );
        assert_eq!(
            parameter_size_from_count(32_000_000_000).as_deref(),
            Some("32B")
        );
    }

    #[test]
    fn derives_synthetic_model_parameter_size_from_gguf_tensors() {
        let path = std::env::temp_dir().join(format!(
            "mesh-llm-profile-parameter-size-{}.gguf",
            std::process::id()
        ));
        write_gguf_with_parameters(&path, 494_000_000);

        let metadata = served_model_metadata_for_path(
            "local-gguf/sha256-74a4da8c9fdbcd15bd1f6e06b796387d397b038d",
            &path,
        )
        .expect("synthetic GGUF should expose metadata");

        assert_eq!(metadata.parameter_size.as_deref(), Some("494M"));
        assert_eq!(metadata.parameter_count_b, Some(0.494));
        let _ = std::fs::remove_file(path);
    }

    fn write_gguf_with_parameters(path: &Path, parameters: u64) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(b"weights");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&parameters.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(path, bytes).expect("write synthetic GGUF");
    }
}
