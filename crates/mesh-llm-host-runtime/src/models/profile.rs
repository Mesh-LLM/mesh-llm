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
            let parameter_size = meta
                .parameter_size
                .clone()
                .or_else(|| parameter_size_from_text(model_name));
            // Authoritative size: sum the GGUF tensor element counts. This is
            // the ONLY source — no name-based fallback. If a served model
            // cannot be summed from its GGUF, it advertises no size and MoA
            // tiering treats it as the lowest-param (weakest) model rather than
            // guessing from a brittle name label (per i386 review).
            //
            // Summed across the whole shard set: `find_model_path` resolves a
            // split GGUF to its first part, and each shard's tensor-info table
            // holds only that shard's weights. Scanning one part reported
            // roughly `total / shard_count` — an ~80B 4-shard model advertised
            // 24.6B — which silently mis-ranked every size-based decision.
            let parameter_count_b = path
                .exists()
                .then(|| crate::models::gguf::scan_gguf_bundle_total_parameters(path))
                .flatten()
                .map(|total| total as f64 / 1e9);
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
            parameter_size: parameter_size_from_text(model_name),
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

fn parameter_size_from_text(text: &str) -> Option<String> {
    static MULTIPLIED_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)([bm])").unwrap());
    static SIMPLE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)([bm])").unwrap());

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
    use super::parameter_size_from_text;

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
}
