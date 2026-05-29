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
            let parameter_count_b = parameter_count_b_from_text(&format!(
                "{} {}",
                model_name,
                meta.parameter_size.as_deref().unwrap_or("")
            ));
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
            parameter_count_b: parameter_count_b_from_text(model_name),
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

fn parameter_count_b_from_text(text: &str) -> Option<f64> {
    static MULTIPLIED_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)([bm])").unwrap());
    static SIMPLE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)([bm])").unwrap());

    let mut best: Option<f64> = None;
    for captures in MULTIPLIED_RE.captures_iter(text) {
        let Some(left) = captures.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) else {
            continue;
        };
        let Some(right) = captures.get(2).and_then(|m| m.as_str().parse::<f64>().ok()) else {
            continue;
        };
        let Some(unit) = captures.get(3).map(|m| m.as_str().to_ascii_lowercase()) else {
            continue;
        };
        let value = match unit.as_str() {
            "b" => left * right,
            "m" => (left * right) / 1000.0,
            _ => continue,
        };
        best = Some(best.map_or(value, |current| current.max(value)));
    }
    for captures in SIMPLE_RE.captures_iter(text) {
        let Some(count) = captures.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) else {
            continue;
        };
        let Some(unit) = captures.get(2).map(|m| m.as_str().to_ascii_lowercase()) else {
            continue;
        };
        let value = match unit.as_str() {
            "b" => count,
            "m" => count / 1000.0,
            _ => continue,
        };
        best = Some(best.map_or(value, |current| current.max(value)));
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{parameter_count_b_from_text, parameter_size_from_text};

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
    fn extracts_total_parameter_count_b() {
        assert_eq!(parameter_count_b_from_text("Qwen3-32B-Q4_K_M"), Some(32.0));
        assert_eq!(parameter_count_b_from_text("mixtral-8x7b"), Some(56.0));
        assert_eq!(parameter_count_b_from_text("235B-A22B"), Some(235.0));
    }
}
