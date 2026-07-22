use std::path::{Path, PathBuf};

/// Strip common GGUF quantization suffixes from a lowercased stem.
/// e.g. "qwen3vl-2b-instruct-q4_k_m" → "qwen3vl-2b-instruct"
fn strip_quant_suffix(stem: &str) -> &str {
    // Quant suffixes are typically the last hyphen-separated component:
    // Q4_K_M, Q8_0, BF16, F16, F32, IQ4_NL, etc.
    if let Some(pos) = stem.rfind('-') {
        let suffix = &stem[pos + 1..];
        // Starts with 'q', 'iq', 'f', or 'bf' followed by a digit → quant suffix
        let is_quant = suffix.starts_with("q")
            || suffix.starts_with("iq")
            || suffix.starts_with("f16")
            || suffix.starts_with("f32")
            || suffix.starts_with("bf16");
        if is_quant {
            return &stem[..pos];
        }
    }
    stem
}

/// Extract the quantization suffix from a lowercased model stem, if present.
/// e.g. "qwen3vl-2b-instruct-q4_k_m" → Some("q4_k_m")
///      "my-model"                    → None
fn extract_quant_suffix(stem: &str) -> Option<String> {
    let stripped = strip_quant_suffix(stem);
    if stripped.len() < stem.len() {
        // +1 to skip the '-' separator; use .get() for safe UTF-8 slicing
        stem.get(stripped.len() + 1..).map(|s| s.to_string())
    } else {
        None
    }
}

/// Return the sole candidate from `candidates` whose lowercased filename
/// contains `quant`, or `None` if zero or multiple candidates match.
fn pick_quant_match(candidates: &[PathBuf], quant: &str) -> Option<PathBuf> {
    let mut matches: Vec<_> = candidates
        .iter()
        .filter(|path| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase().contains(quant))
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

fn is_named_mmproj_match(lower: &str, model_base: &str, model_stem: &str) -> bool {
    // Try pattern: <model>-mmproj... (model name before mmproj)
    if let Some((prefix, _)) = lower
        .split_once("-mmproj")
        .or_else(|| lower.split_once("_mmproj"))
        && (model_base.starts_with(prefix) || model_stem.starts_with(prefix))
    {
        return true;
    }
    // Try pattern: mmproj-<model>... (model name after mmproj)
    if let Some(after) = lower
        .strip_prefix("mmproj-")
        .or_else(|| lower.strip_prefix("mmproj_"))
    {
        let mmproj_model_base = strip_quant_suffix(after);
        if model_base.starts_with(mmproj_model_base) || mmproj_model_base.starts_with(model_base) {
            return true;
        }
    }
    false
}

fn mmproj_precision_variant_key(path: &Path) -> Option<(String, u8)> {
    let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
    let split = stem.rfind(['-', '_'])?;
    let base = stem[..split].trim_end_matches(['-', '_']).to_string();
    let precision = &stem[split + 1..];
    let rank = match precision {
        "bf16" => 0,
        "f16" => 1,
        "f32" => 2,
        _ => return None,
    };
    Some((base, rank))
}

fn choose_mmproj_candidate(candidates: &[PathBuf]) -> Option<PathBuf> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates[0].clone());
    }

    let parsed: Vec<_> = candidates
        .iter()
        .map(|path| mmproj_precision_variant_key(path).map(|(base, rank)| (path, base, rank)))
        .collect::<Option<Vec<_>>>()?;
    let base = &parsed.first()?.1;
    if parsed.iter().any(|(_, other_base, _)| other_base != base) {
        return None;
    }

    parsed
        .into_iter()
        .min_by_key(|(_, _, rank)| *rank)
        .map(|(path, _, _)| path.clone())
}

pub fn find_mmproj_path(_model_name: &str, model_path: &Path) -> Option<PathBuf> {
    // Scan the model's parent directory for a matching mmproj file.
    // This is safe for the HF hub cache because each model lives in its own
    // isolated snapshot subdirectory alongside only its companion files.
    //
    // Preferred resolution order within that exact directory:
    // 1. Model-name-aware matches (single → return immediately).
    // 2. Among multiple name-matched candidates: quant-aware selection —
    //    prefer the mmproj whose filename contains the same quantization as
    //    the model (e.g. Q4_K_M), matching LM Studio's heuristic.
    // 3. Precision-variant fallback: if all remaining candidates are the same
    //    projector in different precisions, prefer BF16 over F16 over F32.
    // 4. Return None when the choice is genuinely ambiguous.
    let parent = model_path.parent()?;
    let model_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())?;
    // Strip the quant suffix from the model stem to get the base model name
    // e.g. "qwen3vl-2b-instruct-q4_k_m" → "qwen3vl-2b-instruct"
    let model_base = strip_quant_suffix(&model_stem);
    // Extract the quantization suffix for quant-aware matching below
    // e.g. "qwen3vl-2b-instruct-q4_k_m" → Some("q4_k_m")
    let model_quant = extract_quant_suffix(&model_stem);
    let mmproj_siblings: Vec<PathBuf> = std::fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != model_path)
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("gguf"))
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| {
                    let lower = stem.to_ascii_lowercase();
                    lower.contains("mmproj")
                })
                .unwrap_or(false)
        })
        .collect();

    let named_matches: Vec<PathBuf> = mmproj_siblings
        .iter()
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| {
                    is_named_mmproj_match(&stem.to_ascii_lowercase(), model_base, &model_stem)
                })
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    if !named_matches.is_empty() {
        // Multiple named matches: try quant-aware selection before precision fallback
        if named_matches.len() > 1
            && let Some(ref quant) = model_quant
            && let Some(candidate) = pick_quant_match(&named_matches, quant)
        {
            return Some(candidate);
        }
        // Single named match, or quant-match failed: precision-variant pick or None
        return choose_mmproj_candidate(&named_matches);
    }

    // No named matches: try quant-aware selection among all siblings, then precision fallback
    if mmproj_siblings.len() > 1
        && let Some(ref quant) = model_quant
        && let Some(candidate) = pick_quant_match(&mmproj_siblings, quant)
    {
        return Some(candidate);
    }
    choose_mmproj_candidate(&mmproj_siblings)
}

#[cfg(test)]
fn resolve_mmproj_path(
    model_name: &str,
    model_path: &Path,
    explicit_mmproj: Option<&Path>,
) -> Option<PathBuf> {
    explicit_mmproj
        .map(Path::to_path_buf)
        .or_else(|| find_mmproj_path(model_name, model_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmproj_path_falls_back_to_single_sibling_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let mmproj = temp.path().join("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&mmproj, b"mmproj").unwrap();

        let found = find_mmproj_path("Qwen3VL-2B-Instruct-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(mmproj.as_path()));
    }

    #[test]
    fn mmproj_path_ignores_ambiguous_sibling_sidecars() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let mmproj_a = temp.path().join("mmproj-a.gguf");
        let mmproj_b = temp.path().join("mmproj-b.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&mmproj_a, b"mmproj").unwrap();
        std::fs::write(&mmproj_b, b"mmproj").unwrap();

        assert!(find_mmproj_path("Qwen3VL-2B-Instruct-Q4_K_M", &model).is_none());
    }

    #[test]
    fn mmproj_path_prefers_bf16_generic_precision_variants() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("Qwen3.5-0.8B-Q4_K_M.gguf");
        let f32 = temp.path().join("mmproj-F32.gguf");
        let f16 = temp.path().join("mmproj-F16.gguf");
        let bf16 = temp.path().join("mmproj-BF16.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&f32, b"mmproj").unwrap();
        std::fs::write(&f16, b"mmproj").unwrap();
        std::fs::write(&bf16, b"mmproj").unwrap();

        let found = find_mmproj_path("Qwen3.5-0.8B-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(bf16.as_path()));
    }

    #[test]
    fn resolve_mmproj_path_prefers_explicit_override() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let sibling = temp.path().join("mmproj-sibling.gguf");
        let explicit = temp.path().join("mmproj-explicit.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&sibling, b"mmproj").unwrap();
        std::fs::write(&explicit, b"mmproj").unwrap();

        let found = resolve_mmproj_path(
            "Qwen3VL-2B-Instruct-Q4_K_M",
            &model,
            Some(explicit.as_path()),
        );
        assert_eq!(found.as_deref(), Some(explicit.as_path()));
    }

    #[test]
    fn mmproj_path_prefers_quant_matched_named_candidate() {
        // When multiple named mmproj candidates exist (model-name prefix matches
        // both), quant-aware selection should pick the one whose filename contains
        // the same quantization as the model (Q4_K_M in this case).
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let q4_mmproj = temp.path().join("mmproj-Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let q8_mmproj = temp.path().join("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&q4_mmproj, b"mmproj").unwrap();
        std::fs::write(&q8_mmproj, b"mmproj").unwrap();

        let found = find_mmproj_path("Qwen3VL-2B-Instruct-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(q4_mmproj.as_path()));
    }

    #[test]
    fn mmproj_path_prefers_quant_matched_generic_sibling() {
        // When there are no model-name-aware matches but the siblings include
        // a projector with the same quant as the model, select that one.
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("my-model-Q4_K_M.gguf");
        // Generic projector names without a matching model prefix
        let q4_mmproj = temp.path().join("mmproj-Q4_K_M.gguf");
        let q8_mmproj = temp.path().join("mmproj-Q8_0.gguf");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&q4_mmproj, b"mmproj").unwrap();
        std::fs::write(&q8_mmproj, b"mmproj").unwrap();

        let found = find_mmproj_path("my-model-Q4_K_M", &model);
        assert_eq!(found.as_deref(), Some(q4_mmproj.as_path()));
    }
}
