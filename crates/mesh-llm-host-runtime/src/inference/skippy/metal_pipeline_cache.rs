use std::fs;

/// Environment variable read by the patched ggml Metal backend to locate the
/// on-disk `MTLBinaryArchive` cache of compiled compute pipeline states.
const GGML_METAL_PIPELINE_CACHE_DIR: &str = "GGML_METAL_PIPELINE_CACHE_DIR";

/// Point the native Metal backend at a per-model on-disk pipeline cache,
/// `~/.cache/mesh-llm/metal/<sanitized-model-id>`.
///
/// The Metal device, library, and pipeline set are process-global in ggml, so
/// only the first configured model wins when several models share one process;
/// later models reuse the already-compiled pipelines. An explicit value set by
/// the user is always left untouched.
pub(crate) fn configure_metal_pipeline_cache(model_id: &str) {
    if std::env::var_os(GGML_METAL_PIPELINE_CACHE_DIR).is_some() {
        return;
    }

    let dir = crate::models::mesh_llm_cache_dir()
        .join("metal")
        .join(sanitize_model_id(model_id));
    if let Err(err) = fs::create_dir_all(&dir) {
        tracing::warn!(
            target: "mesh_llm::inference::skippy::metal_pipeline_cache",
            "cannot create Metal pipeline cache dir {}: {err}",
            dir.display()
        );
        return;
    }

    // SAFETY: UNSAFE CONTRACT — must run before the first native model open
    // initializes the Metal backend, and before concurrent runtime work can
    // access the process environment. The load entrypoints enforce this by
    // calling here at the top of model load.
    unsafe { std::env::set_var(GGML_METAL_PIPELINE_CACHE_DIR, dir) };
}

/// Reduce a model id (e.g. `org/name-GGUF:Q4`) to a safe single path segment.
fn sanitize_model_id(model_id: &str) -> String {
    let mut out = String::with_capacity(model_id.len());
    let mut last_was_replacement = false;
    for ch in model_id.chars() {
        let keep = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.');
        if keep {
            out.push(ch);
            last_was_replacement = false;
        } else if !last_was_replacement {
            out.push('_');
            last_was_replacement = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed.chars().take(128).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_repo_style_ids_to_path_segments() {
        assert_eq!(
            sanitize_model_id("meta-llama/Llama-3.1-8B"),
            "meta-llama_Llama-3.1-8B"
        );
        assert_eq!(sanitize_model_id("model/Q4_K_M::gguf"), "model_Q4_K_M_gguf");
        assert_eq!(sanitize_model_id("///"), "model");
    }
}
