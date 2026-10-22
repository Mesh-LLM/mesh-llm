//! Automatic draft discovery and conservative pairing shared by serving products.
use crate::serving::InferenceOptions;
use skippy_model_artifact::gguf::scan_gguf_compact_meta;
use std::path::{Path, PathBuf};

/// Apply automatic draft selection only when the caller has no explicit plan.
/// An incompatible sibling disables speculation, matching the warn-disable policy.
pub fn apply_auto_speculation(options: &mut InferenceOptions, model_path: &Path) {
    let Some(draft) = discover_sibling_draft_model(model_path) else {
        return;
    };
    if incompatible_draft_pair_reason(model_path, &draft).is_some() {
        options.native_mtp_enabled = false;
        options.native_mtp_max_tokens = skippy_config::local_serving::NATIVE_MTP_DRAFT_TOKENS;
        options.speculative.native_mtp.enabled = false;
        options.speculative.effective_strategy = "disabled".into();
        return;
    }
    if options.native_mtp_enabled {
        options.native_mtp_draft_model_path = Some(draft);
        options.native_mtp_max_tokens = skippy_config::local_serving::DRAFT_MODEL_TOKENS;
        options.speculative.native_mtp.max_draft_tokens = options.native_mtp_max_tokens;
    } else {
        options.draft_model_path = Some(draft);
        options.speculative_window = skippy_config::local_serving::DRAFT_MODEL_TOKENS;
        options.adaptive_speculative_window = true;
    }
}

/// Locate a sibling GGUF draft using the same deterministic ordering in both products.
pub fn discover_sibling_draft_model(model_path: &Path) -> Option<PathBuf> {
    let mut candidates = std::fs::read_dir(model_path.parent()?)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != model_path)
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    let name = name.to_ascii_lowercase();
                    name.contains("draft") || name.contains("eagle")
                })
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

/// Explain why automatic draft pairing cannot be enabled.
pub fn incompatible_draft_pair_reason(
    model_path: &Path,
    draft_model_path: &Path,
) -> Option<String> {
    let target_architecture = model_architecture_from_path(model_path);
    let draft_architecture = model_architecture_from_path(draft_model_path);
    match (target_architecture, draft_architecture) {
        (None, None) => {
            Some("target and draft model architecture metadata is unavailable".to_string())
        }
        (None, Some(_)) => Some("target model architecture metadata is unavailable".to_string()),
        (Some(_), None) => Some("draft model architecture metadata is unavailable".to_string()),
        (Some(target), Some(draft)) if target != draft => Some(format!(
            "target architecture {target} does not match draft architecture {draft}"
        )),
        _ => None,
    }
}

/// Whether `path` is a DFlash or DFlash2 block drafter, which reads target
/// hidden states and so cannot run as an ordinary draft model.
pub fn is_dflash_draft(path: &Path) -> bool {
    model_architecture_from_path(path).as_deref() == Some("dflash")
}

fn model_architecture_from_path(path: &Path) -> Option<String> {
    scan_gguf_compact_meta(path)
        .or_else(|| scan_gguf_compact_meta(&path.join("shared/metadata.gguf")))
        .map(|meta| {
            meta.architecture
                .trim()
                .to_ascii_lowercase()
                .replace('-', "_")
        })
        .filter(|architecture| !architecture.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installed_draft_selection_is_deterministic_and_filters_unrelated_files() {
        let directory = tempfile::tempdir().unwrap();
        for name in [
            "target.gguf",
            "z-draft.gguf",
            "a-eagle.GGUF",
            "draft.txt",
            "weights.gguf",
        ] {
            std::fs::write(directory.path().join(name), []).unwrap();
        }
        let target = directory.path().join("target.gguf");
        assert_eq!(
            discover_sibling_draft_model(&target),
            Some(directory.path().join("a-eagle.GGUF"))
        );
        assert!(discover_sibling_draft_model(Path::new("/missing/model.gguf")).is_none());
    }

    #[test]
    fn automatic_pairing_preserves_native_mtp_without_a_draft_and_disables_unknown_pairings() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.gguf");
        let mut options = InferenceOptions::direct_single_stage_defaults(
            "model".into(),
            skippy_config::local_serving::MAX_OUTPUT_TOKENS,
            4,
            true,
        );
        let original = options.clone();
        apply_auto_speculation(&mut options, &target);
        assert_eq!(options, original);
        std::fs::write(directory.path().join("draft.gguf"), []).unwrap();
        apply_auto_speculation(&mut options, &target);
        assert!(!options.native_mtp_enabled);
        assert!(!options.speculative.native_mtp.enabled);
        assert_eq!(options.speculative.effective_strategy, "disabled");
        assert!(options.draft_model_path.is_none());
        assert!(options.native_mtp_draft_model_path.is_none());
    }
}
