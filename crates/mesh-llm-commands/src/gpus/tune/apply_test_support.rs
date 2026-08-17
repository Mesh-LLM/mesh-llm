use crate::gpus::tune_resolver::{
    ConfigModelMatch, LocalTargetSource, ResolvedTuneTarget, TuneTargetSelection,
};
use model_hf::store::model_ref_for_path;
use std::path::Path;

pub(crate) fn write_local_gguf_file(dir: &Path, name: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, b"GGUF").expect("fixture GGUF should be written");
    path
}

pub(crate) fn configured_target(path: &Path, row_index: usize) -> ResolvedTuneTarget {
    ResolvedTuneTarget {
        requested_input: path.display().to_string(),
        canonical_model_ref: model_ref_for_path(path),
        resolved_path: path.to_path_buf(),
        local_source: LocalTargetSource::FilesystemPath {
            synthetic_model_ref: model_ref_for_path(path),
        },
        config_matches: vec![ConfigModelMatch {
            row_index,
            configured_model: path.display().to_string(),
        }],
        selection: TuneTargetSelection::Configured,
    }
}

pub(crate) fn appended_target(path: &Path) -> ResolvedTuneTarget {
    ResolvedTuneTarget {
        requested_input: path.display().to_string(),
        canonical_model_ref: model_ref_for_path(path),
        resolved_path: path.to_path_buf(),
        local_source: LocalTargetSource::FilesystemPath {
            synthetic_model_ref: model_ref_for_path(path),
        },
        config_matches: Vec::new(),
        selection: TuneTargetSelection::Explicit { configured: false },
    }
}
