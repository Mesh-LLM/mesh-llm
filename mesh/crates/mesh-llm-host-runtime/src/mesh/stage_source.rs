//! Finding a stage load's model source on this node by name, when the load
//! carries no local path that exists here, and refusing the names a peer may
//! not use.

use std::path::Path;

use crate::inference::skippy::{
    StageLoadRequest, StageSourceFetch, names_a_local_path, peer_may_name_package,
};

/// Who asked this node to load a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StageLoadOrigin {
    /// This node's own runtime, over the in-process control channel.
    Local,
    /// Another mesh node, over stage control.
    Peer,
}

/// A peer names a layer package by its `hf://` ref, or by a package directory
/// on this node whose manifest has the load's `manifest_sha256`. The package
/// resolver opens a directory ref as given.
pub(crate) fn ensure_peer_package_ref(
    load: &StageLoadRequest,
    origin: StageLoadOrigin,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        origin == StageLoadOrigin::Local
            || peer_may_name_package(&load.package_ref, &load.manifest_sha256),
        "{} names a package directory on this node without its manifest {}; a peer names a layer package by its hf:// ref or a verified manifest",
        load.package_ref,
        load.manifest_sha256
    );
    Ok(())
}

/// Set `load.model_path` from the load's model names through the model store,
/// the catalog, or a download that `source_fetch` allows, when a runtime-slice
/// load has no path that exists here.
pub(crate) async fn resolve_stage_source_by_name(
    load: &mut StageLoadRequest,
    origin: StageLoadOrigin,
    source_fetch: StageSourceFetch,
) {
    if load.load_mode != skippy_protocol::LoadMode::RuntimeSlice
        || load
            .model_path
            .as_deref()
            .is_some_and(|path| Path::new(path).exists())
    {
        return;
    }
    for candidate in source_name_candidates(&load.model_id, &load.package_ref, origin) {
        if let Some(path) = source_fetch.resolve_model_candidate(&candidate).await {
            load.model_path = Some(path.to_string_lossy().to_string());
            return;
        }
    }
}

fn source_name_candidates(
    model_id: &str,
    package_ref: &str,
    origin: StageLoadOrigin,
) -> Vec<String> {
    [
        model_id,
        package_ref.strip_prefix("gguf://").unwrap_or_default(),
    ]
    .into_iter()
    .filter(|candidate| !candidate.is_empty())
    .filter(|candidate| origin == StageLoadOrigin::Local || !names_a_local_path(candidate))
    .map(str::to_string)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_source_names_cannot_be_local_paths() {
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("model.gguf");
        std::fs::write(&existing, b"GGUF").unwrap();
        let existing = existing.to_string_lossy().into_owned();

        let cases = [
            (
                existing.clone(),
                "gguf:///no/such/model.gguf".to_string(),
                Vec::<&str>::new(),
            ),
            (
                "org/repo:Q4_K_M".to_string(),
                format!("gguf://{existing}"),
                vec!["org/repo:Q4_K_M"],
            ),
            (
                "../../outside/model".to_string(),
                "gguf://./model.gguf".to_string(),
                Vec::new(),
            ),
            (
                "/no/such/model.gguf".to_string(),
                "gguf://org/../../model.gguf".to_string(),
                Vec::new(),
            ),
        ];
        for (model_id, package_ref, expected) in &cases {
            assert_eq!(
                &source_name_candidates(model_id, package_ref, StageLoadOrigin::Peer),
                expected,
                "peer names ({model_id}, {package_ref})"
            );
            assert_eq!(
                source_name_candidates(model_id, package_ref, StageLoadOrigin::Local).len(),
                2,
                "local loads keep every name"
            );
        }
    }

    #[test]
    fn peer_source_names_keep_model_references() {
        assert_eq!(
            source_name_candidates(
                "unsloth/Qwen3-8B-GGUF:Q4_K_M",
                "hf://meshllm/demo",
                StageLoadOrigin::Peer
            ),
            vec!["unsloth/Qwen3-8B-GGUF:Q4_K_M"]
        );
        assert_eq!(
            source_name_candidates(
                "Qwen3-8B-Q4_K_M",
                "gguf://Qwen3-8B-Q4_K_M.gguf",
                StageLoadOrigin::Peer
            ),
            vec!["Qwen3-8B-Q4_K_M", "Qwen3-8B-Q4_K_M.gguf"]
        );
    }
}
