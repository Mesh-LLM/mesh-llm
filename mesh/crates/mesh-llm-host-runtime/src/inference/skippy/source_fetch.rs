//! Whether a stage request may fetch model bytes this node does not have.
//!
//! A split coordinator on another node can ask this node to inventory, then
//! load, a stage of any model. By default the stage fetches what it lacks:
//! layer-package files and GGUFs from Hugging Face, or package artifacts from
//! the coordinator. With `runtime.peer_model_downloads = false` (or
//! `--no-peer-model-downloads`) a peer's request may only use files already on
//! local disk. The node's own loads and split coordination always fetch.

use std::path::{Path, PathBuf};

use super::{StageLoadRequest, is_layer_package_ref};

/// Where a stage request may get model bytes this node does not have.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum StageSourceFetch {
    /// Download from Hugging Face or fetch from the coordinator as needed.
    #[default]
    Allowed,
    /// Use only files already on local disk.
    LocalOnly,
}

impl StageSourceFetch {
    /// The policy for stage requests that arrive from peers.
    pub(crate) fn for_peers(peer_model_downloads: bool) -> Self {
        if peer_model_downloads {
            Self::Allowed
        } else {
            Self::LocalOnly
        }
    }

    pub(crate) fn allows_fetch(self) -> bool {
        self == Self::Allowed
    }

    /// The error for a peer request that needs `what` fetched.
    pub(crate) fn refusal(what: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "{what} is not on local disk, and this node refuses peer model downloads \
             (runtime.peer_model_downloads = false)"
        )
    }

    /// Resolve a stage's model id or GGUF path to a file on disk, downloading
    /// it only when this policy allows fetching.
    pub(crate) async fn resolve_model_candidate(self, candidate: &str) -> Option<PathBuf> {
        if !self.allows_fetch() {
            return local_model_source(candidate);
        }
        crate::models::resolve_model_spec(Path::new(candidate))
            .await
            .ok()
            .filter(|path| path.exists())
    }

    /// Refuse a `LocalOnly` load that names no file on local disk. A layer
    /// package passes here; the stage loop resolves it from cache only.
    pub(crate) fn ensure_local_source(self, load: &StageLoadRequest) -> anyhow::Result<()> {
        if self.allows_fetch()
            || (load.load_mode == skippy_protocol::LoadMode::RuntimeSlice
                && is_layer_package_ref(&load.package_ref))
            || load
                .model_path
                .as_deref()
                .is_some_and(|path| Path::new(path).is_file())
        {
            return Ok(());
        }
        Err(Self::refusal(&format!("model {}", load.model_id)))
    }
}

/// Resolve a model id or path to a GGUF already on local disk, without
/// consulting the catalog or Hugging Face.
fn local_model_source(candidate: &str) -> Option<PathBuf> {
    let direct = Path::new(candidate);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    let cached = crate::models::find_model_path(candidate);
    cached.is_file().then_some(cached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peers_fetch_unless_peer_model_downloads_are_off() {
        assert_eq!(StageSourceFetch::for_peers(true), StageSourceFetch::Allowed);
        assert_eq!(
            StageSourceFetch::for_peers(false),
            StageSourceFetch::LocalOnly
        );
        assert!(StageSourceFetch::default().allows_fetch());
        assert!(!StageSourceFetch::LocalOnly.allows_fetch());
    }

    #[test]
    fn refusal_names_the_missing_source_and_the_setting() {
        let message = StageSourceFetch::refusal("model org/model").to_string();
        assert!(message.contains("model org/model is not on local disk"));
        assert!(message.contains("runtime.peer_model_downloads = false"));
    }

    #[test]
    fn package_metadata_check_never_downloads() {
        let dir = tempfile::tempdir().unwrap();
        super::super::write_test_package_v2_fixture(
            dir.path(),
            "fixture/model",
            &[("layer-0", "layers/layer-0.gguf", "blk.0.attn.weight")],
        )
        .unwrap();

        assert!(super::super::package_metadata_is_local(
            &dir.path().to_string_lossy()
        ));
        assert!(!super::super::package_metadata_is_local(
            "hf://mesh-llm-tests/never-cached-package@main"
        ));
    }

    #[test]
    fn local_model_source_finds_files_and_ignores_missing_refs() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();

        assert_eq!(
            local_model_source(&gguf.to_string_lossy()),
            Some(gguf.clone())
        );
        assert_eq!(local_model_source(&dir.path().to_string_lossy()), None);
        assert_eq!(
            local_model_source("missing-org/missing-model@main:missing.gguf"),
            None
        );
    }
}
