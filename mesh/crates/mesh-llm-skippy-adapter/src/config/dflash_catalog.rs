//! Known DFlash drafts, chosen by default for the targets they were trained on.
//!
//! A target is recognised from GGUF metadata, never its file name or quant, so
//! every quant of a model gets the same draft. Each draft is pinned to a
//! Hugging Face revision and SHA-256; the host fetches it into the Hugging Face
//! cache on first load and serves without speculation if it cannot.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};
use hf_hub::RepoTypeModel;
use skippy_model_artifact::gguf::{
    GgufCompactMeta, scan_gguf_compact_meta, scan_gguf_dflash_target_layers,
};
use skippy_model_hf::store::local::huggingface_snapshot_path;

/// GGUF metadata that identifies a target model.
#[derive(Debug, PartialEq, Eq)]
pub struct DFlashTarget {
    pub architecture: &'static str,
    /// Executable decoder layers, excluding NextN/MTP blocks.
    pub layer_count: u32,
    pub embedding_length: u32,
    /// Routed experts; `0` for a dense model.
    pub expert_count: u32,
}

/// A DFlash draft pinned on Hugging Face.
#[derive(Debug, PartialEq, Eq)]
pub struct DFlashDraftSource {
    pub repo: &'static str,
    pub revision: &'static str,
    pub file: &'static str,
    pub sha256: &'static str,
    /// File size, which the memory plan charges before the draft is fetched.
    pub bytes: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DFlashPairing {
    pub target: DFlashTarget,
    pub draft: DFlashDraftSource,
    /// Default the draft on only under Metal, where it measured faster.
    pub metal_only: bool,
}

pub(super) const DFLASH_PAIRINGS: &[DFlashPairing] = &[
    // Qwen3.5-9B; the draft reads target layers 2..=30 at width 4096.
    DFlashPairing {
        target: DFlashTarget {
            architecture: "qwen35",
            layer_count: 32,
            embedding_length: 4096,
            expert_count: 0,
        },
        draft: DFlashDraftSource {
            repo: "AtomicChat/Qwen3.5-9B-DFlash-GGUF",
            revision: "c7c69923d77a9aa0004b410d3a7a3dbb64591d41",
            file: "Qwen3.5-9B-DFlash.Q8_0.gguf",
            sha256: "27b9d18e605aea9c50ef506e4b63921e0dc624cc91c63c60617a8295f153d436",
            bytes: 1_383_768_224,
        },
        metal_only: false,
    },
    // Qwen3.6-35B-A3B; the draft reads target layers 2..=38 at width 2048.
    DFlashPairing {
        target: DFlashTarget {
            architecture: "qwen35moe",
            layer_count: 40,
            embedding_length: 2048,
            expert_count: 256,
        },
        draft: DFlashDraftSource {
            repo: "Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp",
            revision: "2aa57b642f211d2639083954c5b66d9c1ee503fc",
            file: "Qwen3.6-35B-A3B-DFlash-Q8_0.gguf",
            sha256: "724ac8a102763e0d896cd49bf558fdf5bfdb146ec9f7097d18b035e4a95eac3f",
            bytes: 421_060_800,
        },
        metal_only: false,
    },
    // Qwen3-Coder-Next; the draft reads target layers 4..=44 at width 2048.
    // With 512 experts, verifying 8 rows costs 2.4 decode steps on a DGX Spark
    // (1.15 for Qwen3.5-9B), and every block size measured slower than plain
    // decoding there (59 tok/s off; 38-55 with DFlash). An M3 Ultra gains 1.12x.
    DFlashPairing {
        target: DFlashTarget {
            architecture: "qwen3next",
            layer_count: 48,
            embedding_length: 2048,
            expert_count: 512,
        },
        draft: DFlashDraftSource {
            repo: "AtomicChat/Qwen3-Coder-Next-DFlash-GGUF",
            revision: "f5b4c9d2511678f75a41b3d3de5d9e379dd88768",
            file: "Qwen3-Coder-Next-DFlash.Q8_0.gguf",
            sha256: "139081af2f9f634b3b8bce8f75a0449f92bcb75ae7a2e32abaf1bcdf6f4a7952",
            bytes: 509_674_720,
        },
        metal_only: true,
    },
];

/// The known pairing for the target `meta` describes.
pub(super) fn pairing_for_target(meta: &GgufCompactMeta) -> Option<&'static DFlashPairing> {
    DFLASH_PAIRINGS
        .iter()
        .find(|pairing| pairing.target.matches(meta))
}

/// The default cap on DFlash draft tokens per block. Adaptive block sizing
/// picks each block's length up to it. Held-out prompts put the best cap at 15
/// on CUDA (DGX Spark) and 6 on Metal: an M3 Ultra decodes 150.7 tok/s at cap 6
/// against 135 at cap 15, and an M6 is level between them.
pub(super) fn default_dflash_max_draft_tokens(metal: bool) -> u32 {
    if metal { 6 } else { 15 }
}

impl DFlashPairing {
    /// Whether this pairing is a default on the given backend.
    pub(super) fn default_on(&self, metal: bool) -> bool {
        metal || !self.metal_only
    }
}

impl DFlashTarget {
    fn matches(&self, meta: &GgufCompactMeta) -> bool {
        meta.architecture == self.architecture
            && meta.executable_layer_count() == Some(self.layer_count)
            && meta.embedding_size == self.embedding_length
            && meta.expert_count == self.expert_count
    }
}

impl DFlashDraftSource {
    /// Where the Hugging Face cache keeps this draft once fetched.
    pub fn cache_path(&self) -> PathBuf {
        huggingface_snapshot_path(self.repo, RepoTypeModel, self.revision).join(self.file)
    }
}

impl DFlashPairing {
    /// Refuses a draft that cannot read this target: it must be a DFlash GGUF
    /// as wide as the target and read only layers the target has.
    pub fn check_draft(&self, path: &Path) -> Result<()> {
        let Some(meta) = scan_gguf_compact_meta(path) else {
            bail!("{} is not a readable GGUF", path.display());
        };
        ensure!(
            meta.architecture == "dflash",
            "draft architecture is {}, expected dflash",
            meta.architecture
        );
        ensure!(
            meta.embedding_size == self.target.embedding_length,
            "draft hidden size {} does not match target hidden size {}",
            meta.embedding_size,
            self.target.embedding_length
        );
        let layers = scan_gguf_dflash_target_layers(path).unwrap_or_default();
        ensure!(!layers.is_empty(), "draft names no target layers");
        if let Some(layer) = layers
            .iter()
            .find(|&&layer| layer >= self.target.layer_count)
        {
            bail!(
                "draft reads target layer {layer}, but the target has {} layers",
                self.target.layer_count
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{push_gguf_string, push_string_kv, push_u32_kv, target_meta};
    use super::*;
    use std::io::Write;

    fn draft_file(architecture: &str, embedding: u32, layers: &[u32]) -> tempfile::NamedTempFile {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        push_string_kv(&mut bytes, "general.architecture", architecture);
        push_u32_kv(
            &mut bytes,
            &format!("{architecture}.embedding_length"),
            embedding,
        );
        push_gguf_string(&mut bytes, &format!("{architecture}.target_layers"));
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&(layers.len() as u64).to_le_bytes());
        for layer in layers {
            bytes.extend_from_slice(&layer.to_le_bytes());
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file
    }

    #[test]
    fn targets_match_by_metadata_whatever_the_quant() {
        let repos = |meta: &GgufCompactMeta| pairing_for_target(meta).map(|p| p.draft.repo);
        assert_eq!(
            repos(&target_meta("qwen35", 32, 4096, 0)),
            Some("AtomicChat/Qwen3.5-9B-DFlash-GGUF")
        );
        assert_eq!(
            repos(&target_meta("qwen35moe", 40, 2048, 256)),
            Some("Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp")
        );
        assert_eq!(
            repos(&target_meta("qwen3next", 48, 2048, 512)),
            Some("AtomicChat/Qwen3-Coder-Next-DFlash-GGUF")
        );
        // A NextN block appended to the trunk is not a decoder layer.
        let mut with_mtp = target_meta("qwen35", 33, 4096, 0);
        with_mtp.nextn_predict_layers = 1;
        assert!(pairing_for_target(&with_mtp).is_some());
    }

    #[test]
    fn coder_next_defaults_on_only_under_metal() {
        let coder = pairing_for_target(&target_meta("qwen3next", 48, 2048, 512)).unwrap();
        assert!(coder.default_on(true));
        assert!(!coder.default_on(false));
        for pairing in &DFLASH_PAIRINGS[..2] {
            assert!(pairing.default_on(true) && pairing.default_on(false));
        }
    }

    #[test]
    fn other_shapes_of_the_same_family_do_not_match() {
        for meta in [
            target_meta("qwen35", 64, 5120, 0),
            target_meta("qwen35", 32, 2048, 0),
            target_meta("qwen35moe", 40, 2048, 128),
            target_meta("qwen3", 32, 4096, 0),
        ] {
            assert_eq!(pairing_for_target(&meta), None, "{meta:?}");
        }
    }

    #[test]
    fn every_draft_is_pinned_to_a_commit_and_digest() {
        for pairing in DFLASH_PAIRINGS {
            let draft = &pairing.draft;
            assert_eq!(draft.revision.len(), 40, "{}", draft.repo);
            assert_eq!(draft.sha256.len(), 64, "{}", draft.repo);
            assert!(draft.bytes > 0, "{}", draft.repo);
            assert!(
                draft
                    .cache_path()
                    .ends_with(format!("snapshots/{}/{}", draft.revision, draft.file)),
                "{}",
                draft.repo
            );
        }
    }

    #[test]
    fn draft_cap_defaults_per_backend() {
        assert_eq!(default_dflash_max_draft_tokens(true), 6);
        assert_eq!(default_dflash_max_draft_tokens(false), 15);
    }

    #[test]
    fn a_draft_must_fit_its_target() {
        let pairing = &DFLASH_PAIRINGS[0];
        let fits = draft_file("dflash", 4096, &[2, 6, 10, 14, 18, 22, 26, 30]);
        pairing.check_draft(fits.path()).unwrap();

        for (draft, reason) in [
            (draft_file("qwen35", 4096, &[2, 30]), "architecture"),
            (draft_file("dflash", 2048, &[2, 30]), "hidden size"),
            (draft_file("dflash", 4096, &[2, 38]), "target layer 38"),
            (draft_file("dflash", 4096, &[]), "no target layers"),
        ] {
            let error = pairing.check_draft(draft.path()).unwrap_err().to_string();
            assert!(error.contains(reason), "{error}");
        }
    }
}
