//! Known DFlash drafts, chosen by default for the targets they were trained on.
//!
//! A target is recognised from GGUF metadata, never its file name or quant, so
//! every quant of a model gets the same draft. Each draft is pinned to a
//! Hugging Face revision and SHA-256; the host fetches it into the Hugging Face
//! cache on first load and serves without speculation if it cannot.
//!
//! The default draft comes from, in order: the layer package's own `dflash`
//! proposer, the pairing index this release pins by content hash, and, until
//! that index is cached (or if it cannot be read or fails its hash), the
//! bootstrap table compiled in here.

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, RwLock};

use anyhow::{Context, Result, bail, ensure};
use hf_hub::RepoTypeModel;
use sha2::{Digest, Sha256};
use skippy_model_artifact::gguf::{
    GgufCompactMeta, scan_gguf_compact_meta, scan_gguf_dflash_target_layers,
    scan_gguf_vocab_token_count,
};
use skippy_model_hf::store::local::{
    huggingface_hub_cache_dir, huggingface_repo_folder_name, huggingface_snapshot_path,
};
pub use skippy_package_format::{
    DFlashBackends, DFlashDraft, DFlashPairing, DFlashPairingIndex, DFlashTarget,
};
use skippy_runtime::package::PackageGenerationInfo;

/// The Hugging Face repo that publishes the DFlash pairing index.
pub const DFLASH_PAIRING_INDEX_REPO: &str = "meshllm/dflash-pairings";
/// The SHA-256 of the pairing index this release trusts: the bytes of
/// `data/dflash-pairings/index.json` in this crate. CI publishes that file as
/// `index/<sha256>.json` in [`DFLASH_PAIRING_INDEX_REPO`] when it changes on
/// `main`, so editing the pairings means updating this constant too.
pub const DFLASH_PAIRING_INDEX_SHA256: &str =
    "daec3de5ce231a13e6b3f2f81e6c54993ae055a6501e08215ff37292805b5766";

/// A pairing index named by the SHA-256 of its bytes.
///
/// Published index files are never changed or deleted, so `main` of the repo
/// holds every index any release pins, and a file whose bytes hash to the pin
/// is that index wherever it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFlashPairingIndexPin {
    pub repo: &'static str,
    pub sha256: &'static str,
}

impl DFlashPairingIndexPin {
    /// The revision the index is fetched at.
    pub const REVISION: &'static str = "main";

    /// The index this release trusts.
    pub fn shipped() -> Self {
        Self {
            repo: DFLASH_PAIRING_INDEX_REPO,
            sha256: DFLASH_PAIRING_INDEX_SHA256,
        }
    }

    /// The index's path in the repo: `index/<sha256>.json`.
    pub fn file(&self) -> String {
        format!("index/{}.json", self.sha256)
    }

    /// The index in the Hugging Face cache, in whichever snapshot holds it.
    pub fn cached_path(&self) -> Option<PathBuf> {
        self.cached_path_in(
            &huggingface_hub_cache_dir()
                .join(huggingface_repo_folder_name(self.repo, RepoTypeModel))
                .join("snapshots"),
        )
    }

    /// The index under `snapshots`, a repo's Hugging Face cache snapshot
    /// directory. Its file name is its hash, so any snapshot's copy will do.
    fn cached_path_in(&self, snapshots: &Path) -> Option<PathBuf> {
        let file = self.file();
        let mut found: Vec<PathBuf> = std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path().join(&file)))
            .filter(|path| path.is_file())
            .collect();
        found.sort();
        found.into_iter().next()
    }

    /// Reads the index at `path`, refusing it unless its bytes hash to the pin.
    pub fn load(&self, path: &Path) -> Result<DFlashPairingIndex> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let sha256 = sha256_hex(&bytes);
        ensure!(
            sha256 == self.sha256,
            "{} has sha256 {sha256}, expected {}",
            path.display(),
            self.sha256
        );
        let json = std::str::from_utf8(&bytes)
            .with_context(|| format!("{} is not UTF-8", path.display()))?;
        DFlashPairingIndex::from_json(json)
            .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))
    }
}

/// The lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Reads and validates a pairing index file, without checking its hash.
pub fn load_dflash_pairing_index(path: &Path) -> Result<DFlashPairingIndex> {
    let json = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    DFlashPairingIndex::from_json(&json)
        .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))
}

static INSTALLED_INDEX: RwLock<Option<Arc<Vec<DFlashPairing>>>> = RwLock::new(None);
static BOOTSTRAP_PAIRINGS: LazyLock<Arc<Vec<DFlashPairing>>> =
    LazyLock::new(|| Arc::new(bootstrap_pairings()));

/// Makes `index` the source of default pairings for later loads, in place of
/// the bootstrap table.
pub fn install_dflash_pairing_index(index: DFlashPairingIndex) {
    if let Ok(mut installed) = INSTALLED_INDEX.write() {
        *installed = Some(Arc::new(index.pairings));
    }
}

/// Whether a pairing index has been installed.
pub fn dflash_pairing_index_installed() -> bool {
    INSTALLED_INDEX
        .read()
        .is_ok_and(|installed| installed.is_some())
}

#[cfg(test)]
thread_local! {
    /// Pairings a test resolves against instead of the process-wide source.
    static TEST_PAIRINGS: std::cell::RefCell<Option<Arc<Vec<DFlashPairing>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Resolves on this thread against `pairings` until the guard drops.
#[cfg(test)]
pub(super) fn with_test_pairings(pairings: Vec<DFlashPairing>) -> impl Drop {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_PAIRINGS.with(|cell| cell.borrow_mut().take());
        }
    }
    TEST_PAIRINGS.with(|cell| *cell.borrow_mut() = Some(Arc::new(pairings)));
    Reset
}

/// The pairings plain GGUF loads default from: the installed index, else the
/// bootstrap table.
fn active_pairings() -> Arc<Vec<DFlashPairing>> {
    #[cfg(test)]
    if let Some(pairings) = TEST_PAIRINGS.with(|cell| cell.borrow().clone()) {
        return pairings;
    }
    INSTALLED_INDEX
        .read()
        .ok()
        .and_then(|installed| installed.clone())
        .unwrap_or_else(|| BOOTSTRAP_PAIRINGS.clone())
}

/// Where a default DFlash draft was chosen from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DFlashPairingSource {
    /// The layer package's `dflash` proposer.
    Package,
    /// The published pairing index, or the bootstrap table until it is cached.
    Catalog,
}

/// A draft chosen by default, with the target it must fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultDFlash {
    pub pairing: DFlashPairing,
    /// The target layers a package proposer declares the draft reads; the
    /// draft's header must name exactly these.
    pub target_layers: Option<Vec<u32>>,
    pub source: DFlashPairingSource,
}

/// The pairings this release falls back to until the pinned index is cached,
/// or when it cannot be read or fails its hash.
///
/// TODO: delete once packages declare their drafts; the index
/// (`data/dflash-pairings/index.json`) is the source of truth, and a test
/// checks that it pairs every target here the same way.
pub(super) fn bootstrap_pairings() -> Vec<DFlashPairing> {
    vec![
        // Qwen3.5-9B; the draft reads target layers 2..=30 at width 4096.
        DFlashPairing {
            target: DFlashTarget {
                architecture: "qwen35".to_string(),
                layer_count: 32,
                embedding_length: 4096,
                expert_count: 0,
                family: None,
            },
            draft: DFlashDraft {
                repo: "AtomicChat/Qwen3.5-9B-DFlash-GGUF".to_string(),
                revision: "c7c69923d77a9aa0004b410d3a7a3dbb64591d41".to_string(),
                file: "Qwen3.5-9B-DFlash.Q8_0.gguf".to_string(),
                sha256: "27b9d18e605aea9c50ef506e4b63921e0dc624cc91c63c60617a8295f153d436"
                    .to_string(),
                bytes: 1_383_768_224,
            },
            max_draft_tokens: 15,
            backends: DFlashBackends::All,
        },
        // Qwen3.6-27B; the draft reads target layers 2..=62 at width 5120.
        DFlashPairing {
            target: DFlashTarget {
                architecture: "qwen35".to_string(),
                layer_count: 64,
                embedding_length: 5120,
                expert_count: 0,
                family: None,
            },
            draft: DFlashDraft {
                repo: "AtomicChat/Qwen3.6-27B-DFlash-GGUF".to_string(),
                revision: "a3d4fb1eff6f268cbbdf808042f6902cc17f5458".to_string(),
                file: "Qwen3.6-27B-DFlash.Q8_0.gguf".to_string(),
                sha256: "b9be52167d0b4f5cbaffba6658bcb20dd814ad4e5846532353c165ccf97e47a1"
                    .to_string(),
                bytes: 1_849_481_472,
            },
            max_draft_tokens: 15,
            backends: DFlashBackends::All,
        },
        // Qwen3.8-27B has Qwen3.6-27B's shape (its NextN block is not executed), so
        // only the name tells them apart. A DFlash2 draft reading target layers
        // 6..=62 at width 5120. It replaces the default native MTP: greedy chat
        // decodes 44.0 tok/s on a DGX Spark (20.4 MTP, 12.7 off) and 77.0 on an
        // M3 Ultra (39.0 MTP, 31.2 off); agentic coding on the Spark 28.7 (16.5
        // MTP). Sampled at T=0.7, MTP is slower than plain decoding on both.
        DFlashPairing {
            target: DFlashTarget {
                architecture: "qwen35".to_string(),
                layer_count: 64,
                embedding_length: 5120,
                expert_count: 0,
                family: Some("Qwen3.8".to_string()),
            },
            draft: DFlashDraft {
                repo: "z-lab/Qwen3.8-27B-DFlash2-GGUF".to_string(),
                revision: "2d9571f8ce46e151f61c6499c99dee6079e1d610".to_string(),
                file: "Qwen3.8-27B-DFlash2-Q8_0.gguf".to_string(),
                sha256: "c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e"
                    .to_string(),
                bytes: 2_056_414_816,
            },
            max_draft_tokens: 7,
            backends: DFlashBackends::All,
        },
        // Qwen3.6-35B-A3B; a DFlash2 draft (8-token blocks with a path selector)
        // reading target layers 2..=38 at width 2048. On a DGX Spark it decodes
        // 107.6 tok/s adaptive against 102.6 for the original DFlash draft.
        DFlashPairing {
            target: DFlashTarget {
                architecture: "qwen35moe".to_string(),
                layer_count: 40,
                embedding_length: 2048,
                expert_count: 256,
                family: None,
            },
            draft: DFlashDraft {
                repo: "aminya/Qwen3.6-35B-A3B-DFlash2-GGUF".to_string(),
                revision: "1e26e3499d865b9d2fc55e019d65b3f0e8ab6846".to_string(),
                file: "Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf".to_string(),
                sha256: "69bc81f5344dfeb67bcfc09277dd9617500fca1823f72e6e08d5df92e73a8b44"
                    .to_string(),
                bytes: 570_468_544,
            },
            max_draft_tokens: 7,
            backends: DFlashBackends::NotMetal,
        },
        // Qwen3.6-35B-A3B under Metal keeps the original DFlash draft: at cap 6 an
        // M3 Ultra decodes 168.5 tok/s with it against 160.9 with DFlash2.
        DFlashPairing {
            target: DFlashTarget {
                architecture: "qwen35moe".to_string(),
                layer_count: 40,
                embedding_length: 2048,
                expert_count: 256,
                family: None,
            },
            draft: DFlashDraft {
                repo: "Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp".to_string(),
                revision: "2aa57b642f211d2639083954c5b66d9c1ee503fc".to_string(),
                file: "Qwen3.6-35B-A3B-DFlash-Q8_0.gguf".to_string(),
                sha256: "724ac8a102763e0d896cd49bf558fdf5bfdb146ec9f7097d18b035e4a95eac3f"
                    .to_string(),
                bytes: 421_060_800,
            },
            max_draft_tokens: 15,
            backends: DFlashBackends::Metal,
        },
        // Qwen3-Coder-Next; the draft reads target layers 4..=44 at width 2048.
        // With 512 experts, verifying 8 rows costs 2.4 decode steps on a DGX Spark
        // (1.15 for Qwen3.5-9B), and every block size measured slower than plain
        // decoding there (59 tok/s off; 38-55 with DFlash). An M3 Ultra gains 1.12x.
        DFlashPairing {
            target: DFlashTarget {
                architecture: "qwen3next".to_string(),
                layer_count: 48,
                embedding_length: 2048,
                expert_count: 512,
                family: None,
            },
            draft: DFlashDraft {
                repo: "AtomicChat/Qwen3-Coder-Next-DFlash-GGUF".to_string(),
                revision: "f5b4c9d2511678f75a41b3d3de5d9e379dd88768".to_string(),
                file: "Qwen3-Coder-Next-DFlash.Q8_0.gguf".to_string(),
                sha256: "139081af2f9f634b3b8bce8f75a0449f92bcb75ae7a2e32abaf1bcdf6f4a7952"
                    .to_string(),
                bytes: 509_674_720,
            },
            max_draft_tokens: 15,
            backends: DFlashBackends::Metal,
        },
        // Nemotron-3.5-Bolt-E4B, a Gemma-4-E4B fine-tune with Gemma-4-E4B-It's
        // shape, so the name breaks the tie. NVIDIA's DFlash draft reads target
        // layers 6..=36 at width 2560 with an 8-token block, in sliding-window
        // attention. Greedy chat decodes 142.1 tok/s on a DGX Spark (58.4 off)
        // and 62.3 on an M4 (28.5 off).
        DFlashPairing {
            target: DFlashTarget {
                architecture: "gemma4".to_string(),
                layer_count: 42,
                embedding_length: 2560,
                expert_count: 0,
                family: Some("Nemotron-3.5-Bolt".to_string()),
            },
            draft: DFlashDraft {
                repo: "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF".to_string(),
                revision: "b738967d6da3441a7d1c71fcec15e49e32e16ecd".to_string(),
                file: "dflash-NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf".to_string(),
                sha256: "9a63662ae8395b8bfc1a1b00d7a9ef354120f172704bd6ba88d3870cf22d4d03"
                    .to_string(),
                bytes: 398_892_256,
            },
            max_draft_tokens: 7,
            backends: DFlashBackends::All,
        },
        // Nemotron-3.5-Lightning-30B-A3B has Nemotron-3-Nano-30B-A3B's shape,
        // so the name breaks the tie. NVIDIA's DSpark draft (DFlash with a
        // Markov head) reads target layers 2..=52, layer 52 being the head's
        // input, with an 8-token block. It replaces the default native MTP:
        // on a DGX Spark greedy chat decodes 88.0 tok/s (76.7 MTP, 66.0 off),
        // sampled at T=0.7 87.6 (54.8 MTP) and agentic coding 78.4 (72.8 MTP).
        // Not on Metal, which has no NVFP4 kernels for the draft, so native
        // MTP stays the default there.
        DFlashPairing {
            target: DFlashTarget {
                architecture: "nemotron_h_moe".to_string(),
                layer_count: 52,
                embedding_length: 2688,
                expert_count: 128,
                family: Some("Nemotron-3.5-Lightning".to_string()),
            },
            draft: DFlashDraft {
                repo: "magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF"
                    .to_string(),
                revision: "ed35fc990759330e325ab9e41bcf80737ab4dafc".to_string(),
                file: "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark.gguf".to_string(),
                sha256: "14fcc5296ca9be2fa2156275fd1811de966835e2894729907a03cf7f80634856"
                    .to_string(),
                bytes: 1_357_001_312,
            },
            max_draft_tokens: 7,
            backends: DFlashBackends::NotMetal,
        },
    ]
}

/// The known pairing for the target `meta` describes on this backend.
pub(super) fn pairing_for_target(meta: &GgufCompactMeta, metal: bool) -> Option<DefaultDFlash> {
    pairing_in(&active_pairings(), meta, metal).map(|pairing| DefaultDFlash {
        pairing: pairing.clone(),
        target_layers: None,
        source: DFlashPairingSource::Catalog,
    })
}

fn pairing_in<'a>(
    pairings: &'a [DFlashPairing],
    meta: &GgufCompactMeta,
    metal: bool,
) -> Option<&'a DFlashPairing> {
    let name = meta.model_name().unwrap_or_default();
    pairings.iter().find(|pairing| {
        matches_shape(&pairing.target, meta)
            && pairing.backends.applies_on(metal)
            && match pairing.target.family.as_deref() {
                Some(family) => names_family(name, family),
                None => !pairings.iter().any(|other| {
                    other.target.same_shape(&pairing.target)
                        && other
                            .target
                            .family
                            .as_deref()
                            .is_some_and(|family| names_family(name, family))
                }),
            }
    })
}

/// The draft a layer package declares for `requested_strategy` (`auto` meaning
/// the package default) on this backend: the first proposer of a `dflash`
/// strategy whose backends apply. `None` when that strategy is not `dflash`,
/// no proposer applies here, or the target's shape is unknown.
pub(super) fn package_pairing(
    generation: &PackageGenerationInfo,
    requested_strategy: &str,
    meta: Option<&GgufCompactMeta>,
    metal: bool,
) -> Option<DefaultDFlash> {
    let speculative = generation.speculative_decoding.as_ref()?;
    let name = match requested_strategy {
        "auto" => speculative.default.as_str(),
        name => name,
    };
    let strategy = speculative.strategies.get(name)?;
    if strategy.strategy_type != "dflash" {
        return None;
    }
    let proposer = strategy
        .proposers
        .iter()
        .filter_map(|name| speculative.proposers.get(name)?.dflash.as_ref())
        .find(|proposer| proposer.backends.applies_on(metal))?;
    let meta = meta?;
    Some(DefaultDFlash {
        pairing: DFlashPairing {
            target: DFlashTarget {
                architecture: meta.architecture.clone(),
                layer_count: meta.executable_layer_count()?,
                embedding_length: meta.embedding_size,
                expert_count: meta.expert_count,
                family: None,
            },
            draft: proposer.draft.clone(),
            max_draft_tokens: proposer.max_draft_tokens,
            backends: proposer.backends,
        },
        target_layers: Some(proposer.target_layers.clone()),
        source: DFlashPairingSource::Package,
    })
}

/// Whether a package's selected strategy is `dflash`, so the package, not the
/// catalog, decides the draft.
pub(super) fn package_selects_dflash(
    generation: Option<&PackageGenerationInfo>,
    requested_strategy: &str,
) -> bool {
    generation
        .and_then(|generation| generation.speculative_decoding.as_ref())
        .and_then(|speculative| {
            let name = match requested_strategy {
                "auto" => speculative.default.as_str(),
                name => name,
            };
            speculative.strategies.get(name)
        })
        .is_some_and(|strategy| strategy.strategy_type == "dflash")
}

/// Whether `name` carries `family` as a whole token, ignoring case, so
/// "Huihui-Qwen3.6-27B" names Qwen3.6 but "Qwen3.6x" and "Qwen3.65" do not.
/// Spaces, underscores and hyphens are one separator, so "NVIDIA Nemotron 3.5
/// Bolt E4B" names Nemotron-3.5-Bolt.
fn names_family(name: &str, family: &str) -> bool {
    let name = separators_as_hyphens(name);
    let family = separators_as_hyphens(family);
    let continues = |c: char| c.is_ascii_alphanumeric() || c == '.';
    name.match_indices(&family).any(|(start, _)| {
        !name[..start].ends_with(continues) && !name[start + family.len()..].starts_with(continues)
    })
}

/// `text` lowercased, with spaces and underscores written as hyphens.
fn separators_as_hyphens(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            ' ' | '_' => '-',
            c => c.to_ascii_lowercase(),
        })
        .collect()
}

/// The default cap on DFlash draft tokens per block. Adaptive block sizing
/// picks each block's length up to it. Held-out prompts put the best cap at 15
/// on CUDA (DGX Spark) and 6 on Metal: an M3 Ultra decodes 150.7 tok/s at cap 6
/// against 135 at cap 15, and an M6 is level between them.
pub(super) fn default_dflash_max_draft_tokens(metal: bool) -> u32 {
    if metal { 6 } else { 15 }
}

fn matches_shape(target: &DFlashTarget, meta: &GgufCompactMeta) -> bool {
    meta.architecture == target.architecture
        && meta.executable_layer_count() == Some(target.layer_count)
        && meta.embedding_size == target.embedding_length
        && meta.expert_count == target.expert_count
}

/// Where the Hugging Face cache keeps `draft` once fetched.
pub fn dflash_draft_cache_path(draft: &DFlashDraft) -> PathBuf {
    huggingface_snapshot_path(&draft.repo, RepoTypeModel, &draft.revision).join(&draft.file)
}

impl DefaultDFlash {
    /// Refuses a draft that cannot read this target: it must be a DFlash GGUF
    /// as wide as the target that reads only layers the target has (layer
    /// `layer_count` is the last layer's output, the output head's input),
    /// and exactly the layers a package proposer declares.
    pub fn check_draft(&self, path: &Path) -> Result<()> {
        let target = &self.pairing.target;
        let Some(meta) = scan_gguf_compact_meta(path) else {
            bail!("{} is not a readable GGUF", path.display());
        };
        ensure!(
            meta.architecture == "dflash",
            "draft architecture is {}, expected dflash",
            meta.architecture
        );
        ensure!(
            meta.embedding_size == target.embedding_length,
            "draft hidden size {} does not match target hidden size {}",
            meta.embedding_size,
            target.embedding_length
        );
        let layers = scan_gguf_dflash_target_layers(path).unwrap_or_default();
        ensure!(!layers.is_empty(), "draft names no target layers");
        if let Some(layer) = layers.iter().find(|&&layer| layer > target.layer_count) {
            bail!(
                "draft reads target layer {layer}, but the target has {} layers",
                target.layer_count
            );
        }
        if let Some(declared) = &self.target_layers {
            ensure!(
                &layers == declared,
                "draft reads target layers {layers:?}, but the package declares {declared:?}"
            );
        }
        Ok(())
    }

    /// Refuses a draft whose vocabulary is not the target's: the draft
    /// proposes target token ids, and the runtime refuses the pair when it
    /// attaches. A vocabulary that cannot be read is left to that check.
    pub fn check_draft_vocabulary(&self, draft: &Path, target: &Path) -> Result<()> {
        let (Some(draft_tokens), Some(target_tokens)) = (
            scan_gguf_vocab_token_count(draft),
            scan_gguf_vocab_token_count(target),
        ) else {
            return Ok(());
        };
        ensure!(
            draft_tokens == target_tokens,
            "draft vocabulary has {draft_tokens} tokens but the target has {target_tokens}"
        );
        Ok(())
    }
}

/// Refuses a split final stage's draft unless it is a DFlash GGUF reading
/// exactly the target layers the plan forwards. A plan from a planner without
/// its own copy forwards no layers yet; the draft must then name some, and the
/// final stage reports them back to the planner.
pub fn check_split_dflash_draft(path: &Path, target_layers: &[u32]) -> Result<()> {
    let Some(meta) = scan_gguf_compact_meta(path) else {
        bail!("{} is not a readable GGUF", path.display());
    };
    ensure!(
        meta.architecture == "dflash",
        "draft architecture is {}, expected dflash",
        meta.architecture
    );
    let layers = scan_gguf_dflash_target_layers(path).unwrap_or_default();
    ensure!(!layers.is_empty(), "draft names no target layers");
    ensure!(
        target_layers.is_empty() || layers == target_layers,
        "draft reads target layers {layers:?}, but the plan forwards {target_layers:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        named, push_gguf_string, push_string_kv, push_u32_kv, target_meta,
    };
    use super::*;
    use std::io::Write;

    fn vocabulary_file(tokens: u64) -> tempfile::NamedTempFile {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        push_gguf_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&tokens.to_le_bytes());
        for token in 0..tokens {
            push_gguf_string(&mut bytes, &format!("t{token}"));
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file
    }

    #[test]
    fn a_draft_with_another_vocabulary_is_refused() {
        let pairing = catalog_default(bootstrap_pairings().remove(0));
        let target = vocabulary_file(5);
        let same = vocabulary_file(5);
        let other = vocabulary_file(6);
        let unreadable = tempfile::NamedTempFile::new().unwrap();

        pairing
            .check_draft_vocabulary(same.path(), target.path())
            .unwrap();
        let error = pairing
            .check_draft_vocabulary(other.path(), target.path())
            .unwrap_err();
        assert!(
            error.to_string().contains("6 tokens but the target has 5"),
            "{error}"
        );
        pairing
            .check_draft_vocabulary(unreadable.path(), target.path())
            .unwrap();
    }

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

    fn repo(meta: &GgufCompactMeta, metal: bool) -> Option<String> {
        pairing_in(&bootstrap_pairings(), meta, metal).map(|pairing| pairing.draft.repo.clone())
    }

    const INDEX_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/dflash-pairings/index.json"
    );
    const README_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/dflash-pairings/README.md"
    );

    /// The pin is the hash of the checked-in index, so a change to the
    /// pairings has to update [`DFLASH_PAIRING_INDEX_SHA256`] in the same
    /// change, and CI publishes exactly the bytes this release trusts.
    #[test]
    fn the_pin_is_the_hash_of_the_checked_in_index() {
        let bytes = std::fs::read(INDEX_PATH).unwrap();
        assert_eq!(
            DFLASH_PAIRING_INDEX_SHA256,
            sha256_hex(&bytes),
            "set DFLASH_PAIRING_INDEX_SHA256 to the sha256 of {INDEX_PATH}"
        );
        let pin = DFlashPairingIndexPin::shipped();
        assert_eq!(pin.repo, "meshllm/dflash-pairings");
        assert_eq!(
            pin.file(),
            format!("index/{DFLASH_PAIRING_INDEX_SHA256}.json")
        );
        assert_eq!(
            pin.load(Path::new(INDEX_PATH)).unwrap(),
            load_dflash_pairing_index(Path::new(INDEX_PATH)).unwrap()
        );
    }

    #[test]
    fn an_index_whose_bytes_do_not_hash_to_the_pin_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");
        let mut json = std::fs::read_to_string(INDEX_PATH).unwrap();
        // Still a valid index, but not the one the pin names.
        json.push('\n');
        std::fs::write(&path, &json).unwrap();
        assert!(load_dflash_pairing_index(&path).is_ok());
        let error = DFlashPairingIndexPin::shipped()
            .load(&path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected"), "{error}");
        assert!(error.contains(DFLASH_PAIRING_INDEX_SHA256), "{error}");
    }

    #[test]
    fn a_cached_index_is_found_by_its_hash_in_any_snapshot() {
        let pin = DFlashPairingIndexPin::shipped();
        let snapshots = tempfile::tempdir().unwrap();
        assert_eq!(pin.cached_path_in(&snapshots.path().join("missing")), None);
        // A snapshot without the file, and one (of a later commit) with it.
        std::fs::create_dir_all(snapshots.path().join("a".repeat(40)).join("index")).unwrap();
        let commit = snapshots.path().join("b".repeat(40));
        std::fs::create_dir_all(commit.join("index")).unwrap();
        assert_eq!(pin.cached_path_in(snapshots.path()), None);
        let path = commit.join(pin.file());
        std::fs::copy(INDEX_PATH, &path).unwrap();
        assert_eq!(pin.cached_path_in(snapshots.path()), Some(path.clone()));
        assert!(pin.load(&path).is_ok());
    }

    /// The pairings table in the Hugging Face README, rendered from the index.
    fn readme_table(index: &DFlashPairingIndex) -> String {
        let mut table =
            String::from("| Match | Draft | Cap | Backends |\n| --- | --- | --- | --- |\n");
        for pairing in &index.pairings {
            let target = &pairing.target;
            let mut shape = format!(
                "`{}`, {} layers, {} wide",
                target.architecture, target.layer_count, target.embedding_length
            );
            if target.expert_count > 0 {
                shape.push_str(&format!(", {} experts", target.expert_count));
            }
            if let Some(family) = &target.family {
                shape.push_str(&format!(", name carries `{family}`"));
            }
            let draft = &pairing.draft;
            let backends = match pairing.backends {
                DFlashBackends::All => "all",
                DFlashBackends::Metal => "Metal",
                DFlashBackends::NotMetal => "not Metal",
            };
            table.push_str(&format!(
                "| {shape} | [{repo}](https://huggingface.co/{repo}/blob/{revision}/{file}) `{file}` | {cap} | {backends} |\n",
                repo = draft.repo,
                revision = draft.revision,
                file = draft.file,
                cap = pairing.max_draft_tokens,
            ));
        }
        table
    }

    /// The README CI publishes next to the index lists its pairings.
    /// `UPDATE_DFLASH_PAIRINGS_README=1` rewrites the table from the index.
    #[test]
    fn the_readme_lists_the_checked_in_pairings() {
        const START: &str = "<!-- pairings:start -->\n";
        const END: &str = "<!-- pairings:end -->";
        let index = load_dflash_pairing_index(Path::new(INDEX_PATH)).unwrap();
        let readme = std::fs::read_to_string(README_PATH).unwrap();
        let (head, rest) = readme.split_once(START).expect("README start marker");
        let (_, tail) = rest.split_once(END).expect("README end marker");
        let expected = format!("{head}{START}{}{END}{tail}", readme_table(&index));
        if std::env::var_os("UPDATE_DFLASH_PAIRINGS_README").is_some() {
            std::fs::write(README_PATH, &expected).unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(README_PATH).unwrap(),
            expected,
            "regenerate with UPDATE_DFLASH_PAIRINGS_README=1 cargo test -p mesh-llm-skippy-adapter the_readme_lists"
        );
    }

    /// Every target the bootstrap table pairs by default, named as its GGUF
    /// names it: Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B (with its NextN block),
    /// Qwen3.6-35B-A3B, Qwen3-Coder-Next, Nemotron-3.5-Bolt-E4B and
    /// Nemotron-3.5-Lightning-30B-A3B (with its NextN block).
    fn default_targets() -> Vec<(&'static str, GgufCompactMeta)> {
        let mut qwen38 = named(target_meta("qwen35", 65, 5120, 0), "Qwen3.8-27B");
        qwen38.nextn_predict_layers = 1;
        let mut lightning = named(
            target_meta("nemotron_h_moe", 53, 2688, 128),
            "Nvidia-Nemotron-3.5-Lightning-30B-A3B",
        );
        lightning.nextn_predict_layers = 1;
        vec![
            (
                "Qwen3.5-9B",
                named(target_meta("qwen35", 32, 4096, 0), "Qwen3.5-9B"),
            ),
            (
                "Qwen3.6-27B",
                named(target_meta("qwen35", 64, 5120, 0), "Qwen3.6-27B"),
            ),
            ("Qwen3.8-27B", qwen38),
            (
                "Qwen3.6-35B-A3B",
                named(target_meta("qwen35moe", 40, 2048, 256), "Qwen3.6-35B-A3B"),
            ),
            (
                "Qwen3-Coder-Next",
                named(target_meta("qwen3next", 48, 2048, 512), "Qwen3-Coder-Next"),
            ),
            (
                "Nemotron-3.5-Bolt-E4B",
                named(
                    target_meta("gemma4", 42, 2560, 0),
                    "NVIDIA Nemotron 3.5 Bolt E4B",
                ),
            ),
            ("Nemotron-3.5-Lightning-30B-A3B", lightning),
        ]
    }

    /// The pinned index: the checked-in `index.json`, or the file
    /// `DFLASH_PAIRING_INDEX_FIXTURE` names (a fresh download, say).
    fn published_index() -> DFlashPairingIndex {
        let path = std::env::var_os("DFLASH_PAIRING_INDEX_FIXTURE")
            .map_or_else(|| PathBuf::from(INDEX_PATH), PathBuf::from);
        load_dflash_pairing_index(&path).unwrap()
    }

    #[test]
    fn the_published_index_pairs_every_default_target_as_the_table_does() {
        let index = published_index();
        let table = bootstrap_pairings();
        for (model, meta) in default_targets() {
            for metal in [true, false] {
                assert_eq!(
                    pairing_in(&index.pairings, &meta, metal),
                    pairing_in(&table, &meta, metal),
                    "{model}, metal = {metal}"
                );
                let _index = with_test_pairings(index.pairings.clone());
                let from_index = pairing_for_target(&meta, metal);
                drop(_index);
                let _table = with_test_pairings(table.clone());
                assert_eq!(
                    from_index,
                    pairing_for_target(&meta, metal),
                    "{model}, metal = {metal}"
                );
            }
        }

        let repo = |model: &str, metal| {
            let (_, meta) = default_targets()
                .into_iter()
                .find(|(name, _)| *name == model)
                .unwrap();
            pairing_in(&index.pairings, &meta, metal).map(|pairing| pairing.draft.repo.clone())
        };
        for metal in [true, false] {
            assert_eq!(
                repo("Qwen3.5-9B", metal).as_deref(),
                Some("AtomicChat/Qwen3.5-9B-DFlash-GGUF")
            );
            assert_eq!(
                repo("Qwen3.6-27B", metal).as_deref(),
                Some("AtomicChat/Qwen3.6-27B-DFlash-GGUF")
            );
            assert_eq!(
                repo("Qwen3.8-27B", metal).as_deref(),
                Some("z-lab/Qwen3.8-27B-DFlash2-GGUF")
            );
        }
        assert_eq!(
            repo("Qwen3.6-35B-A3B", false).as_deref(),
            Some("aminya/Qwen3.6-35B-A3B-DFlash2-GGUF")
        );
        assert_eq!(
            repo("Qwen3.6-35B-A3B", true).as_deref(),
            Some("Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp")
        );
        assert_eq!(
            repo("Qwen3-Coder-Next", true).as_deref(),
            Some("AtomicChat/Qwen3-Coder-Next-DFlash-GGUF")
        );
        assert_eq!(repo("Qwen3-Coder-Next", false), None);
        for metal in [true, false] {
            assert_eq!(
                repo("Nemotron-3.5-Bolt-E4B", metal).as_deref(),
                Some("unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF")
            );
        }
        assert_eq!(
            repo("Nemotron-3.5-Lightning-30B-A3B", false).as_deref(),
            Some("magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF")
        );
        // Native MTP stays Lightning's default on Metal.
        assert_eq!(repo("Nemotron-3.5-Lightning-30B-A3B", true), None);
    }

    #[test]
    fn nemotron_lightning_is_told_from_nemotron_3_nano_by_name() {
        const DSPARK: &str = "magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF";
        let with_mtp = |name: &str| {
            let mut meta = named(target_meta("nemotron_h_moe", 53, 2688, 128), name);
            meta.nextn_predict_layers = 1;
            meta
        };
        for name in [
            "Nvidia-Nemotron-3.5-Lightning-30B-A3B",
            "NVIDIA Nemotron 3.5 Lightning 30B A3B",
        ] {
            assert_eq!(
                repo(&with_mtp(name), false).as_deref(),
                Some(DSPARK),
                "{name}"
            );
            assert_eq!(repo(&with_mtp(name), true), None, "{name}");
        }
        // Nemotron-3-Nano-30B-A3B has the same executable shape.
        let nano = named(
            target_meta("nemotron_h_moe", 52, 2688, 128),
            "Nemotron-3-Nano-30B-A3B",
        );
        assert_eq!(repo(&nano, false), None);
        assert_eq!(repo(&nano, true), None);
    }

    #[test]
    fn nemotron_bolt_is_told_from_gemma_4_by_name() {
        const BOLT: &str = "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF";
        let gemma4 = target_meta("gemma4", 42, 2560, 0);
        // As unsloth, bartowski, LM Studio and NVIDIA name their quants.
        for name in [
            "NVIDIA Nemotron 3.5 Bolt E4B",
            "NVIDIA-Nemotron-3.5-Bolt-E4B",
            "Nemotron 3.5 Bolt E4B",
        ] {
            for metal in [true, false] {
                assert_eq!(
                    repo(&named(gemma4.clone(), name), metal).as_deref(),
                    Some(BOLT),
                    "{name}"
                );
            }
        }
        // Gemma-4-E4B-It has the same shape; the Bolt draft is not its default.
        assert_eq!(repo(&named(gemma4.clone(), "Gemma-4-E4B-It"), false), None);
        assert_eq!(repo(&gemma4, false), None);
    }

    #[test]
    fn targets_match_by_metadata_whatever_the_quant() {
        assert_eq!(
            repo(&target_meta("qwen35", 32, 4096, 0), false).as_deref(),
            Some("AtomicChat/Qwen3.5-9B-DFlash-GGUF")
        );
        assert_eq!(
            repo(&target_meta("qwen35", 64, 5120, 0), false).as_deref(),
            Some("AtomicChat/Qwen3.6-27B-DFlash-GGUF")
        );
        assert_eq!(
            repo(&target_meta("qwen35moe", 40, 2048, 256), false).as_deref(),
            Some("aminya/Qwen3.6-35B-A3B-DFlash2-GGUF")
        );
        // A NextN block appended to the trunk is not a decoder layer.
        let mut with_mtp = target_meta("qwen35", 33, 4096, 0);
        with_mtp.nextn_predict_layers = 1;
        assert!(repo(&with_mtp, false).is_some());
    }

    #[test]
    fn qwen38_and_qwen36_27b_share_a_shape_and_are_told_apart_by_name() {
        const QWEN36: &str = "AtomicChat/Qwen3.6-27B-DFlash-GGUF";
        const QWEN38: &str = "z-lab/Qwen3.8-27B-DFlash2-GGUF";
        let with_mtp = |mut meta: GgufCompactMeta| {
            meta.layer_count += 1;
            meta.nextn_predict_layers = 1;
            meta
        };
        let dense = target_meta("qwen35", 64, 5120, 0);

        let qwen38 = with_mtp(named(dense.clone(), "Qwen3.8-27B"));
        for metal in [true, false] {
            assert_eq!(repo(&qwen38, metal).as_deref(), Some(QWEN38));
        }
        let pairings = bootstrap_pairings();
        assert_eq!(
            pairing_in(&pairings, &qwen38, false)
                .unwrap()
                .max_draft_tokens,
            7
        );

        for meta in [
            dense.clone(),
            named(dense.clone(), "Qwen3.6-27B"),
            with_mtp(named(dense.clone(), "Qwen3.6-27B")),
            named(dense.clone(), "Huihui-Qwen3.6-27B-abliterated"),
            named(dense.clone(), "Qwen3.8x-27B"),
        ] {
            assert_eq!(repo(&meta, false).as_deref(), Some(QWEN36), "{meta:?}");
        }

        // Only `general.name` set: still Qwen3.8, never the Qwen3.6 draft.
        let mut name_only = dense.clone();
        name_only.name = Some("qwen3.8 27B Instruct".to_string());
        assert_eq!(repo(&name_only, false).as_deref(), Some(QWEN38));
        // `general.basename` wins over `general.name`.
        let mut basename_first = named(dense, "Qwen3.6-27B");
        basename_first.name = Some("Qwen3.8-27B".to_string());
        assert_eq!(repo(&basename_first, false).as_deref(), Some(QWEN36));
    }

    #[test]
    fn family_names_match_whole_tokens() {
        for (name, family, expected) in [
            ("Qwen3.8-27B", "Qwen3.8", true),
            ("unsloth/qwen3.8-27b", "Qwen3.8", true),
            ("Huihui-Qwen3.6-27B-abliterated", "Qwen3.6", true),
            ("Qwen3.8 27B", "Qwen3.8", true),
            ("Qwen3.8x-27B", "Qwen3.8", false),
            ("Qwen3.80-27B", "Qwen3.8", false),
            ("Qwen3.8.1-27B", "Qwen3.8", false),
            ("MyQwen3.8-27B", "Qwen3.8", false),
            ("Qwen3.6-27B", "Qwen3.8", false),
            ("", "Qwen3.8", false),
            ("NVIDIA Nemotron 3.5 Bolt E4B", "Nemotron-3.5-Bolt", true),
            ("Nemotron 3.5 Bolt E4B", "Nemotron-3.5-Bolt", true),
            (
                "Nvidia-Nemotron-3.5-Lightning-30B-A3B",
                "Nemotron-3.5-Lightning",
                true,
            ),
            ("nemotron_3.5_lightning", "Nemotron-3.5-Lightning", true),
            ("Nemotron-3-Nano-30B-A3B", "Nemotron-3.5-Lightning", false),
            ("Gemma-4-E4B-It", "Nemotron-3.5-Bolt", false),
        ] {
            assert_eq!(names_family(name, family), expected, "{name}");
        }
    }

    #[test]
    fn drafts_are_chosen_per_backend() {
        let coder = target_meta("qwen3next", 48, 2048, 512);
        assert_eq!(
            repo(&coder, true).as_deref(),
            Some("AtomicChat/Qwen3-Coder-Next-DFlash-GGUF")
        );
        assert_eq!(repo(&coder, false), None);
        let moe = target_meta("qwen35moe", 40, 2048, 256);
        assert_eq!(
            repo(&moe, false).as_deref(),
            Some("aminya/Qwen3.6-35B-A3B-DFlash2-GGUF")
        );
        assert_eq!(
            repo(&moe, true).as_deref(),
            Some("Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp")
        );
        for dense in [
            target_meta("qwen35", 32, 4096, 0),
            target_meta("qwen35", 64, 5120, 0),
        ] {
            assert!(repo(&dense, true).is_some() && repo(&dense, false).is_some());
        }
    }

    #[test]
    fn other_shapes_of_the_same_family_do_not_match() {
        for meta in [
            target_meta("qwen35", 48, 5120, 0),
            target_meta("qwen35", 32, 2048, 0),
            target_meta("qwen35moe", 40, 2048, 128),
            target_meta("qwen3", 32, 4096, 0),
        ] {
            assert_eq!(repo(&meta, false), None, "{meta:?}");
            assert_eq!(repo(&meta, true), None, "{meta:?}");
        }
    }

    #[test]
    fn every_draft_is_pinned_to_a_commit_and_digest() {
        DFlashPairingIndex {
            schema_version: skippy_package_format::DFLASH_PAIRING_INDEX_SCHEMA_VERSION,
            pairings: bootstrap_pairings(),
        }
        .validate()
        .unwrap();
        for pairing in bootstrap_pairings() {
            let draft = &pairing.draft;
            assert!(
                dflash_draft_cache_path(draft)
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
    fn every_draft_cap_fits_its_block() {
        let pairings = bootstrap_pairings();
        let moe = pairing_in(&pairings, &target_meta("qwen35moe", 40, 2048, 256), false).unwrap();
        assert_eq!(moe.max_draft_tokens, 7);
        for pairing in &pairings {
            assert!((1..=15).contains(&pairing.max_draft_tokens));
        }
    }

    fn catalog_default(pairing: DFlashPairing) -> DefaultDFlash {
        DefaultDFlash {
            pairing,
            target_layers: None,
            source: DFlashPairingSource::Catalog,
        }
    }

    #[test]
    fn a_draft_must_fit_its_target() {
        let mut pairing = catalog_default(bootstrap_pairings().remove(0));
        let fits = draft_file("dflash", 4096, &[2, 6, 10, 14, 18, 22, 26, 30]);
        pairing.check_draft(fits.path()).unwrap();

        for (draft, reason) in [
            (draft_file("qwen35", 4096, &[2, 30]), "architecture"),
            (draft_file("dflash", 2048, &[2, 30]), "hidden size"),
            (draft_file("dflash", 4096, &[2, 33]), "target layer 33"),
            (draft_file("dflash", 4096, &[]), "no target layers"),
        ] {
            let error = pairing.check_draft(draft.path()).unwrap_err().to_string();
            assert!(error.contains(reason), "{error}");
        }

        // Layer 32 of a 32-layer target is its last layer's output, the
        // output head's input, which Nemotron-3.5 drafts read.
        let head_input = draft_file("dflash", 4096, &[2, 32]);
        pairing.check_draft(head_input.path()).unwrap();

        // A package proposer's declared target layers must be the draft's.
        pairing.target_layers = Some(vec![2, 30]);
        let error = pairing.check_draft(fits.path()).unwrap_err().to_string();
        assert!(error.contains("package declares"), "{error}");
        pairing.target_layers = Some(vec![2, 6, 10, 14, 18, 22, 26, 30]);
        pairing.check_draft(fits.path()).unwrap();
    }

    #[test]
    fn a_split_draft_must_read_the_layers_the_plan_forwards() {
        let draft = draft_file("dflash", 5120, &[2, 62]);
        check_split_dflash_draft(draft.path(), &[2, 62]).unwrap();
        assert!(check_split_dflash_draft(draft.path(), &[2, 30]).is_err());
        let ordinary = draft_file("qwen35", 5120, &[2, 62]);
        assert!(check_split_dflash_draft(ordinary.path(), &[2, 62]).is_err());
        // A planner without its own copy forwards no layers; the final stage
        // reports the draft's.
        check_split_dflash_draft(draft.path(), &[]).unwrap();
        let no_layers = draft_file("dflash", 5120, &[]);
        assert!(check_split_dflash_draft(no_layers.path(), &[]).is_err());
    }

    fn index_pairing(repo: &str, backends: DFlashBackends) -> DFlashPairing {
        let mut pairing = bootstrap_pairings().remove(0);
        pairing.draft.repo = repo.to_string();
        pairing.backends = backends;
        pairing
    }

    #[test]
    fn an_index_entry_is_used_for_a_plain_gguf() {
        let _pairings = with_test_pairings(vec![index_pairing(
            "meshllm/indexed-draft",
            DFlashBackends::All,
        )]);
        let chosen = pairing_for_target(&target_meta("qwen35", 32, 4096, 0), false).unwrap();
        assert_eq!(chosen.pairing.draft.repo, "meshllm/indexed-draft");
        assert_eq!(chosen.source, DFlashPairingSource::Catalog);
        // The index replaces the bootstrap table outright.
        assert!(pairing_for_target(&target_meta("qwen35", 64, 5120, 0), false).is_none());
    }

    #[test]
    fn an_index_backend_policy_is_respected() {
        let _pairings = with_test_pairings(vec![
            index_pairing("meshllm/metal-draft", DFlashBackends::Metal),
            index_pairing("meshllm/cuda-draft", DFlashBackends::NotMetal),
        ]);
        let meta = target_meta("qwen35", 32, 4096, 0);
        let repo = |metal| pairing_for_target(&meta, metal).map(|chosen| chosen.pairing.draft.repo);
        assert_eq!(repo(true).as_deref(), Some("meshllm/metal-draft"));
        assert_eq!(repo(false).as_deref(), Some("meshllm/cuda-draft"));

        let _pairings = with_test_pairings(vec![index_pairing(
            "meshllm/metal-draft",
            DFlashBackends::Metal,
        )]);
        assert!(pairing_for_target(&meta, false).is_none());
    }

    #[test]
    fn a_cached_index_loads_from_the_hugging_face_cache_offline() {
        let cache = tempfile::tempdir().unwrap();
        let path = cache
            .path()
            .join("models--meshllm--dflash-pairings/snapshots/0123456789abcdef0123456789abcdef01234567/index.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut index = DFlashPairingIndex {
            schema_version: 1,
            pairings: vec![index_pairing("meshllm/cached-draft", DFlashBackends::All)],
        };
        std::fs::write(&path, serde_json::to_string(&index).unwrap()).unwrap();
        let loaded = load_dflash_pairing_index(&path).unwrap();
        assert_eq!(loaded, index);
        let pairings = loaded.pairings;
        assert_eq!(
            pairing_in(&pairings, &target_meta("qwen35", 32, 4096, 0), false)
                .map(|pairing| pairing.draft.repo.as_str()),
            Some("meshllm/cached-draft")
        );

        // An index with a malformed digest is refused whole.
        index.pairings[0].draft.sha256 = "not-a-digest".to_string();
        std::fs::write(&path, serde_json::to_string(&index).unwrap()).unwrap();
        let error = load_dflash_pairing_index(&path).unwrap_err().to_string();
        assert!(error.contains("sha256"), "{error}");
    }

    #[test]
    fn the_index_is_pinned_by_a_lowercase_sha256() {
        assert_eq!(DFLASH_PAIRING_INDEX_SHA256.len(), 64);
        assert!(
            DFLASH_PAIRING_INDEX_SHA256
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        );
        assert_eq!(DFlashPairingIndexPin::REVISION, "main");
    }
}
