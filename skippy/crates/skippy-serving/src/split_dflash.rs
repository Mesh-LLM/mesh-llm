//! DFlash speculative decoding across a split target.
//!
//! The draft runs on the final stage, which owns the output head it borrows
//! and already returns native MTP drafts on its replies. Every stage that owns
//! one of the draft's target layers captures that layer's input and forwards
//! it with its activations (see `skippy_runtime::DFlashFeatureRows`). Stage 0
//! drives the decode loop as it does for split native MTP: it sends drafted
//! blocks as verify windows, commits through the first mismatch, and rolls
//! every stage back to the committed position. The final stage drafts the next
//! block right after each decode or verify and returns it on the reply. On a
//! verify window it answers with the verdict first and sends the draft in a
//! second reply, so stage 0 commits and streams the window's tokens while the
//! draft runs (`SKIPPY_DFLASH_EARLY_VERDICT=0` turns that off). Unless
//! `SKIPPY_DFLASH_ADAPTIVE=0`, stage 0 verifies as much of each block as its
//! measured window and serial-step times favour, down to a serial step.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use skippy_protocol::StageConfig;
use skippy_runtime::DFlashDraftOptions;

use crate::frontend::{DFlashProposalConfig, SpeculativeDecodeConfig};
use crate::runtime_state::{RuntimeLaunchOverrides, RuntimeState};

/// Draft tokens in a trained 16-token block, the most a plan without a cap
/// can propose.
const UNCAPPED_DRAFT_TOKENS: usize = 15;
/// The wire carries a requested draft depth in four bits.
const MAX_WIRE_DRAFT_TOKENS: usize = 15;

/// One stage's part in a split DFlash plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitDFlashStage {
    /// Every target layer the draft reads, across all stages.
    pub target_layers: Vec<u32>,
    /// The draft, on the final stage only. A path on the planning node; a
    /// stage on another node only finds it there if `draft_source` is unset.
    pub draft_model_path: Option<PathBuf>,
    pub max_draft_tokens: Option<usize>,
    pub draft_n_gpu_layers: Option<i32>,
    /// The backend device the configured draft runs on, on the final stage
    /// only. `None` puts it on the target's device.
    pub draft_device: Option<String>,
    /// Where a catalog default draft comes from, on the final stage only, so
    /// that stage fetches and verifies its own copy. `None` for a configured
    /// draft, which the operator places, and from older planners.
    pub draft_source: Option<SplitDFlashDraftSource>,
}

/// A default DFlash draft pinned on Hugging Face.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitDFlashDraftSource {
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub sha256: String,
}

impl SplitDFlashStage {
    /// The plan for a stage of a split target served with `dflash`: the
    /// draft's target layers as the plan names them, else from its header.
    pub fn from_draft(
        dflash: &DFlashProposalConfig,
        final_stage: bool,
        draft_n_gpu_layers: Option<i32>,
    ) -> Result<Self> {
        let target_layers = if dflash.target_layers.is_empty() {
            draft_target_layers(&dflash.draft_model_path)?
        } else {
            dflash.target_layers.clone()
        };
        Ok(Self {
            target_layers,
            draft_model_path: final_stage.then(|| dflash.draft_model_path.clone()),
            max_draft_tokens: dflash.max_draft_tokens,
            draft_n_gpu_layers,
            draft_device: None,
            draft_source: None,
        })
    }

    /// Whether this stage drafts: the final stage with a draft to load.
    pub fn drafts(&self) -> bool {
        self.draft_model_path.is_some()
    }

    /// Recurrent rollback depth every stage needs to verify the longest block
    /// in place: every drafted token plus the anchor.
    pub fn recurrent_rollback_tokens(&self) -> u32 {
        let drafted = self.max_draft_tokens.unwrap_or(UNCAPPED_DRAFT_TOKENS);
        u32::try_from(drafted.saturating_add(1)).unwrap_or(u32::MAX)
    }
}

/// The target layers a DFlash draft reads, from its GGUF header.
pub fn draft_target_layers(path: &Path) -> Result<Vec<u32>> {
    let layers = skippy_model_artifact::gguf::scan_gguf_dflash_target_layers(path)
        .with_context(|| format!("read DFlash target layers from {}", path.display()))?;
    if layers.is_empty() {
        bail!("DFlash draft {} names no target layers", path.display());
    }
    Ok(layers)
}

/// The output head of an untied target, a separate tensor from its token
/// embeddings.
const UNTIED_OUTPUT_TENSOR: &str = "output.weight";

/// Adds the target tensors the final stage's draft borrows to the tensors the
/// stage loads, and declares them in its execution contract as borrowed:
/// resident, though no stage graph reads them. A stage slice loads only what
/// its own layers read, and the token embeddings of an untied target live
/// with stage 0: without this the draft embeds every block from memory
/// nothing loaded and drafts noise. A tied target's final stage already reads
/// them for its output head. Other stages, and stages that load every tensor,
/// are left alone. Returns whether the config changed.
pub(crate) fn load_borrowed_target_tensors(
    config: &mut StageConfig,
    plan: &SplitDFlashStage,
) -> Result<bool> {
    if config.downstream.is_some()
        || !plan.drafts()
        || !config
            .resident_tensor_names
            .iter()
            .any(|name| name == UNTIED_OUTPUT_TENSOR)
    {
        return Ok(false);
    }
    let tensor = skippy_runtime::DFLASH_BORROWED_TARGET_TENSOR;
    let mut changed = false;
    if let Err(index) = config
        .resident_tensor_names
        .binary_search_by(|name| name.as_str().cmp(tensor))
    {
        config
            .resident_tensor_names
            .insert(index, tensor.to_string());
        changed = true;
    }
    let mut contract: serde_json::Value = serde_json::from_str(&config.execution_contract)
        .context("parse the stage execution contract")?;
    let names = contract
        .as_object_mut()
        .context("the stage execution contract is not an object")?
        .entry("borrowed")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()))
        .as_array_mut()
        .context("the contract's borrowed tensors are not an array")?;
    if !names.iter().any(|name| name.as_str() == Some(tensor)) {
        names.push(serde_json::Value::String(tensor.to_string()));
        names.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
        config.execution_contract = contract.to_string();
        changed = true;
    }
    Ok(changed)
}

/// The package tensor ids a drafting final stage admits: `resident` plus the
/// target tensors its draft borrows, so the stage fetches the artifacts that
/// hold them. Package-v2 tensor ids are the tensors' GGUF names. `None` when
/// nothing is added: another stage, a draft-less plan, a stage that loads
/// every tensor, or a tied target whose output head already reads them.
pub fn borrowed_resident_tensor_ids(
    plan: &SplitDFlashStage,
    final_stage: bool,
    resident: &[String],
) -> Option<Vec<String>> {
    let tensor = skippy_runtime::DFLASH_BORROWED_TARGET_TENSOR;
    if !final_stage
        || !plan.drafts()
        || !resident.iter().any(|id| id == UNTIED_OUTPUT_TENSOR)
        || resident.iter().any(|id| id == tensor)
    {
        return None;
    }
    let mut ids = resident.to_vec();
    ids.push(tensor.to_string());
    ids.sort();
    Some(ids)
}

/// Attaches this stage's part of a split DFlash plan before any serving
/// session exists: the draft on the final stage, target-layer capture on the
/// others. A final stage without a usable draft serves without one; stage 0
/// then receives no drafts and decodes serially.
///
/// Returns the draft's target layers once a final stage has attached it, so
/// the planner can tell stage 0 the draft is there; `None` otherwise.
pub(crate) fn attach_split_dflash(
    runtime: &mut RuntimeState,
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
    plan: &SplitDFlashStage,
) -> Result<Option<Vec<u32>>> {
    if config.downstream.is_none() {
        let Some(path) = plan.draft_model_path.as_deref() else {
            return Ok(None);
        };
        if !path.is_file() {
            tracing::warn!(
                draft = %path.display(),
                "split DFlash draft is not on this node; serving without speculation"
            );
            return Ok(None);
        }
        let target_config =
            crate::runtime_state::runtime_config_from_stage_config(config, overrides)?;
        let options = DFlashDraftOptions {
            max_draft_tokens: plan.max_draft_tokens,
            n_gpu_layers: plan.draft_n_gpu_layers,
            device: draft_device_on_this_node(plan.draft_device.as_deref()),
        };
        let info = runtime
            .model
            .attach_dflash_draft(path, &options, &target_config)
            .with_context(|| format!("attach split DFlash draft {}", path.display()))?;
        tracing::info!(
            stage = config.stage_id,
            variant = info.variant.as_str(),
            max_draft_tokens = info.max_draft_tokens,
            target_layers = ?info.target_layer_ids,
            "split DFlash draft attached on the final stage"
        );
        return Ok(Some(info.target_layer_ids.clone()));
    }
    let layers = runtime.model.attach_dflash_capture(
        &plan.target_layers,
        config.layer_start,
        config.layer_end,
    )?;
    tracing::info!(
        stage = config.stage_id,
        layers = ?layers,
        "split DFlash target-layer capture attached"
    );
    Ok(None)
}

/// The configured draft device when this node has it. The device is named on
/// the planning node; a final stage on another node may not have it, and then
/// puts the draft on the target's device rather than failing to attach.
fn draft_device_on_this_node(device: Option<&str>) -> Option<String> {
    let device = device?;
    match skippy_runtime::backend_devices() {
        Ok(devices) if devices.iter().any(|known| known.name == device) => Some(device.to_string()),
        Ok(devices) => {
            tracing::warn!(
                device,
                available = ?devices.iter().map(|known| known.name.as_str()).collect::<Vec<_>>(),
                "split DFlash draft device is not on this node; placing the draft with the target"
            );
            None
        }
        Err(error) => {
            tracing::warn!(
                device,
                error = format!("{error:#}"),
                "cannot list this node's devices; placing the split DFlash draft with the target"
            );
            None
        }
    }
}

/// Stage 0's decode plan for a split DFlash target: the final stage's DFlash
/// drafts arrive where native MTP drafts do, so stage 0 drives them through
/// the native MTP verify windows. Each window verifies the whole block.
/// `None` when the plan has no DFlash draft.
pub(crate) fn split_stage0_plan(plan: &SpeculativeDecodeConfig) -> Option<SpeculativeDecodeConfig> {
    let dflash = plan.dflash.as_ref()?;
    let block = dflash
        .max_draft_tokens
        .unwrap_or(UNCAPPED_DRAFT_TOKENS)
        .clamp(1, MAX_WIRE_DRAFT_TOKENS);
    let mut stage0 = plan.clone();
    stage0.dflash = None;
    stage0.ngram = None;
    stage0.extension = None;
    stage0.native_mtp.enabled = true;
    stage0.native_mtp.max_draft_tokens = block;
    stage0.native_mtp.min_draft_tokens = 1;
    stage0.native_mtp.reject_cooldown_tokens = 0;
    stage0.native_mtp.suppress_cooldown_drafts = false;
    stage0.verify_window.min_tokens = block;
    stage0.verify_window.max_tokens = block;
    Some(stage0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dflash_plan(cap: Option<usize>) -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            requested_strategy: "dflash".to_string(),
            effective_strategy: "dflash".to_string(),
            dflash: Some(DFlashProposalConfig {
                draft_model_path: PathBuf::from("/models/draft-dflash.gguf"),
                max_draft_tokens: cap,
                target_layers: Vec::new(),
            }),
            ..SpeculativeDecodeConfig::default()
        }
    }

    #[test]
    fn stage0_verifies_whole_blocks_through_native_mtp_windows() {
        let plan = split_stage0_plan(&dflash_plan(Some(7))).unwrap();
        assert!(plan.native_mtp.enabled);
        assert!(plan.dflash.is_none());
        assert_eq!(plan.native_mtp.max_draft_tokens, 7);
        assert_eq!(plan.native_mtp.min_draft_tokens, 1);
        assert_eq!(
            (plan.verify_window.min_tokens, plan.verify_window.max_tokens),
            (7, 7)
        );
    }

    #[test]
    fn stage0_block_fits_the_wire_depth() {
        let uncapped = split_stage0_plan(&dflash_plan(None)).unwrap();
        assert_eq!(uncapped.native_mtp.max_draft_tokens, 15);
        let wide = split_stage0_plan(&dflash_plan(Some(31))).unwrap();
        assert_eq!(wide.native_mtp.max_draft_tokens, 15);
    }

    #[test]
    fn a_plan_without_dflash_is_left_alone() {
        assert!(split_stage0_plan(&SpeculativeDecodeConfig::default()).is_none());
    }

    fn stage_config(final_stage: bool, resident: &[&str]) -> StageConfig {
        StageConfig {
            layer_start: 32,
            layer_end: 64,
            resident_tensor_names: resident.iter().map(|name| name.to_string()).collect(),
            execution_contract: serde_json::json!({
                "version": 1,
                "layer_start": 32,
                "layer_end": 64,
                "decoder": resident.iter().filter(|name| name.starts_with("blk.")).collect::<Vec<_>>(),
                "mtp": [],
            })
            .to_string(),
            downstream: (!final_stage).then(|| skippy_protocol::PeerConfig {
                stage_id: "stage-2".to_string(),
                stage_index: 2,
                endpoint: "tcp://127.0.0.1:1".to_string(),
                persistent_lane_marker: false,
            }),
            ..StageConfig::default()
        }
    }

    fn draft_plan() -> SplitDFlashStage {
        SplitDFlashStage {
            target_layers: vec![2, 17, 32, 47, 62],
            draft_model_path: Some(PathBuf::from("/models/draft-dflash.gguf")),
            ..SplitDFlashStage::default()
        }
    }

    #[test]
    fn a_drafting_final_stage_loads_the_token_embeddings() {
        // The final stage of an untied target, cut at 32, as the stage
        // planner admits it: its layers and the output head only.
        let mut config = stage_config(true, &["blk.32.attn_norm.weight", "output.weight"]);
        assert!(load_borrowed_target_tensors(&mut config, &draft_plan()).unwrap());
        assert_eq!(
            config.resident_tensor_names,
            [
                "blk.32.attn_norm.weight",
                "output.weight",
                "token_embd.weight"
            ]
        );
        // The native loader admits a resident no stage graph reads only when
        // the contract declares it borrowed.
        let contract: serde_json::Value = serde_json::from_str(&config.execution_contract).unwrap();
        assert_eq!(
            contract["borrowed"],
            serde_json::json!(["token_embd.weight"])
        );
        assert_eq!(
            contract["decoder"],
            serde_json::json!(["blk.32.attn_norm.weight"])
        );
        // A tied target's final stage already holds it.
        assert!(!load_borrowed_target_tensors(&mut config, &draft_plan()).unwrap());
        assert_eq!(config.resident_tensor_names.len(), 3);
    }

    #[test]
    fn stages_without_a_draft_load_only_their_own_tensors() {
        let mut middle = stage_config(false, &["blk.32.attn_norm.weight"]);
        assert!(!load_borrowed_target_tensors(&mut middle, &draft_plan()).unwrap());
        assert_eq!(middle.resident_tensor_names, ["blk.32.attn_norm.weight"]);

        let mut no_draft = stage_config(true, &["output.weight"]);
        let plan = SplitDFlashStage {
            draft_model_path: None,
            ..draft_plan()
        };
        assert!(!load_borrowed_target_tensors(&mut no_draft, &plan).unwrap());
        assert_eq!(no_draft.resident_tensor_names, ["output.weight"]);

        // An unfiltered stage already loads everything.
        let mut whole = stage_config(true, &[]);
        assert!(!load_borrowed_target_tensors(&mut whole, &draft_plan()).unwrap());
        assert!(whole.resident_tensor_names.is_empty());
    }

    #[test]
    fn a_tied_target_loads_nothing_extra() {
        // No separate output head: the final stage reads the embeddings for it.
        let mut tied = stage_config(true, &["blk.32.attn_norm.weight", "token_embd.weight"]);
        let before = tied.clone();
        assert!(!load_borrowed_target_tensors(&mut tied, &draft_plan()).unwrap());
        assert_eq!(tied, before);
    }

    #[test]
    fn admitted_embeddings_of_an_untied_target_are_still_declared_borrowed() {
        // The package closure admitted the embeddings so the stage fetched
        // them; no stage graph reads them, so the contract must say so.
        let mut config = stage_config(
            true,
            &[
                "blk.32.attn_norm.weight",
                "output.weight",
                "token_embd.weight",
            ],
        );
        assert!(load_borrowed_target_tensors(&mut config, &draft_plan()).unwrap());
        let contract: serde_json::Value = serde_json::from_str(&config.execution_contract).unwrap();
        assert_eq!(
            contract["borrowed"],
            serde_json::json!(["token_embd.weight"])
        );
        assert_eq!(config.resident_tensor_names.len(), 3);
    }

    #[test]
    fn a_drafting_final_stage_fetches_the_artifacts_of_borrowed_tensors() {
        let resident = ["blk.32.attn_norm.weight", "output.weight"].map(String::from);
        assert_eq!(
            borrowed_resident_tensor_ids(&draft_plan(), true, &resident),
            Some(
                [
                    "blk.32.attn_norm.weight",
                    "output.weight",
                    "token_embd.weight"
                ]
                .map(String::from)
                .to_vec()
            )
        );
        // Not a drafting final stage, or already admitted, or tied.
        assert_eq!(
            borrowed_resident_tensor_ids(&draft_plan(), false, &resident),
            None
        );
        let no_draft = SplitDFlashStage {
            draft_model_path: None,
            ..draft_plan()
        };
        assert_eq!(
            borrowed_resident_tensor_ids(&no_draft, true, &resident),
            None
        );
        let admitted = ["output.weight", "token_embd.weight"].map(String::from);
        assert_eq!(
            borrowed_resident_tensor_ids(&draft_plan(), true, &admitted),
            None
        );
        let tied = ["blk.32.attn_norm.weight"].map(String::from);
        assert_eq!(
            borrowed_resident_tensor_ids(&draft_plan(), true, &tied),
            None
        );
    }

    #[test]
    fn a_plan_names_its_target_layers_without_the_draft_file() {
        let mut dflash = DFlashProposalConfig {
            draft_model_path: PathBuf::from("/not/on/this/node.gguf"),
            max_draft_tokens: Some(4),
            target_layers: vec![2, 17, 32],
        };
        let plan = SplitDFlashStage::from_draft(&dflash, false, None).unwrap();
        assert_eq!(plan.target_layers, vec![2, 17, 32]);
        assert_eq!(plan.draft_model_path, None);
        dflash.target_layers.clear();
        assert!(SplitDFlashStage::from_draft(&dflash, false, None).is_err());
    }

    #[test]
    fn a_draft_device_this_node_lacks_falls_back_to_the_target_device() {
        assert_eq!(draft_device_on_this_node(None), None);
        assert_eq!(draft_device_on_this_node(Some("NoSuchDevice9")), None);
    }

    #[test]
    fn rollback_depth_covers_the_block_and_the_anchor() {
        let stage = |cap| SplitDFlashStage {
            max_draft_tokens: cap,
            ..SplitDFlashStage::default()
        };
        assert_eq!(stage(Some(7)).recurrent_rollback_tokens(), 8);
        assert_eq!(stage(None).recurrent_rollback_tokens(), 16);
    }
}
