use anyhow::{Context, Result, bail};
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::config::load_json;
use crate::frontend::util::openai_backend_error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeDecodeConfig {
    pub requested_strategy: String,
    pub effective_strategy: String,
    pub native_mtp: NativeMtpProposalConfig,
    pub ngram: Option<NgramProposalConfig>,
    pub extension: Option<NgramExtensionConfig>,
    pub verify_window: VerifyWindowConfig,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMtpProposalConfig {
    pub enabled: bool,
    pub max_draft_tokens: usize,
    pub min_draft_tokens: usize,
    pub reject_cooldown_tokens: usize,
    pub suppress_cooldown_drafts: bool,
    pub suppress_cooldown_draft_limit: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NgramProposerKind {
    Simple,
    Cache,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NgramProposalConfig {
    pub kind: NgramProposerKind,
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub max_proposal_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NgramExtensionConfig {
    pub initial_tokens: usize,
    pub max_tokens: usize,
    pub tail_backoff_proposals: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyWindowConfig {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub pipeline_depth: usize,
}

impl Default for SpeculativeDecodeConfig {
    fn default() -> Self {
        Self {
            requested_strategy: "auto".to_string(),
            effective_strategy: "disabled".to_string(),
            native_mtp: NativeMtpProposalConfig {
                enabled: false,
                max_draft_tokens: 1,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
            },
            ngram: None,
            extension: None,
            verify_window: VerifyWindowConfig {
                min_tokens: 1,
                max_tokens: 4,
                pipeline_depth: 1,
            },
        }
    }
}

impl SpeculativeDecodeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.requested_strategy.trim().is_empty() || self.effective_strategy.trim().is_empty() {
            bail!("speculative decode strategies must not be empty");
        }
        if self.native_mtp.min_draft_tokens > self.native_mtp.max_draft_tokens {
            bail!("native MTP min_draft_tokens must not exceed max_draft_tokens");
        }
        if let Some(ngram) = &self.ngram
            && (ngram.min_ngram == 0
                || ngram.min_ngram > ngram.max_ngram
                || ngram.max_proposal_tokens < ngram.min_ngram)
        {
            bail!(
                "N-gram proposer requires 0 < min_ngram <= max_ngram and max_proposal_tokens >= min_ngram"
            );
        }
        if let Some(ngram) = &self.ngram
            && ngram.kind == NgramProposerKind::Cache
            && ngram.max_ngram > skippy_runtime::NGRAM_CACHE_MAX_NGRAM
        {
            bail!(
                "cache N-gram proposer max_ngram must not exceed llama.cpp limit {}",
                skippy_runtime::NGRAM_CACHE_MAX_NGRAM
            );
        }
        if self.extension.is_some() && (!self.native_mtp.enabled || self.ngram.is_none()) {
            bail!("N-gram extension requires both native MTP and an N-gram proposer");
        }
        if let Some(extension) = &self.extension
            && (extension.initial_tokens == 0
                || extension.initial_tokens > extension.max_tokens
                || extension.max_tokens == 0)
        {
            bail!("N-gram extension requires 0 < initial_tokens <= max_tokens");
        }
        if self.verify_window.min_tokens == 0
            || self.verify_window.min_tokens > self.verify_window.max_tokens
            || self.verify_window.pipeline_depth == 0
        {
            bail!("verify window requires 0 < min_tokens <= max_tokens and pipeline_depth > 0");
        }
        Ok(())
    }

    pub(super) fn insert_telemetry_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        attrs.insert(
            "llama_stage.spec.requested_strategy".to_string(),
            json!(self.requested_strategy),
        );
        attrs.insert(
            "llama_stage.spec.effective_strategy".to_string(),
            json!(self.effective_strategy),
        );
    }
}

pub(super) fn load_standalone_speculative_config(
    path: Option<&PathBuf>,
) -> Result<SpeculativeDecodeConfig> {
    let config = match path {
        Some(path) => load_json(path)
            .with_context(|| format!("load speculative decode config {}", path.display()))?,
        None => SpeculativeDecodeConfig::default(),
    };
    config.validate()?;
    Ok(config)
}

pub(super) fn standalone_simple_ngram_min(config: &SpeculativeDecodeConfig) -> usize {
    config
        .ngram
        .as_ref()
        .filter(|ngram| ngram.kind == NgramProposerKind::Simple)
        .map_or(0, |ngram| ngram.min_ngram)
}

pub(super) fn standalone_simple_ngram_max(config: &SpeculativeDecodeConfig) -> usize {
    config
        .ngram
        .as_ref()
        .filter(|ngram| ngram.kind == NgramProposerKind::Simple)
        .map_or(0, |ngram| ngram.max_proposal_tokens.min(ngram.max_ngram))
}

#[cfg(test)]
mod standalone_speculative_config_tests {
    use super::*;

    #[test]
    fn standalone_speculative_config_rejects_invalid_composite_plan() {
        let config = SpeculativeDecodeConfig {
            extension: Some(NgramExtensionConfig {
                initial_tokens: 2,
                max_tokens: 4,
                tail_backoff_proposals: 1,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let error = config.validate().expect_err("extension requires proposers");

        assert!(
            error
                .to_string()
                .contains("requires both native MTP and an N-gram proposer")
        );
    }

    #[test]
    fn standalone_speculative_config_round_trips_cache_composite() {
        let config = SpeculativeDecodeConfig {
            requested_strategy: "mtp-cache".to_string(),
            effective_strategy: "native-mtp-cache".to_string(),
            native_mtp: NativeMtpProposalConfig {
                enabled: true,
                max_draft_tokens: 2,
                ..SpeculativeDecodeConfig::default().native_mtp
            },
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Cache,
                min_ngram: 2,
                max_ngram: 4,
                max_proposal_tokens: 6,
            }),
            extension: Some(NgramExtensionConfig {
                initial_tokens: 2,
                max_tokens: 6,
                tail_backoff_proposals: 2,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let json = serde_json::to_string(&config).expect("serialize plan");
        let decoded: SpeculativeDecodeConfig = serde_json::from_str(&json).expect("parse plan");

        assert_eq!(decoded, config);
        decoded.validate().expect("valid composite plan");
    }

    #[test]
    fn standalone_speculative_config_rejects_cache_windows_above_llama_limit() {
        let config = SpeculativeDecodeConfig {
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Cache,
                min_ngram: 2,
                max_ngram: skippy_runtime::NGRAM_CACHE_MAX_NGRAM + 1,
                max_proposal_tokens: 6,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let error = config.validate().expect_err("cache max must be bounded");

        assert!(
            error
                .to_string()
                .contains("must not exceed llama.cpp limit 4")
        );
    }
}

#[derive(Clone, Default)]
pub(super) struct OpenAiSpeculativeStats {
    pub(super) windows: usize,
    pub(super) draft_tokens: usize,
    pub(super) accepted_tokens: usize,
    pub(super) rejected_tokens: usize,
    pub(super) full_accept_windows: usize,
    pub(super) accepted_stop_windows: usize,
    pub(super) rejected_windows: usize,
    pub(super) early_reject_windows: usize,
    pub(super) tail_reject_windows: usize,
    pub(super) early_reject_stop_windows: usize,
    pub(super) repair_required_windows: usize,
    pub(super) first_reject_position_sum: usize,
    pub(super) primary_verify_requests: usize,
    pub(super) primary_verify_tokens: usize,
    pub(super) primary_verify_elapsed_ms: f64,
    pub(super) primary_verify_stage0_compute_ms: f64,
    pub(super) primary_verify_runtime_lock_wait_ms: f64,
    pub(super) primary_verify_runtime_lock_hold_ms: f64,
    pub(super) primary_verify_activation_encode_ms: f64,
    pub(super) primary_verify_forward_write_ms: f64,
    pub(super) primary_verify_downstream_wait_ms: f64,
    pub(super) primary_verify_output_activation_bytes: usize,
    pub(super) primary_verify_forward_activation_bytes: usize,
    pub(super) checkpoint_ms: f64,
    pub(super) draft_reset_ms: f64,
    pub(super) draft_propose_ms: f64,
    pub(super) recovery_restores: usize,
    pub(super) recovery_decode_repairs: usize,
    pub(super) recovery_decode_elapsed_ms: f64,
    pub(super) recovery_reverify_tokens: usize,
    pub(super) recovery_ms: f64,
    pub(super) recovery_restore_ms: f64,
    pub(super) recovery_restore_local_ms: f64,
    pub(super) recovery_restore_downstream_write_ms: f64,
    pub(super) recovery_restore_downstream_wait_ms: f64,
    pub(super) recovery_reverify_elapsed_ms: f64,
    pub(super) adaptive_window_start: usize,
    pub(super) adaptive_window_final: usize,
    pub(super) adaptive_window_max: usize,
    pub(super) adaptive_window_min: usize,
    pub(super) adaptive_window_max_seen: usize,
    pub(super) adaptive_window_sum: usize,
    pub(super) adaptive_window_grows: usize,
    pub(super) adaptive_window_shrinks: usize,
    pub(super) adaptive_window_enabled: bool,
}

/// Uses llama.cpp's ngram-simple self-speculative proposer. The accepted
/// history includes the current token; the upstream API keeps it separate
/// from the preceding history internally.
pub(super) fn propose_ngram_tokens(
    history: &[i32],
    min_match_tokens: usize,
    max_proposed_tokens: usize,
) -> OpenAiResult<Vec<i32>> {
    skippy_runtime::ngram_simple_draft(history, min_match_tokens, max_proposed_tokens)
        .map_err(openai_backend_error)
}

/// Request-local, cache-based N-gram proposer. It mirrors only committed
/// history into native state; speculative candidates remain read-only inputs.
pub(super) struct CachedNgramProposer {
    cache: skippy_runtime::NgramCache,
    committed_history: Vec<i32>,
}

impl CachedNgramProposer {
    pub(super) fn from_config(config: &SpeculativeDecodeConfig) -> OpenAiResult<Option<Self>> {
        let Some(ngram) = config.ngram.as_ref() else {
            return Ok(None);
        };
        if ngram.kind != NgramProposerKind::Cache {
            return Ok(None);
        }
        Self::new(ngram.min_ngram, ngram.max_ngram).map(Some)
    }

    pub(super) fn new(ngram_min: usize, ngram_max: usize) -> OpenAiResult<Self> {
        let cache =
            skippy_runtime::NgramCache::new(ngram_min, ngram_max).map_err(openai_backend_error)?;
        Ok(Self {
            cache,
            committed_history: Vec::new(),
        })
    }

    pub(super) fn propose(
        &mut self,
        committed_history: &[i32],
        continuation_prefix: &[i32],
        max_proposed_tokens: usize,
    ) -> OpenAiResult<Vec<i32>> {
        self.sync(committed_history)?;
        self.cache
            .draft_after(continuation_prefix, max_proposed_tokens)
            .map_err(openai_backend_error)
    }

    fn sync(&mut self, committed_history: &[i32]) -> OpenAiResult<()> {
        if committed_history.starts_with(&self.committed_history) {
            let appended = &committed_history[self.committed_history.len()..];
            self.cache.append(appended).map_err(openai_backend_error)?;
        } else {
            self.cache
                .reset(committed_history)
                .map_err(openai_backend_error)?;
        }
        self.committed_history.clear();
        self.committed_history.extend_from_slice(committed_history);
        Ok(())
    }
}

impl OpenAiSpeculativeStats {
    pub(super) fn insert_response_timings(&self, timings: &mut BTreeMap<String, Value>) {
        timings.insert(
            "verify_window_verify_elapsed_ms".to_string(),
            json!(self.primary_verify_elapsed_ms),
        );
        timings.insert(
            "verify_window_stage0_compute_ms".to_string(),
            json!(self.primary_verify_stage0_compute_ms),
        );
        timings.insert(
            "verify_window_forward_write_ms".to_string(),
            json!(self.primary_verify_forward_write_ms),
        );
        timings.insert(
            "verify_window_downstream_wait_ms".to_string(),
            json!(self.primary_verify_downstream_wait_ms),
        );
    }

    pub(super) fn observe_verify_decision(
        &mut self,
        decision: VerifyWindowDecision,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        self.accepted_tokens += decision.accepted_before_reject;
        if decision.rejected() {
            self.rejected_tokens += 1;
        }
        self.adaptive_window_sum += *adaptive_window;
        self.adaptive_window_min = nonzero_min(self.adaptive_window_min, *adaptive_window);
        self.adaptive_window_max_seen = self.adaptive_window_max_seen.max(*adaptive_window);
        match decision.kind {
            VerifyWindowDecisionKind::FullAccept => {
                self.full_accept_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifyWindowDecisionKind::AcceptedStop => {
                self.accepted_stop_windows += 1;
            }
            VerifyWindowDecisionKind::TailReject => {
                self.observe_reject(decision);
                self.tail_reject_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifyWindowDecisionKind::EarlyReject => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.repair_required_windows += 1;
                self.shrink_adaptive_window(adaptive_window, adaptive_enabled, decision);
            }
            VerifyWindowDecisionKind::EarlyRejectStop => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.early_reject_stop_windows += 1;
            }
        }
    }

    pub(super) fn observe_reject(&mut self, decision: VerifyWindowDecision) {
        if let Some(repair_input_count) = decision.repair_input_count {
            self.rejected_windows += 1;
            self.first_reject_position_sum += repair_input_count;
        }
    }

    pub(super) fn grow_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        if adaptive_enabled && *adaptive_window < max_speculative_window {
            *adaptive_window += 1;
            self.adaptive_window_grows += 1;
        }
    }

    pub(super) fn shrink_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        decision: VerifyWindowDecision,
    ) {
        if !adaptive_enabled {
            return;
        }
        let Some(repair_input_count) = decision.repair_input_count else {
            return;
        };
        let next_window = (*adaptive_window)
            .saturating_sub(1)
            .max(repair_input_count)
            .max(1);
        if next_window < *adaptive_window {
            *adaptive_window = next_window;
            self.adaptive_window_shrinks += 1;
        }
    }

    pub(super) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        if self.windows == 0 {
            attrs.insert("llama_stage.spec.enabled".to_string(), json!(false));
            return;
        }
        attrs.insert("llama_stage.spec.enabled".to_string(), json!(true));
        attrs.insert("llama_stage.spec.windows".to_string(), json!(self.windows));
        attrs.insert(
            "llama_stage.spec.proposed".to_string(),
            json!(self.draft_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accepted".to_string(),
            json!(self.accepted_tokens),
        );
        attrs.insert(
            "llama_stage.spec.rejected".to_string(),
            json!(self.rejected_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accept_rate".to_string(),
            json!(if self.draft_tokens == 0 {
                0.0
            } else {
                self.accepted_tokens as f64 / self.draft_tokens as f64
            }),
        );
        attrs.insert(
            "llama_stage.spec.full_accept_windows".to_string(),
            json!(self.full_accept_windows),
        );
        attrs.insert(
            "llama_stage.spec.accepted_stop_windows".to_string(),
            json!(self.accepted_stop_windows),
        );
        attrs.insert(
            "llama_stage.spec.rejected_windows".to_string(),
            json!(self.rejected_windows),
        );
        attrs.insert(
            "llama_stage.spec.early_reject_windows".to_string(),
            json!(self.early_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.tail_reject_windows".to_string(),
            json!(self.tail_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.repair_required_windows".to_string(),
            json!(self.repair_required_windows),
        );
        attrs.insert(
            "llama_stage.spec.draft_reset_ms".to_string(),
            json!(self.draft_reset_ms),
        );
        attrs.insert(
            "llama_stage.spec.draft_propose_ms".to_string(),
            json!(self.draft_propose_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_elapsed_ms".to_string(),
            json!(self.primary_verify_elapsed_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_stage0_compute_ms".to_string(),
            json!(self.primary_verify_stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_wait_ms".to_string(),
            json!(self.primary_verify_runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_hold_ms".to_string(),
            json!(self.primary_verify_runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_activation_encode_ms".to_string(),
            json!(self.primary_verify_activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_write_ms".to_string(),
            json!(self.primary_verify_forward_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_downstream_wait_ms".to_string(),
            json!(self.primary_verify_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_output_activation_bytes".to_string(),
            json!(self.primary_verify_output_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_activation_bytes".to_string(),
            json!(self.primary_verify_forward_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.checkpoint_ms".to_string(),
            json!(self.checkpoint_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restores".to_string(),
            json!(self.recovery_restores),
        );
        attrs.insert(
            "llama_stage.spec.recovery_ms".to_string(),
            json!(self.recovery_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_local_ms".to_string(),
            json!(self.recovery_restore_local_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_write_ms".to_string(),
            json!(self.recovery_restore_downstream_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_wait_ms".to_string(),
            json!(self.recovery_restore_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.adaptive_enabled".to_string(),
            json!(self.adaptive_window_enabled),
        );
        attrs.insert(
            "llama_stage.spec.window_start".to_string(),
            json!(self.adaptive_window_start),
        );
        attrs.insert(
            "llama_stage.spec.window_final".to_string(),
            json!(self.adaptive_window_final),
        );
        attrs.insert(
            "llama_stage.spec.window_max".to_string(),
            json!(self.adaptive_window_max),
        );
        attrs.insert(
            "llama_stage.spec.window_min".to_string(),
            json!(self.adaptive_window_min),
        );
        attrs.insert(
            "llama_stage.spec.window_max_seen".to_string(),
            json!(self.adaptive_window_max_seen),
        );
        attrs.insert(
            "llama_stage.spec.window_grows".to_string(),
            json!(self.adaptive_window_grows),
        );
        attrs.insert(
            "llama_stage.spec.window_shrinks".to_string(),
            json!(self.adaptive_window_shrinks),
        );
    }
}

#[cfg(test)]
mod ngram_tests {
    use super::*;

    #[test]
    fn proposes_tokens_after_latest_matching_suffix() {
        let history = [1, 2, 3, 4, 9, 2, 3, 4];

        assert_eq!(propose_ngram_tokens(&history, 2, 2).unwrap(), vec![9, 2]);
    }

    #[test]
    fn returns_empty_without_enough_history() {
        assert!(propose_ngram_tokens(&[1, 2, 3], 2, 4).unwrap().is_empty());
        assert!(
            propose_ngram_tokens(&[1, 2, 1, 2], 0, 4)
                .unwrap()
                .is_empty()
        );
        assert!(
            propose_ngram_tokens(&[1, 2, 1, 2], 1, 0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn cache_proposer_syncs_only_the_committed_prefix() {
        let mut proposer = CachedNgramProposer::new(2, 2).unwrap();
        let history = [1, 2, 3, 1, 2, 3, 1, 2];

        assert_eq!(proposer.propose(&history, &[], 2).unwrap(), vec![3, 1]);
        assert_eq!(
            proposer.propose(&history, &[9], 2).unwrap(),
            Vec::<i32>::new()
        );
        assert_eq!(proposer.propose(&history, &[], 2).unwrap(), vec![3, 1]);
    }
}

pub(super) fn verify_inputs_for_proposals(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len());
    if proposals.is_empty() {
        return tokens;
    }
    tokens.push(current);
    tokens.extend(proposals.iter().take(proposals.len().saturating_sub(1)));
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerifyWindowDecisionKind {
    FullAccept,
    AcceptedStop,
    TailReject,
    EarlyReject,
    EarlyRejectStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifyWindowDecision {
    pub(super) kind: VerifyWindowDecisionKind,
    pub(super) accepted_before_reject: usize,
    pub(super) repair_input_count: Option<usize>,
    pub(super) commit_count: usize,
}

impl VerifyWindowDecision {
    pub(super) fn rejected(self) -> bool {
        matches!(
            self.kind,
            VerifyWindowDecisionKind::TailReject
                | VerifyWindowDecisionKind::EarlyReject
                | VerifyWindowDecisionKind::EarlyRejectStop
        )
    }

    pub(super) fn requires_repair(self) -> bool {
        self.kind == VerifyWindowDecisionKind::EarlyReject
    }
}

pub(super) fn classify_verify_window<F>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<VerifyWindowDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    if predicted_tokens.len() < draft_tokens.len() {
        return Err(OpenAiError::backend(format!(
            "verify window returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            draft_tokens.len()
        )));
    }

    let mut accepted_before_reject = 0usize;
    let mut commit_count = 0usize;
    for (draft_token, predicted) in draft_tokens.iter().zip(predicted_tokens.iter()) {
        commit_count += 1;
        let accepted = *predicted == *draft_token;
        let reached_eog = token_is_eog(*predicted)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if accepted {
            accepted_before_reject += 1;
            if (reached_eog || reached_limit) && commit_count < draft_tokens.len() {
                return Ok(VerifyWindowDecision {
                    kind: VerifyWindowDecisionKind::AcceptedStop,
                    accepted_before_reject,
                    repair_input_count: None,
                    commit_count,
                });
            }
            continue;
        }

        let repair_input_count = accepted_before_reject + 1;
        let kind = if repair_input_count == draft_tokens.len() {
            VerifyWindowDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifyWindowDecisionKind::EarlyRejectStop
        } else {
            VerifyWindowDecisionKind::EarlyReject
        };
        return Ok(VerifyWindowDecision {
            kind,
            accepted_before_reject,
            repair_input_count: Some(repair_input_count),
            commit_count,
        });
    }

    Ok(VerifyWindowDecision {
        kind: VerifyWindowDecisionKind::FullAccept,
        accepted_before_reject,
        repair_input_count: None,
        commit_count,
    })
}

pub(super) fn repaired_commit_tokens(
    draft_tokens: &[i32],
    accepted_before_reject: usize,
    repair_input_count: usize,
    repaired_predictions: &[i32],
) -> OpenAiResult<Vec<i32>> {
    if repaired_predictions.len() < repair_input_count {
        return Err(OpenAiError::backend(format!(
            "recovery verify returned too few tokens: expected {} got {:?}",
            repair_input_count, repaired_predictions
        )));
    }
    if accepted_before_reject > 0
        && repaired_predictions[..accepted_before_reject] != draft_tokens[..accepted_before_reject]
    {
        eprintln!(
            "recovery verify changed accepted prefix; committing restored target tokens: accepted {:?}, repaired {:?}",
            &draft_tokens[..accepted_before_reject],
            &repaired_predictions[..accepted_before_reject]
        );
    }
    Ok(repaired_predictions[..repair_input_count].to_vec())
}

pub(super) fn nonzero_min(current: usize, candidate: usize) -> usize {
    if current == 0 {
        candidate
    } else {
        current.min(candidate)
    }
}
