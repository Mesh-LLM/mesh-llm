use anyhow::{Context, Result, bail};
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::load_json;
use crate::frontend::util::openai_backend_error;

mod standalone;
mod suffix;

pub(super) use standalone::{propose_configured_ngram_tokens, standalone_ngram_proposal_limit};
use suffix::{SUFFIX_MIN_SEED_LEN, SuffixNgramProposer};

/// Resolved speculative decoding plan for a served model.
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

/// Native multi-token-prediction draft settings.
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

/// Which N-gram draft proposer to run: llama.cpp request-local `cache`, or the
/// pure-Rust longest-suffix `suffix`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NgramProposerKind {
    #[default]
    Cache,
    Suffix,
}

impl NgramProposerKind {
    /// Stable string name used in config and telemetry.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::Suffix => "suffix",
        }
    }
}

/// Longest suffix match window, and upper bound for a suffix proposer's `max_ngram`.
pub const SUFFIX_NGRAM_MAX_WINDOW: usize = 64;

/// N-gram proposer kind and its match-length and draft-length bounds.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NgramProposalConfig {
    /// Defaults to `cache` so speculative plans written before the kind field
    /// existed still deserialize.
    #[serde(default)]
    pub kind: NgramProposerKind,
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub max_proposal_tokens: usize,
}

/// Bounds for extending an MTP prefix with an N-gram tail.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NgramExtensionConfig {
    pub max_tokens: usize,
}

/// Pipelined verify-window sizing.
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
    /// Checks the plan's internal invariants (bounds, proposer limits, extension
    /// and verify-window constraints).
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
                || ngram.max_proposal_tokens == 0)
        {
            bail!(
                "N-gram proposer requires 0 < min_ngram <= max_ngram and max_proposal_tokens > 0"
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
        if let Some(ngram) = &self.ngram
            && ngram.kind == NgramProposerKind::Suffix
            && (ngram.min_ngram < SUFFIX_MIN_SEED_LEN || ngram.max_ngram > SUFFIX_NGRAM_MAX_WINDOW)
        {
            bail!(
                "suffix N-gram proposer requires {SUFFIX_MIN_SEED_LEN} <= min_ngram <= max_ngram <= {SUFFIX_NGRAM_MAX_WINDOW}"
            );
        }
        if self.extension.is_some() && (!self.native_mtp.enabled || self.ngram.is_none()) {
            bail!(
                "composite speculation requires native MTP and an extension policy backed by an N-gram proposer"
            );
        }
        if let Some(extension) = &self.extension
            && extension.max_tokens == 0
        {
            bail!("N-gram extension requires max_tokens > 0");
        }
        if self.verify_window.min_tokens == 0
            || self.verify_window.min_tokens > self.verify_window.max_tokens
            || self.verify_window.pipeline_depth == 0
        {
            bail!("verify window requires 0 < min_tokens <= max_tokens and pipeline_depth > 0");
        }
        Ok(())
    }

    /// Adds the requested and effective strategy names to a telemetry attr map.
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

/// Loads a speculative decode plan from JSON (or the default), then validates it.
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

#[cfg(test)]
mod standalone_speculative_config_tests {
    use super::*;

    #[test]
    fn standalone_speculative_config_rejects_invalid_composite_plan() {
        let config = SpeculativeDecodeConfig {
            extension: Some(NgramExtensionConfig { max_tokens: 4 }),
            ..SpeculativeDecodeConfig::default()
        };

        let error = config.validate().expect_err("extension requires proposers");

        assert!(
            error
                .to_string()
                .contains("requires native MTP and an extension policy")
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
            extension: Some(NgramExtensionConfig { max_tokens: 6 }),
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

    #[test]
    fn standalone_speculative_config_round_trips_suffix() {
        let config = SpeculativeDecodeConfig {
            requested_strategy: "ngram".to_string(),
            effective_strategy: "ngram-suffix".to_string(),
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Suffix,
                min_ngram: 5,
                max_ngram: 32,
                max_proposal_tokens: 48,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let json = serde_json::to_string(&config).expect("serialize plan");
        let decoded: SpeculativeDecodeConfig = serde_json::from_str(&json).expect("parse plan");

        assert_eq!(decoded, config);
        decoded.validate().expect("valid suffix plan");
    }

    #[test]
    fn ngram_proposal_config_without_kind_defaults_to_cache() {
        // Speculative plans written before the kind field existed omit it; they
        // must still deserialize (for --openai-speculative-config) as cache.
        let json = r#"{"min_ngram":2,"max_ngram":4,"max_proposal_tokens":6}"#;
        let config: NgramProposalConfig =
            serde_json::from_str(json).expect("legacy plan without kind should deserialize");
        assert_eq!(config.kind, NgramProposerKind::Cache);
    }

    #[test]
    fn standalone_speculative_config_rejects_suffix_windows_above_limit() {
        let config = SpeculativeDecodeConfig {
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Suffix,
                min_ngram: 3,
                max_ngram: SUFFIX_NGRAM_MAX_WINDOW + 1,
                max_proposal_tokens: 6,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let error = config.validate().expect_err("suffix max must be bounded");

        assert!(error.to_string().contains("min_ngram <= max_ngram <= 64"));
    }

    #[test]
    fn standalone_speculative_config_rejects_suffix_matches_below_seed_length() {
        let config = SpeculativeDecodeConfig {
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Suffix,
                min_ngram: 2,
                max_ngram: 16,
                max_proposal_tokens: 4,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let error = config
            .validate()
            .expect_err("suffix minimum must cover the exact seed");

        assert!(error.to_string().contains("3 <= min_ngram"));
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
    pub(super) draft_reset_ms: f64,
    pub(super) draft_propose_ms: f64,
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

/// Request-local, cache-based N-gram proposer. It mirrors only committed
/// history into native state; speculative candidates remain read-only inputs.
pub(super) struct CachedNgramProposer {
    cache: Option<skippy_runtime::NgramCache>,
    committed_history: Vec<i32>,
    ngram_min: usize,
    ngram_max: usize,
}

impl CachedNgramProposer {
    /// Creates a cache-backed proposer with the given match bounds.
    pub(super) fn new(ngram_min: usize, ngram_max: usize) -> OpenAiResult<Self> {
        if ngram_min == 0
            || ngram_min > ngram_max
            || ngram_max > skippy_runtime::NGRAM_CACHE_MAX_NGRAM
        {
            return Err(OpenAiError::backend(format!(
                "cache N-gram proposer requires 0 < ngram_min <= ngram_max <= {}",
                skippy_runtime::NGRAM_CACHE_MAX_NGRAM
            )));
        }
        Ok(Self {
            cache: None,
            committed_history: Vec::new(),
            ngram_min,
            ngram_max,
        })
    }

    /// Syncs committed history, then drafts after the continuation prefix.
    pub(super) fn propose(
        &mut self,
        committed_history: &[i32],
        continuation_prefix: &[i32],
        max_proposed_tokens: usize,
    ) -> OpenAiResult<Vec<i32>> {
        self.sync(committed_history)?;
        self.cache
            .as_mut()
            .expect("N-gram cache initialized by sync")
            .draft_after(continuation_prefix, max_proposed_tokens)
            .map_err(openai_backend_error)
    }

    /// Mirrors committed history into native cache state (append or reset).
    fn sync(&mut self, committed_history: &[i32]) -> OpenAiResult<()> {
        if self.cache.is_none() {
            self.cache = Some(
                skippy_runtime::NgramCache::new(self.ngram_min, self.ngram_max)
                    .map_err(openai_backend_error)?,
            );
        }
        let cache = self.cache.as_mut().expect("N-gram cache initialized");
        if self.committed_history.is_empty() {
            cache
                .reset(committed_history)
                .map_err(openai_backend_error)?;
        } else if committed_history.starts_with(&self.committed_history) {
            let appended = &committed_history[self.committed_history.len()..];
            cache.append(appended).map_err(openai_backend_error)?;
        } else {
            cache
                .reset(committed_history)
                .map_err(openai_backend_error)?;
        }
        self.committed_history.clear();
        self.committed_history.extend_from_slice(committed_history);
        Ok(())
    }
}

/// The concrete history-backed proposer behind [`HistoryNgramProposer`].
enum HistoryNgramProposerImpl {
    Cache(CachedNgramProposer),
    Suffix(SuffixNgramProposer),
}

/// Aggregate proposer counters surfaced through response timings.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct HistoryNgramProposerStats {
    pub(super) attempts: usize,
    pub(super) hits: usize,
    pub(super) proposed_tokens: usize,
    pub(super) match_length_sum: usize,
    pub(super) match_length_max: usize,
    pub(super) candidates_examined: usize,
    pub(super) appended_tokens: usize,
    pub(super) rebuilds: usize,
    pub(super) sync_us: u64,
    pub(super) lookup_us: u64,
}

/// Config-selected history N-gram proposer (cache or suffix) with running stats.
pub(super) struct HistoryNgramProposer {
    proposer: HistoryNgramProposerImpl,
    stats: HistoryNgramProposerStats,
}

impl HistoryNgramProposer {
    /// Builds the configured history proposer, or `None` for simple/no proposer.
    pub(super) fn from_config(config: &SpeculativeDecodeConfig) -> OpenAiResult<Option<Self>> {
        let Some(ngram) = config.ngram.as_ref() else {
            return Ok(None);
        };
        match ngram.kind {
            NgramProposerKind::Cache => CachedNgramProposer::new(ngram.min_ngram, ngram.max_ngram)
                .map(HistoryNgramProposerImpl::Cache)
                .map(Self::new)
                .map(Some),
            NgramProposerKind::Suffix => SuffixNgramProposer::new(
                ngram.min_ngram,
                ngram.max_ngram,
                ngram.max_proposal_tokens,
            )
            .map_err(OpenAiError::backend)
            .map(HistoryNgramProposerImpl::Suffix)
            .map(Self::new)
            .map(Some),
        }
    }

    /// Wraps a concrete proposer with zeroed stats.
    fn new(proposer: HistoryNgramProposerImpl) -> Self {
        Self {
            proposer,
            stats: HistoryNgramProposerStats::default(),
        }
    }

    /// Runs the proposer and accumulates per-request stats.
    pub(super) fn propose(
        &mut self,
        committed_history: &[i32],
        continuation_prefix: &[i32],
        max_proposed_tokens: usize,
    ) -> OpenAiResult<Vec<i32>> {
        self.stats.attempts += 1;
        let tokens = match &mut self.proposer {
            HistoryNgramProposerImpl::Cache(cache) => {
                let started = Instant::now();
                let tokens =
                    cache.propose(committed_history, continuation_prefix, max_proposed_tokens)?;
                self.stats.lookup_us = self.stats.lookup_us.saturating_add(elapsed_us(started));
                tokens
            }
            HistoryNgramProposerImpl::Suffix(suffix) => {
                let proposal =
                    suffix.propose(committed_history, continuation_prefix, max_proposed_tokens);
                self.stats.match_length_sum = self
                    .stats
                    .match_length_sum
                    .saturating_add(proposal.stats.match_length);
                self.stats.match_length_max =
                    self.stats.match_length_max.max(proposal.stats.match_length);
                self.stats.candidates_examined = self
                    .stats
                    .candidates_examined
                    .saturating_add(proposal.stats.candidates_examined);
                self.stats.appended_tokens = self
                    .stats
                    .appended_tokens
                    .saturating_add(proposal.stats.appended_tokens);
                self.stats.rebuilds += usize::from(proposal.stats.rebuilt);
                self.stats.sync_us = self.stats.sync_us.saturating_add(proposal.stats.sync_us);
                self.stats.lookup_us = self
                    .stats
                    .lookup_us
                    .saturating_add(proposal.stats.lookup_us);
                proposal.tokens
            }
        };
        self.stats.hits += usize::from(!tokens.is_empty());
        self.stats.proposed_tokens = self.stats.proposed_tokens.saturating_add(tokens.len());
        Ok(tokens)
    }

    /// Returns the accumulated proposer stats.
    pub(super) fn stats(&self) -> HistoryNgramProposerStats {
        self.stats
    }

    /// Test-only constructor for the cache variant.
    #[cfg(test)]
    pub(super) fn new_cache(ngram_min: usize, ngram_max: usize) -> OpenAiResult<Self> {
        CachedNgramProposer::new(ngram_min, ngram_max)
            .map(HistoryNgramProposerImpl::Cache)
            .map(Self::new)
    }
}

/// Microseconds elapsed since `started`, saturating into a `u64`.
fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl OpenAiSpeculativeStats {
    pub(super) fn insert_response_timings(&self, timings: &mut BTreeMap<String, Value>) {
        timings.insert("speculative_windows".to_string(), json!(self.windows));
        timings.insert(
            "speculative_proposed_n".to_string(),
            json!(self.draft_tokens),
        );
        timings.insert(
            "speculative_accepted_n".to_string(),
            json!(self.accepted_tokens),
        );
        timings.insert(
            "speculative_rejected_n".to_string(),
            json!(self.rejected_tokens),
        );
        timings.insert(
            "speculative_accept_rate".to_string(),
            json!(if self.draft_tokens == 0 {
                0.0
            } else {
                self.accepted_tokens as f64 / self.draft_tokens as f64
            }),
        );
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
        if decision.rejected() {
            self.rejected_windows += 1;
            self.first_reject_position_sum += decision.commit_count;
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
        if !decision.rejected() {
            return;
        }
        let next_window = (*adaptive_window)
            .saturating_sub(1)
            .max(decision.commit_count)
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
                    commit_count,
                });
            }
            continue;
        }

        let commit_count = accepted_before_reject + 1;
        let kind = if commit_count == draft_tokens.len() {
            VerifyWindowDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifyWindowDecisionKind::EarlyRejectStop
        } else {
            VerifyWindowDecisionKind::EarlyReject
        };
        return Ok(VerifyWindowDecision {
            kind,
            accepted_before_reject,
            commit_count,
        });
    }

    Ok(VerifyWindowDecision {
        kind: VerifyWindowDecisionKind::FullAccept,
        accepted_before_reject,
        commit_count,
    })
}

pub(super) fn nonzero_min(current: usize, candidate: usize) -> usize {
    if current == 0 {
        candidate
    } else {
        current.min(candidate)
    }
}
