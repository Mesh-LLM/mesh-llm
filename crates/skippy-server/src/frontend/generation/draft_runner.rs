use crate::runtime_state::RuntimeState;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use skippy_protocol::StageConfig;
use skippy_runtime::FlashAttentionType as RuntimeFlashAttentionType;
use skippy_runtime::RuntimeConfig;
use skippy_runtime::RuntimeLoadMode;
use skippy_runtime::StageModel;
use skippy_runtime::StageSession;
use skippy_runtime::{ModelInfo, MtpSource};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

pub(in crate::frontend) struct DraftRunner {
    pub(in crate::frontend) path: PathBuf,
    pub(in crate::frontend) window: usize,
    pub(in crate::frontend) _model: StageModel,
    pub(in crate::frontend) session: StageSession,
    /// Tokens currently materialized in the draft session's KV, maintained so
    /// fallback proposals can advance incrementally instead of re-prefilling
    /// the whole context on every call.
    synced: DraftSyncState,
}

/// What a sync to a given context requires of the draft session. Split out
/// from the session I/O because this bookkeeping is load-bearing for KV
/// correctness: claiming a prefix extension the session has not materialized
/// silently corrupts every later proposal.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::frontend) enum DraftSyncPlan {
    /// The session is already at the target; nothing to do.
    AlreadySynced,
    /// The synced tokens are a prefix of the target: prefill only the tail,
    /// given here as a range into the target prefix.
    Extend { from: usize, to: usize },
    /// The synced tokens diverge from the target: reset and prefill the
    /// whole prefix.
    Reset,
}

/// Tokens the draft session has materialized, and the decisions derived from
/// them.
#[derive(Debug, Default)]
pub(in crate::frontend) struct DraftSyncState {
    tokens: Vec<i32>,
}

impl DraftSyncState {
    /// The prefix a context implies: every token but the last, which is the
    /// one a proposal decodes from.
    fn target_len(context_tokens: &[i32]) -> usize {
        context_tokens.len().saturating_sub(1)
    }

    pub(in crate::frontend) fn plan(&self, context_tokens: &[i32]) -> DraftSyncPlan {
        let target = &context_tokens[..Self::target_len(context_tokens)];
        if self.tokens.is_empty() || !target.starts_with(&self.tokens) {
            return DraftSyncPlan::Reset;
        }
        if target.len() == self.tokens.len() {
            return DraftSyncPlan::AlreadySynced;
        }
        DraftSyncPlan::Extend {
            from: self.tokens.len(),
            to: target.len(),
        }
    }

    fn record_extend(&mut self, delta: &[i32]) {
        self.tokens.extend_from_slice(delta);
    }

    fn record_reset(&mut self, prefix: &[i32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prefix);
    }

    /// A proposal decodes from `current`, which the session materializes as
    /// it steps — so it joins the synced prefix and the next sync can extend
    /// instead of resetting.
    fn record_proposal_step(&mut self, current: i32) {
        self.tokens.push(current);
    }
}

impl DraftRunner {
    pub(in crate::frontend) fn open(
        path: &Path,
        config: &StageConfig,
        n_gpu_layers: Option<i32>,
        window: usize,
    ) -> Result<Self> {
        if !path.is_file() {
            bail!("draft model does not exist: {}", path.display());
        }
        let layer_count = model_layer_count(path)?;
        let model = StageModel::open(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size: config.ctx_size,
                lane_count: 1,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: config
                    .selected_device
                    .as_ref()
                    .map(|device| device.backend_device.clone()),
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: RuntimeFlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                mtp_source: MtpSource::Disabled,
                filter_tensors_on_load: false,
            },
        )
        .with_context(|| format!("open draft model {}", path.display()))?;
        let session = model.create_session().context("create draft session")?;
        Ok(Self {
            path: path.to_path_buf(),
            window,
            _model: model,
            session,
            synced: DraftSyncState::default(),
        })
    }

    pub(in crate::frontend) fn reset_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        self.session.reset().context("reset draft session")?;
        self.synced.record_reset(&[]);
        if context_tokens.len() > 1 {
            let prefix = &context_tokens[..context_tokens.len() - 1];
            self.session
                .prefill_chunk(prefix)
                .context("prefill draft context")?;
            self.synced.record_reset(prefix);
        }
        Ok(())
    }

    /// Brings the draft session to `context_tokens` (all but the last token
    /// prefilled, ready to propose from the last). Extends incrementally when
    /// the already-synced tokens are a prefix of the target — the common case
    /// when prior fallback proposals were accepted — and falls back to a full
    /// reset on divergence.
    pub(in crate::frontend) fn sync_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        match self.synced.plan(context_tokens) {
            DraftSyncPlan::AlreadySynced => Ok(()),
            DraftSyncPlan::Extend { from, to } => {
                let delta = &context_tokens[from..to];
                self.session
                    .prefill_chunk(delta)
                    .context("advance draft context")?;
                self.synced.record_extend(delta);
                Ok(())
            }
            DraftSyncPlan::Reset => self.reset_to_context(context_tokens),
        }
    }

    pub(in crate::frontend) fn propose(
        &mut self,
        mut current: i32,
        max_tokens: usize,
    ) -> Result<Vec<i32>> {
        let mut tokens = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            // Record after the step succeeds: a failed decode must not leave
            // the state claiming a token the session does not hold.
            let stepped_from = current;
            current = self
                .session
                .decode_step(current)
                .context("draft decode step")?;
            self.synced.record_proposal_step(stepped_from);
            tokens.push(current);
        }
        Ok(tokens)
    }
}

pub(in crate::frontend) fn open_draft_runner(
    path: Option<&Path>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
    window: usize,
) -> Result<Option<Arc<Mutex<DraftRunner>>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    Ok(Some(Arc::new(Mutex::new(DraftRunner::open(
        path,
        config,
        n_gpu_layers,
        window,
    )?))))
}

pub(in crate::frontend) fn attach_native_mtp_draft_model(
    path: Option<&Path>,
    runtime: &Arc<Mutex<RuntimeState>>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if !path.is_file() {
        bail!("MTP draft model does not exist: {}", path.display());
    }
    let layer_count = model_layer_count(path)?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?;
    runtime
        .model
        .attach_mtp_draft_model(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size: config.ctx_size,
                lane_count: config.lane_count,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: config
                    .selected_device
                    .as_ref()
                    .map(|device| device.backend_device.clone()),
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: RuntimeFlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                mtp_source: MtpSource::External,
                filter_tensors_on_load: false,
            },
        )
        .with_context(|| format!("attach MTP draft model {}", path.display()))
}

pub(in crate::frontend) fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let layer_count = info
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .ok_or_else(|| anyhow!("could not infer layer count for {}", path.display()))?;
    Ok(layer_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(tokens: &[i32]) -> DraftSyncState {
        DraftSyncState {
            tokens: tokens.to_vec(),
        }
    }

    #[test]
    fn an_empty_session_always_resets() {
        assert_eq!(state(&[]).plan(&[1, 2, 3]), DraftSyncPlan::Reset);
        // A context of one token has an empty prefix: still a reset, and
        // `reset_to_context` then prefills nothing.
        assert_eq!(state(&[]).plan(&[1]), DraftSyncPlan::Reset);
    }

    #[test]
    fn a_synced_prefix_extends_by_the_delta_only() {
        // Synced [1, 2]; context [1, 2, 3, 4, 5] has prefix [1, 2, 3, 4].
        assert_eq!(
            state(&[1, 2]).plan(&[1, 2, 3, 4, 5]),
            DraftSyncPlan::Extend { from: 2, to: 4 }
        );
    }

    #[test]
    fn an_exactly_synced_prefix_is_a_no_op() {
        // Synced [1, 2, 3]; context [1, 2, 3, 4] has prefix [1, 2, 3].
        assert_eq!(
            state(&[1, 2, 3]).plan(&[1, 2, 3, 4]),
            DraftSyncPlan::AlreadySynced
        );
    }

    #[test]
    fn divergence_resets_rather_than_extending() {
        // Same length, different token: the KV past that point is wrong.
        assert_eq!(state(&[1, 9]).plan(&[1, 2, 3, 4]), DraftSyncPlan::Reset);
        // A rejected proposal leaves the session longer than the target.
        assert_eq!(state(&[1, 2, 3, 4]).plan(&[1, 2, 3]), DraftSyncPlan::Reset);
    }

    #[test]
    fn proposal_steps_join_the_synced_prefix_so_the_next_sync_extends() {
        let mut synced = state(&[1, 2]);
        // Two proposal steps decoded from 3 then 4.
        synced.record_proposal_step(3);
        synced.record_proposal_step(4);

        // Both accepted, and the caller committed a fifth token: the session
        // already holds [1, 2, 3, 4], so only [5] needs prefilling.
        assert_eq!(
            synced.plan(&[1, 2, 3, 4, 5, 6]),
            DraftSyncPlan::Extend { from: 4, to: 5 }
        );
    }

    #[test]
    fn a_rejected_proposal_step_forces_a_reset() {
        let mut synced = state(&[1, 2]);
        synced.record_proposal_step(3);
        // The verifier rejected 3 and committed 9 instead: the draft KV holds
        // a token the target never accepted, so the prefix cannot be reused.
        assert_eq!(synced.plan(&[1, 2, 9, 10]), DraftSyncPlan::Reset);
    }

    #[test]
    fn recording_a_reset_replaces_the_whole_prefix() {
        let mut synced = state(&[1, 2, 3]);
        synced.record_reset(&[7, 8]);

        assert_eq!(synced.plan(&[7, 8, 9]), DraftSyncPlan::AlreadySynced);
        assert_eq!(synced.plan(&[1, 2, 3]), DraftSyncPlan::Reset);
    }

    #[test]
    fn an_extend_plan_indexes_the_context_the_caller_slices() {
        // The plan's range must address `context_tokens` directly, since
        // sync_to_context slices the context with it.
        let context = [1, 2, 3, 4, 5];
        let DraftSyncPlan::Extend { from, to } = state(&[1, 2]).plan(&context) else {
            panic!("a synced prefix must extend");
        };

        assert_eq!(&context[from..to], &[3, 4]);
    }
}
