use std::time::Instant;

use skippy_runtime::SamplingConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Preempted,
    Finished,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixRestoreKind {
    ResidentKv,
    RecurrentWholeState,
    KvAndRecurrentWholeState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixRestore {
    pub page_id: String,
    pub token_count: usize,
    pub kind: PrefixRestoreKind,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: String,
    pub prompt_tokens: Vec<i32>,
    pub generated_tokens: Vec<i32>,
    pub max_tokens: u32,
    pub sampling: Option<SamplingConfig>,
    pub priority: u64,
    /// Admission reservation for this request across scheduler memory
    /// components. This is deliberately separate from `max_tokens`: the
    /// client value is a generation ceiling, while serving admission reserves
    /// bounded decode headroom plus the concrete prompt.
    pub admission_tokens: usize,
    pub status: SequenceStatus,
    pub prefix_restore: Option<PrefixRestore>,
    pub admitted_at: Option<Instant>,
    pub(crate) prefill_cursor: usize,
}

impl Sequence {
    pub fn new(
        id: String,
        prompt_tokens: Vec<i32>,
        max_tokens: u32,
        sampling: Option<SamplingConfig>,
        priority: u64,
    ) -> Self {
        let admission_tokens = prompt_tokens
            .len()
            .saturating_add(usize::try_from(max_tokens).unwrap_or(usize::MAX));
        Self {
            id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            sampling,
            priority,
            admission_tokens,
            status: SequenceStatus::Waiting,
            prefix_restore: None,
            admitted_at: None,
            prefill_cursor: 0,
        }
    }

    pub fn with_admission_tokens(mut self, admission_tokens: usize) -> Self {
        self.admission_tokens = admission_tokens.max(self.prompt_tokens.len());
        self
    }

    pub fn with_prefix_restore(mut self, restore: PrefixRestore) -> Self {
        // A restored prefix does not carry a sampled next token. Keep one
        // replay token runnable so the native runtime produces logits even
        // when the cache covers the complete prompt.
        let replay_len = self.recompute_tokens().len();
        self.prefill_cursor = restore.token_count.min(replay_len.saturating_sub(1));
        self.prefix_restore = Some(restore);
        self
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.status,
            SequenceStatus::Finished | SequenceStatus::Failed
        )
    }

    pub fn recompute_tokens(&self) -> Vec<i32> {
        let replay_generated = self.generated_tokens.len().saturating_sub(1);
        let mut tokens = Vec::with_capacity(self.prompt_tokens.len() + replay_generated);
        tokens.extend_from_slice(&self.prompt_tokens);
        tokens.extend_from_slice(&self.generated_tokens[..replay_generated]);
        tokens
    }

    pub(crate) fn pending_decode_token(&self) -> Option<i32> {
        self.generated_tokens.last().copied()
    }

    pub(crate) fn reset_for_recompute(&mut self) {
        self.status = SequenceStatus::Preempted;
        self.admitted_at = None;
        self.prefill_cursor = self
            .prefix_restore
            .as_ref()
            .map_or(0, |restore| restore.token_count)
            .min(self.recompute_tokens().len());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationPhase {
    Prefill,
    Recompute,
    Decode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IterationWork {
    pub sequence_id: String,
    pub tokens: Vec<i32>,
    pub positions: Vec<i32>,
    pub sample_last: bool,
    pub phase: IterationPhase,
    pub sampling: Option<SamplingConfig>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IterationPlan {
    pub work: Vec<IterationWork>,
    pub token_count: usize,
    pub admitted: usize,
    pub preempted: usize,
}
