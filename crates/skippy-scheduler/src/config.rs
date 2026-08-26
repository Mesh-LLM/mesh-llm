use std::time::Duration;

use crate::LLAMA_MAX_SEQ;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryComponent {
    pub name: String,
    pub capacity_bytes: u64,
    pub resident_bytes: u64,
    pub bytes_per_token: u64,
    pub bytes_per_sequence: u64,
}

impl MemoryComponent {
    pub fn available_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.resident_bytes)
    }

    pub fn request_bytes(&self, prompt_tokens: usize, max_tokens: u32) -> u64 {
        let token_count = u64::try_from(prompt_tokens)
            .unwrap_or(u64::MAX)
            .saturating_add(u64::from(max_tokens));
        self.bytes_per_sequence
            .saturating_add(self.bytes_per_token.saturating_mul(token_count))
    }

    pub fn reservation_bytes(&self, admission_tokens: usize) -> u64 {
        self.bytes_per_sequence.saturating_add(
            self.bytes_per_token
                .saturating_mul(u64::try_from(admission_tokens).unwrap_or(u64::MAX)),
        )
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_active_sequences: usize,
    pub reserved_sequence_ids: usize,
    pub max_waiting_sequences: usize,
    pub max_tokens_per_iteration: usize,
    pub prefill_chunk_tokens: usize,
    pub max_prefill_sequences_per_iteration: usize,
    /// Maximum prefill/recompute iterations allowed while decode work is live.
    /// `usize::MAX` preserves unbounded prefill-first scheduling.
    pub max_consecutive_prefill_iterations: usize,
    /// Schedule live decode rows first, then use the remaining token budget for
    /// prefill and recompute rows in the same native iteration.
    pub mixed_prefill_decode: bool,
    /// Fairness credit added to a waiting request on every scheduler turn.
    /// Cache-aware admission uses this to prevent cold-prefix starvation.
    pub cache_aging_cost_per_iteration: u64,
    /// Group equal-value waiting requests by shared prompt-prefix subtree.
    pub group_waiting_prefixes: bool,
    pub iteration_interval: Duration,
    pub memory_components: Vec<MemoryComponent>,
}

impl SchedulerConfig {
    pub fn normalized(mut self) -> Self {
        let sequence_ceiling = LLAMA_MAX_SEQ.saturating_sub(self.reserved_sequence_ids);
        self.max_active_sequences = self.max_active_sequences.clamp(1, sequence_ceiling.max(1));
        self.max_waiting_sequences = self.max_waiting_sequences.max(1);
        self.max_tokens_per_iteration = self.max_tokens_per_iteration.max(1);
        self.prefill_chunk_tokens = self
            .prefill_chunk_tokens
            .clamp(1, self.max_tokens_per_iteration);
        self.max_prefill_sequences_per_iteration = self.max_prefill_sequences_per_iteration.max(1);
        self.max_consecutive_prefill_iterations = self.max_consecutive_prefill_iterations.max(1);
        self.cache_aging_cost_per_iteration = self.cache_aging_cost_per_iteration.max(1);
        self
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_active_sequences: 32,
            reserved_sequence_ids: 16,
            max_waiting_sequences: 256,
            max_tokens_per_iteration: 2048,
            prefill_chunk_tokens: 256,
            max_prefill_sequences_per_iteration: usize::MAX,
            max_consecutive_prefill_iterations: usize::MAX,
            mixed_prefill_decode: false,
            cache_aging_cost_per_iteration: 4_096,
            group_waiting_prefixes: true,
            iteration_interval: Duration::from_millis(2),
            memory_components: Vec::new(),
        }
        .normalized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_sequence_limit_accounts_for_reserved_ids() {
        let config = SchedulerConfig {
            max_active_sequences: usize::MAX,
            reserved_sequence_ids: 24,
            ..SchedulerConfig::default()
        }
        .normalized();
        assert_eq!(config.max_active_sequences, LLAMA_MAX_SEQ - 24);
    }

    #[test]
    fn request_memory_includes_fixed_recurrent_state() {
        let component = MemoryComponent {
            name: "recurrent".into(),
            capacity_bytes: 1_000_000,
            resident_bytes: 0,
            bytes_per_token: 0,
            bytes_per_sequence: 4096,
        };
        assert_eq!(component.request_bytes(8192, 512), 4096);
    }
}
