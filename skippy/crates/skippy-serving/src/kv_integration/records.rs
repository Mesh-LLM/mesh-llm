use skippy_runtime::ActivationFrame;

use crate::kv_proto::{KvPageManifest, PageIdentity};
use skippy_cache::{CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayloadKind};

#[derive(Debug, Clone)]
pub struct PrefillKvIdentity {
    pub identity: PageIdentity,
    pub page_id: String,
    pub namespace: String,
    pub token_ids: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct LookupBatchOutcome {
    pub pages: Vec<KvPageManifest>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RecordPageOutcome {
    pub manifest: KvPageManifest,
    pub write_ms: f64,
    pub checksum_ms: f64,
}

#[derive(Debug)]
pub struct AttachedPage {
    pub manifest: KvPageManifest,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixRestore {
    pub page_id: String,
    pub token_count: usize,
    pub seq_id: i32,
    pub entries: usize,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixRecord {
    pub page_id: String,
    pub token_count: usize,
    pub seq_id: i32,
    pub stored: bool,
    pub evicted_entries: usize,
    pub evicted_tokens: u64,
    pub entries: usize,
    pub resident_tokens: u64,
}

/// Why the resident cache kept no entry for a prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentRecordSkip {
    /// The cache does not record, or keeps no resident prefixes.
    RecordingOff,
    /// A recurrent checkpoint needs the session at its exact boundary.
    PositionMismatch {
        session_tokens: u64,
    },
    BelowMinTokens {
        min_tokens: u64,
    },
    /// One checkpoint is larger than the whole checkpoint budget.
    OverCheckpointBudget {
        bytes: u64,
        budget_bytes: u64,
    },
    /// Nothing evictable would make room for the entry.
    NoRoom {
        bytes: u64,
    },
}

impl ResidentRecordSkip {
    pub fn reason(self) -> &'static str {
        match self {
            Self::RecordingOff => "recording_off",
            Self::PositionMismatch { .. } => "position_mismatch",
            Self::BelowMinTokens { .. } => "below_min_tokens",
            Self::OverCheckpointBudget { .. } => "over_checkpoint_budget",
            Self::NoRoom { .. } => "no_room",
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResidentRecordOutcome {
    /// `stored` is false when an identical entry was already cached.
    Recorded(ResidentPrefixRecord),
    Skipped(ResidentRecordSkip),
}

impl ResidentRecordOutcome {
    pub fn into_record(self) -> Option<ResidentPrefixRecord> {
        match self {
            Self::Recorded(record) => Some(record),
            Self::Skipped(_) => None,
        }
    }
}

/// Why a resident lookup restored nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentMissReason {
    LookupOff,
    EmptyCache,
    /// No stored prefix shares even the prompt's first token.
    NoSharedPrefix,
    /// Stored prefixes share tokens with the prompt, but none ends on its
    /// path. Recurrent state resumes only at a checkpoint that is a prefix of
    /// the prompt, so the shared tokens cannot be reused.
    NoCheckpointOnPath,
    BelowMinTokens,
    /// A checkpoint on the path matched but was not restored.
    NotRestored,
}

impl ResidentMissReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LookupOff => "lookup_off",
            Self::EmptyCache => "empty_cache",
            Self::NoSharedPrefix => "no_shared_prefix",
            Self::NoCheckpointOnPath => "no_checkpoint_on_path",
            Self::BelowMinTokens => "below_min_tokens",
            Self::NotRestored => "not_restored",
        }
    }

    pub(crate) fn classify(
        resident_entries: usize,
        shared_prefix_tokens: usize,
        checkpoint_prefix_tokens: usize,
        checkpoint_meets_min_tokens: bool,
    ) -> Self {
        match (
            resident_entries,
            checkpoint_prefix_tokens,
            shared_prefix_tokens,
        ) {
            (0, _, _) => Self::EmptyCache,
            (_, 0, 0) => Self::NoSharedPrefix,
            (_, 0, _) => Self::NoCheckpointOnPath,
            _ if checkpoint_meets_min_tokens => Self::NotRestored,
            _ => Self::BelowMinTokens,
        }
    }
}

/// What the resident cache held for a prompt that missed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentMissDiagnosis {
    pub reason: ResidentMissReason,
    pub resident_entries: usize,
    /// Longest prefix the prompt shares with any stored sequence.
    pub shared_prefix_tokens: usize,
    /// Length of the stored sequence that shares it.
    pub stored_prefix_tokens: usize,
    /// Longest stored checkpoint that is a whole prefix of the prompt.
    pub checkpoint_prefix_tokens: usize,
}

#[cfg(test)]
mod resident_miss_reason_tests {
    use super::ResidentMissReason;

    #[test]
    fn a_shared_prefix_without_a_checkpoint_on_the_path_is_named() {
        assert_eq!(
            ResidentMissReason::classify(3, 4_090, 0, false),
            ResidentMissReason::NoCheckpointOnPath
        );
    }

    #[test]
    fn misses_name_their_cause() {
        for (entries, shared, checkpoint, meets_min, reason) in [
            (0, 0, 0, false, ResidentMissReason::EmptyCache),
            (2, 0, 0, false, ResidentMissReason::NoSharedPrefix),
            (2, 64, 16, false, ResidentMissReason::BelowMinTokens),
            (2, 64, 64, true, ResidentMissReason::NotRestored),
        ] {
            assert_eq!(
                ResidentMissReason::classify(entries, shared, checkpoint, meets_min),
                reason
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResidentActivationRestore {
    pub identity: PrefillKvIdentity,
    pub page_id: String,
    pub token_count: usize,
    pub payload_bytes: usize,
    pub entries: usize,
    pub frame: ActivationFrame,
}

#[derive(Debug, Clone)]
pub struct ResidentActivationRecord {
    pub page_id: String,
    pub token_count: usize,
    pub payload_bytes: usize,
    pub evicted_entries: usize,
    pub evicted_bytes: u64,
    pub entries: usize,
    pub resident_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ExactStateRestore {
    pub page_id: String,
    pub token_count: usize,
    pub payload_kind: ExactStatePayloadKind,
    pub logical_bytes: u64,
    pub entries: usize,
    pub reconstruct_ms: f64,
    pub reconstruct_bytes: u64,
    pub reconstruct_blocks: usize,
    pub lookup_ms: f64,
    pub kv_import_ms: f64,
    pub recurrent_import_ms: f64,
    /// Where the state came from: `"radix"` (RAM) or `"l3"` (disk).
    pub source: &'static str,
    /// Store-side cost of an L3 fill (read, digest verification, assembly),
    /// kept apart from the runtime import above so restore thresholds come
    /// from real numbers. Zero for a radix hit.
    pub fill_ms: f64,
    /// Whether an L3 fill's radix re-warm record was accepted by the worker.
    pub rewarm_enqueued: bool,
}

#[derive(Debug, Clone)]
pub struct ExactStateRecord {
    pub page_id: String,
    pub token_count: usize,
    pub payload_kind: ExactStatePayloadKind,
    pub stored: bool,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub entries: usize,
    pub evicted_entries: usize,
    pub evicted_logical_bytes: u64,
    pub dedupe: CacheDedupeStats,
}

impl AttachedPage {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub(crate) fn add_reconstruct_stats(
    total_ms: &mut f64,
    total_bytes: &mut u64,
    total_blocks: &mut usize,
    stats: CacheBytesReconstructStats,
) {
    *total_ms += stats.reconstruct_ms;
    *total_bytes = total_bytes.saturating_add(stats.reconstruct_bytes);
    *total_blocks = total_blocks.saturating_add(stats.reconstruct_blocks);
}
