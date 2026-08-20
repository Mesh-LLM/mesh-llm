use skippy_runtime::ActivationFrame;

use crate::kv_proto::{KvPageManifest, PageIdentity};
use skippy_cache::{CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayloadKind};

#[derive(Debug, Clone)]
pub struct PrefillKvIdentity {
    pub identity: PageIdentity,
    pub page_id: String,
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
    pub borrowed: bool,
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

/// The tier a restored exact-state payload came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactStateSource {
    Ram,
    Disk,
}

impl ExactStateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ram => "ram",
            Self::Disk => "disk",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExactStateRestore {
    pub page_id: String,
    pub token_count: usize,
    pub payload_kind: ExactStatePayloadKind,
    /// Which tier actually served the payload.
    ///
    /// A hit reported without this is unattributable: RAM hits and disk hits
    /// have completely different costs and completely different failure
    /// modes, and a disk tier that has silently stopped serving looks
    /// identical to a warm RAM cache in the telemetry.
    pub source: ExactStateSource,
    pub logical_bytes: u64,
    pub entries: usize,
    pub reconstruct_ms: f64,
    pub reconstruct_bytes: u64,
    pub reconstruct_blocks: usize,
    pub lookup_ms: f64,
    pub kv_import_ms: f64,
    pub recurrent_import_ms: f64,
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
