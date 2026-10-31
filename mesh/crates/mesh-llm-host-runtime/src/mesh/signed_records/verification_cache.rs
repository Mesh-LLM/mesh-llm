//! Reuse cryptographic verification, never admission decisions.

use mesh_llm_membership::signed_record::{RecordError, RecordKind, VerifiedRecord};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

const MAX_ENTRIES: usize = 4096;
const MAX_SIGNED_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    domain: &'static [u8],
    signature: [u8; 64],
}

struct Entry {
    kind: RecordKind,
    record: VerifiedRecord,
}

#[derive(Default)]
struct Entries {
    records: HashMap<Key, Entry>,
    insertion_order: VecDeque<Key>,
    signed_bytes: usize,
}

/// Per-node FIFO bounded by both record count and retained signed bytes.
/// Failed verification never occupies the cache. Hits still require identical
/// signed bytes, signature and kind; the signature is only a lookup key.
pub(super) struct VerificationCache {
    entries: Mutex<Entries>,
    max_entries: usize,
    max_signed_bytes: usize,
    #[cfg(test)]
    pub(super) verifications: std::sync::atomic::AtomicUsize,
}

impl Default for VerificationCache {
    fn default() -> Self {
        Self {
            entries: Mutex::default(),
            max_entries: MAX_ENTRIES,
            max_signed_bytes: MAX_SIGNED_BYTES,
            #[cfg(test)]
            verifications: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl VerificationCache {
    pub(super) fn verify(
        &self,
        kind: RecordKind,
        signed: &[u8],
        signature: &[u8],
    ) -> Result<VerifiedRecord, RecordError> {
        let key = Key {
            domain: kind.domain_tag,
            signature: signature
                .try_into()
                .map_err(|_| RecordError::InvalidSignatureLength)?,
        };
        {
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = entries.records.get(&key)
                && entry.kind == kind
                && entry.record.signed_bytes() == signed
            {
                return Ok(entry.record.clone());
            }
        }
        // Do expensive signature work outside both the cache and membership
        // locks. Concurrent misses may verify twice, but insertion is idempotent.
        #[cfg(test)]
        self.verifications
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let record = VerifiedRecord::verify(kind, signed, signature)?;
        self.insert(key, kind, &record);
        Ok(record)
    }

    fn insert(&self, key: Key, kind: RecordKind, record: &VerifiedRecord) {
        let bytes = record.signed_bytes().len();
        if self.max_entries == 0 || bytes > self.max_signed_bytes {
            return;
        }
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.records.contains_key(&key) {
            return;
        }
        while entries.records.len() >= self.max_entries
            || entries.signed_bytes > self.max_signed_bytes - bytes
        {
            let Some(oldest) = entries.insertion_order.pop_front() else {
                return;
            };
            if let Some(old) = entries.records.remove(&oldest) {
                entries.signed_bytes -= old.record.signed_bytes().len();
            }
        }
        entries.signed_bytes += bytes;
        entries.insertion_order.push_back(key.clone());
        entries.records.insert(
            key,
            Entry {
                kind,
                record: record.clone(),
            },
        );
    }
}

#[cfg(test)]
mod tests;
