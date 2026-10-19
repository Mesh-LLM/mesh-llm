//! Bounded recurrent snapshots belonging to one lane's attention prefix.
use crate::kv_integration::{KvStageIntegration, PrefillKvIdentity, ResidentRecurrentPolicy};

pub(super) struct ResidentRecurrentCheckpoint {
    pub identity: PrefillKvIdentity,
    pub recurrent: Vec<u8>,
    pub cache: KvStageIntegration,
    pub policy: ResidentRecurrentPolicy,
}

#[derive(Default)]
pub(super) struct RecurrentCheckpoints {
    entries: Vec<ResidentRecurrentCheckpoint>,
}

impl RecurrentCheckpoints {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &ResidentRecurrentCheckpoint> {
        self.entries.iter()
    }

    pub fn retained_bytes(&self) -> u64 {
        self.iter().map(|entry| entry.recurrent.len() as u64).sum()
    }

    /// Admission has already checked that the new entry fits both limits.
    /// Return displaced, still-compatible checkpoints for optional demotion.
    pub fn insert(
        &mut self,
        mut entry: ResidentRecurrentCheckpoint,
        limit: usize,
        byte_limit: u64,
    ) -> Vec<ResidentRecurrentCheckpoint> {
        assert!(limit > 0);
        let bytes = entry.recurrent.len() as u64;
        assert!(byte_limit == 0 || bytes <= byte_limit);
        entry.policy.write_through_l3 |= self.iter().any(|previous| {
            previous.identity.namespace == entry.identity.namespace
                && previous.identity.token_ids == entry.identity.token_ids
                && previous.policy.write_through_l3
        });
        self.entries.retain(|previous| {
            previous.identity.namespace == entry.identity.namespace
                && entry
                    .identity
                    .token_ids
                    .starts_with(&previous.identity.token_ids)
                && previous.identity.token_ids != entry.identity.token_ids
        });
        let mut displaced = Vec::new();
        while self.len() >= limit
            || (byte_limit > 0 && self.retained_bytes().saturating_add(bytes) > byte_limit)
        {
            // Prefer the shared boundary to superseded request continuations.
            let index = self
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, entry)| !entry.policy.write_through_l3)
                .map(|(index, _)| index)
                .unwrap_or(usize::from(self.len() > 1));
            displaced.push(self.entries.remove(index));
        }
        self.entries.push(entry);
        displaced
    }

    /// Later checkpoints refer to attention suffixes removed by this restore.
    pub fn restore(&mut self, index: usize) -> &ResidentRecurrentCheckpoint {
        self.entries.truncate(index + 1);
        &self.entries[index]
    }

    /// Export the shared boundary and latest continuation, avoiding intermediate
    /// full prompt copies. The caller owns native export and admission.
    pub fn take_for_demotion(&mut self) -> Vec<ResidentRecurrentCheckpoint> {
        let mut entries = std::mem::take(&mut self.entries);
        let newest = entries.last().map(|entry| entry.identity.token_ids.len());
        let oldest = entries.first().map(|entry| entry.identity.token_ids.len());
        entries.sort_by_key(|entry| {
            std::cmp::Reverse((
                Some(entry.identity.token_ids.len()) == oldest,
                Some(entry.identity.token_ids.len()) == newest,
                entry.policy.write_through_l3,
                entry.identity.token_ids.len(),
            ))
        });
        entries.truncate(2);
        entries
    }
}

#[cfg(test)]
mod tests;
