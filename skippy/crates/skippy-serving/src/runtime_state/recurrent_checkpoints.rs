//! Bounded recurrent snapshots belonging to one lane's attention prefix.
//!
//! Ownership rule: a checkpoint leaves this set only with a
//! [`RecorderReceipt`] from the recorder, or under a named
//! [`CheckpointDiscard`] policy. A refusal puts its bytes back before the
//! call returns. No accessor hands out an owned checkpoint.
use crate::kv_integration::exact_state_diagnostics::emit_exact_state_diagnostic;
use crate::kv_integration::{
    DemotionRefusal, InPlaceRecord, KvStageIntegration, LeaseLedger, PrefillKvIdentity,
    RecorderReceipt, ResidentDemotion, ResidentRecurrentPolicy,
};
use skippy_runtime::StageSession;

pub(super) struct ResidentRecurrentCheckpoint {
    pub identity: PrefillKvIdentity,
    pub recurrent: Vec<u8>,
    pub cache: KvStageIntegration,
    pub policy: ResidentRecurrentPolicy,
    /// The recorder already holds this checkpoint (an in-place record taken
    /// at park time). It stays resident for its owner's next turn, and a
    /// recycle may drop it without offering it again.
    pub recorded: bool,
}

impl ResidentRecurrentCheckpoint {
    fn position(&self) -> u64 {
        self.identity.token_ids.len() as u64
    }

    /// Hand this checkpoint's bytes to the recorder. Only the bytes move: on
    /// a receipt the caller drops the now-empty entry, on a refusal the bytes
    /// are back in place before this returns.
    fn demote(
        &mut self,
        session: &mut StageSession,
        start: u32,
        end: u32,
    ) -> Result<RecorderReceipt, DemotionRefusal> {
        let recurrent = std::mem::take(&mut self.recurrent);
        match self.cache.demote_resident_recurrent(
            session,
            start..end,
            &self.identity,
            recurrent,
            self.policy,
        ) {
            ResidentDemotion::Queued(receipt) | ResidentDemotion::Lost(receipt) => Ok(receipt),
            ResidentDemotion::Refused { recurrent, reason } => {
                self.recurrent = recurrent;
                Err(reason)
            }
        }
    }

    /// Record this checkpoint from the lane's own cells, keeping it resident.
    /// The attention KV is leased, not copied; the small recurrent half is
    /// cloned because the owner's next turn still imports it from here.
    fn record_in_place(
        &mut self,
        session: &mut StageSession,
        start: u32,
        end: u32,
        ledger: &LeaseLedger,
    ) -> Result<(), DemotionRefusal> {
        match self.cache.record_resident_recurrent_in_place(
            session,
            start..end,
            &self.identity,
            &self.recurrent,
            self.policy,
            ledger,
        ) {
            InPlaceRecord::Queued => {
                self.recorded = true;
                Ok(())
            }
            // The resident copy is intact; the next recycle offers it again.
            InPlaceRecord::Unsupported | InPlaceRecord::Lost => Ok(()),
            InPlaceRecord::Refused(reason) => Err(reason),
        }
    }
}

/// Why a lane is allowed to lose checkpoints the recorder never received.
/// Every variant is a policy decision made at a named call site; there is no
/// "best effort" variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CheckpointDiscard {
    /// Rollback, cancellation, or panic recovery: the request did not finish.
    Cleanup,
    /// The idle pool is at `cache_idle_slots`; a completed lane has nowhere
    /// to park and the recorder refused its export.
    IdlePoolFull,
    /// Explicit eviction under memory pressure.
    MemoryPressure,
    /// Every native lane is occupied and the recorder cannot accept this
    /// lane's remaining checkpoints. Serving takes precedence over caching.
    AdmissionPressure,
    /// The recorder worker has stopped. Nothing can be recorded on this stage
    /// again, so retaining would only turn every lane into "lanes busy".
    RecorderStopped,
    /// The recorder kept refusing this lane past `MAX_RETAINED_REFUSAL`.
    /// Back-pressure is bounded: a recorder that cannot make progress is
    /// treated as failed and the loss is counted, instead of the stage
    /// refusing requests forever.
    RetentionExpired,
}

impl CheckpointDiscard {
    fn event(self) -> &'static str {
        match self {
            Self::Cleanup => "resident_discarded_cleanup",
            Self::IdlePoolFull => "resident_discarded_idle_pool_full",
            Self::MemoryPressure => "resident_discarded_memory_pressure",
            Self::AdmissionPressure => "resident_discarded_admission_pressure",
            Self::RecorderStopped => "resident_discarded_recorder_stopped",
            Self::RetentionExpired => "resident_discarded_retention_expired",
        }
    }
}

/// A checkpoint `insert` pushed out of a full lane. The lane has no slot
/// to return it to, so the caller must either demote it or name it
/// superseded; dropping it silently does not compile past `#[must_use]`.
#[must_use = "a displaced checkpoint must be demoted or marked superseded"]
pub(super) struct DisplacedCheckpoint(ResidentRecurrentCheckpoint);

impl DisplacedCheckpoint {
    pub fn durable(&self) -> bool {
        self.0.policy.write_through_l3 && self.0.cache.has_durable_recurrent_tier()
    }

    /// Offer it to the recorder. A refusal is a loss here: capacity already
    /// evicted it from the lane. A recorded entry is already the recorder's.
    pub fn demote(mut self, session: &mut StageSession, start: u32, end: u32) {
        if self.0.recorded {
            return;
        }
        if self.0.demote(session, start, end).is_err() {
            self.0.cache.notify_demotion_error();
            emit_exact_state_diagnostic(
                "resident_displaced_lost",
                &[("position", self.0.position())],
            );
        }
    }

    /// Hot continuation replacement must not serialize attention KV.
    pub fn superseded(self) {
        if self.0.recorded {
            return;
        }
        emit_exact_state_diagnostic(
            "resident_displaced_superseded",
            &[("position", self.0.position())],
        );
    }
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

    pub fn iter(&self) -> impl Iterator<Item = &ResidentRecurrentCheckpoint> {
        self.entries.iter()
    }

    pub fn retained_bytes(&self) -> u64 {
        self.iter().map(|entry| entry.recurrent.len() as u64).sum()
    }

    /// Offer every unrecorded checkpoint to the recorder, in rank order. The
    /// first refusal ends the pass: it and every unoffered entry stay in the
    /// lane, in prefix order. Only entries with a receipt are dropped;
    /// recorded entries are already the recorder's and stay for their owner.
    /// `Ok` therefore means no unrecorded checkpoint remains.
    pub fn release(
        &mut self,
        session: &mut StageSession,
        start: u32,
        end: u32,
    ) -> Result<(), DemotionRefusal> {
        self.release_with(|entry| entry.demote(session, start, end))
    }

    /// `release` with the recorder abstracted: `offer` returns a receipt for
    /// an entry that may leave, or the refusal that keeps it. Every entry
    /// handed to `offer` is unrecorded.
    fn release_with(
        &mut self,
        mut offer: impl FnMut(
            &mut ResidentRecurrentCheckpoint,
        ) -> Result<RecorderReceipt, DemotionRefusal>,
    ) -> Result<(), DemotionRefusal> {
        let ranked = self.take_for_demotion();
        let mut kept = Vec::with_capacity(ranked.len());
        let mut refusal = None;
        for mut entry in ranked {
            if entry.recorded || refusal.is_some() {
                kept.push(entry);
                continue;
            }
            match offer(&mut entry) {
                Ok(_receipt) => drop(entry),
                Err(reason) => {
                    refusal = Some(reason);
                    kept.push(entry);
                }
            }
        }
        self.retain_resident(kept);
        refusal.map_or(Ok(()), Err)
    }

    /// Record every unrecorded checkpoint from the lane's cells, in rank
    /// order, without moving anything out of the lane. A refusal stops the
    /// pass; the rest record on a later park or at recycle time.
    pub fn record_in_place(
        &mut self,
        session: &mut StageSession,
        start: u32,
        end: u32,
        ledger: &LeaseLedger,
    ) -> Result<usize, DemotionRefusal> {
        let mut ranked = self.take_for_demotion();
        let mut recorded = 0;
        let mut refusal = None;
        for entry in ranked.iter_mut() {
            if entry.recorded || refusal.is_some() {
                continue;
            }
            match entry.record_in_place(session, start, end, ledger) {
                Ok(()) => recorded += usize::from(entry.recorded),
                Err(reason) => refusal = Some(reason),
            }
        }
        self.retain_resident(ranked);
        refusal.map_or(Ok(recorded), Err)
    }

    /// The lane's cells are gone. A `RecyclableLane` only exists once every
    /// unrecorded checkpoint has left (`release`) or been discarded under a
    /// named policy, so everything still here holds a receipt. An unrecorded
    /// entry at this point is a broken invariant, not a policy, and is
    /// reported as one.
    pub fn forget_after_reset(&mut self) {
        for entry in self.entries.drain(..) {
            if !entry.recorded {
                debug_assert!(false, "unrecorded checkpoint survived into reset");
                tracing::error!(
                    position = entry.position(),
                    "unrecorded checkpoint survived into a lane reset; this is a bug"
                );
                entry.cache.notify_demotion_error();
                emit_exact_state_diagnostic(
                    "resident_reset_unrecorded",
                    &[("position", entry.position())],
                );
            }
        }
    }

    /// The only way checkpoints leave a lane without reaching the recorder.
    /// Each loss is counted against the owning stage and named in the log;
    /// an entry the recorder already holds is not a loss.
    pub fn discard(&mut self, reason: CheckpointDiscard) {
        for entry in self.entries.drain(..) {
            if entry.recorded {
                continue;
            }
            entry.cache.notify_demotion_error();
            emit_exact_state_diagnostic(reason.event(), &[("position", entry.position())]);
        }
    }

    /// Admission has already checked that the new entry fits both limits.
    /// Return displaced, still-compatible checkpoints for optional demotion.
    pub fn insert(
        &mut self,
        mut entry: ResidentRecurrentCheckpoint,
        limit: usize,
        byte_limit: u64,
    ) -> Vec<DisplacedCheckpoint> {
        assert!(limit > 0);
        let bytes = entry.recurrent.len() as u64;
        assert!(byte_limit == 0 || bytes <= byte_limit);
        entry.policy.write_through_l3 |= self.iter().any(|previous| {
            previous.identity.namespace == entry.identity.namespace
                && previous.identity.token_ids == entry.identity.token_ids
                && previous.policy.write_through_l3
        });
        self.entries.retain(|previous| {
            let compatible = previous.identity.namespace == entry.identity.namespace
                && entry
                    .identity
                    .token_ids
                    .starts_with(&previous.identity.token_ids)
                && previous.identity.token_ids != entry.identity.token_ids;
            let replacing_same_identity = previous.identity.namespace == entry.identity.namespace
                && previous.identity.token_ids == entry.identity.token_ids;
            if !compatible && !replacing_same_identity {
                emit_exact_state_diagnostic(
                    "resident_branch_discarded",
                    &[("position", previous.position())],
                );
            }
            compatible
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
            let checkpoint = self.entries.remove(index);
            emit_exact_state_diagnostic(
                "resident_capacity_displaced",
                &[("position", checkpoint.position())],
            );
            displaced.push(DisplacedCheckpoint(checkpoint));
        }
        self.entries.push(entry);
        displaced
    }

    /// Later checkpoints refer to attention suffixes removed by this restore.
    pub fn restore(&mut self, index: usize) -> &ResidentRecurrentCheckpoint {
        for checkpoint in self.entries.iter().skip(index + 1) {
            emit_exact_state_diagnostic(
                "resident_restore_suffix_discarded",
                &[("position", checkpoint.position())],
            );
        }
        self.entries.truncate(index + 1);
        &self.entries[index]
    }

    /// Offer durable shared boundaries before request-specific continuations.
    /// A recycle queues a bounded prefix of this order, so the nearest and
    /// preceding shared boundaries must come first: a shortened chat tail can
    /// diverge before the nearest boundary and still reuse the older one.
    /// Later continuations follow as fallbacks.
    fn take_for_demotion(&mut self) -> Vec<ResidentRecurrentCheckpoint> {
        let mut entries = std::mem::take(&mut self.entries);
        entries.sort_by_key(|entry| {
            std::cmp::Reverse((
                entry.policy.write_through_l3,
                entry.identity.token_ids.len(),
            ))
        });
        entries
    }

    /// Entries a demotion pass did not consume go back in prefix order.
    /// `restore` and `insert` both rely on ascending token counts.
    fn retain_resident(&mut self, entries: impl IntoIterator<Item = ResidentRecurrentCheckpoint>) {
        self.entries.extend(entries);
        self.entries
            .sort_by_key(|entry| entry.identity.token_ids.len());
    }
}

#[cfg(test)]
mod tests;
