//! Lane-owned attention KV with exact, bounded recurrent checkpoints.
//!
//! No cache sequence pins these cells a second time. Checkpoints cannot outlive
//! their lane. Reuse on a miss goes through `lane_recycle`: the recorder
//! takes the eligible snapshots or the lane stays parked; cleanup and
//! memory pressure discard under a named policy.
use super::*;
use crate::kv_integration::{
    ExactStateRestore, KvStageIntegration, PrefillKvIdentity, ResidentRecurrentPolicy,
};

use super::lane_recycle::{CheckpointDiscard, Recycle};
use super::recurrent_checkpoints::ResidentRecurrentCheckpoint;

impl RuntimeState {
    /// Opt in only callers that own the successful park/error-reset lifecycle.
    pub(crate) fn enable_resident_recurrent_request(&mut self, session_id: &str) {
        self.resident_recurrent_requests
            .insert(session_id.to_owned());
    }
    /// Release idle resident checkpoints under memory pressure. The recorder
    /// gets first refusal; what it declines is discarded, because freeing
    /// the cells is the point. A failed reset discards the native lane.
    pub fn evict_idle_recurrent(&mut self) -> Result<usize> {
        let mut evicted = 0;
        for index in (0..self.idle_sessions.len()).rev() {
            if self.idle_sessions[index].recurrent_prefixes.is_empty() {
                continue;
            }
            let lane = self.idle_sessions.swap_remove(index);
            let lane = match lane.into_recyclable(self.layer_start, self.layer_end) {
                Recycle::Ready(lane) => lane,
                Recycle::Retained(retained) => retained
                    .lane
                    .into_discarded(CheckpointDiscard::MemoryPressure),
            };
            evicted += 1;
            let lane = self.reset_recycled(lane)?;
            self.idle_sessions.push(lane);
        }
        Ok(evicted)
    }

    pub(crate) fn record_resident_recurrent(
        &mut self,
        session_id: &str,
        identity: &PrefillKvIdentity,
        cache: &KvStageIntegration,
        policy: ResidentRecurrentPolicy,
    ) -> Result<Option<(u64, u64, usize)>> {
        // Checkpoints retain only their owning lane's cells. Admission reuses
        // or resets an idle lane before allocating its replacement request.
        if self.max_idle_sessions == Some(0)
            || !self.resident_recurrent_requests.contains(session_id)
        {
            return Ok(None);
        }
        let count = identity.token_ids.len() as u64;
        if self.session_token_count(session_id) != Some(count)
            || identity.identity.token_count != count
        {
            return Ok(None);
        }
        let Some(lane) = self.sessions.get_mut(session_id) else {
            return Ok(None);
        };
        anyhow::ensure!(
            lane.session.native_position()? == count,
            "resident checkpoint position disagrees with the native lane"
        );
        if !lane.session.has_resident_recurrent_prefix(count) {
            return Ok(None);
        }
        let recurrent = lane.session.export_recurrent_cache_state()?;
        anyhow::ensure!(
            !recurrent.is_empty(),
            "resident checkpoint has no recurrent state"
        );
        let bytes = recurrent.len() as u64;
        let (limit, byte_limit) = cache.resident_recurrent_limits();
        if limit == 0 || (byte_limit > 0 && bytes > byte_limit) {
            return Ok(None);
        }
        let displaced = lane.recurrent_prefixes.insert(
            ResidentRecurrentCheckpoint {
                identity: identity.clone(),
                recurrent,
                cache: cache.clone(),
                policy,
                recorded: false,
            },
            limit,
            byte_limit,
        );
        for entry in displaced {
            if entry.durable() {
                entry.demote(&mut lane.session, self.layer_start, self.layer_end);
            } else {
                entry.superseded();
            }
        }
        let retained_bytes = lane.recurrent_prefixes.retained_bytes();
        Ok(Some((bytes, retained_bytes, lane.recurrent_prefixes.len())))
    }

    /// A successful request may lose its idle slot to another completed lane.
    /// Export eligible checkpoints before ordinary cleanup releases native KV;
    /// with the pool full there is nowhere to keep what the recorder refuses.
    pub(crate) fn demote_completed_recurrent(&mut self, session_id: &str) {
        if let Some(lane) = self.sessions.get_mut(session_id)
            && lane
                .release_checkpoints(self.layer_start, self.layer_end)
                .is_err()
        {
            lane.recurrent_prefixes
                .discard(CheckpointDiscard::IdlePoolFull);
        }
    }

    /// Graceful unload preserves successful parked lanes. Active requests and
    /// panic cleanup remain discard-only; cache storage is always best effort,
    /// and a refused export simply unloads with the lane.
    pub(super) fn demote_idle_recurrent_on_unload(&mut self) {
        if std::thread::panicking() {
            return;
        }
        for lane in &mut self.idle_sessions {
            let _ = lane.release_checkpoints(self.layer_start, self.layer_end);
        }
    }

    /// Only successful, non-cancelled local generation may park a lane.
    /// Ordinary cleanup (including rollback and panic recovery) always resets.
    pub(crate) fn park_resident_recurrent(&mut self, session_id: &str) -> bool {
        if self
            .max_idle_sessions
            .is_some_and(|limit| self.idle_sessions.len() >= limit)
        {
            return false;
        }
        if !self
            .sessions
            .get(session_id)
            .is_some_and(|lane| !lane.recurrent_prefixes.is_empty())
        {
            return false;
        }
        let mut lane = self
            .sessions
            .remove(session_id)
            .expect("checked active lane");
        lane.parked_at = Instant::now();
        // The idle gap is free time for the recorder. On a host-visible stage
        // the cells are the payload, so parking is where recording starts;
        // by recycle time there is usually nothing left to export.
        if self.kv_cell_access() == skippy_runtime::KvCellAccess::HostVisible {
            lane.record_in_place(self.layer_start, self.layer_end);
        }
        self.idle_sessions.push(lane);
        self.session_token_counts.remove(session_id);
        self.session_resident_prefixes.remove(session_id);
        self.resident_recurrent_requests.remove(session_id);
        true
    }

    pub(crate) fn restore_resident_recurrent(
        &mut self,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        if self.sessions.contains_key(session_id)
            || !self.resident_recurrent_requests.contains(session_id)
        {
            return Ok(None);
        }
        self.ensure_not_reset_by_panic(session_id)?;
        let started = Instant::now();
        let selected = self
            .idle_sessions
            .iter()
            .enumerate()
            .flat_map(|(lane_index, lane)| {
                lane.recurrent_prefixes.iter().enumerate().filter_map(
                    move |(checkpoint_index, entry)| {
                        let count = entry.identity.token_ids.len();
                        // Restoring trims to `count`; a recorder reader still
                        // copying a longer page must not be cut underneath.
                        (identities.iter().any(|query| {
                            query.namespace == entry.identity.namespace
                                && query.token_ids.starts_with(&entry.identity.token_ids)
                        }) && lane.session.has_resident_recurrent_prefix(count as u64)
                            && !lane.leases.blocks_trim_to(count as u64))
                        .then_some((count, lane_index, checkpoint_index))
                    },
                )
            })
            .max();
        let Some((count, lane_index, checkpoint_index)) = selected else {
            return Ok(None);
        };
        let mut lane = self.idle_sessions.swap_remove(lane_index);
        let checkpoint = lane.recurrent_prefixes.restore(checkpoint_index);
        let page_id = checkpoint.identity.page_id.clone();
        let bytes = checkpoint.recurrent.len() as u64;
        let lookup_ms = started.elapsed().as_secs_f64() * 1000.0;
        let import_started = Instant::now();
        // Partial restore leaves attention intact. Set position retires any
        // speculative verify state; trim then removes only the obsolete suffix.
        let restored = lane
            .session
            .import_recurrent_cache_state_for_token_count(&checkpoint.recurrent, count as u64)
            .and_then(|()| lane.session.trim_session(count as u64))
            .and_then(|()| {
                lane.session
                    .restore_token_history(&checkpoint.identity.token_ids)
            });
        if let Err(error) = restored {
            self.free_lane_indices.push(lane.index);
            return Err(error);
        }
        let entries = lane.recurrent_prefixes.len();
        self.sessions.insert(session_id.to_owned(), lane);
        self.session_token_counts
            .insert(session_id.to_owned(), count as u64);
        Ok(Some(ExactStateRestore {
            page_id,
            token_count: count,
            payload_kind: skippy_cache::ExactStatePayloadKind::KvRecurrent,
            logical_bytes: bytes,
            entries,
            reconstruct_ms: 0.0,
            reconstruct_bytes: 0,
            reconstruct_blocks: 0,
            lookup_ms,
            kv_import_ms: 0.0,
            recurrent_import_ms: import_started.elapsed().as_secs_f64() * 1000.0,
            source: "resident_recurrent",
            fill_ms: 0.0,
            rewarm_enqueued: false,
        }))
    }
}
