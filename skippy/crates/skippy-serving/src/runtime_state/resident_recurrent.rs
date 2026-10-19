//! Lane-owned attention KV with exact, bounded recurrent checkpoints.
//!
//! No cache sequence pins these cells a second time. Checkpoints cannot outlive
//! their lane: reset/error cleanup drops them; reuse on a miss exports eligible
//! snapshots before resetting the lane.
use super::*;
use crate::kv_integration::{
    ExactStateRestore, KvStageIntegration, PrefillKvIdentity, ResidentRecurrentPolicy,
};

pub(super) struct ResidentRecurrentCheckpoint {
    identity: PrefillKvIdentity,
    recurrent: Vec<u8>,
    cache: KvStageIntegration,
    policy: ResidentRecurrentPolicy,
}

impl RuntimeState {
    /// Opt in only callers that own the successful park/error-reset lifecycle.
    pub(crate) fn enable_resident_recurrent_request(&mut self, session_id: &str) {
        self.resident_recurrent_requests
            .insert(session_id.to_owned());
    }
    /// Release idle resident checkpoints under memory pressure. Admission to
    /// lower tiers is best effort; a failed reset discards the native lane.
    pub fn evict_idle_recurrent(&mut self) -> Result<usize> {
        let mut evicted = 0;
        for index in (0..self.idle_sessions.len()).rev() {
            if self.idle_sessions[index].recurrent_prefixes.is_empty() {
                continue;
            }
            let mut lane = self.idle_sessions.swap_remove(index);
            Self::demote_recurrent_lane(&mut lane, self.layer_start, self.layer_end);
            evicted += 1;
            if let Err(error) = lane.session.reset() {
                self.free_lane_indices.push(lane.index);
                return Err(error);
            }
            self.idle_sessions.push(lane);
        }
        Ok(evicted)
    }

    pub(crate) fn record_resident_recurrent(
        &mut self,
        session_id: &str,
        identity: &PrefillKvIdentity,
        cache: &KvStageIntegration,
        mut policy: ResidentRecurrentPolicy,
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
        let recurrent = lane.session.export_recurrent_state()?;
        anyhow::ensure!(
            !recurrent.is_empty(),
            "resident checkpoint has no recurrent state"
        );
        let bytes = recurrent.len() as u64;
        let (limit, byte_limit) = cache.resident_recurrent_limits();
        if limit == 0 || (byte_limit > 0 && bytes > byte_limit) {
            return Ok(None);
        }
        policy.write_through_l3 |= lane.recurrent_prefixes.iter().any(|entry| {
            entry.identity.namespace == identity.namespace
                && entry.identity.token_ids == identity.token_ids
                && entry.policy.write_through_l3
        });
        lane.recurrent_prefixes.retain(|entry| {
            entry.identity.namespace == identity.namespace
                && identity.token_ids.starts_with(&entry.identity.token_ids)
                && entry.identity.token_ids != identity.token_ids
        });
        while lane.recurrent_prefixes.len() >= limit
            || (byte_limit > 0
                && lane
                    .recurrent_prefixes
                    .iter()
                    .map(|entry| entry.recurrent.len() as u64)
                    .sum::<u64>()
                    .saturating_add(bytes)
                    > byte_limit)
        {
            // Preserve the shared branch boundary ahead of superseded
            // request-specific continuations when the bounded set fills.
            let index = lane
                .recurrent_prefixes
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, entry)| !entry.policy.write_through_l3)
                .map(|(index, _)| index)
                .unwrap_or(usize::from(lane.recurrent_prefixes.len() > 1));
            let entry = lane.recurrent_prefixes.remove(index);
            // Superseded hot checkpoints are cheap to discard. Do not turn
            // every growing turn back into a full attention export merely to
            // populate another L1 representation. Durable boundaries may spill.
            if entry.policy.write_through_l3 && entry.cache.has_durable_recurrent_tier() {
                Self::demote_recurrent_checkpoint(lane, entry, self.layer_start, self.layer_end);
            }
        }
        lane.recurrent_prefixes.push(ResidentRecurrentCheckpoint {
            identity: identity.clone(),
            recurrent,
            cache: cache.clone(),
            policy,
        });
        let retained_bytes = lane
            .recurrent_prefixes
            .iter()
            .map(|entry| entry.recurrent.len() as u64)
            .sum();
        Ok(Some((bytes, retained_bytes, lane.recurrent_prefixes.len())))
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
                        (identities.iter().any(|query| {
                            query.namespace == entry.identity.namespace
                                && query.token_ids.starts_with(&entry.identity.token_ids)
                        }) && lane.session.has_resident_recurrent_prefix(count as u64))
                        .then_some((count, lane_index, checkpoint_index))
                    },
                )
            })
            .max();
        let Some((count, lane_index, checkpoint_index)) = selected else {
            return Ok(None);
        };
        let mut lane = self.idle_sessions.swap_remove(lane_index);
        // Later checkpoints refer to attention suffixes about to be removed.
        lane.recurrent_prefixes.truncate(checkpoint_index + 1);
        let checkpoint = &lane.recurrent_prefixes[checkpoint_index];
        let page_id = checkpoint.identity.page_id.clone();
        let bytes = checkpoint.recurrent.len() as u64;
        let lookup_ms = started.elapsed().as_secs_f64() * 1000.0;
        let import_started = Instant::now();
        // Partial restore leaves attention intact. Set position retires any
        // speculative verify state; trim then removes only the obsolete suffix.
        let restored = lane
            .session
            .import_recurrent_state_for_token_count(&checkpoint.recurrent, count as u64)
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

    pub(super) fn demote_recurrent_lane(lane: &mut RuntimeLaneSession, start: u32, end: u32) {
        // Save the shared boundary first: generated continuations may diverge\n        // from the next tool turn. Avoid exporting intermediate full prompts.
        let mut entries = std::mem::take(&mut lane.recurrent_prefixes);
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
        for entry in entries.into_iter().take(2) {
            Self::demote_recurrent_checkpoint(lane, entry, start, end);
        }
    }

    fn demote_recurrent_checkpoint(
        lane: &mut RuntimeLaneSession,
        entry: ResidentRecurrentCheckpoint,
        start: u32,
        end: u32,
    ) {
        // Only owned bytes go to the recorder, so native cells can safely be
        // released afterward. Optional cache admission may discard the entry.
        if !lane
            .session
            .has_resident_recurrent_prefix(entry.identity.token_ids.len() as u64)
        {
            return;
        }
        if let Err(error) = entry.cache.demote_resident_recurrent(
            &mut lane.session,
            start,
            end,
            &entry.identity,
            entry.recurrent,
            entry.policy,
        ) {
            entry.cache.notify_demotion_error();
            // Errors affect optional cache retention, never generation output.
            tracing::debug!(error = %error, "resident recurrent checkpoint demotion failed");
        }
    }
}
