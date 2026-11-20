//! KV exports a session owes the prefix cache when it ends.
//!
//! An exact-state record for a hybrid model pairs the recurrent state at a
//! token boundary with the KV rows below it. The recurrent state moves on with
//! every decoded token, so it is captured at the boundary. The KV rows below
//! the boundary do not change while the session lives (decode appends, and
//! speculative rollback never reaches below the prompt), so copying them,
//! which takes seconds at long context, waits until the session ends instead
//! of holding the runtime while the request decodes. A sliding-window cache
//! breaks that premise, so a stage with one captures immediately; see
//! [`RuntimeState::kv_rows_stay_resident`].
//!
//! A request that ends normally calls [`RuntimeState::end_session_timed`],
//! which parks the session with its rows intact instead of dropping it, and
//! then schedules [`RuntimeState::complete_parked_session_ends`] as a detached
//! runtime operation, so the copy runs after the response has gone out. Any
//! operation on a session, including lane acquisition, completes parked
//! sessions first: a parked lane is never counted twice, and a parked
//! session's KV cells are released before any other session prefills or
//! decodes into the shared pool, as they were when sessions were dropped. Every other way a session ends or changes (an error drop, a
//! restore, an import, a trim below the boundary, a new session under the
//! same id, panic recovery) abandons the export instead.

use anyhow::Result;

use super::{RuntimeLaneSession, RuntimeSessionDropStats, RuntimeState};

/// Completes a deferred export. Called with the runtime and the session id
/// while the session's rows are still resident, or with `None` when the
/// session changed or ended in a way that no longer guarantees those rows.
pub(crate) type SessionEndKvExportSink = Box<dyn FnOnce(Option<(&mut RuntimeState, &str)>) + Send>;

pub(crate) struct SessionEndKvExport {
    token_count: u64,
    sink: SessionEndKvExportSink,
}

/// A normally ended session whose exports have not run yet.
pub(crate) struct ParkedSessionEnd {
    session_id: String,
    lane: RuntimeLaneSession,
    token_count: Option<u64>,
}

impl RuntimeState {
    /// Whether rows `[0, token_count)` of `session_id` stay resident until
    /// the session ends, which deferring their export relies on.
    ///
    /// Full-attention and recurrent memories only append while a session
    /// decodes. A sliding-window cache may reuse the cells that fall out of
    /// the window, whatever its size (`swa_full` only sizes it), so a session
    /// with one does not qualify, and neither does one whose page layout
    /// cannot be read.
    pub(crate) fn kv_rows_stay_resident(&mut self, session_id: &str, token_count: u64) -> bool {
        #[cfg(test)]
        if self.modelless_for_test {
            return !self.sliding_window_kv_for_test;
        }
        self.kv_page_layout(session_id, 0, token_count)
            .is_ok_and(|layout| !layout.has_sliding_window_component())
    }

    /// Run `sink` when the session ends, while its KV rows
    /// `[0, token_count)` are still resident.
    pub(crate) fn export_kv_at_session_end(
        &mut self,
        session_id: &str,
        token_count: u64,
        sink: SessionEndKvExportSink,
    ) {
        self.session_end_kv_exports
            .entry(session_id.to_string())
            .or_default()
            .push(SessionEndKvExport { token_count, sink });
    }

    /// End a request's session. A session that owes exports is parked with
    /// its rows intact until [`Self::complete_parked_session_ends`] runs;
    /// any other is dropped as by [`Self::drop_session_timed`].
    ///
    /// Ending a session that is already parked (a repeated Stop) changes
    /// nothing, so its exports still run.
    pub fn end_session_timed(&mut self, session_id: &str) -> Result<RuntimeSessionDropStats> {
        if self
            .parked_session_ends
            .iter()
            .any(|parked| parked.session_id == session_id)
        {
            return Ok(self.parked_session_stats());
        }
        if !self.session_end_kv_exports.contains_key(session_id) {
            return self.drop_session_timed(session_id);
        }
        let Some(lane) = self.sessions.remove(session_id) else {
            return self.drop_session_timed(session_id);
        };
        self.dflash_deferred_drafts.remove(session_id);
        self.session_resident_prefixes.remove(session_id);
        let token_count = self.session_token_counts.remove(session_id);
        self.parked_session_ends.push(ParkedSessionEnd {
            session_id: session_id.to_string(),
            lane,
            token_count,
        });
        Ok(self.parked_session_stats())
    }

    fn parked_session_stats(&self) -> RuntimeSessionDropStats {
        RuntimeSessionDropStats {
            reset_session: false,
            reset_ms: 0.0,
            preserved_resident_prefix: false,
            lane_discarded: false,
            lane_discard_reason: None,
            stats_after: self.session_stats(),
        }
    }

    /// Run the exports every parked session owes, then drop it.
    ///
    /// Every runtime operation on a session calls this first (see
    /// `RuntimeState::session`), so no prefill, decode or verify runs while a
    /// parked session still claims its KV cells in the shared pool. Its
    /// sinks reach the runtime through the same path, so a call made while
    /// parked sessions are being completed returns at once.
    pub fn complete_parked_session_ends(&mut self) {
        if self.parked_session_ends.is_empty() || self.completing_parked_session_ends {
            return;
        }
        self.completing_parked_session_ends = true;
        self.finish_parked_session_ends(true);
        self.completing_parked_session_ends = false;
    }

    /// Drop every parked session; `complete` runs their exports first.
    ///
    /// Sessions are taken one at a time, so if a sink panics the ones not yet
    /// finished stay parked, and panic recovery returns their lanes.
    pub(super) fn finish_parked_session_ends(&mut self, complete: bool) {
        while let Some(parked) = self.parked_session_ends.pop() {
            let session_id = parked.session_id;
            // Lane acquisition finishes parked sessions first, so the id is
            // free; never displace a live session if it is not.
            // The lane goes back before any sink runs, so a panicking sink
            // cannot strand it.
            if self.sessions.contains_key(&session_id) {
                let index = parked.lane.index;
                drop(parked.lane);
                self.free_lane_indices.push(index);
                self.abandon_session_end_kv_exports(&session_id);
                continue;
            }
            self.sessions.insert(session_id.clone(), parked.lane);
            if let Some(token_count) = parked.token_count {
                self.session_token_counts
                    .insert(session_id.clone(), token_count);
            }
            if complete {
                self.complete_session_end_kv_exports(&session_id);
            }
            let _ = self.drop_session_timed(&session_id);
        }
    }

    /// Run every export `session_id` owes while its rows are resident.
    ///
    /// The longest runs first. The exports run back to back, so the recorder
    /// is usually still storing the first when the next asks for admission,
    /// and a best-effort one is then declined. The longest prefix is the one
    /// the session's next turn can restore the most of.
    fn complete_session_end_kv_exports(&mut self, session_id: &str) {
        let Some(mut exports) = self.session_end_kv_exports.remove(session_id) else {
            return;
        };
        exports.sort_by_key(|export| std::cmp::Reverse(export.token_count));
        for export in exports {
            if self.has_session_range(session_id, 0, export.token_count) {
                (export.sink)(Some((self, session_id)));
            } else {
                (export.sink)(None);
            }
        }
    }

    /// Give up every export `session_id` owes.
    pub(super) fn abandon_session_end_kv_exports(&mut self, session_id: &str) {
        for export in self
            .session_end_kv_exports
            .remove(session_id)
            .unwrap_or_default()
        {
            (export.sink)(None);
        }
    }

    /// Give up the exports whose rows a trim to `token_count` removes.
    pub(super) fn abandon_session_end_kv_exports_above(
        &mut self,
        session_id: &str,
        token_count: u64,
    ) {
        let Some(exports) = self.session_end_kv_exports.remove(session_id) else {
            return;
        };
        let (kept, abandoned): (Vec<_>, Vec<_>) = exports
            .into_iter()
            .partition(|export| export.token_count <= token_count);
        if !kept.is_empty() {
            self.session_end_kv_exports
                .insert(session_id.to_string(), kept);
        }
        for export in abandoned {
            (export.sink)(None);
        }
    }

    /// Give up every export any session owes, and drop parked sessions.
    pub(super) fn abandon_all_session_end_kv_exports(&mut self) {
        for export in std::mem::take(&mut self.session_end_kv_exports)
            .into_values()
            .flatten()
        {
            (export.sink)(None);
        }
        self.finish_parked_session_ends(false);
    }
}

#[cfg(test)]
#[path = "session_end_exports/tests.rs"]
mod tests;
