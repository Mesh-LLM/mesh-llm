//! Connection-shape classification: no-cursor, in-window, or a replay gap.
//!
//! Reads only the engine's already-public surfaces (`process_instance`,
//! `highest_known_sequence`, `replay().snapshot()`/`replay().frames_after()`)
//! — no reducer or replay logic is reimplemented here.
//!
//! Task 9 (`.omo/plans/event-system-fixes.md`, defect D11): the
//! "did we miss anything, and if so is it a gap" decision for a
//! same-instance cursor now delegates to `ReplayBuffer::frames_after`,
//! which enforces the AGE bound AT READ TIME (a frame can go stale while
//! the engine is otherwise idle, with no push ever running to trigger the
//! buffer's own push-time eviction) in addition to the push-time
//! count/byte eviction this module always observed via `snapshot()`.

use std::sync::Arc;
use std::time::Instant;

use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::replay::{ReplayFrame, ReplayLookup};

use super::cursor::{Cursor, CursorError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GapReason {
    StaleInstance,
    Evicted,
}

pub(super) struct Gap {
    pub(super) reason: GapReason,
    pub(super) requested: Cursor,
    pub(super) oldest_available: Option<u64>,
    pub(super) latest: Option<u64>,
}

pub(super) enum ConnectionShape {
    NoCursor,
    InWindow { frames: Vec<ReplayFrame> },
    Gap(Gap),
}

/// Classify `requested` against the engine's live process instance and
/// replay window. A future sequence for the CURRENT instance is a hard
/// error (400, before headers); a stale instance or an evicted sequence is
/// a `Gap` the caller resolves via `runtime_replay_gap` → `runtime_state` →
/// `runtime_health` rather than an error.
pub(super) fn classify(
    engine: &Arc<RuntimeEventEngine>,
    requested: Option<Cursor>,
) -> Result<ConnectionShape, CursorError> {
    let Some(cursor) = requested else {
        return Ok(ConnectionShape::NoCursor);
    };

    if cursor.process_instance != engine.process_instance() {
        // Purely diagnostic (what does THIS instance currently hold?), so
        // the plain push-time-only snapshot is enough here -- unlike the
        // same-instance path below, there is no "gap relative to this
        // cursor" decision to make against a foreign instance.
        let snapshot = engine.replay().snapshot();
        return Ok(ConnectionShape::Gap(Gap {
            reason: GapReason::StaleInstance,
            requested: cursor,
            oldest_available: oldest_sequence(&snapshot),
            latest: latest_sequence(&snapshot),
        }));
    }

    let highest = engine.highest_known_sequence();
    let future_for_current_instance = match highest {
        Some(highest) => cursor.sequence > highest,
        None => cursor.sequence > 0,
    };
    if future_for_current_instance {
        return Err(CursorError::Malformed);
    }

    let missed_frames_exist = highest.is_some_and(|highest| highest > cursor.sequence);
    if !missed_frames_exist {
        // Nothing has ever been minted past this cursor for the current
        // instance, so there is nothing to look up in replay at all.
        return Ok(ConnectionShape::InWindow { frames: Vec::new() });
    }

    match engine
        .replay()
        .frames_after(cursor.sequence, Instant::now())
    {
        ReplayLookup::InWindow(frames) => Ok(ConnectionShape::InWindow { frames }),
        ReplayLookup::Evicted {
            oldest_available,
            latest,
        } => Ok(ConnectionShape::Gap(Gap {
            reason: GapReason::Evicted,
            requested: cursor,
            oldest_available,
            latest,
        })),
    }
}

fn oldest_sequence(snapshot: &[ReplayFrame]) -> Option<u64> {
    snapshot.first().map(|frame| frame.sequence.get())
}

fn latest_sequence(snapshot: &[ReplayFrame]) -> Option<u64> {
    snapshot.last().map(|frame| frame.sequence.get())
}
