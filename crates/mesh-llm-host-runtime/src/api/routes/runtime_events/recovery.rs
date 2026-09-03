//! Connection-shape classification: no-cursor, in-window, or a replay gap.
//!
//! Reads only the engine's already-public surfaces (`process_instance`,
//! `highest_known_sequence`, `replay().snapshot()`) — no reducer or replay
//! logic is reimplemented here.

use std::sync::Arc;

use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::replay::ReplayFrame;

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

    let snapshot = engine.replay().snapshot();
    if cursor.process_instance != engine.process_instance() {
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
    if missed_frames_exist {
        let oldest = oldest_sequence(&snapshot);
        let evicted = match oldest {
            None => true,
            Some(oldest) => oldest > cursor.sequence.saturating_add(1),
        };
        if evicted {
            return Ok(ConnectionShape::Gap(Gap {
                reason: GapReason::Evicted,
                requested: cursor,
                oldest_available: oldest,
                latest: latest_sequence(&snapshot),
            }));
        }
    }

    let frames = snapshot
        .into_iter()
        .filter(|frame| frame.sequence.get() > cursor.sequence)
        .collect();
    Ok(ConnectionShape::InWindow { frames })
}

fn oldest_sequence(snapshot: &[ReplayFrame]) -> Option<u64> {
    snapshot.first().map(|frame| frame.sequence.get())
}

fn latest_sequence(snapshot: &[ReplayFrame]) -> Option<u64> {
    snapshot.last().map(|frame| frame.sequence.get())
}
