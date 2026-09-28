//! Producer for the `EventSystemHealth` fact family.
//!
//! Engine health is counted in [`EngineHealth`] and delivered out of band
//! (`runtime_health` frames, the `event_system_health` log line). This
//! module also turns growth in those counters into reducer facts, so
//! `runtime_state.node.event_system` reflects what the engine observed.
//!
//! The engine driver calls [`HealthFactProducer::observe`] after every drain
//! pass. For each watched kind it compares the current total against the
//! total it last reported, and submits at most one fact per kind per
//! [`HEALTH_PUBLISH_MIN_INTERVAL`]. Growth inside the interval is not lost:
//! it is reported by the next emission, because the baseline only moves
//! when a fact is submitted.
//!
//! Kind mapping (all counters monotonic):
//!
//! | kind | source counters |
//! | --- | --- |
//! | `events_coalesced` | `coalesced_progress` |
//! | `events_dropped_by_class` | `dropped_progress`, `dropped_diagnostic`, `dropped_native` |
//! | `subscriber_disconnected` | `subscriber_disconnected` |
//! | `reducer_error` | `reducer_rejected`, `state_transition_rejected` |
//! | `unknown_native_event_received` | `rejected_native` |
//!
//! `dropped_native` maps to `events_dropped_by_class` rather than
//! `ingress_queue_pressure`: the counter records records already lost, and
//! the reducer treats pressure as a sticky latest-wins state with no
//! producer-side clear, so reporting a past loss as current pressure would
//! leave `node.event_system.pressure` set forever.
//!
//! Every fact carries only bounded static numeric-summary keys (`delta`,
//! `total`, and one per source counter): no identifiers, no free text.
//!
//! Feedback: these facts are StateTransition-class (fixed by the contract)
//! and go through the unreserved ingress with a fresh root, so the reducer
//! cannot reject them as duplicates or settled. The only counter one of
//! them can move by itself is the one its own [`SubmitOutcome`] names
//! (for example `state_transition_rejected` on a full ring); that increment
//! is credited to the baseline immediately, so a refused health fact never
//! triggers another. Indirect effects (a subscriber disconnected while
//! fanning a health fact out) are bounded by the per-kind interval.

use std::sync::Arc;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, EventSystemHealthEventKind, EventSystemHealthFact, FactData,
    NumericSummary, NumericSummaryKey, NumericValue, OperationId, OperationScope,
    RuntimeEventIngress, RuntimeFact, SubmitOutcome,
};

use super::config::HEALTH_PUBLISH_MIN_INTERVAL;
use super::engine::RuntimeEventEngine;
use super::health::EngineHealthSnapshot;

/// Numeric-summary key carrying the growth since the previous fact.
pub const DELTA_KEY: &str = "delta";
/// Numeric-summary key carrying the running total of the source counters.
pub const TOTAL_KEY: &str = "total";

/// One source counter: its static summary key and how to read it.
type Source = (&'static str, fn(&EngineHealthSnapshot) -> u64);

struct WatchedKind {
    kind: EventSystemHealthEventKind,
    sources: &'static [Source],
}

const WATCHED: &[WatchedKind] = &[
    WatchedKind {
        kind: EventSystemHealthEventKind::EventsCoalesced,
        sources: &[("progress", |s| s.coalesced_progress)],
    },
    WatchedKind {
        kind: EventSystemHealthEventKind::EventsDroppedByClass,
        sources: &[
            ("progress", |s| s.dropped_progress),
            ("diagnostic", |s| s.dropped_diagnostic),
            ("native", |s| s.dropped_native),
        ],
    },
    WatchedKind {
        kind: EventSystemHealthEventKind::SubscriberDisconnected,
        sources: &[("subscribers", |s| s.subscriber_disconnected)],
    },
    WatchedKind {
        kind: EventSystemHealthEventKind::ReducerError,
        sources: &[
            ("reducer_rejected", |s| s.reducer_rejected),
            ("state_transition_rejected", |s| s.state_transition_rejected),
        ],
    },
    WatchedKind {
        kind: EventSystemHealthEventKind::UnknownNativeEventReceived,
        sources: &[("native_rejected", |s| s.rejected_native)],
    },
];

/// Per-kind reporting state: source values as of the last fact, and when
/// that fact was submitted.
#[derive(Debug, Clone, Default)]
struct KindCursor {
    reported: Vec<u64>,
    last_emit: Option<Instant>,
}

/// Diffs engine health against what was last reported and submits
/// `EventSystemHealth` facts for growth. Owned by the engine driver.
#[derive(Debug)]
pub struct HealthFactProducer {
    cursors: Vec<KindCursor>,
}

impl Default for HealthFactProducer {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthFactProducer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cursors: WATCHED
                .iter()
                .map(|watched| KindCursor {
                    reported: vec![0; watched.sources.len()],
                    last_emit: None,
                })
                .collect(),
        }
    }

    /// Submit one fact per watched kind whose counters grew since the last
    /// fact of that kind, unless that kind emitted less than
    /// [`HEALTH_PUBLISH_MIN_INTERVAL`] ago. Returns the number submitted.
    pub fn observe(&mut self, engine: &Arc<RuntimeEventEngine>, now: Instant) -> usize {
        let snapshot = engine.health().snapshot();
        let mut submitted = 0;
        for (index, watched) in WATCHED.iter().enumerate() {
            let cursor = &mut self.cursors[index];
            let current: Vec<u64> = watched
                .sources
                .iter()
                .map(|(_, read)| read(&snapshot))
                .collect();
            if !is_due(cursor, &current, now) {
                continue;
            }
            let fact = health_fact(watched, &cursor.reported, &current);
            cursor.reported = current;
            cursor.last_emit = Some(now);
            let outcome = engine
                .unreserved_ingress(OperationScope::root_only(OperationId::new()))
                .try_submit(fact);
            self.credit_own_outcome(outcome);
            submitted += 1;
        }
        submitted
    }

    /// Move the baseline past the counter this producer's own submission
    /// just incremented, so it is never reported as new growth. A
    /// StateTransition-class submission can move exactly one watched
    /// counter: `state_transition_rejected`, when the ring is full.
    fn credit_own_outcome(&mut self, outcome: SubmitOutcome) {
        if outcome != SubmitOutcome::RejectedCapacity {
            return;
        }
        let Some(index) = WATCHED
            .iter()
            .position(|watched| watched.kind == EventSystemHealthEventKind::ReducerError)
        else {
            return;
        };
        let Some(slot) = WATCHED[index]
            .sources
            .iter()
            .position(|(key, _)| *key == "state_transition_rejected")
        else {
            return;
        };
        let reported = &mut self.cursors[index].reported[slot];
        *reported = reported.saturating_add(1);
    }
}

fn is_due(cursor: &KindCursor, current: &[u64], now: Instant) -> bool {
    let grew = current
        .iter()
        .zip(&cursor.reported)
        .any(|(current, reported)| current > reported);
    let interval_elapsed = cursor
        .last_emit
        .is_none_or(|previous| now.duration_since(previous) >= HEALTH_PUBLISH_MIN_INTERVAL);
    grew && interval_elapsed
}

fn health_fact(watched: &WatchedKind, reported: &[u64], current: &[u64]) -> RuntimeFact {
    let deltas: Vec<u64> = current
        .iter()
        .zip(reported)
        .map(|(current, reported)| current.saturating_sub(*reported))
        .collect();
    let delta = deltas
        .iter()
        .fold(0u64, |sum, value| sum.saturating_add(*value));
    let total = current
        .iter()
        .fold(0u64, |sum, value| sum.saturating_add(*value));
    let mut entries = vec![(DELTA_KEY, delta), (TOTAL_KEY, total)];
    if watched.sources.len() > 1 {
        entries.extend(
            watched
                .sources
                .iter()
                .zip(&deltas)
                .map(|((key, _), value)| (*key, *value)),
        );
    }
    RuntimeFact::EventSystemHealth(EventSystemHealthFact::with_data(
        watched.kind,
        FactData {
            numeric_summaries: summaries(&entries),
            ..FactData::default()
        },
    ))
}

fn summaries(entries: &[(&'static str, u64)]) -> BoundedNumericSummaries {
    let values = entries
        .iter()
        .filter_map(|(key, value)| {
            let key = NumericSummaryKey::new(key).ok()?;
            Some(NumericSummary::new(key, NumericValue::Unsigned(*value)))
        })
        .collect();
    BoundedNumericSummaries::new(values).unwrap_or_default()
}

#[cfg(test)]
mod tests;
