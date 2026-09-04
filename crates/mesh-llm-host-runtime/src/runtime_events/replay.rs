//! Bounded, immutable replay retention.
//!
//! Frames are appended only after the reducer has acknowledged a wake entry,
//! so replay never holds partially applied state. Retention is bounded on
//! all three frozen dimensions (frame count, age, bytes; first limit wins):
//! a push evicts from the front until every bound is satisfied. The byte
//! dimension uses a fixed, per-frame memory-footprint estimate
//! ([`APPROX_FRAME_BYTE_COST`]) rather than a wire-serialized size —
//! `RuntimeFact` deliberately never derives `Serialize` (see `frames.rs` in
//! the API layer), so this bound approximates retained memory, not bytes
//! that will cross the wire.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{EventSequence, OperationScope, RuntimeFact};

use super::config::{REPLAY_MAX_AGE, REPLAY_MAX_BYTES, REPLAY_MAX_FRAMES};

#[derive(Debug, Clone)]
pub struct ReplayFrame {
    pub sequence: EventSequence,
    pub rebuild_generation: u64,
    pub scope: OperationScope,
    pub fact: Arc<RuntimeFact>,
    pub recorded_at: Instant,
}

/// Fixed per-frame memory-footprint estimate for the byte dimension of both
/// replay retention and per-subscriber lag (shared with `subscribers.rs`).
/// Deliberately NOT a wire-serialized size: `RuntimeFact` never derives
/// `Serialize`, so this is `size_of::<ReplayFrame>()` plus the largest
/// family fact's own inline size — a real, deterministic, compiler-computed
/// lower bound on retained memory per frame, not a fabricated constant.
pub(crate) const APPROX_FRAME_BYTE_COST: usize =
    std::mem::size_of::<ReplayFrame>() + std::mem::size_of::<RuntimeFact>();

struct RetainedFrame {
    frame: ReplayFrame,
    byte_cost: usize,
}

struct Inner {
    frames: VecDeque<RetainedFrame>,
    total_bytes: usize,
}

#[derive(Debug)]
pub struct ReplayBuffer {
    inner: Mutex<Inner>,
    capacity: usize,
    max_bytes: usize,
    max_age: Duration,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Inner")
            .field("frame_count", &self.frames.len())
            .field("total_bytes", &self.total_bytes)
            .finish()
    }
}

impl ReplayBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(REPLAY_MAX_FRAMES)
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_bounds(capacity, REPLAY_MAX_BYTES, REPLAY_MAX_AGE)
    }

    /// Full three-dimensional constructor. `with_capacity` delegates here
    /// using the frozen `REPLAY_MAX_BYTES`/`REPLAY_MAX_AGE` values; tests
    /// use this directly to exercise the byte/age dimensions with small,
    /// deterministic bounds.
    #[must_use]
    pub fn with_bounds(capacity: usize, max_bytes: usize, max_age: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                frames: VecDeque::new(),
                total_bytes: 0,
            }),
            capacity,
            max_bytes,
            max_age,
        }
    }

    /// Append `frame`, then evict from the front while ANY of the three
    /// frozen bounds is exceeded (first limit wins — whichever bound is
    /// currently violated drives eviction, not all three at once). Returns
    /// the NUMBER of frames evicted (`0` when none were).
    ///
    /// Task 8-fix E1 (`.omo/plans/event-system-fixes.md`): this used to
    /// return `bool` ("did at least one eviction happen"), which the sole
    /// production caller (`engine::drain::apply_and_publish_fact`) turned
    /// into exactly one `EngineHealth::bump_replay_evicted()` regardless of
    /// how many frames were actually evicted. That undercounts: a single
    /// push CAN evict more than one frame. The count/byte dimensions can
    /// never do this on their own when frames are pushed one at a time
    /// (each push's own eviction loop below restores the invariant before
    /// the next push can violate it again), but the AGE dimension can —
    /// many already-retained frames can all go stale between two pushes,
    /// independent of how many pushes happened while they sat there (see
    /// `tests::age_bound_evicts_every_frame_that_outlives_it_not_just_the_oldest`
    /// below, and `engine::drain::tests` for the engine-level proof through
    /// the real `apply_and_publish_fact` call site). The frozen
    /// `replay_evicted` semantics are "one increment per evicted frame",
    /// so the caller must know the real count, not just "at least one".
    pub fn push(&self, frame: ReplayFrame) -> usize {
        self.push_at(frame, Instant::now())
    }

    /// Same as [`Self::push`] with an explicit "now", so age-bound eviction
    /// is deterministically testable without sleeping on the wall clock.
    pub(crate) fn push_at(&self, frame: ReplayFrame, now: Instant) -> usize {
        let byte_cost = APPROX_FRAME_BYTE_COST;
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.frames.push_back(RetainedFrame { frame, byte_cost });
        inner.total_bytes += byte_cost;

        let mut evicted_count = 0usize;
        while self.exceeds_any_bound(&inner, now) {
            let Some(removed) = inner.frames.pop_front() else {
                break;
            };
            inner.total_bytes -= removed.byte_cost;
            evicted_count += 1;
        }
        evicted_count
    }

    fn exceeds_any_bound(&self, inner: &Inner, now: Instant) -> bool {
        if inner.frames.len() > self.capacity {
            return true;
        }
        if inner.total_bytes > self.max_bytes {
            return true;
        }
        inner.frames.front().is_some_and(|oldest| {
            now.saturating_duration_since(oldest.frame.recorded_at) > self.max_age
        })
    }

    /// Drop every retained frame, used by `rebuild()` to make a fresh
    /// generation's replay window coherent rather than mixing generations.
    pub fn evict_all(&self) -> usize {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let count = inner.frames.len();
        inner.frames.clear();
        inner.total_bytes = 0;
        count
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .frames
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<ReplayFrame> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .frames
            .iter()
            .map(|retained| retained.frame.clone())
            .collect()
    }
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{FamilyFact, NativeRuntimeEventKind, OperationId};

    use super::*;

    fn frame(sequence: u64) -> ReplayFrame {
        frame_at(sequence, Instant::now())
    }

    fn frame_at(sequence: u64, recorded_at: Instant) -> ReplayFrame {
        ReplayFrame {
            sequence: EventSequence::new(sequence),
            rebuild_generation: 0,
            scope: OperationScope::root_only(OperationId::new()),
            fact: Arc::new(RuntimeFact::NativeRuntime(FamilyFact::new(
                NativeRuntimeEventKind::RuntimeStopped,
            ))),
            recorded_at,
        }
    }

    #[test]
    fn push_beyond_capacity_evicts_the_oldest_frame() {
        let buffer = ReplayBuffer::with_capacity(2);
        assert_eq!(buffer.push(frame(0)), 0);
        assert_eq!(buffer.push(frame(1)), 0);
        assert_eq!(buffer.push(frame(2)), 1);

        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2]);
    }

    #[test]
    fn evict_all_clears_the_buffer_and_reports_the_prior_count() {
        let buffer = ReplayBuffer::with_capacity(4);
        buffer.push(frame(0));
        buffer.push(frame(1));

        assert_eq!(buffer.evict_all(), 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn push_beyond_the_byte_bound_evicts_the_oldest_frame_even_under_the_frame_count_bound() {
        // 1,000 frames of headroom on the count dimension; only 3 frames'
        // worth of headroom on the byte dimension, so the byte bound must
        // be the one that fires.
        let buffer = ReplayBuffer::with_bounds(1_000, APPROX_FRAME_BYTE_COST * 3, Duration::MAX);
        assert_eq!(buffer.push(frame(0)), 0);
        assert_eq!(buffer.push(frame(1)), 0);
        assert_eq!(buffer.push(frame(2)), 0);
        assert_eq!(buffer.push(frame(3)), 1);

        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2, 3]);
    }

    #[test]
    fn push_beyond_the_age_bound_evicts_stale_frames_even_under_the_frame_count_bound() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(30));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(0, start), start), 0);
        assert_eq!(
            buffer.push_at(
                frame_at(1, start + Duration::from_secs(10)),
                start + Duration::from_secs(10)
            ),
            0
        );

        // Sequence 0 is now 31s old (past the 30s age bound); sequence 1 is
        // only 21s old (still within it).
        let now = start + Duration::from_secs(31);
        let evicted = buffer.push_at(frame_at(2, now), now);
        assert_eq!(evicted, 1);
        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2]);
    }

    #[test]
    fn age_bound_evicts_every_frame_that_outlives_it_not_just_the_oldest() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(10));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(0, start), start), 0);
        assert_eq!(
            buffer.push_at(
                frame_at(1, start + Duration::from_secs(1)),
                start + Duration::from_secs(1)
            ),
            0
        );

        // Both prior frames are now well past the 10s bound; only the new
        // frame should remain.
        let now = start + Duration::from_secs(50);
        let evicted = buffer.push_at(frame_at(2, now), now);
        assert_eq!(
            evicted, 2,
            "one push must count BOTH stale frames it evicts here, not just \
             report that at least one eviction happened"
        );
        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![2]);
    }

    #[test]
    fn frame_count_bound_still_wins_when_it_is_the_tightest_dimension() {
        // Byte and age bounds are effectively unlimited here; only the
        // frame-count dimension is tight, proving first-limit-wins picks
        // whichever bound actually fires, not always the byte/age ones.
        let buffer = ReplayBuffer::with_bounds(2, usize::MAX, Duration::MAX);
        assert_eq!(buffer.push(frame(0)), 0);
        assert_eq!(buffer.push(frame(1)), 0);
        assert_eq!(buffer.push(frame(2)), 1);
        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2]);
    }
}
