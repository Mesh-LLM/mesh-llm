//! Batches canonical generated tokens into progress-cadence commits.
//!
//! The decode loop produces one token at a time. Before this module, each
//! one became its own `GenerationCommit` -- a `vec![token].into_boxed_slice()`
//! allocation, a `GenerationLifecycleObservation`, and a synchronous
//! submission into whatever ingress is configured, inline with the decode
//! that produced it. At real decode rates that is tens to hundreds of
//! submissions per second per request, all of them carrying a single token.
//!
//! [`GenerationCommit`] is already documented as a delta: `token_ids` is
//! the batch of canonical tokens this commit adds, and
//! `generated_token_count` is the running total *after* applying them. A
//! batch of ten tokens is therefore the same contract as ten batches of
//! one, and the concatenation of every batch is still the exact canonical
//! token sequence in order.
//!
//! Cadence: emit on the first token, then at most once per
//! [`GENERATION_COMMIT_INTERVAL`], plus a mandatory [`Self::flush`] before
//! the generation's terminal so no token is left unreported. The first
//! token is emitted immediately because it is what
//! `first_token_produced` is derived from -- delaying it would delay a
//! latency-critical observation to buy nothing.
//!
//! Cost per token in the steady state: one `Vec::push` into a
//! fixed-capacity buffer and one `Duration` comparison against an elapsed
//! value the decode loop already computed. No allocation: the buffer is
//! preallocated to [`PENDING_CAPACITY`] and a batch that reaches that
//! ceiling flushes early rather than growing.

use std::time::Duration;

use super::generation_receipt::{GenerationCommit, GenerationReceiptConfig};

/// Minimum spacing between commits after the first.
///
/// Mirrors the host runtime-event engine's progress export interval
/// (`runtime_events::config::PROGRESS_EXPORT_INTERVAL`), which coalesces
/// progress to the same cadence on its own. The two constants cannot be
/// shared -- the host runtime depends on this crate, not the other way
/// round -- but they do not need to be equal, only compatible: batching at
/// or above the engine's interval means the engine has nothing left to
/// coalesce, and batching below it would just recreate the per-token
/// submission this module exists to remove.
pub(crate) const GENERATION_COMMIT_INTERVAL: Duration = Duration::from_millis(100);

/// Fixed capacity of the pending-token buffer. Reaching it forces an early
/// commit, so the buffer never reallocates and a fast decode cannot make
/// one batch grow without bound.
pub(crate) const PENDING_CAPACITY: usize = 256;

/// Per-request batching state. Owned by the decode loop; not shared, not
/// locked, and cheap to construct.
///
/// Borrows the receipt config for its whole life so [`Drop`] can flush a
/// residual batch on any exit path, including an early `?` return or a
/// panic unwind. A normal generation still calls [`Self::flush`]
/// explicitly before its terminal, because a batch flushed by `Drop`
/// would land *after* the terminal and be rejected as settled.
pub(crate) struct GenerationCommitBatcher<'config> {
    config: Option<&'config GenerationReceiptConfig>,
    request_id: u64,
    session_id: u64,
    pending: Vec<i32>,
    generated_token_count: usize,
    /// Elapsed-since-request-start at the last emitted commit. `None`
    /// until the first token, which always emits.
    last_emit: Option<Duration>,
}

impl<'config> GenerationCommitBatcher<'config> {
    #[must_use]
    pub(crate) fn new(
        config: Option<&'config GenerationReceiptConfig>,
        request_id: u64,
        session_id: u64,
    ) -> Self {
        Self {
            config,
            request_id,
            session_id,
            // Allocated only when something is actually listening.
            pending: if config.is_some() {
                Vec::with_capacity(PENDING_CAPACITY)
            } else {
                Vec::new()
            },
            generated_token_count: 0,
            last_emit: None,
        }
    }

    /// Record one canonical generated token, emitting a commit when the
    /// cadence is due.
    ///
    /// `elapsed` is time since the request started -- the same value the
    /// decode loop already passes to its receipt observation, so this adds
    /// no clock read.
    pub(crate) fn commit(&mut self, token_id: i32, elapsed: Duration) {
        let Some(config) = self.config else {
            return;
        };
        self.generated_token_count = self.generated_token_count.saturating_add(1);
        self.pending.push(token_id);
        if self.is_due(elapsed) {
            emit(
                config,
                self.request_id,
                self.session_id,
                self.generated_token_count,
                &mut self.pending,
            );
            self.last_emit = Some(elapsed);
        }
    }

    /// Emit whatever is still pending. Call this before the generation's
    /// terminal (`finished` or `abort`) so the final partial batch is not
    /// silently dropped, and so it publishes before the terminal rather
    /// than after it.
    pub(crate) fn flush(&mut self) {
        let Some(config) = self.config else {
            return;
        };
        if self.pending.is_empty() {
            return;
        }
        // Deliberately does not move `last_emit`: there is no further
        // token to space away from this one.
        emit(
            config,
            self.request_id,
            self.session_id,
            self.generated_token_count,
            &mut self.pending,
        );
    }

    /// Total canonical tokens recorded so far, including any still pending.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn generated_token_count(&self) -> usize {
        self.generated_token_count
    }

    fn is_due(&self, elapsed: Duration) -> bool {
        if self.pending.len() >= PENDING_CAPACITY {
            return true;
        }
        match self.last_emit {
            // First token of the generation: emit immediately, because
            // `first_token_produced` is derived from it.
            None => true,
            Some(last) => elapsed.saturating_sub(last) >= GENERATION_COMMIT_INTERVAL,
        }
    }
}

impl Drop for GenerationCommitBatcher<'_> {
    /// Safety net for exit paths that never reach an explicit
    /// [`GenerationCommitBatcher::flush`] -- an early `?` return or a panic
    /// unwind out of the decode loop. A no-op after a normal flush.
    fn drop(&mut self) {
        self.flush();
    }
}

/// Hand the pending batch to `config` and reset the buffer.
///
/// Copies out and then `clear`s rather than replacing the `Vec`: `clear`
/// keeps the preallocated capacity, so the buffer is allocated once per
/// generation instead of once per commit, and `Box::from(&slice)`
/// allocates the exact length with no shrink-to-fit step.
fn emit(
    config: &GenerationReceiptConfig,
    request_id: u64,
    session_id: u64,
    generated_token_count: usize,
    pending: &mut Vec<i32>,
) {
    let token_ids: Box<[i32]> = Box::from(pending.as_slice());
    pending.clear();
    config.committed(GenerationCommit {
        request_id,
        session_id,
        generated_token_count,
        token_ids,
    });
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::Result;

    use super::*;
    use crate::frontend::generation_receipt::{
        GenerationLifecycleIngress, GenerationLifecycleObservation,
    };

    #[derive(Default)]
    struct RecordingIngress {
        commits: Mutex<Vec<(usize, Vec<i32>)>>,
    }

    impl GenerationLifecycleIngress for RecordingIngress {
        fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
            if let GenerationLifecycleObservation::Committed(commit) = observation {
                self.commits
                    .lock()
                    .unwrap()
                    .push((commit.generated_token_count, commit.token_ids.to_vec()));
            }
            Ok(())
        }
    }

    impl RecordingIngress {
        fn commits(&self) -> Vec<(usize, Vec<i32>)> {
            self.commits.lock().unwrap().clone()
        }
    }

    fn config(ingress: &Arc<RecordingIngress>) -> GenerationReceiptConfig {
        GenerationReceiptConfig::from_lifecycle_ingress(Arc::clone(ingress) as Arc<_>)
    }

    fn at(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    /// The first token emits immediately rather than waiting out a cadence
    /// window: `first_token_produced` is derived from it, and delaying that
    /// would delay a latency-critical observation to buy nothing.
    #[test]
    fn the_first_token_emits_immediately() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = config(&ingress);
        let mut batcher = GenerationCommitBatcher::new(Some(&config), 7, 9);

        batcher.commit(11, at(0));

        assert_eq!(ingress.commits(), vec![(1, vec![11])]);
    }

    /// Tokens inside one cadence window accumulate into a single commit,
    /// and the running total is the total after applying that batch.
    #[test]
    fn tokens_inside_one_window_accumulate_into_one_commit() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = config(&ingress);
        let mut batcher = GenerationCommitBatcher::new(Some(&config), 7, 9);

        batcher.commit(1, at(0));
        for (index, token) in [2, 3, 4].into_iter().enumerate() {
            batcher.commit(token, at(10 * (index as u64 + 1)));
        }
        batcher.commit(5, at(100));

        assert_eq!(
            ingress.commits(),
            vec![(1, vec![1]), (5, vec![2, 3, 4, 5])],
            "the second commit carries the whole window and the total after it"
        );
    }

    /// The concatenation of every batch is the exact canonical token
    /// sequence in order -- the property that makes batching contract-safe
    /// for a consumer that needs canonical model evidence.
    #[test]
    fn concatenated_batches_reproduce_the_token_sequence_exactly() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = config(&ingress);
        let tokens: Vec<i32> = (0..500).collect();

        {
            let mut batcher = GenerationCommitBatcher::new(Some(&config), 7, 9);
            for (index, token) in tokens.iter().enumerate() {
                // 3 ms per token: several cadence windows across the run.
                batcher.commit(*token, at(3 * index as u64));
            }
            batcher.flush();
            assert_eq!(batcher.generated_token_count(), tokens.len());
        }

        let commits = ingress.commits();
        let observed: Vec<i32> = commits
            .iter()
            .flat_map(|(_, batch)| batch.iter().copied())
            .collect();
        assert_eq!(observed, tokens);
        assert!(
            commits.len() < tokens.len(),
            "batching must produce fewer commits than tokens; got {} for {}",
            commits.len(),
            tokens.len()
        );

        let totals: Vec<usize> = commits.iter().map(|(total, _)| *total).collect();
        let mut running = 0usize;
        let expected: Vec<usize> = commits
            .iter()
            .map(|(_, batch)| {
                running += batch.len();
                running
            })
            .collect();
        assert_eq!(
            totals, expected,
            "generated_token_count must be the running total after each batch"
        );
    }

    /// A batch that reaches the fixed buffer capacity commits early, so the
    /// buffer never reallocates no matter how fast decode runs.
    #[test]
    fn reaching_the_buffer_capacity_commits_early() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = config(&ingress);
        let mut batcher = GenerationCommitBatcher::new(Some(&config), 7, 9);

        // All inside one cadence window, so only the capacity ceiling can
        // force the second commit.
        for token in 0..=PENDING_CAPACITY as i32 {
            batcher.commit(token, at(0));
        }

        let commits = ingress.commits();
        assert_eq!(commits.len(), 2, "first token, then the capacity ceiling");
        assert_eq!(commits[1].1.len(), PENDING_CAPACITY);
    }

    /// `flush` emits the residual batch and does not move the cadence, so a
    /// terminal flush cannot suppress a later window. Flushing twice is a
    /// no-op.
    #[test]
    fn flush_emits_the_residual_batch_exactly_once() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = config(&ingress);
        let mut batcher = GenerationCommitBatcher::new(Some(&config), 7, 9);

        batcher.commit(1, at(0));
        batcher.commit(2, at(5));
        batcher.flush();
        batcher.flush();

        assert_eq!(ingress.commits(), vec![(1, vec![1]), (2, vec![2])]);
    }

    /// Dropping without an explicit flush still emits the residual batch.
    /// This is the safety net for an early `?` return out of a decode loop.
    #[test]
    fn dropping_flushes_the_residual_batch() {
        let ingress = Arc::new(RecordingIngress::default());
        let config = config(&ingress);

        {
            let mut batcher = GenerationCommitBatcher::new(Some(&config), 7, 9);
            batcher.commit(1, at(0));
            batcher.commit(2, at(5));
        }

        assert_eq!(ingress.commits(), vec![(1, vec![1]), (2, vec![2])]);
    }

    /// With no receipt config there is nothing to batch and nothing to
    /// count: the batcher retains no tokens at all.
    #[test]
    fn no_receipt_config_retains_nothing() {
        let mut batcher = GenerationCommitBatcher::new(None, 7, 9);
        batcher.commit(1, at(0));
        batcher.commit(2, at(500));
        batcher.flush();
        assert_eq!(batcher.generated_token_count(), 0);
    }
}
