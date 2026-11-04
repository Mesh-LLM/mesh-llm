//! Which challenge or rows request a stage works on next, and which it
//! turns away.
//!
//! An opening replays a turn and a rows request reads stored rows; both
//! hold the runtime's lock while the stage serves requests, so a stage
//! works on [`RUNNING`] of them at a time. The others wait in a bounded
//! queue: at most [`MAX_WAITING`] in all and [`MAX_WAITING_PER_KEY`] of one
//! client key, each for at most [`MAX_WAIT`]. When several wait, the key
//! that has used the least time since the stage was last idle goes next,
//! so a client key that sends many challenges cannot hold the stage while
//! another client's challenge waits. The turns bound to no key count as one
//! key.
//!
//! A request past these bounds gets [`Busy`]: HTTP 503 with a
//! `Retry-After`. It is not signed, as a cannot-open answer is: the
//! receipt format has no busy answer a node signs. The challenge is not
//! admitted, so the client can send it again.

use std::collections::HashMap;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Openings and rows requests a stage works on at a time.
pub(super) const RUNNING: usize = 1;
/// Requests that wait for the stage, at most. With [`RUNNING`] and replays
/// of about a second each, the last one waits a few seconds. The turn a
/// waiting challenge names stays pinned while it waits and is opened
/// (`Recorder::hold_for_opening`).
pub(super) const MAX_WAITING: usize = 8;
/// Requests of one client key that wait, at most: a challenge and its rows
/// request, so one key cannot fill the queue.
pub(super) const MAX_WAITING_PER_KEY: usize = 2;
/// How long a request waits, at most, before it gets [`Busy`]: an answer
/// later than this would reach a client past the challenge window anyway.
pub(super) const MAX_WAIT: Duration = Duration::from_secs(8);
/// When a turned-away client should try again.
pub(super) const RETRY_AFTER: Duration = Duration::from_secs(2);

/// The client key a request is queued under; `None` for a turn bound to no
/// key.
pub(super) type QueueKey = Option<[u8; 32]>;

/// The stage turned a request away before admitting it; the client may
/// send it again after `retry_after`.
#[derive(Debug)]
pub(super) struct Busy {
    pub reason: String,
    pub retry_after: Duration,
}

impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}; try again in {} s",
            self.reason,
            self.retry_after.as_secs()
        )
    }
}

impl std::error::Error for Busy {}

fn busy(reason: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Busy {
        reason: reason.into(),
        retry_after: RETRY_AFTER,
    })
}

struct Waiter {
    ticket: u64,
    key: QueueKey,
}

#[derive(Default)]
struct QueueState {
    running: usize,
    /// In arrival order.
    waiting: Vec<Waiter>,
    /// By key, the time its requests ran since the stage was last idle.
    used: HashMap<QueueKey, Duration>,
    next_ticket: u64,
}

impl QueueState {
    /// The waiter that goes next: of the key that used the least time, the
    /// one that came first.
    fn next(&self) -> Option<u64> {
        self.waiting
            .iter()
            .min_by_key(|w| (self.used.get(&w.key).copied().unwrap_or_default(), w.ticket))
            .map(|w| w.ticket)
    }

    fn leave(&mut self, ticket: u64) {
        self.waiting.retain(|w| w.ticket != ticket);
    }
}

#[derive(Default)]
pub(super) struct OpeningQueue {
    state: Mutex<QueueState>,
    changed: Condvar,
}

/// A request's turn at the stage; the next one goes when it is dropped.
pub(super) struct Slot<'a> {
    queue: &'a OpeningQueue,
    key: QueueKey,
    started: Instant,
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let mut state = self.queue.lock();
        state.running -= 1;
        *state.used.entry(self.key).or_default() += self.started.elapsed();
        if state.running == 0 && state.waiting.is_empty() {
            // the stage is idle: every key starts even again
            state.used.clear();
        }
        self.queue.changed.notify_all();
    }
}

impl OpeningQueue {
    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Waits for a request of `key` to have its turn at the stage, or
    /// fails with [`Busy`].
    pub(super) fn enter(&self, key: QueueKey) -> anyhow::Result<Slot<'_>> {
        self.enter_within(key, MAX_WAIT)
    }

    fn enter_within(&self, key: QueueKey, wait: Duration) -> anyhow::Result<Slot<'_>> {
        let mut state = self.lock();
        if state.running < RUNNING && state.waiting.is_empty() {
            return Ok(self.start(&mut state, key));
        }
        if state.waiting.len() >= MAX_WAITING {
            return Err(busy(format!(
                "the stage is opening other turns, and {MAX_WAITING} requests wait"
            )));
        }
        if state.waiting.iter().filter(|w| w.key == key).count() >= MAX_WAITING_PER_KEY {
            return Err(busy(format!(
                "{MAX_WAITING_PER_KEY} requests of this client key wait already"
            )));
        }
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        state.waiting.push(Waiter { ticket, key });
        let deadline = Instant::now() + wait;
        loop {
            if state.running < RUNNING && state.next() == Some(ticket) {
                state.leave(ticket);
                return Ok(self.start(&mut state, key));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                state.leave(ticket);
                // another waiter may be next now
                self.changed.notify_all();
                return Err(busy(format!(
                    "the stage was opening other turns for {} s",
                    wait.as_secs_f64()
                )));
            }
            state = self
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    fn start(&self, state: &mut QueueState, key: QueueKey) -> Slot<'_> {
        state.running += 1;
        Slot {
            queue: self,
            key,
            started: Instant::now(),
        }
    }

    #[cfg(test)]
    pub(super) fn waiting(&self) -> usize {
        self.lock().waiting.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc;

    use super::*;

    const A: QueueKey = Some([1; 32]);
    const B: QueueKey = Some([2; 32]);

    fn is_busy(result: &anyhow::Result<Slot<'_>>) -> bool {
        result
            .as_ref()
            .err()
            .is_some_and(|e| e.downcast_ref::<Busy>().is_some())
    }

    /// Waits until `n` requests wait in `queue`.
    fn until_waiting(queue: &OpeningQueue, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while queue.waiting() != n {
            assert!(
                Instant::now() < deadline,
                "{} wait, not {n}",
                queue.waiting()
            );
            std::thread::yield_now();
        }
    }

    // One key's flood waits behind no more than its share, and the
    // challenge of another key goes before the flood's next request.
    #[test]
    fn a_key_that_floods_the_stage_does_not_starve_another_key() {
        let queue = Arc::new(OpeningQueue::default());
        let running = queue.enter(A).unwrap();
        let (order, finished) = mpsc::channel();
        let waiter = |key: QueueKey, name: &'static str| {
            let (queue, order) = (queue.clone(), order.clone());
            std::thread::spawn(move || {
                let slot = queue.enter_within(key, Duration::from_secs(60));
                order.send((name, slot.is_ok())).unwrap();
            })
        };
        let flood = [waiter(A, "a1"), waiter(A, "a2")];
        until_waiting(&queue, 2);
        // past its share, the flood is turned away at once
        assert!(is_busy(&queue.enter_within(A, Duration::from_secs(60))));
        let other = waiter(B, "b");
        until_waiting(&queue, 3);
        drop(running);
        for thread in flood.into_iter().chain([other]) {
            thread.join().unwrap();
        }
        let order: Vec<_> = finished.try_iter().collect();
        assert_eq!(order[0], ("b", true), "{order:?}");
        assert!(order.iter().all(|(_, ok)| *ok), "{order:?}");
    }

    // The queue is bounded in all, and a request that waits too long is
    // turned away; a turned-away request leaves no trace.
    #[test]
    fn a_full_queue_and_a_long_wait_answer_busy() {
        let queue = Arc::new(OpeningQueue::default());
        let running = queue.enter(None).unwrap();
        let threads: Vec<_> = (0..MAX_WAITING)
            .map(|i| {
                let queue = queue.clone();
                std::thread::spawn(move || {
                    let key = Some([u8::try_from(i).unwrap() + 10; 32]);
                    queue
                        .enter_within(key, Duration::from_secs(60))
                        .map(|_| ())
                        .is_ok()
                })
            })
            .collect();
        until_waiting(&queue, MAX_WAITING);
        let error = queue.enter(B).err().unwrap();
        let busy = error.downcast_ref::<Busy>().unwrap();
        assert_eq!(busy.retry_after, RETRY_AFTER);
        drop(running);
        assert!(threads.into_iter().all(|t| t.join().unwrap()));
        let running = queue.enter(A).unwrap();
        assert!(is_busy(&queue.enter_within(B, Duration::ZERO)));
        assert_eq!(queue.waiting(), 0);
        drop(running);
        assert!(queue.enter(B).is_ok());
    }
}
