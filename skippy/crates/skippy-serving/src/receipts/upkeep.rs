//! One thread per stage for its timed receipt chores: pinning cells anew
//! when a turn's pin runs out, and freeing the replay context once replays
//! stop. It ends when the stage's receipts are dropped, which happens when
//! the model is unloaded.

use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

#[derive(Default)]
struct Signal {
    /// Whether to stop, and whether a chore was asked for since the last run.
    state: Mutex<(bool, bool)>,
    wake: Condvar,
}

impl Signal {
    fn lock(&self) -> std::sync::MutexGuard<'_, (bool, bool)> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Waits until `deadline`, a poke or a stop; `false` on a stop.
    fn wait(&self, deadline: Option<Instant>) -> bool {
        let mut state = self.lock();
        loop {
            if state.0 {
                return false;
            }
            if std::mem::take(&mut state.1) {
                return true;
            }
            state = match deadline {
                None => self
                    .wake
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return true;
                    };
                    self.wake
                        .wait_timeout(state, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
    }
}

/// Runs chores on a thread until dropped.
pub(super) struct Upkeep {
    signal: Arc<Signal>,
    thread: Option<JoinHandle<()>>,
}

impl Upkeep {
    /// Starts a thread that runs `chores` when poked and at the time the
    /// last run asked for. `chores` gets the current time and returns when
    /// to run next without a poke, or `None` to end the thread: the stage
    /// it works for is gone.
    pub(super) fn start(
        mut chores: impl FnMut(Instant) -> Option<Option<Instant>> + Send + 'static,
    ) -> std::io::Result<Self> {
        let signal = Arc::new(Signal::default());
        let shared = signal.clone();
        let thread = std::thread::Builder::new()
            .name("skippy-receipts-upkeep".into())
            .spawn(move || {
                let mut next = None;
                while shared.wait(next) {
                    match chores(Instant::now()) {
                        Some(deadline) => next = deadline,
                        None => return,
                    }
                }
            })?;
        Ok(Self {
            signal,
            thread: Some(thread),
        })
    }

    /// Runs the chores soon: what they time has changed.
    pub(super) fn poke(&self) {
        self.signal.lock().1 = true;
        self.signal.wake.notify_all();
    }
}

impl Drop for Upkeep {
    fn drop(&mut self) {
        self.signal.lock().0 = true;
        self.signal.wake.notify_all();
        let Some(thread) = self.thread.take() else {
            return;
        };
        // the chores may hold the last reference to what owns this
        if thread.thread().id() != std::thread::current().id() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    const GENEROUS: Duration = Duration::from_secs(30);

    #[test]
    fn chores_run_when_poked_and_when_due_and_end_with_their_owner() {
        let (ran, runs) = mpsc::channel();
        let mut due_once = true;
        let upkeep = Upkeep::start(move |now| {
            ran.send(()).ok()?;
            // the first run asks for one more, at once; later ones wait
            Some(std::mem::take(&mut due_once).then_some(now))
        })
        .expect("thread");
        upkeep.poke();
        runs.recv_timeout(GENEROUS).expect("the poked run");
        runs.recv_timeout(GENEROUS).expect("the run that was due");
        upkeep.poke();
        runs.recv_timeout(GENEROUS).expect("another poked run");
        drop(upkeep);
        // the thread ended and dropped its sender
        assert!(runs.recv_timeout(GENEROUS).is_err());
    }
}
