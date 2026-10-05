//! Test seam: hold a drain pass inside its critical section.
//!
//! The property this subsystem has to prove is that a producer never waits
//! on the consumer. That is hard to test with timing alone, because a fast
//! drain and a non-blocking drain look identical. This seam makes the
//! difference unmissable: a test installs a hold, a drain pass parks inside
//! whatever it is holding for a fixed duration, and producers submit
//! against it. If a producer is coupled to the drain it waits the whole
//! hold; if it is not, it returns immediately. The signal is the hold
//! duration, not jitter.
//!
//! The hold point sits inside the drain's ingress critical section
//! (`engine::drain::drain_up_to_inner_with_work_budget`) on purpose. While
//! producers still take `ingress_gate`, a hold blocks them -- which is the
//! defect. Once ingress no longer has a producer-visible gate, the same
//! hold point sits inside the drain's own exclusive section and producers
//! are unaffected.
//!
//! Test-only. Nothing here is compiled into a non-test build.

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// One installed hold. Reusable across passes: `entered` counts how many
/// drain passes have reached the hold point, so a test can wait for the
/// first one and then know the drain is parked.
#[derive(Debug)]
pub(crate) struct DrainHold {
    entered: Mutex<usize>,
    signal: Condvar,
    duration: Duration,
}

impl DrainHold {
    #[must_use]
    pub(crate) fn new(duration: Duration) -> Self {
        Self {
            entered: Mutex::new(0),
            signal: Condvar::new(),
            duration,
        }
    }

    /// Drain side. Announce arrival, then park for the hold duration while
    /// still holding everything the drain took to get here.
    pub(crate) fn hold(&self) {
        {
            let mut entered = self
                .entered
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *entered += 1;
        }
        self.signal.notify_all();
        std::thread::sleep(self.duration);
    }

    /// Test side. Block until a drain pass has reached the hold point, or
    /// `timeout` elapses. Returns whether a pass arrived.
    pub(crate) fn wait_until_entered(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut entered = self
            .entered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *entered == 0 {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (guard, _) = self
                .signal
                .wait_timeout(entered, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            entered = guard;
        }
        true
    }

    #[must_use]
    pub(crate) fn duration(&self) -> Duration {
        self.duration
    }
}
