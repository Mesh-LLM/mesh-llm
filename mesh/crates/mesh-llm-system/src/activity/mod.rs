//! Cross-platform host activity abstraction and process priority control.
//!
//! This module intentionally does not inspect raw detector internals in logs or
//! telemetry. Backends should surface only coarse `HostActivity` values.

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "windows")]
mod windows;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod unsupported;

use std::time::{Duration, Instant};

/// Coarse host activity inferred from platform signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostActivity {
    /// Recent user-visible activity is detected.
    Active,
    /// No activity has been observed for the configured threshold.
    Idle,
    /// Activity could not be inferred (unsupported/headless/platform/operation error).
    Unknown,
}

/// Session-only manual override for activity state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivityOverride {
    /// Use detector output.
    #[default]
    Auto,
    /// Force active state for this process session.
    Active,
    /// Force idle state for this process session.
    Idle,
}

impl ActivityOverride {
    pub fn resolve(self, detected: HostActivity) -> HostActivity {
        match self {
            Self::Auto => detected,
            Self::Active => HostActivity::Active,
            Self::Idle => HostActivity::Idle,
        }
    }
}

/// Provides host activity samples from the platform.
pub trait HostActivityDetector {
    fn sample(&mut self) -> HostActivity;
}

/// Provides an injectable clock used by the debounce policy.
pub trait ActivityClock {
    fn now(&self) -> Instant;
}

/// Minimal clock implementation backed by monotonic process time.
#[derive(Default, Debug, Clone, Copy)]
pub struct SystemClock;

impl ActivityClock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Provides capture/apply/restore around low-priority execution mode.
pub trait PriorityController {
    /// Capture the current priority and best-effort apply reduced priority.
    fn reduce_priority(&mut self) -> Result<(), PriorityFailure>;

    /// Best-effort restore priority state captured by `reduce_priority`.
    fn restore_priority(&mut self) -> Result<(), PriorityFailure>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriorityFailure {
    /// Platform back-end is unsupported for this process.
    Unsupported,
    /// Reduction could not be applied.
    ApplyFailed,
    /// Previously reduced state could not be restored.
    RestoreFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriorityStatus {
    /// Priority control currently healthy.
    Healthy,
    /// Priority control is degraded and best-effort behavior is active.
    Degraded(PriorityFailure),
}

impl PriorityStatus {
    pub const fn is_degraded(self) -> bool {
        matches!(self, Self::Degraded(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivityPolicy {
    /// Host considered idle after this amount of wall observed idle time.
    pub idle_after: Duration,
    /// Debounce duration before accepting an idle transition.
    pub resume_debounce: Duration,
}

impl ActivityPolicy {
    pub const fn new(idle_after: Duration, resume_debounce: Duration) -> Self {
        Self {
            idle_after,
            resume_debounce,
        }
    }

    fn transition_delay(&self) -> Duration {
        self.idle_after.saturating_add(self.resume_debounce)
    }
}

impl Default for ActivityPolicy {
    fn default() -> Self {
        Self {
            idle_after: Duration::from_secs(300),
            resume_debounce: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostActivitySample {
    pub observed: HostActivity,
    pub effective: HostActivity,
    pub changed: bool,
}

/// Debounces host-activity transitions and applies session-only override policy.
pub struct HostActivityMonitor<D, C>
where
    D: HostActivityDetector,
    C: ActivityClock,
{
    detector: D,
    clock: C,
    policy: ActivityPolicy,
    override_mode: ActivityOverride,
    effective_activity: HostActivity,
    detected_idle_since: Option<Instant>,
}

impl<D, C> HostActivityMonitor<D, C>
where
    D: HostActivityDetector,
    C: ActivityClock,
{
    pub fn new(detector: D, clock: C, policy: ActivityPolicy) -> Self {
        Self {
            detector,
            clock,
            policy,
            override_mode: ActivityOverride::Auto,
            effective_activity: HostActivity::Unknown,
            detected_idle_since: None,
        }
    }

    pub fn set_override(&mut self, override_mode: ActivityOverride) {
        self.override_mode = override_mode;
    }

    pub fn activity_override(&self) -> ActivityOverride {
        self.override_mode
    }

    pub fn effective_activity(&self) -> HostActivity {
        self.effective_activity
    }

    pub fn sample(&mut self) -> HostActivitySample {
        let observed = self.override_mode.resolve(self.detector.sample());
        let next = self.next_activity(observed, self.clock.now());
        let changed = next != self.effective_activity;
        self.effective_activity = next;

        HostActivitySample {
            observed,
            effective: next,
            changed,
        }
    }

    fn next_activity(&mut self, observed: HostActivity, now: Instant) -> HostActivity {
        match observed {
            HostActivity::Active => {
                self.detected_idle_since = None;
                HostActivity::Active
            }
            HostActivity::Unknown => {
                self.detected_idle_since = None;
                HostActivity::Unknown
            }
            HostActivity::Idle => {
                if self.effective_activity == HostActivity::Idle {
                    return HostActivity::Idle;
                }

                let started_idle = self.detected_idle_since.get_or_insert(now);
                if now.duration_since(*started_idle) >= self.policy.transition_delay() {
                    HostActivity::Idle
                } else {
                    self.effective_activity
                }
            }
        }
    }
}

/// Applies/reverts priority reduction safely with fail-closed reporting.
pub struct PrioritySession<C>
where
    C: PriorityController,
{
    controller: C,
    reduced: bool,
    status: PriorityStatus,
}

impl<C> PrioritySession<C>
where
    C: PriorityController,
{
    pub fn new(controller: C) -> Self {
        Self {
            controller,
            reduced: false,
            status: PriorityStatus::Healthy,
        }
    }

    pub fn status(&self) -> PriorityStatus {
        self.status
    }

    pub fn is_reduced(&self) -> bool {
        self.reduced
    }

    pub fn is_degraded(&self) -> bool {
        self.status.is_degraded()
    }

    pub fn reduce(&mut self) -> PriorityStatus {
        if self.reduced {
            return self.status;
        }

        if let Err(reason) = self.controller.reduce_priority() {
            self.status = PriorityStatus::Degraded(reason);
            return self.status;
        }

        self.reduced = true;
        self.status = PriorityStatus::Healthy;
        self.status
    }

    pub fn restore(&mut self) -> PriorityStatus {
        if !self.reduced {
            return self.status;
        }

        if let Err(reason) = self.controller.restore_priority() {
            self.status = PriorityStatus::Degraded(reason);
            return self.status;
        }

        self.reduced = false;
        self.status = PriorityStatus::Healthy;
        self.status
    }

    pub fn clear_degraded(&mut self) {
        self.status = PriorityStatus::Healthy;
    }
}

#[cfg(target_os = "linux")]
pub type NativeHostActivityDetector = linux::NativeHostActivityDetector;

#[cfg(target_os = "linux")]
pub type NativePriorityController = linux::NativePriorityController;

#[cfg(target_os = "macos")]
pub type NativeHostActivityDetector = macos::MacHostActivityDetector;

#[cfg(target_os = "macos")]
pub type NativePriorityController = macos::MacPriorityController;

#[cfg(target_os = "windows")]
pub type NativeHostActivityDetector = windows::WindowsHostActivityDetector;

#[cfg(target_os = "windows")]
pub type NativePriorityController = windows::WindowsPriorityController;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub type NativeHostActivityDetector = unsupported::UnsupportedHostActivityDetector;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub type NativePriorityController = unsupported::UnsupportedPriorityController;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    #[derive(Debug, Clone)]
    struct FakeClock {
        now: Rc<Cell<Instant>>,
    }

    impl FakeClock {
        fn new(start: Instant) -> Self {
            Self {
                now: Rc::new(Cell::new(start)),
            }
        }

        fn advance(&self, by: Duration) {
            self.now.set(self.now.get() + by);
        }
    }

    impl ActivityClock for FakeClock {
        fn now(&self) -> Instant {
            self.now.get()
        }
    }

    #[derive(Debug)]
    struct FakeHostActivityDetector {
        sequence: VecDeque<HostActivity>,
    }

    impl FakeHostActivityDetector {
        fn new(sequence: &[HostActivity]) -> Self {
            Self {
                sequence: VecDeque::from(sequence.to_vec()),
            }
        }
    }

    impl HostActivityDetector for FakeHostActivityDetector {
        fn sample(&mut self) -> HostActivity {
            self.sequence.pop_front().unwrap_or(HostActivity::Unknown)
        }
    }

    struct FakePriorityController {
        reduce_results: VecDeque<Result<(), PriorityFailure>>,
        restore_results: VecDeque<Result<(), PriorityFailure>>,
    }

    impl FakePriorityController {
        fn new(
            reduce_results: Vec<Result<(), PriorityFailure>>,
            restore_results: Vec<Result<(), PriorityFailure>>,
        ) -> Self {
            Self {
                reduce_results: VecDeque::from(reduce_results),
                restore_results: VecDeque::from(restore_results),
            }
        }
    }

    impl PriorityController for FakePriorityController {
        fn reduce_priority(&mut self) -> Result<(), PriorityFailure> {
            self.reduce_results.pop_front().unwrap_or(Ok(()))
        }

        fn restore_priority(&mut self) -> Result<(), PriorityFailure> {
            self.restore_results.pop_front().unwrap_or(Ok(()))
        }
    }

    #[test]
    fn activity_detector_debounce_and_override() {
        let detector = FakeHostActivityDetector::new(&[
            HostActivity::Active,
            HostActivity::Idle,
            HostActivity::Active,
            HostActivity::Idle,
            HostActivity::Idle,
            HostActivity::Idle,
        ]);

        let start = Instant::now();
        let clock = FakeClock::new(start);
        let mut monitor = HostActivityMonitor::new(
            detector,
            clock.clone(),
            ActivityPolicy::new(Duration::from_secs(2), Duration::from_secs(2)),
        );

        let sample = monitor.sample();
        assert_eq!(sample.observed, HostActivity::Active);
        assert_eq!(sample.effective, HostActivity::Active);
        assert!(sample.changed);

        monitor.set_override(ActivityOverride::Idle);
        let sample = monitor.sample();
        assert_eq!(sample.observed, HostActivity::Idle);
        assert_eq!(sample.effective, HostActivity::Active);
        assert_eq!(monitor.activity_override(), ActivityOverride::Idle);

        monitor.set_override(ActivityOverride::Auto);
        let sample = monitor.sample();
        assert_eq!(sample.observed, HostActivity::Active);
        assert_eq!(sample.effective, HostActivity::Active);

        let sample = monitor.sample();
        assert_eq!(sample.observed, HostActivity::Idle);
        assert_eq!(sample.effective, HostActivity::Active);

        clock.advance(Duration::from_secs(3));
        let sample = monitor.sample();
        assert!(!sample.changed);

        clock.advance(Duration::from_secs(2));
        let sample = monitor.sample();
        assert!(sample.changed);
        assert_eq!(sample.effective, HostActivity::Idle);

        let sample = monitor.sample();
        assert!(sample.changed);
        assert_eq!(sample.effective, HostActivity::Unknown);
    }

    #[test]
    fn activity_detector_and_priority_fail_closed_without_leak() {
        let detector = FakeHostActivityDetector::new(&[HostActivity::Unknown]);
        let mut monitor = HostActivityMonitor::new(
            detector,
            SystemClock,
            ActivityPolicy::new(Duration::from_secs(1), Duration::from_secs(1)),
        );

        let sample = monitor.sample();
        assert_eq!(sample.effective, HostActivity::Unknown);

        let mut priority = PrioritySession::new(FakePriorityController::new(
            vec![Ok(()), Err(PriorityFailure::Unsupported)],
            vec![Err(PriorityFailure::RestoreFailed), Ok(())],
        ));

        // Given reduced priority with a transient restore failure queued.
        let reduce = priority.reduce();
        assert_eq!(reduce, PriorityStatus::Healthy);
        assert!(priority.is_reduced());

        let reduce = priority.reduce();
        assert_eq!(reduce, PriorityStatus::Healthy);

        // When restore fails, then the session stays reduced and retryable.
        let restore = priority.restore();
        assert_eq!(
            restore,
            PriorityStatus::Degraded(PriorityFailure::RestoreFailed)
        );
        assert!(priority.is_reduced());

        // When a later restore succeeds, then reduced state is cleared.
        let restore = priority.restore();
        assert_eq!(restore, PriorityStatus::Healthy);
        assert!(!priority.is_reduced());
        assert_eq!(priority.status(), PriorityStatus::Healthy);

        let mut priority = PrioritySession::new(FakePriorityController::new(
            vec![Err(PriorityFailure::Unsupported)],
            vec![Err(PriorityFailure::RestoreFailed)],
        ));

        let reduced = priority.reduce();
        assert_eq!(
            reduced,
            PriorityStatus::Degraded(PriorityFailure::Unsupported)
        );
        assert!(priority.is_degraded());

        let restored = priority.restore();
        assert_eq!(
            restored,
            PriorityStatus::Degraded(PriorityFailure::Unsupported)
        );

        let rendered = format!("{:?}", priority.status());
        assert!(rendered.contains("Unsupported"));
        assert!(!rendered.contains("SENTINEL"));
        assert!(!rendered.contains("raw-error:"));
        assert!(!rendered.contains("panic"));

        let mut retry = PrioritySession::new(FakePriorityController::new(
            vec![Err(PriorityFailure::ApplyFailed), Ok(())],
            vec![Ok(())],
        ));
        assert!(retry.reduce().is_degraded());
        assert_eq!(retry.reduce(), PriorityStatus::Healthy);
        assert!(retry.is_reduced());
    }

    #[test]
    fn default_platform_aliases_are_type_checkable() {
        let _detector = NativeHostActivityDetector::default();
        let _ = _detector;
        let _controller = NativePriorityController::default();
        let _ = _controller;
        let _ = std::mem::size_of::<NativeHostActivityDetector>();
        let _ = std::mem::size_of::<NativePriorityController>();
    }
}
