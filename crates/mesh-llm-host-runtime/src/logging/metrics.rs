//! Process-local, privacy-safe logging metric adapter.
//!
//! Logging owns the closed metric vocabulary and never imports OTLP. Runtime
//! telemetry may install an optional sink at startup; absent, poisoned, or
//! panicking sinks are intentionally ignored so logging and request serving
//! remain fail-open.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, RwLock};

/// Closed terminal outcomes suitable for metric labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoggingTerminalOutcome {
    Completed,
    Failed,
    Rejected,
    Cancelled,
    Dropped,
}

impl LoggingTerminalOutcome {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
        }
    }
}

/// Closed cleanup outcomes suitable for metric labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoggingCleanupOutcome {
    Completed,
    Failed,
    SkippedUnavailable,
}

impl LoggingCleanupOutcome {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::SkippedUnavailable => "skipped_unavailable",
        }
    }
}

/// Closed durable webhook terminal states suitable for metric labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoggingWebhookDeliveryOutcome {
    Delivered,
    RetryScheduled,
    DeadLettered,
    FencedOut,
}

impl LoggingWebhookDeliveryOutcome {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Delivered => "delivered",
            Self::RetryScheduled => "retry_scheduled",
            Self::DeadLettered => "dead_lettered",
            Self::FencedOut => "fenced_out",
        }
    }
}

/// Closed durable webhook attempt states suitable for metric labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoggingWebhookAttemptState {
    Claimed,
}

impl LoggingWebhookAttemptState {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Claimed => "claimed",
        }
    }
}

/// Closed artifact capture states suitable for metric labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoggingArtifactCaptureStatus {
    Written,
    Disabled,
    Failed,
}

impl LoggingArtifactCaptureStatus {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Written => "written",
            Self::Disabled => "disabled",
            Self::Failed => "failed",
        }
    }
}

/// Metrics emitted by logging. Every field is a bounded enum or a count; this
/// type deliberately has no ID, URL, path, payload, token, or hash fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoggingMetric {
    LifecycleTerminal {
        outcome: LoggingTerminalOutcome,
    },
    PersistenceQueueDropped {
        count: u64,
    },
    PersistenceFailure {
        count: u64,
    },
    PersistenceShutdownLoss {
        count: u64,
    },
    PersistenceOutstanding {
        current: u64,
    },
    ReplayEvicted {
        count: u64,
    },
    ReplayGap {
        count: u64,
    },
    ReplayDropped {
        count: u64,
    },
    Cleanup {
        outcome: LoggingCleanupOutcome,
    },
    WebhookDelivery {
        outcome: LoggingWebhookDeliveryOutcome,
    },
    WebhookAttempt {
        state: LoggingWebhookAttemptState,
    },
    ArtifactCapture {
        status: LoggingArtifactCaptureStatus,
    },
}

/// Optional process-local consumer for the closed logging metric vocabulary.
///
/// Implementations must not block or make logging persistence dependent on
/// telemetry delivery. The host runtime's survey adapter only queues metrics
/// in a bounded in-memory buffer.
pub(crate) trait LoggingMetricsSink: Send + Sync {
    fn record(&self, metric: LoggingMetric);
}

/// Shared optional sink handle used by logging producers and the replay bus.
#[derive(Clone, Default)]
pub(crate) struct LoggingMetrics {
    sink: Arc<RwLock<Option<Arc<dyn LoggingMetricsSink>>>>,
}

impl LoggingMetrics {
    pub(crate) fn set_sink(&self, sink: Option<Arc<dyn LoggingMetricsSink>>) {
        *self
            .sink
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = sink;
    }

    /// Emit synchronously to the optional in-process adapter. Contended or
    /// poisoned handles are dropped instead of waiting, and any sink panic is
    /// contained at this boundary.
    pub(crate) fn record(&self, metric: LoggingMetric) {
        let Ok(sink) = self.sink.try_read() else {
            return;
        };
        let sink = sink.clone();
        let Some(sink) = sink else {
            return;
        };
        let _ = catch_unwind(AssertUnwindSafe(|| sink.record(metric)));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct RecordingSink {
        calls: AtomicUsize,
    }

    impl LoggingMetricsSink for RecordingSink {
        fn record(&self, _: LoggingMetric) {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct PanickingSink;

    impl LoggingMetricsSink for PanickingSink {
        fn record(&self, _: LoggingMetric) {
            panic!("test logging metric sink panic");
        }
    }

    #[test]
    fn absent_sink_emits_nothing() {
        let metrics = LoggingMetrics::default();
        let recording = Arc::new(RecordingSink {
            calls: AtomicUsize::new(0),
        });
        metrics.set_sink(Some(recording.clone()));
        metrics.set_sink(None);

        metrics.record(LoggingMetric::ReplayDropped { count: 1 });

        assert_eq!(recording.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn panicking_sink_is_fail_open() {
        let metrics = LoggingMetrics::default();
        metrics.set_sink(Some(Arc::new(PanickingSink)));

        metrics.record(LoggingMetric::ReplayDropped { count: 1 });
    }
}
