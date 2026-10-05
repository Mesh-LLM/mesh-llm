//! OTLP mapping for the logging-owned, closed metric vocabulary.
//!
//! This module is deliberately the only place where logging metrics meet
//! telemetry. It exports no logging payload fields, identifiers, paths, URLs,
//! prompts, completions, tokens, or hashes.

use crate::logging::{LoggingMetric, LoggingMetricsSink};
use opentelemetry::KeyValue;

use super::{
    SurveyEvent, SurveyRecorder, SurveyTelemetry, debug_assert_telemetry_attrs_allowlisted,
};

impl LoggingMetricsSink for SurveyTelemetry {
    fn record(&self, metric: LoggingMetric) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        inner.queue.try_push(SurveyEvent::LoggingMetric { metric });
    }
}

impl SurveyRecorder {
    pub(super) fn record_logging_metric(&mut self, metric: LoggingMetric) {
        let attrs = logging_metric_key_values(metric);
        match metric {
            LoggingMetric::LifecycleTerminal { .. } => {
                self.logging_lifecycle_terminal_total.add(1, &attrs);
            }
            LoggingMetric::PersistenceQueueDropped { count } => {
                self.logging_persistence_queue_dropped_total
                    .add(count, &attrs);
            }
            LoggingMetric::PersistenceFailure { count } => {
                self.logging_persistence_failure_total.add(count, &attrs);
            }
            LoggingMetric::PersistenceShutdownLoss { count } => {
                self.logging_persistence_shutdown_loss_total
                    .add(count, &attrs);
            }
            LoggingMetric::PersistenceOutstanding { current } => {
                self.logging_persistence_outstanding.record(current, &attrs);
            }
            LoggingMetric::ReplayEvicted { count } => {
                self.logging_replay_evicted_total.add(count, &attrs);
            }
            LoggingMetric::ReplayGap { count } => {
                self.logging_replay_gap_total.add(count, &attrs);
            }
            LoggingMetric::ReplayDropped { count } => {
                self.logging_replay_dropped_total.add(count, &attrs);
            }
            LoggingMetric::Cleanup { .. } => {
                self.logging_cleanup_total.add(1, &attrs);
            }
            LoggingMetric::WebhookDelivery { .. } => {
                self.logging_webhook_delivery_total.add(1, &attrs);
            }
            LoggingMetric::WebhookAttempt { .. } => {
                self.logging_webhook_attempt_total.add(1, &attrs);
            }
            LoggingMetric::ArtifactCapture { .. } => {
                self.logging_artifact_capture_total.add(1, &attrs);
            }
        }
    }
}

pub(super) fn logging_metric_key_values(metric: LoggingMetric) -> Vec<KeyValue> {
    match metric {
        LoggingMetric::LifecycleTerminal { outcome } => {
            logging_metric_attr("mesh_llm.logging_terminal_outcome", outcome.label())
        }
        LoggingMetric::PersistenceQueueDropped { .. }
        | LoggingMetric::PersistenceFailure { .. }
        | LoggingMetric::PersistenceShutdownLoss { .. }
        | LoggingMetric::PersistenceOutstanding { .. }
        | LoggingMetric::ReplayEvicted { .. }
        | LoggingMetric::ReplayGap { .. }
        | LoggingMetric::ReplayDropped { .. } => Vec::new(),
        LoggingMetric::Cleanup { outcome } => {
            logging_metric_attr("mesh_llm.logging_cleanup_outcome", outcome.label())
        }
        LoggingMetric::WebhookDelivery { outcome } => {
            logging_metric_attr("mesh_llm.logging_webhook_delivery_outcome", outcome.label())
        }
        LoggingMetric::WebhookAttempt { state } => {
            logging_metric_attr("mesh_llm.logging_webhook_attempt_state", state.label())
        }
        LoggingMetric::ArtifactCapture { status } => {
            logging_metric_attr("mesh_llm.logging_artifact_capture_status", status.label())
        }
    }
}

fn logging_metric_attr(key: &'static str, value: &'static str) -> Vec<KeyValue> {
    let attrs = vec![KeyValue::new(key, value)];
    debug_assert_telemetry_attrs_allowlisted(&attrs);
    attrs
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::logging::{
        LoggingArtifactCaptureStatus, LoggingCleanupOutcome, LoggingTerminalOutcome,
        LoggingWebhookAttemptState, LoggingWebhookDeliveryOutcome,
    };

    use super::*;

    #[test]
    fn disabled_telemetry_exposes_no_logging_sink() {
        assert!(SurveyTelemetry::disabled().logging_sink().is_none());
    }

    #[test]
    fn logging_metric_attributes_are_bounded_and_exclude_private_values() {
        let metrics = [
            LoggingMetric::LifecycleTerminal {
                outcome: LoggingTerminalOutcome::Rejected,
            },
            LoggingMetric::PersistenceQueueDropped { count: 1 },
            LoggingMetric::PersistenceFailure { count: 1 },
            LoggingMetric::PersistenceShutdownLoss { count: 1 },
            LoggingMetric::PersistenceOutstanding { current: 3 },
            LoggingMetric::ReplayEvicted { count: 1 },
            LoggingMetric::ReplayGap { count: 1 },
            LoggingMetric::ReplayDropped { count: 1 },
            LoggingMetric::Cleanup {
                outcome: LoggingCleanupOutcome::SkippedUnavailable,
            },
            LoggingMetric::WebhookDelivery {
                outcome: LoggingWebhookDeliveryOutcome::RetryScheduled,
            },
            LoggingMetric::WebhookAttempt {
                state: LoggingWebhookAttemptState::Claimed,
            },
            LoggingMetric::ArtifactCapture {
                status: LoggingArtifactCaptureStatus::Failed,
            },
        ];
        let attrs: Vec<_> = metrics
            .into_iter()
            .flat_map(logging_metric_key_values)
            .collect();
        let keys: BTreeSet<_> = attrs.iter().map(|attr| attr.key.to_string()).collect();
        assert_eq!(
            keys,
            BTreeSet::from([
                "mesh_llm.logging_artifact_capture_status".to_string(),
                "mesh_llm.logging_cleanup_outcome".to_string(),
                "mesh_llm.logging_terminal_outcome".to_string(),
                "mesh_llm.logging_webhook_attempt_state".to_string(),
                "mesh_llm.logging_webhook_delivery_outcome".to_string(),
            ])
        );

        let rendered = attrs
            .iter()
            .map(|attr| format!("{}={}", attr.key, attr.value))
            .collect::<Vec<_>>()
            .join("\n");
        for private_value in [
            "prompt: ignore all previous instructions",
            "completion: secret answer",
            "https://collector.example/private?token=secret",
            "/private/operator/logs/request.json",
            "raw-node-id-1234",
            "Bearer super-secret-token",
            "sha256:private-prompt-hash",
        ] {
            assert!(
                !rendered.contains(private_value),
                "private logging value escaped into telemetry attributes: {private_value}"
            );
        }
    }
}
