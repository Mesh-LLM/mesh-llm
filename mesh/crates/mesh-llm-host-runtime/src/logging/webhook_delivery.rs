//! Bounded asynchronous delivery of already-durable terminal webhook records.
//!
//! This module intentionally has no startup loop or request-path hook. A later
//! runtime owner schedules [`WebhookDeliveryWorker::process_next`]; that one
//! step claims a record, performs one bounded HTTP attempt, and persists the
//! fenced terminal transition. The payload is built solely from the delivery
//! record, never by re-reading lifecycle payloads or artifacts.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, SecondsFormat, Utc};
use mesh_llm_config::LoggingWebhookConfig;
use mesh_llm_log_store::{
    LogStore, LogStoreError, WebhookDeliveryErrorCode, WebhookDeliveryRecord, WebhookRetryOutcome,
    WebhookTerminalOutcome,
};
use rand::RngExt;
use serde::Serialize;
use thiserror::Error;
use url::Url;

use super::metrics::{
    LoggingMetric, LoggingMetrics, LoggingWebhookAttemptState, LoggingWebhookDeliveryOutcome,
};

const BASE_RETRY_DELAY: Duration = Duration::from_secs(1);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const MAX_JITTER_MILLIS: u64 = 1_000;
const LEASE_SAFETY_MARGIN: Duration = Duration::from_secs(5);

/// A deliberately small terminal notification. In particular, it never
/// carries prompt/completion content, artifact metadata, a filesystem path,
/// endpoint URL, transport error, or lifecycle event JSON.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct WebhookTerminalPayload {
    request_id: String,
    outcome: WebhookTerminalOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    status_code: Option<u16>,
}

impl WebhookTerminalPayload {
    fn from_record(record: &WebhookDeliveryRecord) -> Option<Self> {
        record.request_id.as_ref().map(|request_id| Self {
            request_id: request_id.clone(),
            outcome: record.terminal_outcome,
            status_code: record.terminal_status_code,
        })
    }
}

/// Result returned by an injected transport. Transport errors intentionally
/// carry no endpoint, response body, or raw error text so they cannot be
/// accidentally persisted by the worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WebhookTransportError {
    Timeout,
    Transport,
}

/// HTTP transport seam for deterministic tests and alternate runtime clients.
#[async_trait]
pub(crate) trait WebhookTransport: Send + Sync {
    async fn post_terminal(
        &self,
        endpoint: &Url,
        payload: &WebhookTerminalPayload,
        timeout: Duration,
    ) -> Result<u16, WebhookTransportError>;
}

/// Default HTTP transport. The endpoint is constructed only from validated
/// logging configuration, and a timeout is set on every individual request.
pub(crate) struct ReqwestWebhookTransport {
    client: reqwest::Client,
}

impl ReqwestWebhookTransport {
    /// Disable redirects: only the explicitly configured endpoint is allowed
    /// to receive a terminal notification.
    pub(crate) fn new() -> Result<Self, reqwest::Error> {
        reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map(|client| Self { client })
    }
}

#[async_trait]
impl WebhookTransport for ReqwestWebhookTransport {
    async fn post_terminal(
        &self,
        endpoint: &Url,
        payload: &WebhookTerminalPayload,
        timeout: Duration,
    ) -> Result<u16, WebhookTransportError> {
        let response = self
            .client
            .post(endpoint.clone())
            .timeout(timeout)
            .json(payload)
            .send()
            .await
            .map_err(|error| {
                if error.is_timeout() {
                    WebhookTransportError::Timeout
                } else {
                    WebhookTransportError::Transport
                }
            })?;
        Ok(response.status().as_u16())
    }
}

/// Timestamp source kept independent from the request logging clock so a
/// worker test can make retry and lease times exact without wall-clock sleeps.
pub(crate) trait WebhookWorkerClock: Send + Sync {
    fn now(&self) -> String;
}

/// Production timestamp source for webhook worker scheduling.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SystemWebhookWorkerClock;

impl WebhookWorkerClock for SystemWebhookWorkerClock {
    fn now(&self) -> String {
        Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
    }
}

/// Bounded random source used only to decorrelate retry wakeups. Tests inject
/// a fixed implementation, so retry scheduling remains deterministic there.
pub(crate) trait WebhookJitter: Send + Sync {
    fn millis_up_to(&self, inclusive_maximum: u64) -> u64;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RandomWebhookJitter;

impl WebhookJitter for RandomWebhookJitter {
    fn millis_up_to(&self, inclusive_maximum: u64) -> u64 {
        if inclusive_maximum == 0 {
            0
        } else {
            rand::rng().random_range(0..=inclusive_maximum)
        }
    }
}

/// Construction errors are deliberately static: configuration diagnostics own
/// user-facing detail, and worker errors must not expose an endpoint value.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum WebhookWorkerConfigError {
    #[error("webhook delivery is disabled")]
    Disabled,
    #[error("webhook delivery requires an endpoint")]
    MissingEndpoint,
    #[error("webhook delivery endpoint is invalid")]
    InvalidEndpoint,
    #[error("webhook delivery timeout is outside the supported range")]
    InvalidTimeout,
}

#[derive(Debug, Error)]
pub(crate) enum WebhookWorkerError {
    #[error("webhook delivery store operation failed")]
    Store,
    #[error("webhook delivery clock produced an invalid timestamp")]
    InvalidTimestamp,
    #[error("webhook delivery blocking worker failed")]
    BlockingWorker,
}

/// The outcome of one non-blocking worker step. A runtime loop can use this to
/// choose its next wakeup without the request-serving path awaiting delivery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WebhookWorkerOutcome {
    Idle,
    Delivered {
        delivery_id: String,
        status_code: u16,
    },
    RetryScheduled {
        delivery_id: String,
    },
    DeadLettered {
        delivery_id: String,
    },
    FencedOut {
        delivery_id: String,
    },
}

/// One-at-a-time executor for durable webhook deliveries. It does not spawn a
/// loop; callers must explicitly schedule each invocation on a worker owner.
pub(crate) struct WebhookDeliveryWorker {
    store: Arc<LogStore>,
    endpoint: Url,
    timeout: Duration,
    transport: Arc<dyn WebhookTransport>,
    clock: Arc<dyn WebhookWorkerClock>,
    jitter: Arc<dyn WebhookJitter>,
    metrics: LoggingMetrics,
}

impl WebhookDeliveryWorker {
    /// Build a worker from the already-loaded logging configuration, repeating
    /// endpoint safety checks as a defense-in-depth boundary before it reaches
    /// the HTTP client.
    pub(crate) fn from_config(
        store: Arc<LogStore>,
        config: &LoggingWebhookConfig,
        transport: Arc<dyn WebhookTransport>,
        clock: Arc<dyn WebhookWorkerClock>,
        jitter: Arc<dyn WebhookJitter>,
    ) -> Result<Self, WebhookWorkerConfigError> {
        if !config.enabled {
            return Err(WebhookWorkerConfigError::Disabled);
        }
        if !(1..=60).contains(&config.timeout_secs) {
            return Err(WebhookWorkerConfigError::InvalidTimeout);
        }
        let endpoint = config
            .url
            .as_deref()
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .ok_or(WebhookWorkerConfigError::MissingEndpoint)
            .and_then(validate_endpoint)?;

        Ok(Self {
            store,
            endpoint,
            timeout: Duration::from_secs(config.timeout_secs),
            transport,
            clock,
            jitter,
            metrics: LoggingMetrics::default(),
        })
    }

    /// Attach the process-local metrics handle owned by `LoggingService`.
    /// The worker still performs no telemetry I/O and keeps delivery fail-open.
    pub(crate) fn with_metrics(mut self, metrics: LoggingMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Claim and process at most one eligible delivery. All SQLite operations
    /// use Tokio's blocking pool, while the HTTP request remains async, so this
    /// future cannot park a request-serving executor worker on SQLite or I/O.
    pub(crate) async fn process_next(&self) -> Result<WebhookWorkerOutcome, WebhookWorkerError> {
        let claimed_at = self.clock.now();
        let lease_expires_at = timestamp_after(&claimed_at, lease_duration(self.timeout))?;
        let claimed = self
            .run_store(move |store| {
                store.claim_next_webhook_delivery(&claimed_at, &lease_expires_at)
            })
            .await?;
        let Some(record) = claimed else {
            return Ok(WebhookWorkerOutcome::Idle);
        };
        self.metrics.record(LoggingMetric::WebhookAttempt {
            state: LoggingWebhookAttemptState::Claimed,
        });

        let outcome = if let Some(payload) = WebhookTerminalPayload::from_record(&record) {
            match self
                .transport
                .post_terminal(&self.endpoint, &payload, self.timeout)
                .await
            {
                Ok(status_code) if (200..=299).contains(&status_code) => {
                    self.record_success(&record, status_code).await?
                }
                Ok(status_code) => {
                    self.record_failure(
                        &record,
                        error_code_for_status(status_code),
                        Some(status_code),
                    )
                    .await?
                }
                Err(WebhookTransportError::Timeout) => {
                    self.record_failure(&record, WebhookDeliveryErrorCode::Timeout, None)
                        .await?
                }
                Err(WebhookTransportError::Transport) => {
                    self.record_failure(&record, WebhookDeliveryErrorCode::Transport, None)
                        .await?
                }
            }
        } else {
            self.record_failure(&record, WebhookDeliveryErrorCode::Configuration, None)
                .await?
        };
        self.record_delivery_outcome(&outcome);
        Ok(outcome)
    }

    fn record_delivery_outcome(&self, outcome: &WebhookWorkerOutcome) {
        let outcome = match outcome {
            WebhookWorkerOutcome::Idle => return,
            WebhookWorkerOutcome::Delivered { .. } => LoggingWebhookDeliveryOutcome::Delivered,
            WebhookWorkerOutcome::RetryScheduled { .. } => {
                LoggingWebhookDeliveryOutcome::RetryScheduled
            }
            WebhookWorkerOutcome::DeadLettered { .. } => {
                LoggingWebhookDeliveryOutcome::DeadLettered
            }
            WebhookWorkerOutcome::FencedOut { .. } => LoggingWebhookDeliveryOutcome::FencedOut,
        };
        self.metrics
            .record(LoggingMetric::WebhookDelivery { outcome });
    }

    async fn record_success(
        &self,
        record: &WebhookDeliveryRecord,
        status_code: u16,
    ) -> Result<WebhookWorkerOutcome, WebhookWorkerError> {
        let delivery_id = record.delivery_id.clone();
        let completed_at = self.clock.now();
        let claim_generation = record.claim_generation;
        let completed = self
            .run_store(move |store| {
                store.complete_webhook_delivery(
                    &delivery_id,
                    claim_generation,
                    &completed_at,
                    status_code,
                )
            })
            .await?;
        if completed {
            Ok(WebhookWorkerOutcome::Delivered {
                delivery_id: record.delivery_id.clone(),
                status_code,
            })
        } else {
            Ok(WebhookWorkerOutcome::FencedOut {
                delivery_id: record.delivery_id.clone(),
            })
        }
    }

    async fn record_failure(
        &self,
        record: &WebhookDeliveryRecord,
        error_code: WebhookDeliveryErrorCode,
        response_status_code: Option<u16>,
    ) -> Result<WebhookWorkerOutcome, WebhookWorkerError> {
        let delivery_id = record.delivery_id.clone();
        let updated_at = self.clock.now();
        let retry_delay = retry_delay(record.attempt_number, self.jitter.as_ref());
        let next_attempt_at = timestamp_after(&updated_at, retry_delay)?;
        let claim_generation = record.claim_generation;
        let result = self
            .run_store(move |store| {
                store.retry_or_dead_letter_webhook_delivery(
                    &delivery_id,
                    claim_generation,
                    &updated_at,
                    &next_attempt_at,
                    error_code,
                    response_status_code,
                )
            })
            .await?;
        match result {
            Some(WebhookRetryOutcome::RetryScheduled) => Ok(WebhookWorkerOutcome::RetryScheduled {
                delivery_id: record.delivery_id.clone(),
            }),
            Some(WebhookRetryOutcome::DeadLettered) => Ok(WebhookWorkerOutcome::DeadLettered {
                delivery_id: record.delivery_id.clone(),
            }),
            None => Ok(WebhookWorkerOutcome::FencedOut {
                delivery_id: record.delivery_id.clone(),
            }),
        }
    }

    async fn run_store<T>(
        &self,
        operation: impl FnOnce(&LogStore) -> Result<T, LogStoreError> + Send + 'static,
    ) -> Result<T, WebhookWorkerError>
    where
        T: Send + 'static,
    {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || operation(&store))
            .await
            .map_err(|_| WebhookWorkerError::BlockingWorker)?
            .map_err(|_| WebhookWorkerError::Store)
    }
}

fn validate_endpoint(value: &str) -> Result<Url, WebhookWorkerConfigError> {
    let endpoint = Url::parse(value).map_err(|_| WebhookWorkerConfigError::InvalidEndpoint)?;
    let is_http = endpoint.scheme() == "http" || endpoint.scheme() == "https";
    if !is_http
        || endpoint.host().is_none()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(WebhookWorkerConfigError::InvalidEndpoint);
    }
    Ok(endpoint)
}

fn lease_duration(timeout: Duration) -> Duration {
    timeout
        .checked_add(LEASE_SAFETY_MARGIN)
        .unwrap_or(LEASE_SAFETY_MARGIN)
}

fn retry_delay(attempt_number: u32, jitter: &dyn WebhookJitter) -> Duration {
    let exponent = attempt_number.saturating_sub(1).min(6);
    let exponential = BASE_RETRY_DELAY.saturating_mul(1_u32 << exponent);
    let capped = exponential.min(MAX_RETRY_DELAY);
    let jitter_limit = capped
        .as_millis()
        .min(u128::from(MAX_JITTER_MILLIS))
        .try_into()
        .unwrap_or(MAX_JITTER_MILLIS)
        / 4;
    let remaining = MAX_RETRY_DELAY.saturating_sub(capped);
    let jitter = Duration::from_millis(
        jitter
            .millis_up_to(jitter_limit)
            .min(remaining.as_millis() as u64),
    );
    capped.checked_add(jitter).unwrap_or(MAX_RETRY_DELAY)
}

fn timestamp_after(value: &str, duration: Duration) -> Result<String, WebhookWorkerError> {
    let timestamp = DateTime::parse_from_rfc3339(value)
        .map_err(|_| WebhookWorkerError::InvalidTimestamp)?
        .with_timezone(&Utc);
    let milliseconds =
        i64::try_from(duration.as_millis()).map_err(|_| WebhookWorkerError::InvalidTimestamp)?;
    timestamp
        .checked_add_signed(ChronoDuration::milliseconds(milliseconds))
        .map(|timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Millis, true))
        .ok_or(WebhookWorkerError::InvalidTimestamp)
}

fn error_code_for_status(status_code: u16) -> WebhookDeliveryErrorCode {
    if (400..=499).contains(&status_code) {
        WebhookDeliveryErrorCode::Http4xx
    } else {
        WebhookDeliveryErrorCode::Http5xx
    }
}

#[cfg(test)]
mod tests;
