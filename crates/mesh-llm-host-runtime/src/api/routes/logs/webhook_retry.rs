//! Trusted-local, audited webhook dead-letter retry endpoint.
//!
//! This route deliberately exposes no delivery record fields. It converts the
//! durable state machine into fixed idempotent outcome labels, while the
//! scheduler/worker retain all endpoint and payload ownership.

use mesh_llm_log_store::WebhookManualRetryOutcome;
use serde::Serialize;
use tokio::net::TcpStream;

use super::{LoggingQueryFacade, LoggingRuntimeState, LogsError, run_blocking};

const MANUAL_RETRY_AUDIT_ACTION: &str = "log_webhook_manual_retry";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WebhookRetryResponse {
    outcome: WebhookRetryOutcome,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum WebhookRetryOutcome {
    Scheduled,
    AlreadyScheduled,
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &LoggingRuntimeState,
    delivery_id: &str,
    path: &str,
    body: &str,
) -> Result<(), LogsError> {
    let request = super::parse::webhook_retry_request(delivery_id, path, body)?;
    let facade = super::query_facade(state)?;
    let audit_facade = facade.clone();
    let reason = request.reason.clone();
    let result = run_blocking(move || retry_delivery(&facade, request)).await;
    let result_label = if result.is_ok() {
        "succeeded"
    } else {
        "failed"
    };
    // This audit is deliberately best-effort. The shared writer's recursion
    // guard prevents a rejected audit insert from generating more logging.
    let _ = audit_facade.write_operator_audit(MANUAL_RETRY_AUDIT_ACTION, reason, result_label);
    let response = result?;
    crate::api::http::respond_json(stream, 200, &response)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

fn retry_delivery(
    facade: &LoggingQueryFacade,
    request: super::parse::WebhookRetryRequest,
) -> Result<WebhookRetryResponse, LogsError> {
    match facade.manually_retry_webhook_delivery(&request.delivery_id)? {
        WebhookManualRetryOutcome::Scheduled => Ok(WebhookRetryResponse {
            outcome: WebhookRetryOutcome::Scheduled,
        }),
        WebhookManualRetryOutcome::AlreadyScheduled => Ok(WebhookRetryResponse {
            outcome: WebhookRetryOutcome::AlreadyScheduled,
        }),
        WebhookManualRetryOutcome::NotRetryable => Err(LogsError::WebhookNotRetryable),
        WebhookManualRetryOutcome::NotFound => Err(LogsError::NotFound),
    }
}
