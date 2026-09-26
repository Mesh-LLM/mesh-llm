//! Planned requests use the same protocol-aware response relay as direct routes.
use super::common::{RouteAttemptLoggingContext, RouteAttemptResult};
use super::pipeline::{PipelineCapsuleNonce, PipelineProxyResult};
use crate::network::openai::client_stream::ClientStream;
use tokio::io::AsyncWriteExt;

const PLANNED_REQUEST_WRITE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

pub(super) async fn relay_planned_request(
    client: &mut ClientStream,
    port: u16,
    body: &serde_json::Value,
    nonce: &PipelineCapsuleNonce,
    logging: RouteAttemptLoggingContext<'_>,
) -> PipelineProxyResult {
    let Ok(mut upstream) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await else {
        return PipelineProxyResult::FallbackToDirect;
    };
    let body = body.to_string();
    let mut header = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    super::probe::append_capsule_nonce_headers(
        &mut header,
        nonce.client_nonce.as_deref(),
        nonce.nonce_origin.as_deref(),
    );
    header.push_str("\r\n");
    let write_result = tokio::time::timeout(PLANNED_REQUEST_WRITE_TIMEOUT, async {
        upstream.write_all(header.as_bytes()).await?;
        upstream.write_all(body.as_bytes()).await
    })
    .await;
    if !matches!(write_result, Ok(Ok(()))) {
        return PipelineProxyResult::FallbackToDirect;
    }
    let Ok(probe) = super::probe::probe_http_response_local(&mut upstream).await else {
        return PipelineProxyResult::FallbackToDirect;
    };
    match super::dispatch::relay_probed_response(
        client,
        &mut upstream,
        probe,
        logging.request_id,
        logging.retry_policy,
        logging.response_adapter,
        logging.served_by,
        logging.route_observer,
    )
    .await
    {
        Ok(RouteAttemptResult::Delivered {
            status_code,
            usage: Some(usage),
            ..
        }) => PipelineProxyResult::RespondedWithUsage { status_code, usage },
        Ok(RouteAttemptResult::Delivered { status_code, .. }) => {
            PipelineProxyResult::Responded(status_code)
        }
        _ => PipelineProxyResult::Dropped,
    }
}
