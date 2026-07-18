use super::common::{ResponseRetryPolicy, RouteAttemptResult, retryable_route_result_from_error};
use super::dispatch::relay_attempted_response;
use super::probe::{probe_http_response, probe_http_response_local};
use crate::mesh;
use crate::network::openai::request_normalize::ResponseAdapter;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(in crate::network::openai) async fn route_local_attempt(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    port: u16,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match TcpStream::connect(format!("127.0.0.1:{port}")).await {
        Ok(mut upstream) => {
            let _inflight = node.begin_inflight_request();
            let _ = upstream.set_nodelay(true);
            if let Err(err) = upstream.write_all(prefetched).await {
                tracing::warn!(
                    "API proxy: failed to forward buffered request to local OpenAI surface on {port}: {err}"
                );
                return RouteAttemptResult::RetryableUnavailable;
            }
            route_local_attempt_after_forward(
                tcp_stream,
                &mut upstream,
                port,
                retry_policy,
                response_adapter,
            )
            .await
        }
        Err(err) => {
            tracing::warn!("API proxy: can't reach local OpenAI surface on {port}: {err}");
            RouteAttemptResult::RetryableUnavailable
        }
    }
}

async fn route_local_attempt_after_forward(
    tcp_stream: &mut TcpStream,
    upstream: &mut TcpStream,
    port: u16,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match probe_http_response_local(upstream).await {
        Ok(probe) => {
            let result = relay_attempted_response(
                tcp_stream,
                upstream,
                probe,
                retry_policy,
                response_adapter,
                "API proxy (local): downstream client disconnected during relay",
                "API proxy (local) ended after commit",
            )
            .await;
            if matches!(result, RouteAttemptResult::ClientDisconnected) {
                let _ = upstream.shutdown().await;
            }
            result
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to read local response from OpenAI surface on {port}: {err}"
            );
            retryable_route_result_from_error(&err)
        }
    }
}

pub(in crate::network::openai) async fn route_remote_attempt(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match node.open_http_tunnel(host_id).await {
        Ok((mut quic_send, mut quic_recv)) => {
            if let Err(err) = quic_send.write_all(prefetched).await {
                tracing::warn!(
                    "API proxy: failed to forward buffered request to host {}: {err}",
                    host_id.fmt_short()
                );
                return RouteAttemptResult::RetryableUnavailable;
            }
            route_remote_attempt_after_forward(
                tcp_stream,
                &mut quic_recv,
                host_id,
                retry_policy,
                response_adapter,
            )
            .await
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: can't tunnel to host {}: {err}",
                host_id.fmt_short()
            );
            retryable_route_result_from_error(&err)
        }
    }
}

async fn route_remote_attempt_after_forward(
    tcp_stream: &mut TcpStream,
    quic_recv: &mut iroh::endpoint::RecvStream,
    host_id: iroh::EndpointId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    match probe_http_response(quic_recv).await {
        Ok(probe) => {
            relay_attempted_response(
                tcp_stream,
                quic_recv,
                probe,
                retry_policy,
                response_adapter,
                "API proxy (remote): downstream client disconnected during relay",
                "API proxy (remote) ended after commit",
            )
            .await
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to read response from host {}: {err}",
                host_id.fmt_short()
            );
            retryable_route_result_from_error(&err)
        }
    }
}
