use super::common::{
    ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult,
    retryable_route_result_from_error,
};
use super::dispatch::{RelayAttemptContext, relay_attempted_response};
use super::probe::{probe_http_response, probe_http_response_local};
use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::openai::forwarded_request::prepare_peer_forwarded_request;
use crate::network::openai::request_normalize::ResponseAdapter;
use mesh_llm_events::logging::identifiers::RequestId;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

pub(in crate::network::openai) async fn route_local_attempt(
    node: &mesh::Node,
    tcp_stream: &mut TcpStream,
    port: u16,
    prefetched: &[u8],
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let RouteAttemptLoggingContext {
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
    } = logging;
    let Ok((_instance_request, mut upstream)) = acquire_local_attempt_upstream(node, port).await
    else {
        return RouteAttemptResult::RetryableUnavailable;
    };
    let _inflight = node.begin_inflight_request();
    let _ = upstream.set_nodelay(true);
    if let Err(err) = forward_buffered_request(&mut upstream, prefetched).await {
        tracing::warn!(
            "API proxy: failed to forward buffered request to local OpenAI surface on {port}: {err}"
        );
        return RouteAttemptResult::RetryableUnavailable;
    }
    route_local_attempt_after_forward(
        tcp_stream,
        &mut upstream,
        port,
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
    )
    .await
}

async fn acquire_local_attempt_upstream(
    node: &mesh::Node,
    port: u16,
) -> Result<(Option<crate::runtime::InstanceRequestGuard>, TcpStream), ()> {
    let instance_request = node
        .begin_runtime_instance_request(port)
        .await
        .map_err(|error| {
            tracing::debug!(%error, port, "local runtime instance rejected new work");
        })?;
    let upstream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .map_err(|error| {
            tracing::warn!("API proxy: can't reach local OpenAI surface on {port}: {error}");
        })?;
    Ok((instance_request, upstream))
}

async fn route_local_attempt_after_forward(
    tcp_stream: &mut TcpStream,
    upstream: &mut TcpStream,
    port: u16,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_http_response_local(upstream).await {
        Ok(probe) => {
            let result = relay_attempted_response(
                tcp_stream,
                upstream,
                probe,
                RelayAttemptContext {
                    request_id,
                    disconnect_message: "API proxy (local): downstream client disconnected during relay",
                    commit_message: "API proxy (local) ended after commit",
                    route_observer,
                },
                retry_policy,
                response_adapter,
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
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let RouteAttemptLoggingContext {
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
    } = logging;
    let (mut quic_send, mut quic_recv) = match node.open_http_tunnel(host_id).await {
        Ok(tunnel) => tunnel,
        Err(err) => {
            tracing::warn!(
                "API proxy: can't tunnel to host {}: {err}",
                host_id.fmt_short()
            );
            return retryable_route_result_from_error(&err);
        }
    };

    if forward_peer_request(&mut quic_send, host_id, prefetched)
        .await
        .is_err()
    {
        return RouteAttemptResult::RetryableUnavailable;
    }

    route_remote_attempt_after_forward(
        tcp_stream,
        &mut quic_recv,
        host_id,
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
    )
    .await
}

async fn forward_peer_request(
    quic_send: &mut iroh::endpoint::SendStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
) -> Result<(), ()> {
    // Caller credentials are meaningful at ingress, not on a remote peer.
    let peer_request = match prepare_peer_forwarded_request(prefetched) {
        Ok(request) => request,
        Err(err) => {
            tracing::warn!(
                "API proxy: refusing to forward malformed request to host {}: {err}",
                host_id.fmt_short()
            );
            return Err(());
        }
    };

    if let Err(err) = quic_send.write_all(&peer_request).await {
        tracing::warn!(
            "API proxy: failed to forward buffered request to host {}: {err}",
            host_id.fmt_short()
        );
        return Err(());
    }

    Ok(())
}

async fn forward_buffered_request<W: AsyncWrite + Unpin>(
    upstream: &mut W,
    prefetched: &[u8],
) -> std::io::Result<()> {
    upstream.write_all(prefetched).await
}

async fn route_remote_attempt_after_forward(
    tcp_stream: &mut TcpStream,
    quic_recv: &mut iroh::endpoint::RecvStream,
    host_id: iroh::EndpointId,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_http_response(quic_recv).await {
        Ok(probe) => {
            relay_attempted_response(
                tcp_stream,
                quic_recv,
                probe,
                RelayAttemptContext {
                    request_id,
                    disconnect_message: "API proxy (remote): downstream client disconnected during relay",
                    commit_message: "API proxy (remote) ended after commit",
                    route_observer,
                },
                retry_policy,
                response_adapter,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn local_and_remote_forwarding_preserve_the_canonical_request_id_bytes() {
        const REQUEST: &[u8] = b"POST /v1/chat/completions HTTP/1.1\r\nx-request-id: 4c3ca94d-bc1f-4759-912d-f4f6d77d5515\r\nContent-Length: 2\r\n\r\n{}";

        for _route in ["local", "remote"] {
            let (mut upstream, mut received) = tokio::io::duplex(REQUEST.len());
            let forwarding = tokio::spawn(async move {
                forward_buffered_request(&mut upstream, REQUEST)
                    .await
                    .unwrap();
            });
            let mut forwarded = Vec::new();
            received.read_to_end(&mut forwarded).await.unwrap();
            forwarding.await.unwrap();

            assert_eq!(forwarded, REQUEST);
        }
    }
}
