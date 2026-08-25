use super::cancellation::{CancelUpstream, cancel_upstream_if_client_disconnected};
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
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
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
            cancel_upstream_if_client_disconnected(result, upstream).await
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

async fn route_remote_attempt_after_forward<R: AsyncRead + Unpin + CancelUpstream>(
    tcp_stream: &mut TcpStream,
    quic_recv: &mut R,
    host_id: iroh::EndpointId,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_http_response(quic_recv).await {
        Ok(probe) => {
            let result = relay_attempted_response(
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
            .await;
            cancel_upstream_if_client_disconnected(result, quic_recv).await
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
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    use tokio::io::AsyncReadExt;
    use tokio::io::ReadBuf;
    use tokio::net::TcpListener;
    use tokio::sync::Notify;
    use tokio::sync::oneshot;
    use tokio::time::{Duration, timeout};

    /// A real duplex pipe as the upstream half of `CancelUpstream`, wrapped to
    /// signal a `Notify` the moment a read finds nothing buffered yet.
    ///
    /// Unlike `ScriptedUpstream` below, this can genuinely block waiting for
    /// more bytes, which is what lets a test prove the route arm is actively
    /// relaying (header consumed, now waiting on the body) at a specific
    /// moment, rather than assuming it from timing.
    struct DuplexUpstream {
        inner: tokio::io::DuplexStream,
        cancels: Arc<AtomicUsize>,
        waiting_for_more: Arc<Notify>,
    }

    impl AsyncRead for DuplexUpstream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            let result = Pin::new(&mut self.inner).poll_read(cx, buf);
            if result.is_pending() {
                self.waiting_for_more.notify_one();
            }
            result
        }
    }

    impl CancelUpstream for DuplexUpstream {
        async fn cancel(&mut self) {
            self.cancels.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// An upstream that hands out a fixed script of reads, then records whether
    /// the route arm cancelled it.
    struct ScriptedUpstream {
        steps: std::collections::VecDeque<Result<Vec<u8>, io::ErrorKind>>,
        cancels: Arc<AtomicUsize>,
    }

    impl ScriptedUpstream {
        fn new(steps: Vec<Result<Vec<u8>, io::ErrorKind>>) -> Self {
            Self {
                steps: steps.into(),
                cancels: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl AsyncRead for ScriptedUpstream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            match self.steps.pop_front() {
                Some(Ok(bytes)) => {
                    buf.put_slice(&bytes);
                    Poll::Ready(Ok(()))
                }
                Some(Err(kind)) => Poll::Ready(Err(io::Error::new(kind, "scripted failure"))),
                None => Poll::Ready(Ok(())),
            }
        }
    }

    impl CancelUpstream for ScriptedUpstream {
        async fn cancel(&mut self) {
            self.cancels.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A remote attempt that loses its client must tell the worker peer to stop.
    ///
    /// The local arm has always shut its upstream down here. The remote arm did
    /// not, so a worker on another node kept generating a response for a client
    /// that had already hung up — observed in rc6 validation as the worker
    /// logging `request_stream_completed` after the gateway had already logged
    /// `request_dropped`.
    #[tokio::test]
    async fn a_remote_attempt_cancels_the_peer_tunnel_when_the_client_disconnects() {
        let (mut upstream_writer, upstream_reader) = tokio::io::duplex(64 * 1024);
        let cancels = Arc::new(AtomicUsize::new(0));
        let waiting_for_more = Arc::new(Notify::new());
        let mut upstream = DuplexUpstream {
            inner: upstream_reader,
            cancels: Arc::clone(&cancels),
            waiting_for_more: Arc::clone(&waiting_for_more),
        };
        let host_id = iroh::SecretKey::generate().public();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (disconnected_tx, disconnected_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let client_std = client.into_std().unwrap();
            let observer_std = client_std.try_clone().unwrap();
            let mut client = TcpStream::from_std(client_std).unwrap();
            let observer = TcpStream::from_std(observer_std).unwrap();
            let observer_task = tokio::spawn(async move {
                let mut peeked = [0; 1];
                loop {
                    if observer.readable().await.is_err() {
                        break;
                    }
                    match observer.peek(&mut peeked).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                }
                let _ = disconnected_tx.send(());
            });
            let result = route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                OpenAiRouteObserver::default(),
            )
            .await;
            observer_task.await.unwrap();
            result
        });
        let client_socket = TcpStream::connect(address).await.unwrap();

        let body = "x".repeat(64);
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        // Send only the header. The route arm buffers it, then blocks asking
        // upstream for body bytes that have not arrived yet.
        upstream_writer.write_all(header.as_bytes()).await.unwrap();
        waiting_for_more.notified().await;

        // The route arm is now genuinely waiting on upstream body bytes, not
        // still parsing the header -- closing the client here means the write
        // it makes once the body arrives lands on an actually disconnected
        // socket, rather than an upstream read failure standing in for one.
        //
        // A graceful close (a plain `drop`) sends only a FIN; wait until the
        // accepted socket observes EOF/reset before releasing the body. This
        // closes the propagation window without consuming route data.
        client_socket.set_zero_linger().unwrap();
        drop(client_socket);
        timeout(Duration::from_secs(5), disconnected_rx)
            .await
            .unwrap()
            .unwrap();

        upstream_writer.write_all(body.as_bytes()).await.unwrap();

        let result = task.await.unwrap();

        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
        assert_eq!(
            cancels.load(Ordering::SeqCst),
            1,
            "the peer tunnel was left running after the client disconnected"
        );
    }

    /// The counterpart: a normal delivery must not cancel anything.
    #[tokio::test]
    async fn a_remote_attempt_that_delivers_leaves_the_peer_tunnel_alone() {
        let body = "x".repeat(8);
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut upstream = ScriptedUpstream::new(vec![Ok(header.into_bytes())]);
        let cancels = Arc::clone(&upstream.cancels);
        let host_id = iroh::SecretKey::generate().public();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (mut client, _) = listener.accept().await.unwrap();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let mut socket = TcpStream::connect(address).await.unwrap();
        let mut relayed = Vec::new();
        socket.read_to_end(&mut relayed).await.unwrap();

        let result = task.await.unwrap();

        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        assert_eq!(cancels.load(Ordering::SeqCst), 0);
    }

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
