use super::cancellation::{CancelUpstream, cancel_upstream_if_client_disconnected};
use super::common::{
    PeerCapsuleIdSink, ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult,
    retryable_route_result_from_error,
};
use super::dispatch::{RelayAttemptContext, relay_attempted_response};
use super::probe::{
    PEER_CAPSULE_ID_HEADER, ResponseProbe, peer_response_header_value, probe_http_response,
    probe_http_response_local,
};
use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::forwarded_request::prepare_peer_forwarded_request;
use crate::network::openai::request_normalize::ResponseAdapter;
use anyhow::Result;
use mesh_llm_events::logging::identifiers::RequestId;
use std::future::Future;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

pub(in crate::network::openai) async fn route_local_attempt(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    port: u16,
    prefetched: &[u8],
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let RouteAttemptLoggingContext {
        exchange_id: _,
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
        served_by,
        peer_capsule_id,
    } = logging;
    #[cfg(feature = "payments")]
    {
        if !super::paid::is_local_origin(tcp_stream) {
            let model = super::super::request_parse::parse_json_body_from_http_request(prefetched)
                .and_then(|body| {
                    body.get("model")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                });
            match node.advertised_payment_offers().await {
                Ok(prices)
                    if model
                        .as_ref()
                        .is_some_and(|model| prices.contains_key(model)) =>
                {
                    return super::paid::payment_error(
                        tcp_stream,
                        "this provider requires the Lightning payment protocol",
                    )
                    .await;
                }
                Err(_) => {
                    return super::paid::payment_error(
                        tcp_stream,
                        "seller payment state unavailable",
                    )
                    .await;
                }
                _ => {}
            }
        }
    }
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
        served_by,
        peer_capsule_id,
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

#[allow(clippy::too_many_arguments)]
pub(super) async fn route_local_attempt_after_forward<U: AsyncRead + Unpin + CancelUpstream>(
    tcp_stream: &mut ClientStream,
    upstream: &mut U,
    port: u16,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    peer_capsule_id: Option<&PeerCapsuleIdSink>,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_with_downstream_disconnect(tcp_stream, probe_http_response_local(upstream)).await {
        ProbeOutcome::ClientDisconnected => {
            cancel_upstream_if_client_disconnected(RouteAttemptResult::ClientDisconnected, upstream)
                .await
        }
        ProbeOutcome::Response(Ok(probe)) => {
            record_peer_capsule_id(peer_capsule_id, &probe);
            let result = relay_attempted_response(
                tcp_stream,
                upstream,
                probe,
                RelayAttemptContext {
                    request_id,
                    disconnect_message: "API proxy (local): downstream client disconnected during relay",
                    commit_message: "API proxy (local) ended after commit",
                    served_by,
                    route_observer,
                },
                retry_policy,
                response_adapter,
            )
            .await;
            cancel_upstream_if_client_disconnected(result, upstream).await
        }
        ProbeOutcome::Response(Err(err)) => {
            tracing::warn!(
                "API proxy: failed to read local response from OpenAI surface on {port}: {err}"
            );
            retryable_route_result_from_error(&err)
        }
    }
}

enum ProbeOutcome {
    ClientDisconnected,
    Response(Result<ResponseProbe>),
}

/// Race the complete upstream response probe against a non-consuming
/// downstream close check. Pipelined TCP bytes leave the watcher pending, while
/// a TCP reset or QUIC STOP_SENDING reports a client disconnect promptly.
async fn probe_with_downstream_disconnect<F>(client: &ClientStream, probe: F) -> ProbeOutcome
where
    F: Future<Output = Result<ResponseProbe>>,
{
    tokio::pin!(probe);
    tokio::select! {
        biased;
        disconnected = client.wait_for_response_disconnect() => {
            debug_assert!(disconnected);
            ProbeOutcome::ClientDisconnected
        }
        response = &mut probe => ProbeOutcome::Response(response),
    }
}

#[allow(clippy::too_many_arguments)]
async fn route_remote_attempt_after_forward<R: AsyncRead + Unpin + CancelUpstream>(
    tcp_stream: &mut ClientStream,
    quic_recv: &mut R,
    host_id: iroh::EndpointId,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    peer_capsule_id: Option<&PeerCapsuleIdSink>,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_with_downstream_disconnect(tcp_stream, probe_http_response(quic_recv)).await {
        ProbeOutcome::ClientDisconnected => {
            cancel_upstream_if_client_disconnected(
                RouteAttemptResult::ClientDisconnected,
                quic_recv,
            )
            .await
        }
        ProbeOutcome::Response(Ok(probe)) => {
            record_peer_capsule_id(peer_capsule_id, &probe);
            let result = relay_attempted_response(
                tcp_stream,
                quic_recv,
                probe,
                RelayAttemptContext {
                    request_id,
                    disconnect_message: "API proxy (remote): downstream client disconnected during relay",
                    commit_message: "API proxy (remote) ended after commit",
                    served_by,
                    route_observer,
                },
                retry_policy,
                response_adapter,
            )
            .await;
            cancel_upstream_if_client_disconnected(result, quic_recv).await
        }
        ProbeOutcome::Response(Err(err)) => {
            tracing::warn!(
                "API proxy: failed to read response from host {}: {err}",
                host_id.fmt_short()
            );
            retryable_route_result_from_error(&err)
        }
    }
}

pub(in crate::network::openai) async fn route_remote_attempt(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let RouteAttemptLoggingContext {
        exchange_id: _,
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
        served_by,
        peer_capsule_id,
    } = logging;
    #[cfg(feature = "payments")]
    {
        if let Ok(request) = crate::network::payments::request::PaidRequest::parse(prefetched)
            && let Some(price) = node.peer_payment_offer(host_id, &request.model).await
        {
            return super::paid::route(node, tcp_stream, host_id, prefetched, price, logging).await;
        }
    }
    #[cfg(feature = "payments")]
    let sanitized = match crate::network::payments::request::strip_intent(prefetched) {
        Ok(raw) => raw,
        Err(_) => {
            return super::paid::payment_error(tcp_stream, "invalid request payment intent").await;
        }
    };
    #[cfg(not(feature = "payments"))]
    let sanitized = prefetched.to_vec();
    let prefetched = sanitized.as_slice();
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
        served_by,
        peer_capsule_id,
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
    #[cfg(feature = "payments")]
    let sanitized = crate::network::payments::request::strip_intent(prefetched)
        .map_err(std::io::Error::other)?;
    #[cfg(not(feature = "payments"))]
    let sanitized = prefetched.to_vec();
    upstream.write_all(&sanitized).await
}

/// Peek the just-probed, untouched response headers for a peer's
/// `X-Capsule-Id` and stash it in `sink`, before any relay/adapter rewrite
/// runs. A no-op when `sink` is `None` (every attempt except the
/// `RemoteMesh` dispatch path) or the header is absent — never invents a
/// value.
fn record_peer_capsule_id(sink: Option<&PeerCapsuleIdSink>, probe: &ResponseProbe) {
    let Some(sink) = sink else {
        return;
    };
    if let Some(capsule_id) =
        peer_response_header_value(&probe.buffered, probe.header_end, PEER_CAPSULE_ID_HEADER)
    {
        sink.set(capsule_id);
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
    use tokio::sync::oneshot;
    use tokio::time::{Duration, timeout};

    /// A real silent duplex pipe as the upstream half of `CancelUpstream`.
    struct DuplexUpstream {
        inner: tokio::io::DuplexStream,
        cancels: Arc<AtomicUsize>,
    }

    impl AsyncRead for DuplexUpstream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            Pin::new(&mut self.inner).poll_read(cx, buf)
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

    /// Both route arms must cancel their silent upstream as soon as the real
    /// downstream socket is reset, without waiting for the response probe.
    #[tokio::test]
    async fn a_local_attempt_cancels_the_upstream_when_the_client_disconnects() {
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_address = upstream_listener.local_addr().unwrap();
        let (upstream_ready_tx, upstream_ready_rx) = oneshot::channel();
        let upstream_task = tokio::spawn(async move {
            let (mut upstream_peer, _) = upstream_listener.accept().await.unwrap();
            upstream_ready_tx.send(()).unwrap();
            let mut bytes = [0u8; 1];
            timeout(Duration::from_secs(2), upstream_peer.read(&mut bytes))
                .await
                .expect("local cancellation must reach the upstream socket")
                .expect("reading the cancelled upstream socket must succeed")
        });

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            let mut upstream = TcpStream::connect(upstream_address).await.unwrap();
            route_local_attempt_after_forward(
                &mut client,
                &mut upstream,
                0,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let client_socket = TcpStream::connect(address).await.unwrap();
        upstream_ready_rx.await.unwrap();
        client_socket.set_zero_linger().unwrap();
        drop(client_socket);

        let result = timeout(Duration::from_secs(2), task)
            .await
            .expect("local route must notice a reset promptly")
            .unwrap();

        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
        assert_eq!(
            upstream_task.await.unwrap(),
            0,
            "local cancellation must shut down the concrete TCP upstream"
        );
    }

    #[tokio::test]
    async fn a_remote_attempt_cancels_the_peer_tunnel_when_the_client_disconnects() {
        let (_upstream_writer, upstream_reader) = tokio::io::duplex(64 * 1024);
        let cancels = Arc::new(AtomicUsize::new(0));
        let mut upstream = DuplexUpstream {
            inner: upstream_reader,
            cancels: Arc::clone(&cancels),
        };
        let host_id = iroh::SecretKey::generate().public();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let client_socket = TcpStream::connect(address).await.unwrap();
        client_socket.set_zero_linger().unwrap();
        drop(client_socket);

        let result = timeout(Duration::from_secs(2), task)
            .await
            .expect("remote route must notice a reset promptly")
            .unwrap();

        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_quic_downstream_reset_cancels_the_peer_tunnel_during_probe() {
        const TEST_ALPN: &[u8] = b"mesh-llm/routing-cancellation-test/1";

        let server = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(iroh::SecretKey::generate())
            .alpns(vec![TEST_ALPN.to_vec()])
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .unwrap()
            .bind()
            .await
            .unwrap();
        let server_endpoint = server.clone();
        let cancels = Arc::new(AtomicUsize::new(0));
        let route_cancels = Arc::clone(&cancels);
        let host_id = iroh::SecretKey::generate().public();
        let route = tokio::spawn(async move {
            let incoming = server_endpoint.accept().await.expect("connection arrives");
            let connection = incoming.await.expect("connection negotiates");
            let (send, recv) = connection.accept_bi().await.expect("stream arrives");
            let mut downstream = ClientStream::from_quic_with_prefix(recv, send, Vec::new());
            let (_upstream_writer, upstream_reader) = tokio::io::duplex(64 * 1024);
            let mut upstream = DuplexUpstream {
                inner: upstream_reader,
                cancels: route_cancels,
            };
            route_remote_attempt_after_forward(
                &mut downstream,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });

        let client = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(iroh::SecretKey::generate())
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .unwrap()
            .bind()
            .await
            .unwrap();
        let connection = client.connect(server.addr(), TEST_ALPN).await.unwrap();
        let (_request_send, mut response_recv) = connection.open_bi().await.unwrap();
        response_recv.stop(42u32.into()).unwrap();

        let result = timeout(Duration::from_secs(2), route)
            .await
            .expect("QUIC reset must interrupt the pending response probe")
            .unwrap();
        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
        assert_eq!(cancels.load(Ordering::SeqCst), 1);

        client.close().await;
        server.close().await;
    }

    #[tokio::test]
    async fn pipelined_downstream_bytes_are_not_consumed_by_disconnect_watch() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (downstream, _) = listener.accept().await.unwrap();
            let mut downstream: ClientStream = downstream.into();
            assert!(
                timeout(
                    Duration::from_millis(100),
                    downstream.wait_for_response_disconnect()
                )
                .await
                .is_err(),
                "pipelined bytes are not a disconnect"
            );
            let mut pipelined = [0u8; 1];
            timeout(
                Duration::from_secs(1),
                downstream.read_exact(&mut pipelined),
            )
            .await
            .expect("pipelined byte must remain available after probing")
            .unwrap();
            pipelined
        });

        let mut client = TcpStream::connect(address).await.unwrap();
        client.write_all(b"x").await.unwrap();

        assert_eq!(task.await.unwrap(), [b'x']);
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
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                None,
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

    /// A `RemoteMesh` attempt records the
    /// peer's `X-Capsule-Id` response header into the caller-supplied sink —
    /// this is how the routing node learns what to thread onto its own
    /// terminal plugin event (see `ingress.rs`'s remote-mesh routing).
    #[tokio::test]
    async fn a_remote_attempt_records_the_peers_capsule_id_into_the_sink() {
        let body = "x".repeat(8);
        let header = format!(
            "HTTP/1.1 200 OK\r\nX-Capsule-Id: cap-peer-1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut upstream = ScriptedUpstream::new(vec![Ok(header.into_bytes())]);
        let host_id = iroh::SecretKey::generate().public();
        let sink = PeerCapsuleIdSink::new();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                Some(&sink),
                OpenAiRouteObserver::default(),
            )
            .await;
            sink
        });
        let mut socket = TcpStream::connect(address).await.unwrap();
        let mut relayed = Vec::new();
        socket.read_to_end(&mut relayed).await.unwrap();

        let sink = task.await.unwrap();
        assert_eq!(sink.take(), Some("cap-peer-1".to_string()));
    }

    /// Mutant: no `X-Capsule-Id` on the peer's response — the sink stays
    /// empty. Never invented.
    #[tokio::test]
    async fn a_remote_attempt_without_a_capsule_id_header_leaves_the_sink_empty() {
        let body = "x".repeat(8);
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut upstream = ScriptedUpstream::new(vec![Ok(header.into_bytes())]);
        let host_id = iroh::SecretKey::generate().public();
        let sink = PeerCapsuleIdSink::new();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                Some(&sink),
                OpenAiRouteObserver::default(),
            )
            .await;
            sink
        });
        let mut socket = TcpStream::connect(address).await.unwrap();
        let mut relayed = Vec::new();
        socket.read_to_end(&mut relayed).await.unwrap();

        let sink = task.await.unwrap();
        assert_eq!(sink.take(), None);
    }

    #[tokio::test]
    async fn a_write_half_closed_downstream_still_receives_the_upstream_response() {
        let body = "half-close response";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut upstream = ScriptedUpstream::new(vec![Ok(response.into_bytes())]);
        let host_id = iroh::SecretKey::generate().public();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });

        let mut socket = TcpStream::connect(address).await.unwrap();
        socket.shutdown().await.unwrap();
        let mut relayed = Vec::new();
        timeout(Duration::from_secs(1), socket.read_to_end(&mut relayed))
            .await
            .expect("write-half-closed client should still receive a response")
            .unwrap();
        let result = timeout(Duration::from_secs(1), task)
            .await
            .expect("route should finish after relaying the response")
            .unwrap();

        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        assert!(String::from_utf8_lossy(&relayed).contains(body));
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

    /// A remote request for a model this node does not charge for must reach
    /// local inference — even when the node runs with no `payments.v1`
    /// provider at all, the documented free-only configuration. Treating that
    /// absence as "seller payment state unavailable" answered 402 before any
    /// inference happened.
    #[cfg(feature = "payments")]
    #[tokio::test]
    async fn a_remote_free_request_reaches_local_inference_without_a_payments_provider() {
        const TEST_ALPN: &[u8] = b"mesh-llm/free-routing-regression/1";
        const BODY: &str = r#"{"model":"free-model","prompt":"Hi","max_tokens":8}"#;

        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
            .await
            .unwrap();
        // Free-only: a manager is installed, but nothing serves `payments.v1`.
        let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(8);
        let manager = crate::plugin::PluginManager::start_with_in_process(
            &crate::plugin::ResolvedPlugins {
                externals: Vec::new(),
                inactive: Vec::new(),
            },
            crate::plugin::PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            mesh_tx,
            crate::plugin::InProcessPlugins::default(),
        )
        .await
        .unwrap();
        node.set_plugin_manager(manager).await;

        // The node's local OpenAI surface, which proves inference was reached.
        let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = backend.local_addr().unwrap().port();
        let (inference_reached_tx, inference_reached_rx) = oneshot::channel();
        let backend_task = tokio::spawn(async move {
            let (mut stream, _) = backend.accept().await.expect("local inference connection");
            let mut raw = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream
                    .read(&mut buffer)
                    .await
                    .expect("local inference request");
                if read == 0 {
                    break;
                }
                raw.extend_from_slice(&buffer[..read]);
                if raw.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            inference_reached_tx.send(()).unwrap();
            let body = r#"{"id":"free-1","model":"free-model","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("local inference response");
            stream.shutdown().await.expect("local inference shutdown");
        });

        let server = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(iroh::SecretKey::generate())
            .alpns(vec![TEST_ALPN.to_vec()])
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .unwrap()
            .bind()
            .await
            .unwrap();
        let server_address = server.addr();
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{BODY}",
            BODY.len()
        );
        // The caller dials in over QUIC and reads the relayed response. A QUIC
        // ingress exposes no socket address, so this request has a remote
        // origin and the payments branch runs for it.
        let caller_request = request.clone();
        let caller_task = tokio::spawn(async move {
            let caller = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
                .secret_key(iroh::SecretKey::generate())
                .relay_mode(iroh::endpoint::RelayMode::Disabled)
                .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
                .unwrap()
                .bind()
                .await
                .unwrap();
            let connection = caller.connect(server_address, TEST_ALPN).await.unwrap();
            let (mut request_send, mut response_recv) = connection.open_bi().await.unwrap();
            request_send
                .write_all(caller_request.as_bytes())
                .await
                .expect("the caller sends its request over the QUIC stream");
            let relayed = response_recv
                .read_to_end(16 * 1024)
                .await
                .expect("the caller must receive the relayed response");
            caller.close().await;
            relayed
        });

        let incoming = server.accept().await.expect("the remote caller arrives");
        let connection = incoming.await.expect("the remote caller negotiates");
        let (send, recv) = connection
            .accept_bi()
            .await
            .expect("the remote request stream");
        let mut client = ClientStream::from_quic_with_prefix(recv, send, Vec::new());

        let result = timeout(
            Duration::from_secs(10),
            route_local_attempt(
                &node,
                &mut client,
                port,
                request.as_bytes(),
                RouteAttemptLoggingContext {
                    exchange_id: None,
                    request_id: RequestId::new(),
                    retry_policy: ResponseRetryPolicy::next_target_available(false),
                    response_adapter: ResponseAdapter::None,
                    route_observer: OpenAiRouteObserver::default(),
                    served_by: None,
                    peer_capsule_id: None,
                },
            ),
        )
        .await
        .expect("a free remote request must not stall");

        assert!(
            matches!(
                result,
                RouteAttemptResult::Delivered {
                    status_code: 200,
                    ..
                }
            ),
            "{result:?}"
        );

        // `connection` stays alive here so the caller can still read the
        // relayed response before the stream is torn down.
        let relayed = timeout(Duration::from_secs(5), caller_task)
            .await
            .expect("the caller must receive its response")
            .unwrap();
        assert!(
            String::from_utf8_lossy(&relayed).starts_with("HTTP/1.1 200"),
            "{}",
            String::from_utf8_lossy(&relayed)
        );
        drop(connection);
        timeout(Duration::from_secs(5), inference_reached_rx)
            .await
            .expect("the request must reach local inference")
            .expect("the local inference task must report acceptance");
        backend_task.await.unwrap();
        drop(client);
        node.endpoint.close().await;
        server.close().await;
    }
}
