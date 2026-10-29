//! Live transcription sessions (`GET /v1/realtime` websocket upgrades).
//!
//! A session is a long-lived duplex stream pinned to one speech-recognition
//! host, so it bypasses request/response routing. Local ingress picks the host,
//! preferring this node, then the peer with the lowest measured RTT, since live
//! captions are latency-bound. It relays bytes in both directions: over TCP to
//! a local model, or through the peer's HTTP tunnel. The realtime protocol
//! itself stays in the host's Skippy backend.
//!
//! A session that arrives from a peer was already assigned to this host: it is
//! served by a local model or refused, never forwarded again.
//!
//! Sessions draw on their own bounded budget and end after a quiet spell, so
//! parked websockets cannot starve ordinary requests. Exchange-admission
//! plugins (payments, policy) see request/response exchanges only, so a node
//! that runs them refuses live sessions rather than bypassing them.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Semaphore;

use crate::inference::election::{self, InferenceTarget};
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::request_parse::{header_values_from_raw, internal_model_for_public_id};
use crate::network::openai::transport as proxy;
use crate::network::openai::workload_routing::eligible_targets;

pub(super) const REALTIME_PATH: &str = "/v1/realtime";
/// Live sessions one node relays at once, out of the shared ingress budget.
const MAX_REALTIME_SESSIONS: usize = 64;
/// A session with no bytes in either direction for this long is closed.
const REALTIME_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
/// Websocket Close frame (server to client, unmasked) with status 1011,
/// "internal error", sent when the host side fails mid-session.
const CLOSE_INTERNAL_ERROR: [u8; 4] = [0x88, 0x02, 0x03, 0xf3];

fn session_budget() -> &'static Arc<Semaphore> {
    static BUDGET: OnceLock<Arc<Semaphore>> = OnceLock::new();
    BUDGET.get_or_init(|| Arc::new(Semaphore::new(MAX_REALTIME_SESSIONS)))
}

/// Whether a request opens a realtime websocket session.
pub(super) fn is_realtime_upgrade(request: &proxy::BufferedHttpRequest) -> bool {
    opens_realtime_session(&request.method, &request.path, &request.raw)
}

fn opens_realtime_session(method: &str, path: &str, raw: &[u8]) -> bool {
    method.eq_ignore_ascii_case("GET")
        && path.split('?').next() == Some(REALTIME_PATH)
        && header_values_from_raw(raw, "upgrade")
            .unwrap_or_default()
            .iter()
            .flat_map(|value| value.split(','))
            .any(|token| token.trim().eq_ignore_ascii_case("websocket"))
}

/// Relays one realtime session to its host. `requested_by` is the peer that
/// assigned this session to us, if it came over the mesh.
pub(super) async fn relay(
    node: &mesh::Node,
    client: ClientStream,
    request: &proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    ingress_type: crate::runtime::IngressType,
    requested_by: Option<iroh::EndpointId>,
) {
    let _inflight = node.begin_inflight_request();
    if let Err(message) = admit(node, ingress_type).await {
        let _ = crate::network::openai::send_503(client, &message).await;
        return;
    }
    let Ok(_slot) = session_budget().clone().try_acquire_owned() else {
        let _ = crate::network::openai::send_503(
            client,
            "too many live transcription sessions on this node; try again later",
        )
        .await;
        return;
    };
    let (model, target) = match choose_host(node, request, targets, requested_by.is_some()).await {
        Ok(chosen) => chosen,
        Err(message) => {
            let _ = crate::network::openai::send_503(client, &message).await;
            return;
        }
    };
    run_session(node, client, &request.raw, &model, target).await;
}

/// Relays an admitted session to its host and logs how it ended.
async fn run_session(
    node: &mesh::Node,
    mut client: ClientStream,
    raw: &[u8],
    model: &str,
    target: InferenceTarget,
) {
    let opening = upgrade_request(raw, model);
    let started = Instant::now();
    tracing::info!(model, target = ?target, "realtime transcription session started");
    let result = match target {
        InferenceTarget::Local(port) => relay_local(&mut client, &opening, port).await,
        InferenceTarget::Remote(peer) => relay_remote(node, &mut client, &opening, peer).await,
        InferenceTarget::None => Err(anyhow::anyhow!("no target")),
    };
    log_session_end(model, started.elapsed().as_secs(), result);
}

fn log_session_end(model: &str, seconds: u64, result: anyhow::Result<SessionEnd>) {
    match result {
        Ok(ended) => tracing::info!(
            model,
            seconds,
            bytes_to_host = ended.to_host,
            bytes_to_client = ended.to_client,
            reason = ended.reason,
            "realtime transcription session ended"
        ),
        Err(error) => {
            tracing::warn!(model, seconds, %error, "realtime transcription session failed")
        }
    }
}

/// Live sessions honour activity pauses, and are refused where exchange
/// admission plugins would otherwise be bypassed.
async fn admit(node: &mesh::Node, ingress_type: crate::runtime::IngressType) -> Result<(), String> {
    if let crate::runtime::AdmissionResult::Paused { reason, .. } =
        node.activity_policy_guard.check_admission(ingress_type)
    {
        return Err(format!("inference paused: {reason}"));
    }
    if let Some(manager) = node.plugin_manager().await
        && manager.has_exchange_hooks().await
    {
        return Err(
            "live transcription is unavailable on a node with exchange admission plugins".into(),
        );
    }
    Ok(())
}

/// The model and host for a session. The requested model goes through the
/// same public-alias and `#profile` handling as other requests; with none, the
/// first live-capable model is used. A session assigned by a peer may only be
/// served here; otherwise the nearest capable, unblocked peer takes it.
async fn choose_host(
    node: &mesh::Node,
    request: &proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    assigned_by_peer: bool,
) -> Result<(String, InferenceTarget), String> {
    let model = match query_model(&request.path) {
        Some(requested) => resolve_model(node, targets, &requested, assigned_by_peer).await,
        None => first_realtime_model(node, targets)
            .await
            .ok_or("no served model supports live transcription")?,
    };
    let mut candidates =
        eligible_targets(node, &model, REALTIME_PATH, &targets.candidates(&model)).await;
    let has_local = candidates
        .iter()
        .any(|target| matches!(target, InferenceTarget::Local(_)));
    if assigned_by_peer {
        candidates.retain(|target| matches!(target, InferenceTarget::Local(_)));
    } else if !has_local {
        let mut hosts = node.hosts_for_model(&model).await;
        node.peer_blocks
            .retain_unblocked(&mut hosts, crate::network::peer_blocks::now_ms());
        let remote = hosts
            .into_iter()
            .map(InferenceTarget::Remote)
            .collect::<Vec<_>>();
        candidates = eligible_targets(node, &model, REALTIME_PATH, &remote).await;
    }
    let target = nearest(node, candidates).await.ok_or_else(|| {
        if assigned_by_peer {
            format!("{model} is not served by this host for live transcription")
        } else {
            format!("no host serves {model} for live transcription")
        }
    })?;
    Ok((model, target))
}

/// Maps a public model id to the internal one and drops a `#profile` suffix.
/// A peer-assigned session carries the internal name, because the relay
/// forwards the resolved model in place of the client's.
async fn resolve_model(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    requested: &str,
    assigned_by_peer: bool,
) -> String {
    let model = if assigned_by_peer {
        requested.to_string()
    } else {
        let callable = super::callable_models(targets);
        let descriptors = node.all_served_model_descriptors().await;
        internal_model_for_public_id(requested, &callable, &descriptors)
            .unwrap_or_else(|| requested.to_string())
    };
    super::parse_model_with_profile(&model).0.to_string()
}

/// How a relayed session ended, for the session log.
struct SessionEnd {
    to_host: u64,
    to_client: u64,
    reason: &'static str,
}

async fn relay_local(
    client: &mut ClientStream,
    opening: &[u8],
    port: u16,
) -> anyhow::Result<SessionEnd> {
    let mut upstream = tokio::net::TcpStream::connect(("127.0.0.1", port)).await?;
    upstream.set_nodelay(true)?;
    upstream.write_all(opening).await?;
    relay_bytes(client, upstream, REALTIME_IDLE_TIMEOUT).await
}

async fn relay_remote(
    node: &mesh::Node,
    client: &mut ClientStream,
    opening: &[u8],
    peer: iroh::EndpointId,
) -> anyhow::Result<SessionEnd> {
    let (mut send, recv) = node.open_http_tunnel(peer).await?;
    send.write_all(opening).await?;
    let upstream = ClientStream::from_quic_with_prefix(recv, send, Vec::new());
    relay_bytes(client, upstream, REALTIME_IDLE_TIMEOUT).await
}

/// Copies bytes both ways until either side closes or nothing moves for
/// `idle`. If the host side fails, the client gets a websocket Close frame
/// rather than a bare connection reset.
async fn relay_bytes<C, U>(
    client: &mut C,
    upstream: U,
    idle: Duration,
) -> anyhow::Result<SessionEnd>
where
    C: AsyncRead + AsyncWrite + Unpin,
    U: AsyncRead + AsyncWrite + Unpin,
{
    let (mut host_read, mut host_write) = tokio::io::split(upstream);
    let (mut client_read, mut client_write) = tokio::io::split(client);
    let mut to_host = 0u64;
    let mut to_client = 0u64;
    let mut from_client = vec![0u8; 16 * 1024];
    let mut from_host = vec![0u8; 16 * 1024];
    let reason = loop {
        tokio::select! {
            read = tokio::time::timeout(idle, client_read.read(&mut from_client)) => match read {
                Err(_) => break "idle",
                Ok(Ok(0)) | Ok(Err(_)) => break "client closed",
                Ok(Ok(n)) => {
                    if !matches!(tokio::time::timeout(idle, host_write.write_all(&from_client[..n])).await, Ok(Ok(()))) {
                        let _ = tokio::time::timeout(idle, client_write.write_all(&CLOSE_INTERNAL_ERROR)).await;
                        break "host write failed";
                    }
                    to_host += n as u64;
                }
            },
            read = tokio::time::timeout(idle, host_read.read(&mut from_host)) => match read {
                Err(_) => break "idle",
                Ok(Ok(0)) => break "host closed",
                Ok(Err(_)) => {
                    let _ = tokio::time::timeout(idle, client_write.write_all(&CLOSE_INTERNAL_ERROR)).await;
                    break "host read failed";
                }
                Ok(Ok(n)) => {
                    if !matches!(tokio::time::timeout(idle, client_write.write_all(&from_host[..n])).await, Ok(Ok(()))) {
                        break "client write failed";
                    }
                    to_client += n as u64;
                }
            },
        }
    };
    let _ = tokio::time::timeout(idle, client_write.flush()).await;
    let _ = tokio::time::timeout(idle, host_write.shutdown()).await;
    Ok(SessionEnd {
        to_host,
        to_client,
        reason,
    })
}

/// Local first; among peers, the lowest measured direct RTT, then the rest.
async fn nearest(node: &mesh::Node, candidates: Vec<InferenceTarget>) -> Option<InferenceTarget> {
    if let Some(local) = candidates
        .iter()
        .find(|target| matches!(target, InferenceTarget::Local(_)))
    {
        return Some(local.clone());
    }
    let state = node.state.lock().await;
    candidates
        .into_iter()
        .filter(|target| matches!(target, InferenceTarget::Remote(_)))
        .min_by_key(|target| match target {
            InferenceTarget::Remote(peer) => state
                .peers
                .get(peer)
                .and_then(|info| info.current_direct_rtt_ms())
                .unwrap_or(u32::MAX),
            _ => u32::MAX,
        })
}

/// With no `?model=`, the first served model that supports live
/// transcription, local models first.
async fn first_realtime_model(
    node: &mesh::Node,
    targets: &election::ModelTargets,
) -> Option<String> {
    for model in super::callable_models(targets) {
        if !eligible_targets(node, &model, REALTIME_PATH, &targets.candidates(&model))
            .await
            .is_empty()
        {
            return Some(model);
        }
    }
    let now = crate::network::peer_blocks::now_ms();
    let state = node.state.lock().await;
    state.peers.iter().find_map(|(id, peer)| {
        if node.peer_blocks.is_blocked(id, now) {
            return None;
        }
        peer.served_model_descriptors
            .iter()
            .find(|descriptor| {
                crate::network::openai::workload_routing::model_satisfies_request(
                    &descriptor.identity.model_name,
                    REALTIME_PATH,
                    std::slice::from_ref(*descriptor),
                )
            })
            .map(|descriptor| descriptor.identity.model_name.clone())
    })
}

fn query_model(path: &str) -> Option<String> {
    let query = path.split_once('?')?.1;
    query.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (key == "model")
            .then(|| percent_decode(value))
            .filter(|model| !model.is_empty())
    })
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let escaped = (bytes[index] == b'%')
            .then(|| bytes.get(index + 1..index + 3))
            .flatten()
            .filter(|hex| hex.iter().all(u8::is_ascii_hexdigit))
            .and_then(|hex| u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok());
        match (bytes[index], escaped) {
            (_, Some(decoded)) => {
                out.push(decoded);
                index += 3;
            }
            (b'+', None) => {
                out.push(b' ');
                index += 1;
            }
            (byte, None) => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The bytes to send the host: the request head with the resolved `model`
/// in its query and `Connection: Upgrade` restored (ingress parsing
/// normalizes it for request/response forwarding), then any websocket frames
/// the client sent together with its handshake. Ingress rebuilds every head
/// with CRLF line ends.
fn upgrade_request(raw: &[u8], model: &str) -> Vec<u8> {
    let head_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap_or(raw.len());
    let text = String::from_utf8_lossy(&raw[..head_end]);
    let mut lines = text.split("\r\n");
    let mut head = with_model_query(lines.next().unwrap_or_default(), model);
    head.push_str("\r\nConnection: Upgrade");
    for line in lines.filter(|line| {
        !line
            .split_once(':')
            .is_some_and(|(key, _)| key.trim().eq_ignore_ascii_case("connection"))
    }) {
        head.push_str("\r\n");
        head.push_str(line);
    }
    head.push_str("\r\n\r\n");
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(raw.get(head_end + 4..).unwrap_or_default());
    bytes
}

/// Sets the request line's `model` query value, keeping the other parameters.
fn with_model_query(line: &str, model: &str) -> String {
    let mut parts = line.splitn(3, ' ');
    let (Some(method), Some(target), Some(version)) = (parts.next(), parts.next(), parts.next())
    else {
        return line.to_string();
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut pairs = query
        .split('&')
        .filter(|pair| !pair.is_empty() && pair.split('=').next() != Some("model"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    pairs.push(format!("model={}", percent_encode(model)));
    format!("{method} {path}?{} {version}", pairs.join("&"))
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                char::from(byte).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod quic_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_models_are_percent_decoded() {
        assert_eq!(
            query_model(
                "/v1/realtime?intent=transcription&model=ggerganov%2Fwhisper.cpp%3Aggml-tiny.bin"
            )
            .as_deref(),
            Some("ggerganov/whisper.cpp:ggml-tiny.bin")
        );
        assert_eq!(query_model("/v1/realtime?model="), None);
        assert_eq!(query_model("/v1/realtime"), None);
        assert_eq!(percent_decode("a%2"), "a%2");
        assert_eq!(percent_decode("%+1é%zz"), "% 1é%zz");
    }

    #[test]
    fn forwarded_heads_name_the_upgrade_once_and_keep_early_frames() {
        let raw = b"GET /v1/realtime HTTP/1.1\r\nHost: x\r\nconnection: close\r\nUpgrade: websocket\r\n\r\n\x81\x05early";
        assert_eq!(
            upgrade_request(raw, "m"),
            b"GET /v1/realtime?model=m HTTP/1.1\r\nConnection: Upgrade\r\nHost: x\r\nUpgrade: websocket\r\n\r\n\x81\x05early"
        );
    }

    #[test]
    fn forwarded_heads_carry_the_resolved_model() {
        let line = "GET /v1/realtime?intent=transcription&model=public%23fast&x=1 HTTP/1.1";
        let forwarded = with_model_query(line, "ggerganov/whisper.cpp:ggml-base.en.bin");
        assert_eq!(
            forwarded,
            "GET /v1/realtime?intent=transcription&x=1&model=ggerganov%2Fwhisper.cpp%3Aggml-base.en.bin HTTP/1.1"
        );
        assert_eq!(
            query_model(forwarded.split(' ').nth(1).unwrap()).as_deref(),
            Some("ggerganov/whisper.cpp:ggml-base.en.bin")
        );
    }

    #[test]
    fn only_websocket_gets_on_the_realtime_path_open_sessions() {
        let upgrade = b"GET /v1/realtime HTTP/1.1\r\nHost: x\r\nConnection: keep-alive, Upgrade\r\nUPGRADE: WebSocket\r\n\r\n";
        let plain = b"GET /v1/realtime HTTP/1.1\r\nHost: x\r\n\r\n";
        assert!(opens_realtime_session(
            "GET",
            "/v1/realtime?model=m",
            upgrade
        ));
        assert!(!opens_realtime_session("POST", "/v1/realtime", upgrade));
        assert!(!opens_realtime_session("GET", "/v1/realtime", plain));
        assert!(!opens_realtime_session("GET", "/v1/models", upgrade));
    }

    #[tokio::test]
    async fn bytes_flow_both_ways_until_a_side_closes() {
        let (mut client, mut client_side) = tokio::io::duplex(1024);
        let (host_side, mut host) = tokio::io::duplex(1024);
        let relay = tokio::spawn(async move {
            relay_bytes(&mut client_side, host_side, Duration::from_secs(5)).await
        });
        client.write_all(b"hello").await.unwrap();
        let mut buf = [0u8; 5];
        host.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"hello");
        host.write_all(b"world").await.unwrap();
        client.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"world");
        drop(host);
        let ended = relay.await.unwrap().unwrap();
        assert_eq!((ended.to_host, ended.to_client), (5, 5));
        assert_eq!(ended.reason, "host closed");
    }

    #[tokio::test]
    async fn a_quiet_session_is_closed_after_the_idle_timeout() {
        let (_client, mut client_side) = tokio::io::duplex(1024);
        let (host_side, _host) = tokio::io::duplex(1024);
        let ended = relay_bytes(&mut client_side, host_side, Duration::from_millis(50))
            .await
            .unwrap();
        assert_eq!(ended.reason, "idle");
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_host_write_ends_the_session() {
        let (mut client, mut client_side) = tokio::io::duplex(16);
        let (host_side, _host) = tokio::io::duplex(1);
        client.write_all(b"hello").await.unwrap();
        let ended = tokio::time::timeout(
            Duration::from_secs(5),
            relay_bytes(&mut client_side, host_side, Duration::from_secs(1)),
        )
        .await
        .expect("the stalled write must time out")
        .unwrap();
        assert_eq!(ended.reason, "host write failed");
        assert_eq!(ended.to_host, 0);
        let mut frame = [0; 4];
        client.read_exact(&mut frame).await.unwrap();
        assert_eq!(frame, CLOSE_INTERNAL_ERROR);
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_client_write_ends_the_session() {
        let (_client, mut client_side) = tokio::io::duplex(1);
        let (host_side, mut host) = tokio::io::duplex(16);
        host.write_all(b"hello").await.unwrap();
        let ended = tokio::time::timeout(
            Duration::from_secs(5),
            relay_bytes(&mut client_side, host_side, Duration::from_secs(1)),
        )
        .await
        .expect("the stalled write must time out")
        .unwrap();
        assert_eq!(ended.reason, "client write failed");
        assert_eq!(ended.to_client, 0);
    }

    /// A host stream whose reads fail, as a dropped QUIC tunnel does.
    struct FailingHost;

    impl AsyncRead for FailingHost {
        fn poll_read(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            _buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Err(std::io::ErrorKind::ConnectionReset.into()))
        }
    }

    impl AsyncWrite for FailingHost {
        fn poll_write(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &[u8],
        ) -> std::task::Poll<std::io::Result<usize>> {
            std::task::Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn poll_shutdown(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn a_failing_host_gets_the_client_a_close_frame() {
        let (mut client, mut client_side) = tokio::io::duplex(1024);
        let ended = relay_bytes(&mut client_side, FailingHost, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(ended.reason, "host read failed");
        drop(client_side);
        let mut frame = Vec::new();
        client.read_to_end(&mut frame).await.unwrap();
        assert_eq!(frame, CLOSE_INTERNAL_ERROR);
    }
}
