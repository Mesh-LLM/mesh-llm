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

use tokio::io::AsyncWriteExt;

use crate::inference::election::{self, InferenceTarget};
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::transport as proxy;
use crate::network::openai::workload_routing::eligible_targets;

pub(super) const REALTIME_PATH: &str = "/v1/realtime";

/// Whether a request opens a realtime websocket session.
pub(super) fn is_realtime_upgrade(request: &proxy::BufferedHttpRequest) -> bool {
    request.method.eq_ignore_ascii_case("GET")
        && request.path.split('?').next() == Some(REALTIME_PATH)
        && header_values(&request.raw, "upgrade")
            .iter()
            .any(|value| value.eq_ignore_ascii_case("websocket"))
}

/// Relays one realtime session to its host. `requested_by` is the peer that
/// assigned this session to us, if it came over the mesh.
pub(super) async fn relay(
    node: &mesh::Node,
    mut client: ClientStream,
    request: &proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    ingress_type: crate::runtime::IngressType,
    requested_by: Option<iroh::EndpointId>,
) {
    let _inflight = node.begin_inflight_request();
    if let crate::runtime::AdmissionResult::Paused { reason, .. } =
        node.activity_policy_guard.check_admission(ingress_type)
    {
        let _ =
            crate::network::openai::send_503(client, &format!("inference paused: {reason}")).await;
        return;
    }
    let target = match choose_host(node, request, targets, requested_by.is_some()).await {
        Ok(target) => target,
        Err(message) => {
            let _ = crate::network::openai::send_503(client, &message).await;
            return;
        }
    };
    let head = upgrade_head(&request.raw);
    let result = match target {
        InferenceTarget::Local(port) => relay_local(&mut client, &head, port).await,
        InferenceTarget::Remote(peer) => relay_remote(node, &mut client, &head, peer).await,
        InferenceTarget::None => return,
    };
    if let Err(error) = result {
        tracing::debug!(%error, "realtime transcription session ended");
    }
}

/// The host for a session: the requested (or first live-capable) model, on
/// this node if it serves it. A session assigned by a peer may only be served
/// here; otherwise the nearest capable peer takes it.
async fn choose_host(
    node: &mesh::Node,
    request: &proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    assigned_by_peer: bool,
) -> Result<InferenceTarget, String> {
    let model = match query_model(&request.path) {
        Some(model) => model,
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
        let remote = node
            .hosts_for_model(&model)
            .await
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
    tracing::info!(model = %model, target = ?target, "relaying realtime transcription session");
    Ok(target)
}

async fn relay_local(client: &mut ClientStream, head: &[u8], port: u16) -> anyhow::Result<()> {
    let mut upstream = tokio::net::TcpStream::connect(("127.0.0.1", port)).await?;
    upstream.set_nodelay(true)?;
    upstream.write_all(head).await?;
    tokio::io::copy_bidirectional(client, &mut upstream).await?;
    Ok(())
}

async fn relay_remote(
    node: &mesh::Node,
    client: &mut ClientStream,
    head: &[u8],
    peer: iroh::EndpointId,
) -> anyhow::Result<()> {
    let (mut send, recv) = node.open_http_tunnel(peer).await?;
    send.write_all(head).await?;
    let mut upstream = ClientStream::from_quic_with_prefix(recv, send, Vec::new());
    tokio::io::copy_bidirectional(client, &mut upstream).await?;
    Ok(())
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
    let state = node.state.lock().await;
    state.peers.values().find_map(|peer| {
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

/// The request head to forward. Ingress parsing normalizes `Connection` for
/// request/response forwarding; a websocket handshake needs it to name the
/// upgrade again.
fn upgrade_head(raw: &[u8]) -> Vec<u8> {
    let head_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap_or(raw.len());
    let text = String::from_utf8_lossy(&raw[..head_end]);
    let mut lines = text.split("\r\n");
    let mut head = lines.next().unwrap_or_default().to_string();
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
    head.into_bytes()
}

/// Values of one header in a raw request head, case-insensitively.
fn header_values(raw: &[u8], name: &str) -> Vec<String> {
    let head_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..head_end])
        .split("\r\n")
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .filter(|(key, _)| key.trim().eq_ignore_ascii_case(name))
        .flat_map(|(_, value)| {
            value
                .split(',')
                .map(|token| token.trim().to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

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
    fn forwarded_heads_name_the_upgrade_once() {
        let raw = b"GET /v1/realtime HTTP/1.1\r\nHost: x\r\nconnection: close\r\nUpgrade: websocket\r\n\r\n";
        assert_eq!(
            String::from_utf8(upgrade_head(raw)).unwrap(),
            "GET /v1/realtime HTTP/1.1\r\nConnection: Upgrade\r\nHost: x\r\nUpgrade: websocket\r\n\r\n"
        );
    }

    #[test]
    fn upgrade_headers_are_matched_case_insensitively_across_tokens() {
        let raw = b"GET /v1/realtime HTTP/1.1\r\nHost: x\r\nConnection: keep-alive, Upgrade\r\nUPGRADE: WebSocket\r\n\r\n";
        assert_eq!(header_values(raw, "upgrade"), ["WebSocket"]);
        assert_eq!(header_values(raw, "connection"), ["keep-alive", "Upgrade"]);
        assert!(header_values(raw, "x-missing").is_empty());
    }
}
