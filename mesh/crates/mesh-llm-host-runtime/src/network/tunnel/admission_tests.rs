//! Regression coverage for #2206: the QUIC HTTP tunnel ingress must refuse
//! streams that exceed its quota instead of spawning unbounded per-stream
//! work.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use super::Manager;
use crate::mesh::{Node, NodeRole, QuicBindSelection, RelayConfig, RelayPolicy, TunnelChannels};
use iroh::endpoint::RecvStream;

/// Documented per-peer cap for concurrent tunneled requests (#2206): one
/// authenticated peer must not keep more than this many tunnel request
/// handlers alive at once.
const MAX_TUNNELED_PER_PEER: u64 = 32;
/// How many stalled partial requests the peer opens, past the per-peer cap.
const STALLED_STREAMS: usize = MAX_TUNNELED_PER_PEER as usize + 12;
/// Sampling window during which every stalled handler must still be alive:
/// shorter than the ingress' 5 s header-prefix deadline.
const SAMPLE_WINDOW: Duration = Duration::from_secs(2);
/// Refusals must land while the admitted streams are still stalled on their
/// 5 s header-prefix deadline; the refused-stream read window below ends
/// before it fires.
const REFUSAL_WINDOW: Duration = Duration::from_secs(2);

async fn start_node() -> (Node, TunnelChannels) {
    let relay_urls = Vec::new();
    let relay_auths = HashMap::new();
    Node::start(
        NodeRole::Client,
        RelayConfig {
            urls: &relay_urls,
            auths: &relay_auths,
            policy: RelayPolicy::Disabled,
        },
        QuicBindSelection {
            ip: Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            port: Some(0),
        },
        Some(0.0),
        false,
        None,
        None,
        crate::MeshRequirements::unrestricted(),
    )
    .await
    .expect("start tunnel admission test node")
}

/// Waits until the response side of one tunneled stream ends: an error (the
/// peer reset it) or EOF.
async fn stream_closed_early(recv: &mut RecvStream) {
    let mut buf = [0u8; 1024];
    loop {
        match recv.read(&mut buf).await {
            Ok(None) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

#[tokio::test]
async fn tunneled_http_ingress_caps_concurrent_requests_per_peer() {
    let (sender, _sender_channels) = start_node().await;
    let (receiver, receiver_channels) = start_node().await;
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .expect("connect tunnel admission test nodes");

    // Drive the receiver's tunnel ingress exactly the way the runtime wires
    // it: every tunneled HTTP stream is handled in a spawned task.
    let manager_node = receiver.clone();
    let http_rx = receiver_channels.http;
    let stage_rx = receiver_channels.stage;
    let legacy_rx = receiver_channels.rpc;
    tokio::spawn(async move {
        let _manager = Manager::start(manager_node, legacy_rx, http_rx, stage_rx)
            .await
            .expect("start the receiver's tunnel manager");
    });

    // The peer opens more stalled partial requests than its quota: each one
    // sends nothing after the stream type byte, so a handler that admitted it
    // stays stalled until its 5 s header deadline.
    let mut streams = Vec::with_capacity(STALLED_STREAMS);
    for _ in 0..STALLED_STREAMS {
        let (send, recv) = sender
            .open_http_tunnel(receiver.id())
            .await
            .expect("open a tunneled request stream");
        streams.push((send, recv));
    }

    // While the streams are stalled, watch how many request handlers the
    // receiver keeps alive concurrently.
    let mut peak_inflight = 0u64;
    let sample_end = tokio::time::Instant::now() + SAMPLE_WINDOW;
    while tokio::time::Instant::now() < sample_end {
        peak_inflight = peak_inflight.max(receiver.inflight_requests());
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The refused streams must be closed while the admitted ones are still
    // stalled: an admitted stream only dies at its 5 s header deadline.
    let mut closed_early = Vec::new();
    for (_send, mut recv) in streams {
        closed_early.push(
            tokio::time::timeout(REFUSAL_WINDOW, stream_closed_early(&mut recv))
                .await
                .is_ok(),
        );
    }
    let refused = closed_early.iter().filter(|closed| **closed).count();

    assert!(
        peak_inflight >= MAX_TUNNELED_PER_PEER,
        "the receiver admitted fewer than the per-peer cap (peak inflight {peak_inflight}); \
         the test needs live stalled handlers to observe the admission bound"
    );
    assert!(
        peak_inflight <= MAX_TUNNELED_PER_PEER,
        "the receiver ran {peak_inflight} concurrent tunneled request handlers; \
         the per-peer cap is {MAX_TUNNELED_PER_PEER}"
    );
    assert!(
        refused == STALLED_STREAMS - MAX_TUNNELED_PER_PEER as usize,
        "expected {} streams refused, saw {refused} closed within {REFUSAL_WINDOW:?}",
        STALLED_STREAMS - MAX_TUNNELED_PER_PEER as usize
    );
}
