//! Owned, bounded ingress for a full-model serving attachment.
//!
//! This task and every child are cancelled on detach. Capacity is claimed
//! before parsing HTTP bodies; request admission is a separate fair queue.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Semaphore, watch};
use tokio::task::{JoinHandle, JoinSet};

use crate::inference::election::ModelTargets;
use crate::mesh::{Node, TunnelChannels};
use crate::network::{affinity::AffinityRouter, openai::client_stream::ClientStream};

/// Serve only admitted HTTP tunnels through this membership's local target map.
///
/// Claim per-membership and aggregate ingress permits before reading request bodies.
/// The caller must abort the task on detach; its owned request set cancels child
/// futures when the task exits. RPC and stage receivers are deliberately closed.
pub(crate) fn start_serving_attachment(
    node: Node,
    channels: TunnelChannels,
    targets: watch::Receiver<ModelTargets>,
    global_slots: Arc<Semaphore>,
    membership_capacity: usize,
) -> JoinHandle<()> {
    let mut incoming = channels.http;
    drop((channels.rpc, channels.stage));
    let ingress = super::HttpIngress {
        targets,
        affinity: AffinityRouter::new(),
    };
    let member_slots = Arc::new(Semaphore::new(membership_capacity));
    tokio::spawn(async move {
        let mut requests = JoinSet::new();
        loop {
            tokio::select! {
                _ = requests.join_next(), if !requests.is_empty() => {},
                next = incoming.recv() => {
                    let Some((remote, send, recv)) = next else { break };
                    let permits = member_slots.clone().try_acquire_owned()
                        .and_then(|member| global_slots.clone().try_acquire_owned().map(|global| (member, global)));
                    let Ok(permits) = permits else {
                        let stream = ClientStream::from_quic_with_prefix(recv, send, Vec::new());
                        let _ = tokio::time::timeout(Duration::from_secs(1),
                            crate::network::openai::send_503(stream, "shared ingress capacity exhausted; retry later")).await;
                        continue;
                    };
                    let node = node.clone();
                    let ingress = ingress.clone();
                    requests.spawn(async move {
                        let _permits = permits;
                        if let Err(error) = super::inbound_http::handle_inbound_http_stream(
                            node, remote, send, recv, Some(ingress),
                        ).await {
                            tracing::debug!(%error, "serving attachment request ended");
                        }
                    });
                }
            }
        }
        // JoinSet aborts in-flight request futures when this owner is dropped.
    })
}
