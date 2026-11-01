//! QUIC tunnel management for delivering remote OpenAI HTTP traffic and
//! forwarding stage transport streams.

pub(crate) use mesh_llm_transport::stage_link_delay;

use crate::mesh::Node;
use crate::protocol::read_len_prefixed;
use anyhow::{Context, Result};
use iroh::EndpointId;
use prost::Message;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;

mod admission;
mod inbound_http;
mod serving_attachment;
pub(crate) use serving_attachment::start_serving_attachment;

#[cfg(test)]
#[path = "tunnel/admission_tests.rs"]
mod admission_tests;

fn quic_response_first_byte_timeout() -> Duration {
    Duration::from_secs(5 * 60)
}

#[derive(Clone)]
pub(super) struct HttpIngress {
    targets: tokio::sync::watch::Receiver<crate::inference::election::ModelTargets>,
    affinity: crate::network::affinity::AffinityRouter,
}

/// Manages all tunnels for a node
#[derive(Clone)]
pub struct Manager {
    node: Node,
    http_ingress: Arc<std::sync::RwLock<Option<HttpIngress>>>,
}

impl Manager {
    /// Start the tunnel manager.
    /// The API proxy port for inbound HTTP tunnels is set by the runtime once
    /// the node begins serving.
    pub async fn start(
        node: Node,
        _legacy_tunnel_rx: tokio::sync::mpsc::Receiver<(
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
        mut tunnel_http_rx: tokio::sync::mpsc::Receiver<(
            EndpointId,
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
        mut stage_transport_rx: tokio::sync::mpsc::Receiver<(
            EndpointId,
            iroh::endpoint::SendStream,
            iroh::endpoint::RecvStream,
        )>,
    ) -> Result<Self> {
        let mgr = Manager {
            node: node.clone(),
            http_ingress: Arc::new(std::sync::RwLock::new(None)),
        };

        // Handle inbound HTTP tunnel streams in-process when the runtime has
        // installed direct ingress; otherwise reject the request.
        let http_ingress_ref = mgr.http_ingress.clone();
        let http_node = mgr.node.clone();
        let admission = admission::TunnelHttpAdmission::new();
        tokio::spawn(async move {
            while let Some((remote, mut send, mut recv)) = tunnel_http_rx.recv().await {
                match admission.claim(remote) {
                    admission::TunnelHttpClaim::Full => {
                        // Above the global or per-peer quota: close the stream
                        // pair fast and spawn no request work.
                        tracing::warn!(
                            peer = %remote.fmt_short(),
                            "Refusing tunneled HTTP request stream: the per-peer or global quota is exhausted"
                        );
                        let _ = recv.stop(0u32.into());
                        let _ = send.reset(0u32.into());
                    }
                    admission::TunnelHttpClaim::Granted(permit) => {
                        let ingress = http_ingress_ref.read().ok().and_then(|guard| guard.clone());
                        let node = http_node.clone();
                        tokio::spawn(async move {
                            let _permit = permit;
                            if let Err(e) = inbound_http::handle_inbound_http_stream(
                                node, remote, send, recv, ingress,
                            )
                            .await
                            {
                                tracing::warn!("Inbound HTTP tunnel stream error: {e}");
                            }
                        });
                    }
                }
            }
        });

        let stage_node = mgr.node.clone();
        tokio::spawn(async move {
            while let Some((remote, send, recv)) = stage_transport_rx.recv().await {
                let node = stage_node.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_stage_transport(node, remote, send, recv).await {
                        tracing::warn!(
                            "Inbound stage transport stream error from {}: {e}",
                            remote.fmt_short()
                        );
                    }
                });
            }
        });

        Ok(mgr)
    }

    /// Install the in-process model-aware ingress used by remote HTTP tunnels.
    pub fn set_http_ingress(
        &self,
        targets: tokio::sync::watch::Receiver<crate::inference::election::ModelTargets>,
        affinity: crate::network::affinity::AffinityRouter,
    ) {
        if let Ok(mut ingress) = self.http_ingress.write() {
            *ingress = Some(HttpIngress { targets, affinity });
        }
        tracing::info!("Tunnel manager: direct HTTP ingress enabled");
    }
}

/// Most peer requests one membership serves at once. Further streams wait in
/// the channel, which pushes back on the connection.
const MEMBERSHIP_INGRESS_STREAMS: usize = 256;

/// Serves a mesh membership's inbound HTTP tunnel streams through the same
/// in-process ingress as the node it belongs to, until the channel closes or
/// the task is aborted. Aborting it also cancels every stream it is serving,
/// so none outlives the membership. A membership carries no stage transport,
/// so it has no [`Manager`].
pub(crate) async fn serve_membership_http_ingress(
    node: Node,
    mut tunnel_http_rx: tokio::sync::mpsc::Receiver<(
        EndpointId,
        iroh::endpoint::SendStream,
        iroh::endpoint::RecvStream,
    )>,
    targets: tokio::sync::watch::Receiver<crate::inference::election::ModelTargets>,
    affinity: crate::network::affinity::AffinityRouter,
) {
    let ingress = HttpIngress { targets, affinity };
    let slots = Arc::new(tokio::sync::Semaphore::new(MEMBERSHIP_INGRESS_STREAMS));
    let mut streams = tokio::task::JoinSet::new();
    loop {
        let Ok(slot) = slots.clone().acquire_owned().await else {
            return;
        };
        let Some((remote, send, recv)) = tunnel_http_rx.recv().await else {
            return;
        };
        let node = node.clone();
        let ingress = ingress.clone();
        streams.spawn(async move {
            let _slot = slot;
            if let Err(e) =
                inbound_http::handle_inbound_http_stream(node, remote, send, recv, Some(ingress))
                    .await
            {
                tracing::warn!("Inbound membership HTTP tunnel stream error: {e}");
            }
        });
        while streams.try_join_next().is_some() {}
    }
}

async fn handle_inbound_stage_transport(
    node: Node,
    remote: EndpointId,
    quic_send: iroh::endpoint::SendStream,
    mut quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    let buf = read_len_prefixed(&mut quic_recv).await?;
    let open = skippy_protocol::proto::stage::StageTransportOpen::decode(buf.as_slice())
        .map_err(|e| anyhow::anyhow!("StageTransportOpen decode error: {e}"))?;
    skippy_protocol::validate_stage_transport_open(&open)
        .map_err(|e| anyhow::anyhow!("StageTransportOpen validation error: {e}"))?;
    if open.requester_id.as_slice() != remote.as_bytes() {
        anyhow::bail!("stage transport requester_id does not match QUIC peer identity");
    }
    if !node.stage_transport_allowed(remote, &open).await {
        anyhow::bail!(
            "stage transport requester is not part of topology {} / {}",
            open.topology_id,
            open.run_id
        );
    }

    let bind_addr = resolve_stage_transport_bind_addr(&node, &open).await?;
    let tcp_stream = TcpStream::connect(&bind_addr).await?;
    tcp_stream.set_nodelay(true)?;
    tracing::info!(
        "Inbound stage transport stream {} → {}",
        remote.fmt_short(),
        bind_addr
    );
    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv, None).await
}

async fn resolve_stage_transport_bind_addr(
    node: &Node,
    open: &skippy_protocol::proto::stage::StageTransportOpen,
) -> Result<String> {
    let status_result = node
        .query_local_stage_status(crate::inference::skippy::StageStatusFilter {
            topology_id: Some(open.topology_id.clone()),
            run_id: Some(open.run_id.clone()),
            stage_id: Some(open.stage_id.clone()),
        })
        .await;
    match status_result {
        Ok(statuses) => {
            if let Some(status) = statuses.into_iter().find(|status| {
                status.topology_id == open.topology_id
                    && status.run_id == open.run_id
                    && status.stage_id == open.stage_id
            }) {
                if status.state != crate::inference::skippy::StageRuntimeState::Ready {
                    anyhow::bail!(
                        "stage {} / {} / {} is not ready: {:?}",
                        status.topology_id,
                        status.run_id,
                        status.stage_id,
                        status.state
                    );
                }
                return Ok(status.bind_addr);
            }
        }
        Err(error) => {
            if let Some(bind_addr) = node
                .stage_transport_alias(&open.topology_id, &open.run_id, &open.stage_id)
                .await
            {
                return Ok(bind_addr);
            }
            return Err(error).with_context(|| {
                format!(
                    "query local stage status for {} / {} / {}",
                    open.topology_id, open.run_id, open.stage_id
                )
            });
        }
    }
    if let Some(bind_addr) = node
        .stage_transport_alias(&open.topology_id, &open.run_id, &open.stage_id)
        .await
    {
        return Ok(bind_addr);
    }
    anyhow::bail!(
        "stage {} / {} / {} is not loaded locally",
        open.topology_id,
        open.run_id,
        open.stage_id
    )
}

/// Relay admitted streams using Mesh's existing first-response timeout policy.
pub async fn relay_bidirectional(
    tcp_read: tokio::io::ReadHalf<TcpStream>,
    tcp_write: tokio::io::WriteHalf<TcpStream>,
    quic_send: iroh::endpoint::SendStream,
    quic_recv: iroh::endpoint::RecvStream,
    link_delay: Option<Duration>,
) -> Result<()> {
    mesh_llm_transport::relay_bidirectional(
        tcp_read,
        tcp_write,
        quic_send,
        quic_recv,
        quic_response_first_byte_timeout(),
        link_delay,
    )
    .await
}
