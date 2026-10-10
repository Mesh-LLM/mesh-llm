//! QUIC tunnel management for delivering remote OpenAI HTTP traffic and
//! forwarding stage transport streams.

pub(crate) use mesh_llm_transport::stage_link_delay;

use crate::mesh::Node;
use crate::protocol::read_len_prefixed;
use anyhow::{Context, Result};
use iroh::EndpointId;
use prost::Message;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;
use tokio::net::TcpStream;

mod inbound_http;
mod remote_origin;
pub(crate) use remote_origin::RemoteBridge;
#[cfg(feature = "payments")]
pub(crate) use remote_origin::is_remote_bridge;

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
    http_port: Arc<AtomicU16>,
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
            http_port: Arc::new(AtomicU16::new(0)),
            http_ingress: Arc::new(std::sync::RwLock::new(None)),
        };

        // Handle inbound HTTP tunnel streams in-process when the runtime has
        // installed direct ingress; retain the legacy port fallback for embedders.
        let http_port_ref = mgr.http_port.clone();
        let http_ingress_ref = mgr.http_ingress.clone();
        let http_node = mgr.node.clone();
        tokio::spawn(async move {
            while let Some((remote, send, recv)) = tunnel_http_rx.recv().await {
                let ingress = http_ingress_ref.read().ok().and_then(|guard| guard.clone());
                let port = http_port_ref.load(Ordering::Relaxed);
                if ingress.is_none() && port == 0 {
                    tracing::warn!("Inbound HTTP tunnel but no OpenAI surface running, dropping");
                    continue;
                }
                let node = http_node.clone();
                tokio::spawn(async move {
                    if let Err(e) = inbound_http::handle_inbound_http_stream(
                        node, remote, send, recv, port, ingress,
                    )
                    .await
                    {
                        tracing::warn!("Inbound HTTP tunnel stream error: {e}");
                    }
                });
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

    /// Update the compatibility API proxy port used until direct ingress is installed.
    pub fn set_http_port(&self, port: u16) {
        self.http_port.store(port, Ordering::Relaxed);
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
