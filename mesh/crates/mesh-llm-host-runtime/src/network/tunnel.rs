//! QUIC tunnel management for delivering remote OpenAI HTTP traffic and
//! forwarding stage transport streams.

use crate::mesh::Node;
use crate::protocol::read_len_prefixed;
use anyhow::Result;
use iroh::EndpointId;
use prost::Message;
use std::sync::Arc;

pub(crate) mod application_stage;
mod inbound_http;

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
        tokio::spawn(async move {
            while let Some((remote, send, recv)) = tunnel_http_rx.recv().await {
                let ingress = http_ingress_ref.read().ok().and_then(|guard| guard.clone());
                let node = http_node.clone();
                tokio::spawn(async move {
                    if let Err(e) =
                        inbound_http::handle_inbound_http_stream(node, remote, send, recv, ingress)
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
}

async fn handle_inbound_stage_transport(
    node: Node,
    remote: EndpointId,
    quic_send: iroh::endpoint::SendStream,
    mut quic_recv: iroh::endpoint::RecvStream,
) -> Result<()> {
    node.ensure_default_stage_transport_allowed()?;
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

    node.deliver_direct_stage(
        remote,
        &open,
        mesh_llm_transport::protocol::ProtocolStream::from_iroh(quic_send, quic_recv),
    )
    .await
}
