//! Invoice exchange over authenticated peer tunnels. Wallet authorization is
//! owned by local ingress; remote forwarding cannot spend the relay's wallet.

pub(crate) mod client;
mod delivery;
mod gate;
pub(crate) mod node_ext;
pub(crate) mod request;
mod server;
pub(crate) mod wallet_plugin;

pub(crate) use node_ext::{PaymentsSlot, in_process_plugins};
pub(crate) use server::serve;

pub(crate) fn is_payment_upgrade(prefix: &[u8]) -> bool {
    prefix.starts_with(b"POST /mesh/payment/v1 HTTP/1.1\r\n")
}

/// What the tunnel should do after payments has looked at the request head.
pub(crate) enum Inbound {
    /// Payments consumed the stream (payment upgrade served, or refused).
    Handled,
    /// Not a payments concern: continue with ordinary ingress.
    Continue(
        Vec<u8>,
        iroh::endpoint::RecvStream,
        iroh::endpoint::SendStream,
    ),
}

/// Tunnel ingress hook: serve payment upgrades and refuse legacy bridge
/// callers that would bypass seller payment enforcement.
pub(crate) async fn intercept_inbound(
    node: &crate::mesh::Node,
    remote: iroh::EndpointId,
    ingress_targets: Option<
        &tokio::sync::watch::Receiver<crate::inference::election::ModelTargets>,
    >,
    prefix: Vec<u8>,
    quic_recv: iroh::endpoint::RecvStream,
    quic_send: iroh::endpoint::SendStream,
) -> anyhow::Result<Inbound> {
    use anyhow::Context;
    use tokio::io::AsyncReadExt;

    if is_payment_upgrade(&prefix) {
        let (offset, _) = crate::network::openai::request_parse::http_header_terminator(&prefix)
            .context("incomplete payment upgrade")?;
        let remainder = std::io::Cursor::new(prefix[offset..].to_vec());
        let targets = ingress_targets
            .context("payment ingress unavailable")?
            .borrow()
            .clone();
        serve(
            node.clone(),
            remote,
            remainder.chain(quic_recv),
            quic_send,
            targets,
        )
        .await?;
        return Ok(Inbound::Handled);
    }
    // Legacy bridge callers cannot bypass seller payment enforcement.
    if ingress_targets.is_none() && legacy_bridge_requires_payment_ingress(node).await? {
        let stream = crate::network::openai::client_stream::ClientStream::from_quic_with_prefix(
            quic_recv, quic_send, prefix,
        );
        crate::network::openai::send_error(stream, 402, "payment-capable peer required").await?;
        return Ok(Inbound::Handled);
    }
    Ok(Inbound::Continue(prefix, quic_recv, quic_send))
}

async fn legacy_bridge_requires_payment_ingress(node: &crate::mesh::Node) -> anyhow::Result<bool> {
    if !node.advertised_payment_offers().await?.is_empty() {
        return Ok(true);
    }
    // A loopback TCP bridge loses remote provenance. A wallet-enabled node
    // must use direct QUIC ingress, where spending authority remains remote.
    let directory = node.config_state.lock().await.payment_directory();
    if mesh_llm_payments::provisioning::has_persisted_wallet(&directory) {
        return Ok(true);
    }
    Ok(node
        .payments
        .get()
        .is_some_and(|service| service.has_wallet()))
}

#[cfg(test)]
mod tests;
