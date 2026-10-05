//! The paid serving path's exchange events.
//!
//! The seller's paid serving path forwards the backend's raw bytes to the
//! payer rather than running the relay, so it publishes the free path's
//! `openai.exchange.v1` effective and terminal events itself, through the same
//! terminal builder (`ingress::publish_raw_proxy_terminal`).

use crate::mesh;
use crate::network::openai::ingress::{RawProxyTerminalFacts, publish_raw_proxy_terminal};
use crate::network::openai::transport as proxy;
use crate::plugin::openai_exchange::{
    OpenAiExchangeChannel, OpenAiExchangeDispatchPath, OpenAiExchangeEnvelope,
};

/// The paid serving path's exchange events, the same ones the free
/// host-served path publishes: an effective event at admission and a terminal
/// event after delivery, under one host-minted exchange id, with this node's
/// serving provenance (it served on its own weights), the real request digest
/// and the served response's usage and digests. Without it, a plugin that
/// subscribes to exchange events sees free exchanges but never a paid one.
pub(crate) struct PaidServedExchange {
    channel: std::sync::Arc<dyn OpenAiExchangeChannel>,
    exchange_id: String,
}

impl PaidServedExchange {
    /// Whether a plugin subscribes to this node's exchange events.
    pub(crate) async fn is_subscribed(node: &mesh::Node) -> bool {
        match paid_exchange_channel(node).await {
            Some(channel) => channel.has_subscriber().await,
            None => false,
        }
    }

    /// Publish the effective event under `exchange_id`, the id the serving
    /// path minted for this request; `None` (nothing published) without an
    /// id or when no plugin subscribes to exchange events, as on the free path.
    pub(crate) async fn begin(
        node: &mesh::Node,
        model_name: &str,
        exchange_id: Option<&str>,
    ) -> Option<Self> {
        let exchange_id = exchange_id?.to_owned();
        let channel = paid_exchange_channel(node).await?;
        if !channel.has_subscriber().await {
            return None;
        }
        channel
            .publish(&OpenAiExchangeEnvelope::effective(
                exchange_id.clone(),
                OpenAiExchangeDispatchPath::RawProxy,
                model_name,
            ))
            .await;
        Some(Self {
            channel,
            exchange_id,
        })
    }

    /// `payer` is the node that asked, the payments stream's QUIC-authenticated
    /// remote id (never taken from the request), named on the terminal as
    /// `requested_by_node_id` the way the free path names the tunnel's.
    pub(crate) async fn finish(
        &self,
        node: &mesh::Node,
        model_name: &str,
        outcome: &proxy::RouteDispatchOutcome,
        request_digest: Option<&str>,
        payer: iroh::EndpointId,
    ) {
        let requested_by_node_id = hex::encode(payer.as_bytes());
        // Served here, on this node's own weights.
        publish_raw_proxy_terminal(
            node,
            self.channel.as_ref(),
            &self.exchange_id,
            model_name,
            outcome,
            RawProxyTerminalFacts {
                served_locally: true,
                request_digest,
                requested_by_node_id: Some(&requested_by_node_id),
            },
        )
        .await;
    }
}

/// The node's plugin manager, which broadcasts to subscribing plugins.
#[cfg(not(test))]
async fn paid_exchange_channel(
    node: &mesh::Node,
) -> Option<std::sync::Arc<dyn OpenAiExchangeChannel>> {
    let manager = node.plugin_manager().await?;
    Some(std::sync::Arc::new(manager))
}

/// Under test, a channel registered for this node (see
/// [`record_paid_exchanges_for_test`]); else the plugin manager.
#[cfg(test)]
async fn paid_exchange_channel(
    node: &mesh::Node,
) -> Option<std::sync::Arc<dyn OpenAiExchangeChannel>> {
    let registered = PAID_EXCHANGE_TEST_CHANNELS
        .lock()
        .unwrap()
        .get(&node.id())
        .cloned();
    if registered.is_some() {
        return registered;
    }
    let manager = node.plugin_manager().await?;
    Some(std::sync::Arc::new(manager))
}

#[cfg(test)]
static PAID_EXCHANGE_TEST_CHANNELS: std::sync::Mutex<
    std::collections::BTreeMap<iroh::EndpointId, std::sync::Arc<dyn OpenAiExchangeChannel>>,
> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Capture the paid exchange events `node` publishes, for a test.
#[cfg(test)]
pub(crate) fn record_paid_exchanges_for_test(
    node: &mesh::Node,
) -> std::sync::Arc<crate::plugin::openai_exchange::test_support::RecordingChannel> {
    let channel = std::sync::Arc::new(
        crate::plugin::openai_exchange::test_support::RecordingChannel::default(),
    );
    PAID_EXCHANGE_TEST_CHANNELS
        .lock()
        .unwrap()
        .insert(node.id(), channel.clone());
    channel
}
