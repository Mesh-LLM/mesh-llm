//! The receipt audit (`network/receipt_audit`) of one attempt on a peer: the
//! request the peer is sent names this node's receipt key, and the response
//! the client is written is collected for the audit.

use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::receipt_audit;
use crate::network::request_trust::RequestTrust;

/// The watch on one attempt; empty when this node does not audit the turn.
pub(super) struct ReceiptWatch(Option<receipt_audit::Watch>);

impl ReceiptWatch {
    /// Watches the turn `request` asks `host_id` for, when this node audits
    /// it. A local client's chat completion names this node's receipt key
    /// when this node audits its peers' receipts. Whether it does was
    /// decided with the request's trust policy, so a request that is not
    /// audited pays for no second read of the settings or the head here.
    pub(super) async fn begin(
        node: &mesh::Node,
        trust: RequestTrust,
        client: &ClientStream,
        host_id: iroh::EndpointId,
        request: &[u8],
    ) -> Self {
        Self(match trust.audits_peers() && !client.is_mesh_peer() {
            true => receipt_audit::watch(node, host_id, request).await,
            false => None,
        })
    }

    /// The request to forward: `request`, or the one that names this node's
    /// receipt key when the turn is watched.
    pub(super) fn request<'a>(&'a self, request: &'a [u8]) -> &'a [u8] {
        self.0
            .as_ref()
            .map_or(request, |watch| watch.request.as_slice())
    }

    /// Collects what `client` is written from here on for the audit.
    pub(super) fn observe(&self, client: &mut ClientStream) {
        if let Some(watch) = &self.0 {
            client.observe(watch.observer.clone());
        }
    }

    /// Stops collecting: the attempt is over.
    pub(super) fn stop_observing(&self, client: &mut ClientStream) {
        if let Some(watch) = &self.0 {
            client.stop_observing(&watch.observer);
        }
    }
}
