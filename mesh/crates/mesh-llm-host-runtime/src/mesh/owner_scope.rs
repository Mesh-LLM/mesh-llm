//! Private routing: keep one request on nodes signed by this node's owner.
//!
//! A client asks for it with `x-mesh-scope: private` or a `@private` model
//! suffix (see `network::openai::routing_scope`). The node the client talks to
//! decides the scope once, from its own verified owner certificate, and runs
//! the rest of that request inside [`with_owner_scope`]. While the scope is
//! set:
//!
//! - [`Node::routable_peers`] and [`Node::models_being_served`] only see peers
//!   whose ownership certificate verifies against the same owner id, is bound
//!   to the peer's QUIC endpoint id, and has not expired;
//! - [`Node::open_http_tunnel`] refuses any other peer, so a path that built
//!   its candidates some other way still cannot send the request out of scope.
//!
//! Peers never re-route a forwarded request (peer ingress executes locally),
//! so the check at the entry node is the whole boundary. Model names, node
//! labels and hostnames are never consulted: they are peer-supplied hints.

use crate::crypto::{OwnershipStatus, OwnershipSummary};
use iroh::EndpointId;

tokio::task_local! {
    static OWNER_SCOPE: OwnerScope;
}

/// The owner whose nodes one private request may reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OwnerScope {
    owner_id: String,
}

impl OwnerScope {
    pub(crate) fn new(owner_id: impl Into<String>) -> Self {
        Self {
            owner_id: owner_id.into(),
        }
    }

    /// Whether a peer with this verified ownership belongs to the scope.
    pub(crate) fn admits(&self, summary: &OwnershipSummary, now_unix_ms: u64) -> bool {
        summary.status == OwnershipStatus::Verified
            && summary.owner_id.as_deref() == Some(self.owner_id.as_str())
            && summary
                .expires_at_unix_ms
                .is_some_and(|expires| expires > now_unix_ms)
    }
}

/// Run `future` with `scope` as the current owner scope; `None` runs it
/// unscoped.
pub(crate) async fn with_owner_scope<F: std::future::Future>(
    scope: Option<OwnerScope>,
    future: F,
) -> F::Output {
    match scope {
        Some(scope) => OWNER_SCOPE.scope(scope, future).await,
        None => future.await,
    }
}

/// The owner scope of the request this task is serving, if it is private.
pub(crate) fn current_owner_scope() -> Option<OwnerScope> {
    OWNER_SCOPE.try_with(Clone::clone).ok()
}

pub(crate) fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

impl super::Node {
    /// Whether `peer` may receive the current request. Always true outside a
    /// private request; inside one, only for a known peer owned by the scope's
    /// owner.
    pub(crate) async fn peer_in_current_owner_scope(&self, peer: EndpointId) -> bool {
        let Some(scope) = current_owner_scope() else {
            return true;
        };
        let state = self.state.lock().await;
        state
            .peers
            .get(&peer)
            .is_some_and(|info| scope.admits(&info.owner_summary, now_unix_ms()))
    }

    /// Drop remote targets outside `scope` from an election snapshot. Local
    /// targets stay: a private request only enters at a node its owner signed.
    pub(crate) async fn targets_in_owner_scope(
        &self,
        scope: &OwnerScope,
        targets: &crate::inference::election::ModelTargets,
    ) -> crate::inference::election::ModelTargets {
        use crate::inference::election::InferenceTarget;
        let now = now_unix_ms();
        let in_scope: std::collections::HashSet<EndpointId> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|peer| scope.admits(&peer.owner_summary, now))
                .map(|peer| peer.id)
                .collect()
        };
        let mut scoped = targets.clone();
        for hosts in scoped.targets.values_mut() {
            hosts.retain(|target| {
                !matches!(target, InferenceTarget::Remote(peer) if !in_scope.contains(peer))
            });
        }
        scoped
    }

    /// The owner scope for a private request entering at this node: its own
    /// verified owner. `None` when this node is not signed by an owner key,
    /// in which case "private" has no meaning here and the caller must refuse.
    pub(crate) async fn private_owner_scope(&self) -> Option<OwnerScope> {
        self.local_verified_owner_id().await.map(OwnerScope::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(owner: &str, status: OwnershipStatus, expires: Option<u64>) -> OwnershipSummary {
        OwnershipSummary {
            owner_id: Some(owner.to_string()),
            verified: status == OwnershipStatus::Verified,
            status,
            expires_at_unix_ms: expires,
            ..OwnershipSummary::default()
        }
    }

    #[test]
    fn only_a_live_verified_cert_from_the_same_owner_is_in_scope() {
        let scope = OwnerScope::new("me");
        let now = 1_000;
        assert!(scope.admits(&summary("me", OwnershipStatus::Verified, Some(2_000)), now));
        assert!(!scope.admits(
            &summary("them", OwnershipStatus::Verified, Some(2_000)),
            now
        ));
        assert!(!scope.admits(&summary("me", OwnershipStatus::Verified, Some(1_000)), now));
        assert!(!scope.admits(&summary("me", OwnershipStatus::Verified, None), now));
        for status in [
            OwnershipStatus::Unsigned,
            OwnershipStatus::Expired,
            OwnershipStatus::InvalidSignature,
            OwnershipStatus::MismatchedNodeId,
            OwnershipStatus::RevokedOwner,
            OwnershipStatus::RevokedCert,
            OwnershipStatus::RevokedNodeId,
            OwnershipStatus::UntrustedOwner,
        ] {
            assert!(
                !scope.admits(&summary("me", status.clone(), Some(2_000)), now),
                "{status:?} must not be in scope"
            );
        }
        assert!(!scope.admits(&OwnershipSummary::default(), now));
    }

    #[tokio::test]
    async fn the_scope_is_visible_only_inside_the_scoped_future() {
        assert_eq!(current_owner_scope(), None);
        let seen =
            with_owner_scope(Some(OwnerScope::new("me")), async { current_owner_scope() }).await;
        assert_eq!(seen, Some(OwnerScope::new("me")));
        assert_eq!(current_owner_scope(), None);
        assert_eq!(
            with_owner_scope(None, async { current_owner_scope() }).await,
            None
        );
    }
}
