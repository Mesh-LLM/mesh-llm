//! Private routing: keep one request inside the entry node's trust circle.
//!
//! A client asks for it with `x-mesh-scope: private` or a `@private` model
//! suffix (see `network::openai::routing_scope`). The node the client talks to
//! decides the scope once, from its own verified owner certificate and its
//! trust store, and runs the rest of that request inside [`with_owner_scope`].
//!
//! The circle is this node's own owner plus every owner in its trust store's
//! `trusted_owners` (`mesh-llm auth trust add`, `--trust-owner`). With an
//! empty trust store it is just "nodes signed by my owner key". While the
//! scope is set:
//!
//! - [`Node::routable_peers`] and [`Node::models_being_served`] only see peers
//!   whose ownership certificate verifies against an owner in the circle, is
//!   bound to the peer's QUIC endpoint id, has not expired and is not revoked;
//! - [`Node::open_http_tunnel`] refuses any other peer, so a path that built
//!   its candidates some other way still cannot send the request out of scope.
//!
//! Peers never re-route a forwarded request (peer ingress executes locally),
//! so the check at the entry node is the whole boundary. Model names, node
//! labels and hostnames are never consulted: they are peer-supplied hints.

use crate::crypto::{OwnershipStatus, OwnershipSummary};
use crate::runtime::IngressType;
use iroh::EndpointId;
use std::collections::HashSet;

tokio::task_local! {
    static OWNER_SCOPE: OwnerScope;
}

/// The owners whose nodes one private request may reach: the entry node's
/// own owner and the owners it trusts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OwnerScope {
    owner_ids: Vec<String>,
}

impl OwnerScope {
    /// A scope of one owner's nodes.
    pub(crate) fn new(owner_id: impl Into<String>) -> Self {
        Self {
            owner_ids: vec![owner_id.into()],
        }
    }

    /// Exactly these owners' nodes; no owners admits no peer at all.
    fn of(owner_ids: impl IntoIterator<Item = String>) -> Self {
        let mut scope = Self {
            owner_ids: Vec::new(),
        };
        for owner in owner_ids {
            if !scope.owner_ids.contains(&owner) {
                scope.owner_ids.push(owner);
            }
        }
        scope
    }

    /// `owner_id`'s nodes and those of every owner in `trusted`.
    pub(crate) fn with_trusted<'a>(
        owner_id: impl Into<String>,
        trusted: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        Self::of(std::iter::once(owner_id.into()).chain(trusted.into_iter().map(str::to_string)))
    }

    /// Whether a peer with this verified ownership belongs to the scope.
    /// A revoked owner, certificate or node id is never `Verified`, so the
    /// trust store's revocations apply here too.
    pub(crate) fn admits(&self, summary: &OwnershipSummary, now_unix_ms: u64) -> bool {
        summary.status == OwnershipStatus::Verified
            && summary
                .owner_id
                .as_deref()
                .is_some_and(|owner| self.owner_ids.iter().any(|id| id == owner))
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

    pub(crate) async fn ensure_peer_in_current_owner_scope(
        &self,
        peer: EndpointId,
    ) -> anyhow::Result<()> {
        if self.peer_in_current_owner_scope(peer).await {
            Ok(())
        } else {
            anyhow::bail!(
                "peer {} is not in this private request's trust circle",
                peer.fmt_short()
            )
        }
    }

    /// Drop targets outside `scope` from an election snapshot: remote hosts
    /// not owned by the scope's owner, and a local split whose stages run on
    /// any such node (its activations and KV would leave the scope).
    pub(crate) async fn targets_in_owner_scope(
        &self,
        scope: &OwnerScope,
        targets: &crate::inference::election::ModelTargets,
    ) -> crate::inference::election::ModelTargets {
        use crate::inference::election::InferenceTarget;
        let in_scope = self.peers_in_scope(scope).await;
        let me = self.endpoint.id();
        let leaky_splits: HashSet<String> = {
            let topologies = self.stage_topologies.lock().await;
            topologies
                .topologies
                .values()
                .filter(|topology| {
                    topology
                        .stages
                        .iter()
                        .any(|stage| stage.node_id != me && !in_scope.contains(&stage.node_id))
                })
                .map(|topology| topology.model_id.clone())
                .collect()
        };
        let mut scoped = targets.clone();
        for (model, hosts) in &mut scoped.targets {
            let local_leaks = leaky_splits.contains(model);
            hosts.retain(|target| match target {
                InferenceTarget::Remote(peer) => in_scope.contains(peer),
                InferenceTarget::Local(_) => !local_leaks,
                InferenceTarget::None => true,
            });
        }
        scoped
    }

    async fn peers_in_scope(&self, scope: &OwnerScope) -> HashSet<EndpointId> {
        let now = now_unix_ms();
        let state = self.state.lock().await;
        state
            .peers
            .values()
            .filter(|peer| scope.admits(&peer.owner_summary, now))
            .map(|peer| peer.id)
            .collect()
    }

    /// Whether `peer` is verified as owned by the same owner as this node.
    /// Trusted owners do not count: this decides admission priority.
    pub(crate) async fn peer_shares_my_owner(&self, peer: EndpointId) -> bool {
        let Some(scope) = self.local_verified_owner_id().await.map(OwnerScope::new) else {
            return false;
        };
        self.peer_admitted_by(&scope, peer).await
    }

    async fn peer_admitted_by(&self, scope: &OwnerScope, peer: EndpointId) -> bool {
        let state = self.state.lock().await;
        state
            .peers
            .get(&peer)
            .is_some_and(|info| scope.admits(&info.owner_summary, now_unix_ms()))
    }

    /// The scope for a private request that `requester`, its entry node,
    /// forwarded here: this node's owner and the requester's, when verified.
    /// Those are the only owners known to be in the entry node's circle.
    /// This node's own trusted owners are not: this node trusting a third
    /// owner says nothing about whether the entry node does. The marker only
    /// ever narrows what this node does with the request, so it is honoured
    /// from any peer; with neither owner verified nothing outside this node
    /// is in scope.
    pub(crate) async fn forwarded_private_scope(&self, requester: EndpointId) -> OwnerScope {
        let now = now_unix_ms();
        let requester_owner = self
            .state
            .lock()
            .await
            .peers
            .get(&requester)
            .and_then(|peer| {
                let summary = &peer.owner_summary;
                (summary.status == OwnershipStatus::Verified
                    && summary
                        .expires_at_unix_ms
                        .is_some_and(|expires| expires > now))
                .then(|| summary.owner_id.clone())
                .flatten()
            });
        OwnerScope::of(
            self.local_verified_owner_id()
                .await
                .into_iter()
                .chain(requester_owner),
        )
    }

    /// The ingress class activity policy admits a request as. A request from
    /// one of this owner's own nodes is the owner's request, so a node that
    /// pauses remote inference while its owner is busy still serves it.
    pub(crate) async fn owner_admission_ingress(
        &self,
        ingress_type: IngressType,
        requester: Option<EndpointId>,
    ) -> IngressType {
        match requester {
            Some(peer)
                if ingress_type == IngressType::RemoteQuicHttp
                    && self.peer_shares_my_owner(peer).await =>
            {
                IngressType::LocalOpenAi
            }
            _ => ingress_type,
        }
    }

    /// The owner scope for a private request entering at this node: its
    /// trust circle, its own verified owner plus the trust store's trusted
    /// owners. `None` when this node is not signed by an owner key, in which
    /// case "private" has no meaning here and the caller must refuse.
    pub(crate) async fn private_owner_scope(&self) -> Option<OwnerScope> {
        let owner = self.local_verified_owner_id().await?;
        let trust_store = self.trust_store.lock().await;
        Some(OwnerScope::with_trusted(
            owner,
            trust_store
                .trusted_owners
                .iter()
                .map(|trusted| trusted.owner_id.as_str()),
        ))
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

    #[test]
    fn a_trust_circle_admits_trusted_owners_and_nobody_else() {
        let scope = OwnerScope::with_trusted("me", ["friend", "me"]);
        let now = 1_000;
        for owner in ["me", "friend"] {
            assert!(scope.admits(&summary(owner, OwnershipStatus::Verified, Some(2_000)), now));
        }
        assert!(!scope.admits(
            &summary("stranger", OwnershipStatus::Verified, Some(2_000)),
            now
        ));
        assert!(!scope.admits(
            &summary("friend", OwnershipStatus::RevokedOwner, Some(2_000)),
            now
        ));
        assert_eq!(scope, OwnerScope::with_trusted("me", ["friend"]));
        assert!(
            !OwnerScope::of([]).admits(&summary("me", OwnershipStatus::Verified, Some(2_000)), now)
        );
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
