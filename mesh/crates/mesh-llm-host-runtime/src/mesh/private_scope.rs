//! Private routing: keep one request on nodes the entry node lists.
//!
//! A client asks for it with `x-mesh-scope: private` or a `@private` model
//! suffix (see `network::openai::routing_scope`). The node the client talks to
//! decides the scope once and runs the rest of that request inside
//! [`with_private_scope`].
//!
//! The scope is this node plus the node ids in its `[private] nodes` config
//! list (see [`super::private_nodes`]). With none, a private request never
//! leaves this node.
//! While the scope is set:
//!
//! - [`Node::routable_peers`] and [`Node::models_being_served`] only see listed
//!   peers that advertise private routing;
//! - [`Node::open_http_tunnel`] refuses any other peer, so a path that built
//!   its candidates some other way still cannot send the request out of scope.
//!
//! A peer's node id is its QUIC endpoint id, which the connection proves.
//! Model names, node labels and hostnames are never consulted.

use crate::runtime::IngressType;
use iroh::EndpointId;
use std::collections::HashSet;

tokio::task_local! {
    static CURRENT_PRIVATE_SCOPE: PrivateScope;
}

/// The nodes one private request may reach, by hex node id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrivateScope {
    node_ids: Vec<String>,
}

/// The form a node id takes in a scope: lowercase hex, as
/// `mesh-llm auth status` prints it.
pub(crate) fn node_key(id: &EndpointId) -> String {
    hex::encode(id.as_bytes())
}

impl PrivateScope {
    /// A scope of one node.
    pub(crate) fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_ids: vec![node_id.into()],
        }
    }

    /// Exactly these nodes; none admits no node at all.
    pub(crate) fn of(node_ids: impl IntoIterator<Item = String>) -> Self {
        let mut seen = HashSet::new();
        Self {
            node_ids: node_ids
                .into_iter()
                .map(|id| id.to_ascii_lowercase())
                .filter(|id| seen.insert(id.clone()))
                .collect(),
        }
    }

    pub(crate) fn node_ids(&self) -> &[String] {
        &self.node_ids
    }

    /// The nodes in both scopes.
    pub(crate) fn intersect(&self, other: &PrivateScope) -> PrivateScope {
        let other: HashSet<&String> = other.node_ids.iter().collect();
        Self {
            node_ids: self
                .node_ids
                .iter()
                .filter(|id| other.contains(id))
                .cloned()
                .collect(),
        }
    }

    pub(crate) fn contains(&self, id: &EndpointId) -> bool {
        let key = node_key(id);
        self.node_ids.contains(&key)
    }

    /// Whether a private request may go to `peer`: it is listed and
    /// advertises private routing. A node too old to know the private
    /// marker would serve the request as public (a split, a consultation),
    /// so it gets none, listed or not.
    pub(crate) fn admits_peer(&self, peer: &super::PeerInfo) -> bool {
        peer.private_routing_supported && self.contains(&peer.id)
    }
}

/// Run `future` with `scope` as the current private scope; `None` runs it
/// unscoped.
pub(crate) async fn with_private_scope<F: std::future::Future>(
    scope: Option<PrivateScope>,
    future: F,
) -> F::Output {
    match scope {
        Some(scope) => CURRENT_PRIVATE_SCOPE.scope(scope, future).await,
        None => future.await,
    }
}

/// The private scope of the request this task is serving, if it is private.
pub(crate) fn current_private_scope() -> Option<PrivateScope> {
    CURRENT_PRIVATE_SCOPE.try_with(Clone::clone).ok()
}

impl super::Node {
    /// Whether `peer` may receive the current request. Always true outside a
    /// private request; inside one, only for a known peer the scope admits.
    pub(crate) async fn peer_in_current_private_scope(&self, peer: EndpointId) -> bool {
        let Some(scope) = current_private_scope() else {
            return true;
        };
        let state = self.state.lock().await;
        state
            .peers
            .get(&peer)
            .is_some_and(|info| scope.admits_peer(info))
    }

    pub(crate) async fn ensure_peer_in_current_private_scope(
        &self,
        peer: EndpointId,
    ) -> anyhow::Result<()> {
        if self.peer_in_current_private_scope(peer).await {
            Ok(())
        } else {
            anyhow::bail!(
                "peer {} is not one of this private request's nodes",
                peer.fmt_short()
            )
        }
    }

    /// Drop targets outside `scope` from an election snapshot: remote hosts
    /// it does not admit, and a local split whose stages run on any such
    /// node (its activations and KV would leave the scope).
    pub(crate) async fn targets_in_private_scope(
        &self,
        scope: &PrivateScope,
        targets: &crate::inference::election::ModelTargets,
    ) -> crate::inference::election::ModelTargets {
        use crate::inference::election::InferenceTarget;
        let in_scope = self.peers_in_scope(scope).await;
        let me = self.endpoint.id();
        // A narrowed scope (a plugin's nested call) may leave out this node.
        let me_in_scope = scope.contains(&me);
        // Splits with a stage out of scope, by model and the node that runs
        // stage 0 (the one a target names). A split this node only takes part
        // in, coordinated elsewhere, leaves this node's own target alone. A
        // known split behind a listed peer's target drops that target too,
        // which guards against a listed peer too old to honour the marker.
        let leaky_splits: HashSet<(String, EndpointId)> = {
            let topologies = self.stage_topologies.lock().await;
            topologies
                .visible_topologies()
                .into_iter()
                .filter(|topology| {
                    topology.stages.iter().any(|stage| {
                        if stage.node_id == me {
                            !me_in_scope
                        } else {
                            !in_scope.contains(&stage.node_id)
                        }
                    })
                })
                .filter_map(|topology| {
                    let coordinator = topology
                        .stages
                        .iter()
                        .find(|stage| stage.stage_index == 0)?
                        .node_id;
                    Some((topology.model_id, coordinator))
                })
                .collect()
        };
        let leaks =
            |model: &str, node: EndpointId| leaky_splits.contains(&(model.to_string(), node));
        let mut scoped = targets.clone();
        for (model, hosts) in &mut scoped.targets {
            hosts.retain(|target| match target {
                InferenceTarget::Remote(peer) => in_scope.contains(peer) && !leaks(model, *peer),
                InferenceTarget::Local(_) => me_in_scope && !leaks(model, me),
                InferenceTarget::None => true,
            });
        }
        scoped
    }

    async fn peers_in_scope(&self, scope: &PrivateScope) -> HashSet<EndpointId> {
        let state = self.state.lock().await;
        state
            .peers
            .values()
            .filter(|peer| scope.admits_peer(peer))
            .map(|peer| peer.id)
            .collect()
    }

    /// Whether this node itself is in `scope`.
    pub(crate) fn local_in_private_scope(&self, scope: &PrivateScope) -> bool {
        scope.contains(&self.endpoint.id())
    }

    /// [`Self::targets_in_private_scope`] for the current request, if it is
    /// private: for targets a route builds after the ingress snapshot.
    pub(crate) async fn targets_in_current_private_scope(
        &self,
        targets: crate::inference::election::ModelTargets,
    ) -> crate::inference::election::ModelTargets {
        match current_private_scope() {
            Some(scope) => self.targets_in_private_scope(&scope, &targets).await,
            None => targets,
        }
    }

    /// This node's id, as a scope names it.
    pub(crate) fn local_scope_node_id(&self) -> String {
        node_key(&self.endpoint.id())
    }

    /// Whether `peer` is on this node's private node list.
    pub(crate) fn is_private_node(&self, peer: EndpointId) -> bool {
        self.private_nodes.contains(&peer)
    }

    /// The scope for a private request that `requester`, its entry node,
    /// forwarded here: this node and the requester. Those are the only nodes
    /// known to be in the entry node's scope; this node's own list says
    /// nothing about the entry node's. The marker only ever narrows what this
    /// node does with the request, so it is honoured from any peer.
    pub(crate) fn forwarded_private_scope(&self, requester: EndpointId) -> PrivateScope {
        PrivateScope::of([self.local_scope_node_id(), node_key(&requester)])
    }

    /// The ingress class activity policy admits a request as. A request from
    /// one of this node's private nodes is the owner's request, so a node
    /// that pauses remote inference while its owner is busy still serves it.
    pub(crate) fn private_node_admission_ingress(
        &self,
        ingress_type: IngressType,
        requester: Option<EndpointId>,
    ) -> IngressType {
        match requester {
            Some(peer)
                if ingress_type == IngressType::RemoteQuicHttp && self.is_private_node(peer) =>
            {
                IngressType::LocalOpenAi
            }
            _ => ingress_type,
        }
    }

    /// The scope for a private request entering at this node: this node and
    /// its private node list.
    pub(crate) fn entry_private_scope(&self) -> PrivateScope {
        PrivateScope::of(
            std::iter::once(self.local_scope_node_id())
                .chain(self.private_nodes.current().iter().map(node_key)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scope_admits_exactly_its_listed_nodes() {
        let scope = PrivateScope::of(["AA".to_string(), "bb".to_string(), "aa".to_string()]);
        assert_eq!(scope.node_ids(), ["aa", "bb"]);
        assert_eq!(
            scope.intersect(&PrivateScope::of(["bb".to_string(), "cc".to_string()])),
            PrivateScope::new("bb")
        );
        assert!(PrivateScope::of([]).node_ids().is_empty());
    }

    #[tokio::test]
    async fn the_scope_is_visible_only_inside_the_scoped_future() {
        assert_eq!(current_private_scope(), None);
        let seen = with_private_scope(Some(PrivateScope::new("me")), async {
            current_private_scope()
        })
        .await;
        assert_eq!(seen, Some(PrivateScope::new("me")));
        assert_eq!(current_private_scope(), None);
        assert_eq!(
            with_private_scope(None, async { current_private_scope() }).await,
            None
        );
    }
}
