use super::node::Node;
use super::peer_state::NodeRole;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum HostRoleClaim {
    LocalModel,
    PluginInference,
}

#[derive(Default)]
pub(crate) struct HostRoleClaims(BTreeMap<HostRoleClaim, usize>);

impl HostRoleClaims {
    fn claim(&mut self, claim: HostRoleClaim) {
        *self.0.entry(claim).or_default() += 1;
    }

    fn release(&mut self, claim: HostRoleClaim) -> bool {
        let Some(count) = self.0.get_mut(&claim) else {
            return false;
        };
        if *count > 1 {
            *count -= 1;
        } else {
            self.0.remove(&claim);
        }
        true
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Node {
    pub async fn claim_host_role(&self, claim: HostRoleClaim, http_port: u16) {
        let transitioned = {
            let mut claims = self.host_role_claims.lock().await;
            claims.claim(claim);
            let mut role = self.role.lock().await;
            if matches!(*role, NodeRole::Worker) {
                *role = NodeRole::Host { http_port };
                true
            } else {
                false
            }
        };
        if transitioned {
            self.regossip().await;
        }
    }

    pub async fn release_host_role(&self, claim: HostRoleClaim) {
        let transitioned = {
            let mut claims = self.host_role_claims.lock().await;
            if !claims.release(claim) || !claims.is_empty() {
                false
            } else {
                let mut role = self.role.lock().await;
                if matches!(*role, NodeRole::Host { .. }) {
                    *role = NodeRole::Worker;
                    true
                } else {
                    false
                }
            }
        };
        if transitioned {
            self.regossip().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn host_role_claims_are_reference_counted_across_sources() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

        node.claim_host_role(HostRoleClaim::LocalModel, 9337).await;
        node.claim_host_role(HostRoleClaim::PluginInference, 9337)
            .await;
        assert_eq!(node.role().await, NodeRole::Host { http_port: 9337 });

        node.release_host_role(HostRoleClaim::PluginInference).await;
        assert_eq!(node.role().await, NodeRole::Host { http_port: 9337 });

        node.claim_host_role(HostRoleClaim::LocalModel, 9337).await;
        node.release_host_role(HostRoleClaim::LocalModel).await;
        assert_eq!(node.role().await, NodeRole::Host { http_port: 9337 });

        node.release_host_role(HostRoleClaim::LocalModel).await;
        assert_eq!(node.role().await, NodeRole::Worker);
    }

    /// A worker that serves a model is not routable, so claiming the host
    /// role is what actually publishes a runtime-loaded model to the mesh.
    ///
    /// This is asserted through a peer's own view of the announcement rather
    /// than on the local role, because `accepts_http_inference` filters on
    /// the *gossiped* role: a node can serve a model, advertise it in
    /// `serving_models`, and still be skipped by every peer's router.
    #[tokio::test]
    async fn serving_worker_is_only_routable_once_it_claims_the_host_role() {
        use crate::mesh::PeerInfo;

        let server = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        // Mirror a registered runtime model: both lists are populated, as
        // observed on a live node's gossip.
        server.set_serving_models(vec!["chat".into()]).await;
        server.set_hosted_models(vec!["chat".into()]).await;

        let observed = |node: &Node| {
            let node = node.clone();
            async move {
                let data = node.snapshot_local_announcement_data().await;
                let ann = node.build_local_announcement(data);
                PeerInfo::from_announcement(node.id(), ann.addr.clone(), &ann, Default::default())
            }
        };

        // Serving, advertised, and still invisible to routing.
        let peer = observed(&server).await;
        assert_eq!(peer.serving_models, vec!["chat".to_string()]);
        assert!(
            peer.http_routable_models().is_empty(),
            "a Worker must not attract HTTP inference"
        );
        assert!(!peer.routes_http_model("chat"));

        server
            .claim_host_role(HostRoleClaim::LocalModel, 19337)
            .await;

        let peer = observed(&server).await;
        assert_eq!(peer.role, NodeRole::Host { http_port: 19337 });
        assert!(
            peer.routes_http_model("chat"),
            "a host serving the model must be routable"
        );

        // Releasing on unload withdraws routability with it, so peers stop
        // sending inference to a port with nothing behind it.
        server.release_host_role(HostRoleClaim::LocalModel).await;
        let peer = observed(&server).await;
        assert!(!peer.routes_http_model("chat"));
    }
}
