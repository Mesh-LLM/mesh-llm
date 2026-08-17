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
}
