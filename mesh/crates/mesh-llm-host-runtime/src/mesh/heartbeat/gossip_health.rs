//! Classify gossip outcomes without treating a local encoding limit as a
//! remote failure or manufacturing evidence of peer liveness.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GossipHealth {
    Alive,
    Failed,
    Deferred,
}

impl GossipHealth {
    pub(crate) fn from_exchange(
        result: Result<anyhow::Result<()>, tokio::time::error::Elapsed>,
    ) -> Self {
        match result {
            Ok(Ok(())) => Self::Alive,
            Ok(Err(error))
                if error
                    .downcast_ref::<crate::protocol::GossipFrameBudgetExceeded>()
                    .is_some() =>
            {
                tracing::warn!(%error, "heartbeat deferred: local gossip exceeds the frame budget");
                Self::Deferred
            }
            _ => Self::Failed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Node, NodeRole};
    use std::collections::HashMap;

    #[tokio::test]
    async fn deferred_exchange_neither_increments_failures_nor_recovers_a_peer() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let peer = iroh::SecretKey::from_bytes(&[23; 32]).public();
        let mut failures = HashMap::from([(peer, 2)]);
        node.record_heartbeat_result(peer, GossipHealth::Deferred, &mut failures)
            .await;
        assert_eq!(failures.get(&peer), Some(&2));
        assert!(!node.state.lock().await.peers.contains_key(&peer));
        node.record_heartbeat_result(peer, GossipHealth::Alive, &mut failures)
            .await;
        assert!(!failures.contains_key(&peer));
    }

    #[test]
    fn remote_protocol_errors_remain_failures() {
        assert_eq!(
            GossipHealth::from_exchange(Ok(Err(anyhow::anyhow!("oversized remote frame")))),
            GossipHealth::Failed
        );
        assert_eq!(GossipHealth::from_exchange(Ok(Ok(()))), GossipHealth::Alive);
    }
}
