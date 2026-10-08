//! Release node listeners when startup fails before normal runtime shutdown owns them.

use crate::mesh::Node;

pub(super) async fn cleanup_failed_node_start(node: &Node, error: anyhow::Error) -> anyhow::Error {
    node.shutdown_control_listener().await;
    node.close_endpoint().await;
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{OwnershipStatus, OwnershipSummary};
    use crate::mesh::NodeRole;
    use std::time::Duration;

    #[tokio::test]
    async fn failed_start_closes_listeners_and_stage_control_in_live_runtime() -> anyhow::Result<()>
    {
        let (node, secret_key) = Node::new_for_tests_with_secret(NodeRole::Worker).await?;
        *node.owner_summary.lock().await = OwnershipSummary {
            owner_id: Some("fixture-owner".into()),
            status: OwnershipStatus::Verified,
            verified: true,
            ..Default::default()
        };
        node.maybe_start_control_listener(secret_key, None, None)
            .await?;
        let control_endpoint = node
            .control_listener
            .lock()
            .await
            .as_ref()
            .expect("verified owner control listener")
            .endpoint
            .clone();
        let stage = crate::inference::skippy::spawn_stage_control_loop(
            super::super::run_auto::skippy_telemetry_options(&crate::RuntimeOptions::default()),
        );
        let stage_sender = stage.sender();
        node.set_stage_control_handle(stage).await;
        let accepting_node = node.clone();
        let accept_loop = tokio::spawn(async move { accepting_node.accept_loop().await });
        node.start_accepting();
        assert!(!node.endpoint.is_closed());
        assert!(!control_endpoint.is_closed());

        let error =
            cleanup_failed_node_start(&node, anyhow::anyhow!("provider handshake failed")).await;

        assert_eq!(error.to_string(), "provider handshake failed");
        assert!(node.endpoint.is_closed());
        assert!(control_endpoint.is_closed());
        assert!(node.control_endpoint().await.is_none());
        assert!(node.stage_control_tx.lock().await.is_none());
        assert!(stage_sender.is_closed());
        tokio::time::timeout(Duration::from_secs(5), accept_loop)
            .await
            .expect("failed startup must release the detached accept-loop node clone")?;
        // These assertions run before the Tokio runtime tears down, so runtime
        // teardown cannot hide a leaked listener or stage-control task.
        Ok(())
    }
}
