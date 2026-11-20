use super::*;

#[tokio::test]
async fn peer_evidence_oversized_local_metadata_is_not_signed() -> Result<()> {
    let node = make_test_node(LocalRole::Worker).await?;
    let models: Vec<_> = (0..256)
        .map(|n| format!("{n:03}{}", "m".repeat(505)))
        .collect();
    node.set_serving_models(models.clone()).await;
    node.set_requested_models(models).await;
    let gossip = node.collect_own_outbound_gossip().await;
    assert!(
        gossip
            .announcements
            .iter()
            .any(|ann| ann.addr.id == node.id())
    );
    assert!(gossip.signed_records.is_empty());
    node.close_endpoint().await;
    Ok(())
}
