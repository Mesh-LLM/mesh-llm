//! Policy adoption and compact gossip over real loopback QUIC connections.

use super::*;
use crate::mesh::NodeRole as LocalRole;

#[tokio::test]
async fn strict_policy_is_adopted_and_two_servers_relay_models_and_demand() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let relay = make_test_node(LocalRole::Host { http_port: 9338 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    let mut policy = requirement_policy_without_release_attestation();
    policy.requirements.require_signed_gossip = true;
    install_requirement_policy(&origin, &policy).await?;
    origin.set_serving_models(vec!["origin-model".into()]).await;
    relay.set_serving_models(vec!["relay-model".into()]).await;
    origin.merge_remote_demand(&HashMap::from([(
        "requested-model".into(),
        ModelDemand {
            last_active: now_secs(),
            request_count: 4,
        },
    )]));
    for node in [&origin, &relay, &observer] {
        node.start_accepting();
    }
    let token = origin.invite_token().await;
    relay.join(&token).await?;
    // The relay inherited the signed genesis policy despite having default
    // local creation settings. Its invitation carries the same policy.
    assert!(relay.requires_signed_gossip().await);
    observer.join(&relay.invite_token().await).await?;
    assert!(observer.requires_signed_gossip().await);
    for node in [&relay, &observer] {
        assert!(
            node.peers()
                .await
                .iter()
                .any(|p| p.id == origin.id() && p.serving_models.contains(&"origin-model".into()))
        );
        assert_eq!(node.get_demand()["requested-model"].request_count, 4);
    }
    assert!(
        observer
            .peers()
            .await
            .iter()
            .any(|p| p.id == relay.id() && p.serving_models.contains(&"relay-model".into()))
    );
    let outbound = relay.collect_outbound_gossip().await;
    assert_eq!(outbound.announcements.len(), 1);
    assert!(outbound.signed_records.len() >= 2);
    for node in [&observer, &relay, &origin] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn rejected_unsigned_join_cannot_seed_strict_gossip_records() -> Result<()> {
    let host = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let outsider = make_test_node(LocalRole::Worker).await?;
    let mut policy = requirement_policy_without_release_attestation();
    policy.requirements.require_signed_gossip = true;
    install_requirement_policy(&host, &policy).await?;
    host.start_accepting();
    outsider.start_accepting();
    outsider
        .connect_to_peer(host.endpoint_addr_for_advertisement())
        .await
        .expect_err("unsigned connection must not bypass signed bootstrap admission");
    let state = host.state.lock().await;
    assert!(!state.node_records.contains(&outsider.id()));
    assert!(!state.cache_affinity_records.contains(&outsider.id()));
    assert!(!state.peers.contains_key(&outsider.id()));
    drop(state);
    outsider.close_endpoint().await;
    host.close_endpoint().await;
    Ok(())
}
