//! Mesh memberships: a node serving its models into a further mesh keeps that
//! mesh's peers, admission and gossip apart from its own.

use super::*;
use crate::crypto::OwnerKeypair;
use crate::mesh::node::MembershipNodeConfig;

/// A node with its own owner that admits only the `trusted` owners.
async fn owned_node(owner: &OwnerKeypair, trusted: &[&OwnerKeypair]) -> Result<Node> {
    let mut node = make_test_node(super::super::NodeRole::Worker).await?;
    let ownership = sign_node_ownership(
        owner,
        node.id().as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )?;
    let mut trust_store = TrustStore::default();
    for trusted in trusted {
        trust_store.add_trusted_owner(trusted.owner_id(), None);
    }
    node.owner_keypair = Some(owner.clone());
    node.quic_bind.ip = Some(std::net::Ipv4Addr::LOCALHOST.into());
    node.trust_policy = TrustPolicy::Allowlist;
    *node.owner_attestation.lock().await = Some(ownership);
    *node.trust_store.lock().await = trust_store;
    node.start_accepting();
    Ok(node)
}

async fn membership_of(node: &Node, trusted: &[&OwnerKeypair]) -> Result<Node> {
    let owners: Vec<String> = trusted.iter().map(|owner| owner.owner_id()).collect();
    let (membership, _channels) = node
        .start_membership_node(MembershipNodeConfig {
            secret_key: SecretKey::generate(),
            relay: RelayConfig {
                urls: &[],
                auths: &HashMap::new(),
                policy: RelayPolicy::ExplicitlyDisabled,
            },
            trusted_owners: &owners,
            enumerate_host: false,
        })
        .await?;
    node.memberships
        .insert(
            "team".into(),
            super::super::MembershipEntry {
                node: membership.clone(),
                api_port: 0,
                tasks: Vec::new(),
            },
        )
        .await;
    Ok(membership)
}

async fn peer_ids(node: &Node) -> HashSet<EndpointId> {
    node.peers().await.into_iter().map(|peer| peer.id).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_membership_keeps_its_peers_apart_from_the_nodes_own() -> Result<()> {
    let me = test_owner_keypair(0x51, 0x52);
    let team = test_owner_keypair(0x61, 0x62);
    let node = owned_node(&me, &[&me]).await?;
    let my_other_node = owned_node(&me, &[&me]).await?;
    my_other_node.join(&node.invite_token().await).await?;
    wait_for_peer(&node, my_other_node.id()).await;

    let membership = membership_of(&node, &[&team]).await?;
    let teammate = owned_node(&team, &[&team, &me]).await?;
    teammate.join(&membership.invite_token().await).await?;
    wait_for_peer(&membership, teammate.id()).await;
    wait_for_peer(&teammate, membership.id()).await;

    assert_eq!(peer_ids(&teammate).await, HashSet::from([membership.id()]));
    assert_eq!(peer_ids(&membership).await, HashSet::from([teammate.id()]));
    assert_eq!(peer_ids(&node).await, HashSet::from([my_other_node.id()]));
    assert_eq!(peer_ids(&my_other_node).await, HashSet::from([node.id()]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_membership_admits_only_its_own_allowlist() -> Result<()> {
    let me = test_owner_keypair(0x51, 0x52);
    let team = test_owner_keypair(0x61, 0x62);
    let node = owned_node(&me, &[&me]).await?;
    let membership = membership_of(&node, &[&team]).await?;

    // Trusted by the node itself, but not on the membership's allowlist.
    let my_other_node = owned_node(&me, &[&me]).await?;
    let _ = my_other_node.join(&membership.invite_token().await).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !membership
            .state
            .lock()
            .await
            .policy_rejected_peers
            .contains_key(&my_other_node.id())
        {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("the membership should refuse an owner its allowlist lacks");
    assert!(peer_ids(&membership).await.is_empty());
    assert!(peer_ids(&node).await.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_membership_advertises_what_the_node_serves() -> Result<()> {
    let me = test_owner_keypair(0x51, 0x52);
    let team = test_owner_keypair(0x61, 0x62);
    let node = owned_node(&me, &[&me]).await?;
    let membership = membership_of(&node, &[&team]).await?;
    let teammate = owned_node(&team, &[&team, &me]).await?;
    teammate.join(&membership.invite_token().await).await?;
    wait_for_peer(&teammate, membership.id()).await;

    // Loaded after the membership started: the node's regossip reaches the
    // membership's peers too.
    node.set_serving_models(vec!["shared-model".into()]).await;
    node.regossip().await;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let advertised = teammate
                .peers()
                .await
                .into_iter()
                .find(|peer| peer.id == membership.id())
                .is_some_and(|peer| peer.serving_models == ["shared-model"]);
            if advertised {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("the membership's peers should see the node's models");
    Ok(())
}

#[tokio::test]
async fn a_membership_needs_an_owner() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let error = membership_of(&node, &[])
        .await
        .err()
        .expect("a node without an owner key cannot start a membership");
    assert!(format!("{error:#}").contains("owner key"), "{error:#}");
    Ok(())
}
