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
    named_membership_of(node, "team", trusted).await
}

async fn named_membership_of(node: &Node, name: &str, trusted: &[&OwnerKeypair]) -> Result<Node> {
    let owners: Vec<String> = trusted.iter().map(|owner| owner.owner_id()).collect();
    let (membership, _channels) = node
        .start_membership_node(MembershipNodeConfig {
            mesh_name: name,
            secret_key: SecretKey::generate(),
            relay: RelayConfig {
                urls: &[],
                auths: &HashMap::new(),
                policy: RelayPolicy::ExplicitlyDisabled,
            },
            quic_port: None,
            trusted_owners: &owners,
            enumerate_host: false,
        })
        .await?;
    let entry = super::super::MembershipEntry {
        node: membership.clone(),
        api_port: 0,
        key_path: std::path::PathBuf::from(format!("{name}.key")),
        config: crate::runtime::MeshMembershipConfig {
            name: name.into(),
            join_tokens: Vec::new(),
            trusted_owners: owners.clone(),
            api_port: 0,
            node_key_path: None,
            enumerate_host: false,
        },
        join_error: Default::default(),
        tasks: Vec::new(),
    };
    assert!(node.memberships.insert(name.into(), entry).await.is_ok());
    Ok(membership)
}

async fn peer_ids(node: &Node) -> HashSet<EndpointId> {
    node.peers().await.into_iter().map(|peer| peer.id).collect()
}

#[tokio::test]
async fn gossip_verification_work_is_isolated_between_memberships() -> Result<()> {
    use crate::protocol::InboundGossip;

    let owner = test_owner_keypair(0x51, 0x52);
    let node = owned_node(&owner, &[&owner]).await?;
    let busy = named_membership_of(&node, "busy", &[&owner]).await?;
    let other = named_membership_of(&node, "other", &[&owner]).await?;
    let sender = make_test_node(super::super::NodeRole::Worker).await?;
    let record = sender
        .collect_outbound_gossip()
        .await
        .signed_records
        .remove(0);
    let frame = |records| InboundGossip {
        announcements: Vec::new(),
        signed_records: records,
        signed_cache_affinity: Vec::new(),
    };

    // Invalid records still consume the authenticated sender's work quota.
    // Keep this below the frame byte limit and avoid expensive cryptography.
    let mut invalid = record.clone();
    invalid.signed.clear();
    assert!(
        busy.prepare_gossip_records(sender.id(), &mut frame(vec![invalid; 4096]))
            .await
            .is_some()
    );
    assert!(
        busy.prepare_gossip_records(sender.id(), &mut frame(vec![record.clone()]))
            .await
            .is_none()
    );
    for independent in [&node, &other] {
        let prepared = independent
            .prepare_gossip_records(sender.id(), &mut frame(vec![record.clone()]))
            .await
            .expect("another mesh must retain its verification allowance");
        assert_eq!(prepared.nodes.len(), 1);
    }
    node.close_memberships().await;
    node.close_endpoint().await;
    sender.close_endpoint().await;
    Ok(())
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
    // Exercise owner admission over the bound loopback socket. Invite-token
    // addresses depend on iroh's asynchronous address discovery; discarding a
    // failed dial would leave this test waiting for a rejection of a peer the
    // membership never saw.
    let socket = membership
        .endpoint
        .bound_sockets()
        .into_iter()
        .find(|addr| addr.is_ipv4())
        .expect("membership binds an IPv4 socket");
    assert!(socket.ip().is_loopback());
    let address = EndpointAddr::new(membership.id()).with_ip_addr(socket);
    let dial_result = my_other_node.connect_to_peer(address).await;
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
    .unwrap_or_else(|_| {
        panic!("membership must record its owner-policy rejection: {dial_result:?}")
    });
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_membership_skips_its_own_token_and_refuses_signed_ones() -> Result<()> {
    let me = test_owner_keypair(0x51, 0x52);
    let node = owned_node(&me, &[&me]).await?;
    let membership = membership_of(&node, &[&me]).await?;
    assert!(
        !membership
            .join_membership(&membership.invite_token().await)
            .await?
    );

    let host = owned_node(&me, &[&me]).await?;
    install_requirement_policy(&host, &requirement_policy("trusted-signer")).await?;
    let error = membership
        .join_membership(&host.invite_token().await)
        .await
        .expect_err("a signed token carries requirements a membership lacks");
    assert!(format!("{error:#}").contains("unsigned"), "{error:#}");
    assert!(membership.requirement_mesh_state.lock().await.is_none());
    assert!(peer_ids(&membership).await.is_empty());
    Ok(())
}

/// A membership renews its own certificate as it heartbeats and leaves the
/// node's ownership file alone; the node's own renewal still saves it there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_membership_renews_its_certificate_without_saving_the_nodes_file() -> Result<()> {
    const DAY_MS: u64 = 24 * 60 * 60 * 1000;
    let me = test_owner_keypair(0x51, 0x52);
    let dir = tempfile::tempdir()?;
    let ownership_file = dir.path().join("node-ownership.json");
    let mut node = owned_node(&me, &[&me]).await?;
    node.owner_attestation_path = Some(ownership_file.clone());
    let membership = membership_of(&node, &[&me]).await?;
    // A day from expiry: inside the renewal window.
    let due = |id: EndpointId| {
        sign_node_ownership(
            &me,
            id.as_bytes(),
            current_time_unix_ms() + DAY_MS,
            None,
            None,
        )
    };

    *membership.owner_attestation.lock().await = Some(due(membership.id())?);
    membership.run_heartbeat_cycle(&mut HashMap::new()).await;
    let renewed = membership
        .owner_attestation
        .lock()
        .await
        .clone()
        .expect("the membership keeps a certificate");
    assert!(renewed.claim.expires_at_unix_ms > current_time_unix_ms() + 6 * DAY_MS);
    assert!(
        !ownership_file.exists(),
        "a membership's renewal must not write the node's ownership file"
    );

    *node.owner_attestation.lock().await = Some(due(node.id())?);
    node.run_heartbeat_cycle(&mut HashMap::new()).await;
    let saved: SignedNodeOwnership =
        serde_json::from_str(&std::fs::read_to_string(&ownership_file)?)?;
    assert_eq!(
        Some(&saved),
        node.owner_attestation.lock().await.as_ref(),
        "the node's own renewal is what the file holds"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_membership_never_peers_with_its_node_or_its_siblings() -> Result<()> {
    let me = test_owner_keypair(0x51, 0x52);
    let node = owned_node(&me, &[&me]).await?;
    let team = named_membership_of(&node, "team", &[&me]).await?;
    let family = named_membership_of(&node, "family", &[&me]).await?;

    for (from, to) in [(&team, &node), (&team, &family)] {
        let error = from
            .join_membership(&to.invite_token().await)
            .await
            .expect_err("a membership must not join another mesh its node serves");
        assert!(format!("{error:#}").contains("another mesh"), "{error:#}");
    }
    // The node itself refuses to dial its membership, whatever the token says.
    assert!(node.join(&team.invite_token().await).await.is_err());
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    for member in [&node, &team, &family] {
        assert!(peer_ids(member).await.is_empty());
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_membership_refuses_a_peer_announcing_another_mesh() -> Result<()> {
    let me = test_owner_keypair(0x51, 0x52);
    let node = owned_node(&me, &[&me]).await?;
    let membership = membership_of(&node, &[&me]).await?;
    // Trusted by owner, but already in a different mesh.
    let elsewhere = owned_node(&me, &[&me]).await?;
    elsewhere.set_mesh_id_force("another-mesh".into()).await;
    let _ = elsewhere.join(&membership.invite_token().await).await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert!(!peer_ids(&membership).await.contains(&elsewhere.id()));
    assert!(
        !membership
            .state
            .lock()
            .await
            .node_records
            .contains(&elsewhere.id()),
        "a rejected mesh must not seed the membership's signed records"
    );

    // A node not yet in any mesh is welcome, and adopts this one.
    let newcomer = owned_node(&me, &[&me]).await?;
    newcomer.join(&membership.invite_token().await).await?;
    wait_for_peer(&membership, newcomer.id()).await;
    assert_eq!(newcomer.mesh_id().await, membership.mesh_id().await);
    Ok(())
}
