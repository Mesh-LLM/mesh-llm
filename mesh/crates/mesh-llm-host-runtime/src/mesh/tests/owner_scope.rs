//! Private routing keeps a request on nodes signed by the entry node's owner,
//! whichever routing path picks the peer.

use super::*;
use crate::crypto::OwnershipSummary;
use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::{OwnerScope, with_owner_scope};

const MODEL: &str = "Qwen3-8B-Q4_K_M";

fn owned_host(seed: u8, owner: &str, status: OwnershipStatus) -> PeerInfo {
    let mut peer = departed_peer_never_seen_again(make_test_endpoint_id(seed));
    peer.display_rtt = Some(DirectLatencyObservation {
        rtt_ms: 5,
        observed_at: std::time::Instant::now(),
    });
    peer.owner_summary = OwnershipSummary {
        owner_id: Some(owner.to_string()),
        verified: status == OwnershipStatus::Verified,
        status,
        expires_at_unix_ms: Some(u64::MAX),
        ..OwnershipSummary::default()
    };
    peer
}

async fn node_with_mine_and_theirs() -> (Node, EndpointId, EndpointId, EndpointId) {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let mine = owned_host(61, "me", OwnershipStatus::Verified);
    let theirs = owned_host(62, "them", OwnershipStatus::Verified);
    // Claims my owner id but its certificate did not verify.
    let forged = owned_host(63, "me", OwnershipStatus::InvalidSignature);
    let ids = (mine.id, theirs.id, forged.id);
    for peer in [mine, theirs, forged] {
        node.insert_test_peer_without_liveness(peer).await;
    }
    (node, ids.0, ids.1, ids.2)
}

#[tokio::test]
async fn a_private_request_only_routes_to_peers_signed_by_the_same_owner() {
    let (node, mine, theirs, forged) = node_with_mine_and_theirs().await;

    let public = node.hosts_for_model(MODEL).await;
    assert!(public.contains(&mine) && public.contains(&theirs) && public.contains(&forged));

    let scope = Some(OwnerScope::new("me"));
    let private = with_owner_scope(scope.clone(), node.hosts_for_model(MODEL)).await;
    assert_eq!(private, vec![mine]);

    let served = with_owner_scope(scope, node.models_being_served()).await;
    assert!(served.iter().any(|name| name == MODEL));
    let served_by_stranger =
        with_owner_scope(Some(OwnerScope::new("nobody")), node.models_being_served()).await;
    assert!(!served_by_stranger.iter().any(|name| name == MODEL));
}

#[tokio::test]
async fn a_private_request_cannot_open_a_tunnel_to_a_stranger() {
    let (node, _mine, theirs, forged) = node_with_mine_and_theirs().await;
    for peer in [theirs, forged] {
        let error = with_owner_scope(Some(OwnerScope::new("me")), node.open_http_tunnel(peer))
            .await
            .expect_err("a stranger's node must be refused");
        assert!(
            error
                .to_string()
                .contains("not in this private request's trust circle"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn election_targets_keep_local_and_owned_hosts_only() {
    let (node, mine, theirs, forged) = node_with_mine_and_theirs().await;
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        MODEL.to_string(),
        vec![
            InferenceTarget::Local(9000),
            InferenceTarget::Remote(mine),
            InferenceTarget::Remote(theirs),
            InferenceTarget::Remote(forged),
        ],
    );
    let scoped = node
        .targets_in_owner_scope(&OwnerScope::new("me"), &targets)
        .await;
    assert_eq!(
        scoped.targets[MODEL],
        vec![InferenceTarget::Local(9000), InferenceTarget::Remote(mine)]
    );
}

#[tokio::test]
async fn a_local_split_with_a_stranger_stage_is_not_private() {
    let (node, mine, theirs, _forged) = node_with_mine_and_theirs().await;
    let split = |model: &str, stage_node: EndpointId| crate::mesh::StageTopologyInstance {
        topology_id: format!("topology-{model}"),
        run_id: "run".to_string(),
        model_id: model.to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        admissions: Default::default(),
        stages: vec![crate::mesh::StageAssignment {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            node_id: stage_node,
            layer_start: 12,
            layer_end: 24,
            endpoint: crate::mesh::StageEndpoint {
                bind_addr: "127.0.0.1:0".to_string(),
            },
        }],
    };
    {
        let mut topologies = node.stage_topologies.lock().await;
        topologies.record_topology(split("leaky", theirs));
        topologies.record_topology(split("owned", mine));
    }
    let mut targets = ModelTargets::default();
    for model in ["leaky", "owned"] {
        targets
            .targets
            .insert(model.to_string(), vec![InferenceTarget::Local(9000)]);
    }
    let scope = OwnerScope::new("me");
    let scoped = node.targets_in_owner_scope(&scope, &targets).await;
    assert!(scoped.targets["leaky"].is_empty());
    assert_eq!(scoped.targets["owned"], vec![InferenceTarget::Local(9000)]);
}

#[tokio::test]
async fn my_own_nodes_are_admitted_as_me_when_remote_is_paused() {
    use crate::runtime::IngressType;
    let (node, mine, theirs, forged) = node_with_mine_and_theirs().await;
    let remote = IngressType::RemoteQuicHttp;

    // Unsigned entry node: nobody is "me".
    assert_eq!(
        node.owner_admission_ingress(remote, Some(mine)).await,
        remote
    );

    *node.owner_summary.lock().await = OwnershipSummary {
        owner_id: Some("me".to_string()),
        verified: true,
        status: OwnershipStatus::Verified,
        expires_at_unix_ms: Some(u64::MAX),
        ..OwnershipSummary::default()
    };
    assert_eq!(
        node.owner_admission_ingress(remote, Some(mine)).await,
        IngressType::LocalOpenAi
    );
    for stranger in [Some(theirs), Some(forged), None] {
        assert_eq!(node.owner_admission_ingress(remote, stranger).await, remote);
    }
}

async fn sign_as_me(node: &Node) {
    *node.owner_summary.lock().await = OwnershipSummary {
        owner_id: Some("me".to_string()),
        verified: true,
        status: OwnershipStatus::Verified,
        expires_at_unix_ms: Some(u64::MAX),
        ..OwnershipSummary::default()
    };
}

#[tokio::test]
async fn trusted_owners_join_the_private_scope_but_not_owner_priority() {
    use crate::runtime::IngressType;
    let (node, mine, theirs, forged) = node_with_mine_and_theirs().await;
    sign_as_me(&node).await;
    node.trust_store
        .lock()
        .await
        .add_trusted_owner("them".to_string(), Some("friend".to_string()));

    let scope = node.private_owner_scope().await.unwrap();
    let hosts = with_owner_scope(Some(scope), node.hosts_for_model(MODEL)).await;
    assert!(hosts.contains(&mine) && hosts.contains(&theirs));
    assert!(!hosts.contains(&forged));

    // A trusted owner's node is in the circle, but is not "me".
    let remote = IngressType::RemoteQuicHttp;
    assert_eq!(
        node.owner_admission_ingress(remote, Some(mine)).await,
        IngressType::LocalOpenAi
    );
    assert_eq!(
        node.owner_admission_ingress(remote, Some(theirs)).await,
        remote
    );
}

#[tokio::test]
async fn a_forwarded_private_request_keeps_to_this_and_the_requesters_owner() {
    let (node, mine, theirs, forged) = node_with_mine_and_theirs().await;
    let friend = owned_host(64, "friend", OwnershipStatus::Verified);
    let friend_id = friend.id;
    node.insert_test_peer_without_liveness(friend).await;
    sign_as_me(&node).await;
    // This node trusts "friend"; the entry node ("them") might not.
    node.trust_store
        .lock()
        .await
        .add_trusted_owner("friend".to_string(), None);

    let scope = node.forwarded_private_scope(theirs).await;
    let hosts = with_owner_scope(Some(scope), node.hosts_for_model(MODEL)).await;
    assert!(hosts.contains(&mine) && hosts.contains(&theirs));
    assert!(!hosts.contains(&friend_id) && !hosts.contains(&forged));

    // An unverified requester adds nobody.
    let scope = node.forwarded_private_scope(forged).await;
    let hosts = with_owner_scope(Some(scope), node.hosts_for_model(MODEL)).await;
    assert!(hosts.contains(&mine));
    assert!(!hosts.contains(&theirs) && !hosts.contains(&forged));
}

/// Entry node A ("them") trusts this node B ("me"); B trusts C ("friend"),
/// A does not. B's plugin handles A's forwarded private request, and its
/// nested call must not reach C.
#[tokio::test]
async fn a_plugins_nested_call_does_not_reach_owners_the_entry_node_does_not_trust() {
    use crate::network::openai::{request_parse, routing_scope};
    use crate::plugin::private_invocations::NestedScope;
    use tokio::io::AsyncWriteExt;

    let (node, mine, theirs, _) = node_with_mine_and_theirs().await;
    let friend = owned_host(64, "friend", OwnershipStatus::Verified);
    let friend_id = friend.id;
    node.insert_test_peer_without_liveness(friend).await;
    sign_as_me(&node).await;
    node.trust_store
        .lock()
        .await
        .add_trusted_owner("friend".to_string(), None);

    let plugins = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    let forwarded = node.forwarded_private_scope(theirs).await;
    let _turn = plugins.begin_private_invocation("moa", Some(forwarded));
    let Some(NestedScope::Owners(owners)) = plugins.private_nested_scope("moa") else {
        panic!("the turn's owners must travel with its nested calls");
    };

    // The nested call, as the bridge sends it to this node's own API.
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         {}: {}\r\n{}: {}\r\nContent-Length: 2\r\n\r\n{{}}",
        routing_scope::MESH_SCOPE_HEADER,
        routing_scope::PRIVATE_SCOPE,
        routing_scope::MESH_SCOPE_OWNERS_HEADER,
        owners.owner_ids().join(","),
    );
    let (mut reader, mut writer) = tokio::io::duplex(wire.len() + 1);
    writer.write_all(wire.as_bytes()).await.unwrap();
    let mut request = request_parse::read_http_request(&mut reader).await.unwrap();
    let scope = routing_scope::resolve(&node, &mut request)
        .await
        .unwrap()
        .unwrap();

    let hosts = with_owner_scope(Some(scope), node.hosts_for_model(MODEL)).await;
    assert!(hosts.contains(&mine));
    assert!(!hosts.contains(&friend_id), "C is outside A's circle");
    // Narrowing only: B does not trust A, so A's other nodes stay out too.
    assert!(!hosts.contains(&theirs));
}
