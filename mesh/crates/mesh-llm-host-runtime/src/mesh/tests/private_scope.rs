//! Private routing keeps a request on the entry node and the nodes it
//! lists, whichever routing path picks the peer.

use super::*;
use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::private_scope::node_key;
use crate::mesh::{PrivateScope, with_private_scope};

const MODEL: &str = "Qwen3-8B-Q4_K_M";

fn host(seed: u8) -> PeerInfo {
    let mut peer = departed_peer_never_seen_again(make_test_endpoint_id(seed));
    peer.display_rtt = Some(DirectLatencyObservation {
        rtt_ms: 5,
        observed_at: std::time::Instant::now(),
    });
    peer.private_routing_supported = true;
    peer
}

/// A node that lists `mine` as a private node; `theirs` is not listed.
async fn node_with_mine_and_theirs() -> (Node, EndpointId, EndpointId) {
    let mut node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let (mine, theirs) = (host(61), host(62));
    let ids = (mine.id, theirs.id);
    node.private_nodes = std::sync::Arc::new([mine.id].into());
    for peer in [mine, theirs] {
        node.insert_test_peer_without_liveness(peer).await;
    }
    (node, ids.0, ids.1)
}

fn scope_of(ids: &[EndpointId]) -> PrivateScope {
    PrivateScope::of(ids.iter().map(node_key))
}

#[tokio::test]
async fn a_private_request_only_routes_to_listed_peers() {
    let (node, mine, theirs) = node_with_mine_and_theirs().await;

    let public = node.hosts_for_model(MODEL).await;
    assert!(public.contains(&mine) && public.contains(&theirs));

    let scope = Some(node.entry_private_scope());
    let private = with_private_scope(scope.clone(), node.hosts_for_model(MODEL)).await;
    assert_eq!(private, vec![mine]);

    let served = with_private_scope(scope, node.models_being_served()).await;
    assert!(served.iter().any(|name| name == MODEL));
    let served_by_nobody =
        with_private_scope(Some(PrivateScope::of([])), node.models_being_served()).await;
    assert!(!served_by_nobody.iter().any(|name| name == MODEL));
}

#[tokio::test]
async fn by_default_a_private_request_stays_on_this_node() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.insert_test_peer_without_liveness(host(61)).await;
    let scope = node.entry_private_scope();
    assert_eq!(scope, PrivateScope::new(node.local_scope_node_id()));
    assert!(node.local_in_private_scope(&scope));
    assert!(
        with_private_scope(Some(scope), node.hosts_for_model(MODEL))
            .await
            .is_empty()
    );
}

/// A node too old to know the private marker would serve a private request
/// as public, so it gets none even when it is listed.
#[tokio::test]
async fn a_listed_peer_without_private_routing_gets_no_private_request() {
    let (mut node, mine, _theirs) = node_with_mine_and_theirs().await;
    let mut old = host(64);
    old.private_routing_supported = false;
    let old_id = old.id;
    node.private_nodes = std::sync::Arc::new([mine, old_id].into());
    node.insert_test_peer_without_liveness(old).await;
    let scope = Some(node.entry_private_scope());
    let private = with_private_scope(scope.clone(), node.hosts_for_model(MODEL)).await;
    assert_eq!(private, vec![mine]);
    assert!(node.hosts_for_model(MODEL).await.contains(&old_id));
    assert!(
        with_private_scope(scope, node.open_http_tunnel(old_id))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn a_private_request_cannot_open_a_tunnel_to_an_unlisted_peer() {
    let (node, _mine, theirs) = node_with_mine_and_theirs().await;
    let error = with_private_scope(
        Some(node.entry_private_scope()),
        node.open_http_tunnel(theirs),
    )
    .await
    .expect_err("an unlisted node must be refused");
    assert!(
        error
            .to_string()
            .contains("not one of this private request's nodes"),
        "{error}"
    );
}

#[tokio::test]
async fn election_targets_keep_local_and_listed_hosts_only() {
    let (node, mine, theirs) = node_with_mine_and_theirs().await;
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        MODEL.to_string(),
        vec![
            InferenceTarget::Local(9000),
            InferenceTarget::Remote(mine),
            InferenceTarget::Remote(theirs),
        ],
    );
    let scoped = node
        .targets_in_private_scope(&node.entry_private_scope(), &targets)
        .await;
    assert_eq!(
        scoped.targets[MODEL],
        vec![InferenceTarget::Local(9000), InferenceTarget::Remote(mine)]
    );

    // A scope narrowed to other nodes leaves this node out.
    let scoped = node
        .targets_in_private_scope(&scope_of(&[mine]), &targets)
        .await;
    assert_eq!(scoped.targets[MODEL], vec![InferenceTarget::Remote(mine)]);
}

fn stage(index: u32, node: EndpointId) -> crate::mesh::StageAssignment {
    crate::mesh::StageAssignment {
        stage_id: format!("stage-{index}"),
        stage_index: index,
        node_id: node,
        layer_start: index * 12,
        layer_end: (index + 1) * 12,
        endpoint: crate::mesh::StageEndpoint {
            bind_addr: "127.0.0.1:0".to_string(),
        },
    }
}

fn split(
    id: &str,
    model: &str,
    stages: Vec<crate::mesh::StageAssignment>,
) -> crate::mesh::StageTopologyInstance {
    crate::mesh::StageTopologyInstance {
        topology_id: format!("topology-{id}"),
        run_id: "run".to_string(),
        model_id: model.to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        admissions: Default::default(),
        stages,
    }
}

#[tokio::test]
async fn a_local_split_with_an_unlisted_stage_is_not_private() {
    let (node, mine, theirs) = node_with_mine_and_theirs().await;
    let me = node.endpoint.id();
    {
        let mut topologies = node.stage_topologies.lock().await;
        topologies.record_topology(split(
            "leaky",
            "leaky",
            vec![stage(0, me), stage(1, theirs)],
        ));
        topologies.record_topology(split(
            "listed",
            "listed",
            vec![stage(0, me), stage(1, mine)],
        ));
        // This node is only a stage in someone else's split of a model it
        // also serves in full: its own target stays.
        topologies.record_topology(split(
            "theirs",
            "full",
            vec![stage(0, theirs), stage(1, me)],
        ));
    }
    let mut targets = ModelTargets::default();
    for model in ["leaky", "listed", "full"] {
        targets
            .targets
            .insert(model.to_string(), vec![InferenceTarget::Local(9000)]);
    }
    let scoped = node
        .targets_in_private_scope(&node.entry_private_scope(), &targets)
        .await;
    assert!(scoped.targets["leaky"].is_empty());
    assert_eq!(scoped.targets["listed"], vec![InferenceTarget::Local(9000)]);
    assert_eq!(scoped.targets["full"], vec![InferenceTarget::Local(9000)]);
}

/// A listed peer too old to honour the private marker could serve a
/// forwarded request from a split with an unlisted stage. A split this
/// node knows of drops that peer's target.
#[tokio::test]
async fn a_listed_peers_known_leaky_split_is_not_private() {
    let (node, mine, theirs) = node_with_mine_and_theirs().await;
    let me = node.endpoint.id();
    node.stage_topologies.lock().await.record_topology(split(
        "peer",
        MODEL,
        vec![stage(0, mine), stage(1, me), stage(2, theirs)],
    ));
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        MODEL.to_string(),
        vec![InferenceTarget::Local(9000), InferenceTarget::Remote(mine)],
    );
    let scoped = node
        .targets_in_private_scope(&node.entry_private_scope(), &targets)
        .await;
    assert_eq!(scoped.targets[MODEL], vec![InferenceTarget::Local(9000)]);
}

#[tokio::test]
async fn listed_nodes_are_admitted_as_mine_when_remote_is_paused() {
    use crate::runtime::IngressType;
    let (node, mine, theirs) = node_with_mine_and_theirs().await;
    let remote = IngressType::RemoteQuicHttp;
    assert_eq!(
        node.private_node_admission_ingress(remote, Some(mine)),
        IngressType::LocalOpenAi
    );
    for other in [Some(theirs), None] {
        assert_eq!(node.private_node_admission_ingress(remote, other), remote);
    }
}

#[tokio::test]
async fn a_forwarded_private_request_keeps_to_this_node_and_the_requester() {
    let (node, mine, theirs) = node_with_mine_and_theirs().await;
    // The entry node ("theirs") forwarded it; this node's own list says
    // nothing about the entry node's.
    let scope = node.forwarded_private_scope(theirs);
    assert_eq!(scope, scope_of(&[node.endpoint.id(), theirs]));
    let hosts = with_private_scope(Some(scope), node.hosts_for_model(MODEL)).await;
    assert_eq!(hosts, vec![theirs]);
    assert!(!hosts.contains(&mine));
}

/// Entry node A lists this node B; B lists C, A does not. B's plugin
/// handles A's forwarded private request, and its nested call must not
/// reach C.
#[tokio::test]
async fn a_plugins_nested_call_does_not_reach_nodes_the_entry_node_does_not_list() {
    use crate::network::openai::{request_parse, routing_scope};
    use crate::plugin::private_invocations::NestedScope;
    use tokio::io::AsyncWriteExt;

    let (node, mine, theirs) = node_with_mine_and_theirs().await;
    let plugins = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    let forwarded = node.forwarded_private_scope(theirs);
    let _turn = plugins.begin_private_invocation("moa", Some(forwarded));
    let Some(NestedScope::Nodes(nodes)) = plugins.private_nested_scope("moa") else {
        panic!("the turn's nodes must travel with its nested calls");
    };

    // The nested call, as the bridge sends it to this node's own API.
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         {}: {}\r\n{}: {}\r\nContent-Length: 2\r\n\r\n{{}}",
        routing_scope::MESH_SCOPE_HEADER,
        routing_scope::PRIVATE_SCOPE,
        routing_scope::MESH_SCOPE_NODES_HEADER,
        nodes.node_ids().join(","),
    );
    let (mut reader, mut writer) = tokio::io::duplex(wire.len() + 1);
    writer.write_all(wire.as_bytes()).await.unwrap();
    let mut request = request_parse::read_http_request(&mut reader).await.unwrap();
    let scope = routing_scope::resolve(&node, &mut request)
        .await
        .unwrap()
        .unwrap();

    assert!(node.local_in_private_scope(&scope));
    let hosts = with_private_scope(Some(scope), node.hosts_for_model(MODEL)).await;
    assert!(!hosts.contains(&mine), "C is outside A's scope");
    // Narrowing only: B does not list A, so A stays out too.
    assert!(!hosts.contains(&theirs));
}

async fn parsed(headers: &str) -> crate::network::openai::request_parse::BufferedHttpRequest {
    use tokio::io::AsyncWriteExt;
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         {headers}Content-Length: 2\r\n\r\n{{}}"
    );
    let (mut reader, mut writer) = tokio::io::duplex(wire.len() + 1);
    writer.write_all(wire.as_bytes()).await.unwrap();
    crate::network::openai::request_parse::read_http_request(&mut reader)
        .await
        .unwrap()
}

/// A peer's private marker fails closed: repeated headers count as marked,
/// an unknown requester adds no node, and the entry node's list narrows.
#[tokio::test]
async fn a_forwarded_private_marker_fails_closed_and_only_narrows() {
    use crate::network::openai::routing_scope::peer_private_scope;
    let (node, _mine, theirs) = node_with_mine_and_theirs().await;
    let me = node.endpoint.id();

    let unmarked = parsed("").await;
    assert_eq!(
        peer_private_scope(&node, &unmarked, Some(theirs)).await,
        None
    );

    let repeated = parsed("x-mesh-scope: public\r\nx-mesh-scope: public\r\n").await;
    assert_eq!(
        peer_private_scope(&node, &repeated, Some(theirs)).await,
        Some(scope_of(&[me, theirs]))
    );

    let marked = parsed("x-mesh-scope: private\r\n").await;
    assert_eq!(
        peer_private_scope(&node, &marked, None).await,
        Some(scope_of(&[me]))
    );

    // The entry node ("theirs") scoped the request to itself only.
    let narrowed = parsed(&format!(
        "x-mesh-scope: private\r\nx-mesh-scope-nodes: {},other\r\n",
        node_key(&theirs)
    ))
    .await;
    assert_eq!(
        peer_private_scope(&node, &narrowed, Some(theirs)).await,
        Some(scope_of(&[theirs]))
    );
}

#[tokio::test]
async fn a_node_limit_alone_selects_private_routing() {
    use crate::network::openai::routing_scope;
    let (node, _mine, _theirs) = node_with_mine_and_theirs().await;
    let me = node.local_scope_node_id();
    let mut request = parsed(&format!("x-mesh-scope-nodes: {me}\r\n")).await;
    assert_eq!(
        routing_scope::resolve(&node, &mut request).await,
        Ok(Some(PrivateScope::new(me)))
    );
    let mut request = parsed("").await;
    assert_eq!(routing_scope::resolve(&node, &mut request).await, Ok(None));
}
