//! Mesh memberships through actual peer HTTP ingress: a membership's peers run
//! requests on the node's own model, and a membership routes only among its
//! own mesh.
use super::duplicate_alias_repro::{duplicate_descriptor, duplicate_peer};
use super::*;
use crate::crypto::{DEFAULT_NODE_CERT_LIFETIME_SECS, OwnerKeypair, sign_node_ownership};

const MODEL: &str = "Falcon-H1-1.5B-Instruct-Q4_K_M";

fn owner(seed: u8) -> OwnerKeypair {
    OwnerKeypair::from_bytes(&[seed; 32], &[seed.wrapping_add(1); 32]).unwrap()
}

/// Gives `node` an owner and the certificate that proves it.
async fn own(node: &mut mesh::Node, owner: &OwnerKeypair) {
    let ownership = sign_node_ownership(
        owner,
        node.id().as_bytes(),
        mesh::current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        None,
        None,
    )
    .unwrap();
    node.owner_keypair = Some(owner.clone());
    *node.owner_attestation.lock().await = Some(ownership);
}

async fn membership_of(
    node: &mesh::Node,
    trusted: &[&OwnerKeypair],
) -> (mesh::Node, mesh::TunnelChannels) {
    let owners: Vec<String> = trusted.iter().map(|owner| owner.owner_id()).collect();
    node.start_membership_node(mesh::MembershipNodeConfig {
        mesh_name: "team",
        secret_key: iroh::SecretKey::generate(),
        relay: mesh::RelayConfig {
            urls: &[],
            auths: &HashMap::new(),
            policy: mesh::RelayPolicy::Disabled,
        },
        quic_port: None,
        trusted_owners: &owners,
        enumerate_host: false,
    })
    .await
    .unwrap()
}

fn chat_request(model: &str) -> String {
    let body = serde_json::json!({"model":model,"messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true}).to_string();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

#[tokio::test]
async fn membership_peer_request_executes_on_the_nodes_local_model() {
    let (team, me) = (owner(0x71), owner(0x73));
    let (mut node, _node_channels, _node_dir) = start_http_tunnel_test_node().await;
    own(&mut node, &me).await;
    let (port, backend, backend_task) = spawn_tunnel_capture().await;
    // Served by the node itself; the membership shares it.
    node.set_hosted_models(vec![MODEL.to_owned()]).await;
    node.set_served_model_descriptors(vec![duplicate_descriptor(MODEL)])
        .await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.to_owned(),
        vec![election::InferenceTarget::Local(port)],
    );
    let (_targets_tx, targets_rx) = watch::channel(targets);

    let (membership, channels) = membership_of(&node, &[&team]).await;
    let ingress = tokio::spawn(crate::network::tunnel::serve_membership_http_ingress(
        membership.clone(),
        channels.http,
        targets_rx,
        AffinityRouter::new(),
    ));
    let (mut teammate, _teammate_channels, _teammate_dir) = start_http_tunnel_test_node().await;
    own(&mut teammate, &team).await;
    teammate.start_accepting();
    teammate
        .connect_to_peer(membership.endpoint_addr_for_advertisement())
        .await
        .unwrap();

    let (mut send, _recv) = teammate.open_http_tunnel(membership.id()).await.unwrap();
    send.write_all(chat_request(MODEL).as_bytes())
        .await
        .unwrap();
    send.finish().unwrap();
    let captured = tokio::time::timeout(Duration::from_secs(5), backend).await;
    ingress.abort();
    backend_task.abort();
    teammate.close_endpoint().await;
    membership.close_endpoint().await;
    node.close_endpoint().await;
    let captured = captured
        .expect("the membership should run the request on the node's model")
        .unwrap();
    assert!(String::from_utf8_lossy(&captured).contains(MODEL));
}

#[tokio::test]
async fn membership_routes_only_among_its_own_mesh() {
    let me = owner(0x73);
    let (mut node, _node_channels, _node_dir) = start_http_tunnel_test_node().await;
    own(&mut node, &me).await;
    let (membership, _channels) = membership_of(&node, &[&owner(0x71)]).await;
    // A peer of the node's own mesh serves a model this node does not.
    let (my_other_node, _other_channels, _other_dir) = start_http_tunnel_test_node().await;
    node.insert_test_peer(duplicate_peer(my_other_node.id(), MODEL))
        .await;
    let targets = election::ModelTargets::default();
    let candidates = |node: &mesh::Node| {
        let node = node.clone();
        let targets = targets.clone();
        async move {
            crate::network::openai::workload_routing::ingress_candidates(
                &node,
                MODEL,
                "/v1/chat/completions",
                &targets,
            )
            .await
        }
    };
    assert_eq!(
        candidates(&node).await,
        vec![election::InferenceTarget::Remote(my_other_node.id())]
    );
    assert!(candidates(&membership).await.is_empty());
    membership.close_endpoint().await;
    my_other_node.close_endpoint().await;
    node.close_endpoint().await;
}
