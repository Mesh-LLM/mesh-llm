use super::super::tests::{ingress_client_socket_pair, test_endpoint_id, test_remote_peer};
use super::*;
use mesh_llm_events::logging::identifiers::RequestId;

const MODEL: &str = "acme/chat-model:Q4_K_M";

fn request(path: &str, trust: Option<&str>) -> proxy::BufferedHttpRequest {
    let trust = trust.map_or(String::new(), |value| format!("x-mesh-trust: {value}\r\n"));
    let raw = format!("POST {path} HTTP/1.1\r\nHost: t\r\n{trust}Content-Length: 2\r\n\r\n{{}}");
    proxy::BufferedHttpRequest {
        raw: raw.into_bytes(),
        method: "POST".to_owned(),
        path: path.to_owned(),
        client_path: path.to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 2,
        completion_tokens: None,
        stream: None,
        model_name: Some(MODEL.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
        exchange_observation_id: None,
    }
}

/// A peer serving `MODEL`, with or without receipts.
fn peer(seed: u32, receipts: bool) -> mesh::PeerInfo {
    let mut peer = test_remote_peer(seed, MODEL);
    peer.served_model_descriptors = vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: MODEL.to_owned(),
            ..Default::default()
        },
        receipts,
        ..Default::default()
    }];
    peer
}

struct Mesh {
    node: mesh::Node,
    receipted: iroh::EndpointId,
    plain: iroh::EndpointId,
    targets: election::ModelTargets,
    affinity: affinity::AffinityRouter,
}

/// This node serves `MODEL` without receipts beside a peer that serves it
/// with them and one that does not.
async fn mesh_with_one_receipted_peer() -> Mesh {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node should start");
    let receipted = peer(0x31, true);
    let plain = peer(0x32, false);
    let (receipted_id, plain_id) = (receipted.id, plain.id);
    node.insert_test_peer(receipted).await;
    node.insert_test_peer(plain).await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.to_owned(),
        vec![
            election::InferenceTarget::Local(9000),
            election::InferenceTarget::Remote(receipted_id),
            election::InferenceTarget::Remote(plain_id),
        ],
    );
    Mesh {
        node,
        receipted: receipted_id,
        plain: plain_id,
        targets,
        affinity: affinity::AffinityRouter::new(),
    }
}

impl Mesh {
    fn ctx(&self) -> IngressRouteContext<'_> {
        IngressRouteContext {
            node: &self.node,
            targets: &self.targets,
            affinity: &self.affinity,
            plugin_manager: None,
            requested_by_node_id: None,
            exchange_channel: None,
        }
    }
}

#[tokio::test]
async fn a_receipts_request_routes_only_to_receipted_runtimes() {
    let mesh = mesh_with_one_receipted_peer().await;
    let mut excluded = vec![test_endpoint_id(0x07)];
    let request = request("/v1/chat/completions", Some("receipts"));
    let Ok(Some(targets)) = restrict(&mesh.ctx(), &request, MODEL, &mut excluded).await else {
        panic!("a receipted peer serves the model");
    };
    assert_eq!(
        targets.targets[MODEL],
        vec![election::InferenceTarget::Remote(mesh.receipted)]
    );
    assert!(excluded.contains(&mesh.node.id()));
    assert!(excluded.contains(&mesh.plain));
    assert!(excluded.contains(&test_endpoint_id(0x07)));
    assert!(!excluded.contains(&mesh.receipted));

    mesh.node.set_issues_receipts(MODEL, true);
    let mut excluded = Vec::new();
    let Ok(Some(targets)) = restrict(&mesh.ctx(), &request, MODEL, &mut excluded).await else {
        panic!("this node serves the model with receipts");
    };
    assert_eq!(targets.targets[MODEL].len(), 2);
    assert_eq!(excluded, vec![mesh.plain]);
}

#[tokio::test]
async fn requests_without_the_policy_route_as_before() {
    let mesh = mesh_with_one_receipted_peer().await;
    let mut excluded = Vec::new();
    for trust in [None, Some("any")] {
        let request = request("/v1/chat/completions", trust);
        let restricted = restrict(&mesh.ctx(), &request, MODEL, &mut excluded).await;
        assert!(matches!(restricted, Ok(None)));
    }
    assert!(excluded.is_empty());
}

/// What the client reads when `request` for `model` is rejected under its
/// trust policy.
async fn rejection_answer(
    mesh: &Mesh,
    request: &proxy::BufferedHttpRequest,
    model: &str,
) -> String {
    use tokio::io::AsyncReadExt;
    let Err(rejection) = restrict(&mesh.ctx(), request, model, &mut Vec::new()).await else {
        panic!("the policy must reject this request");
    };
    let (server, mut client) = ingress_client_socket_pair().await;
    reject(
        ClientStream::from(server),
        &mesh.ctx(),
        model,
        rejection,
        OpenAiRouteObserver::default(),
    )
    .await;
    let mut answer = String::new();
    client
        .read_to_string(&mut answer)
        .await
        .expect("read the rejection");
    answer
}

#[tokio::test]
async fn the_policy_fails_closed() {
    let mesh = mesh_with_one_receipted_peer().await;
    // no candidate satisfies the policy: 409 with the code clients match on
    for (path, model) in [
        ("/v1/embeddings", MODEL),
        ("/v1/chat/completions", "other-model"),
    ] {
        let answer = rejection_answer(&mesh, &request(path, Some("receipts")), model).await;
        assert!(answer.starts_with("HTTP/1.1 409"), "{answer}");
        assert!(answer.contains(request_trust::UNSATISFIED_CODE), "{answer}");
    }

    // a policy this node cannot read: 400, without that code
    let unknown = request("/v1/chat/completions", Some("fast"));
    let answer = rejection_answer(&mesh, &unknown, MODEL).await;
    assert!(answer.starts_with("HTTP/1.1 400"), "{answer}");
    assert!(
        !answer.contains(request_trust::UNSATISFIED_CODE),
        "{answer}"
    );
}

#[tokio::test]
async fn a_request_a_peer_sent_is_not_routed_under_a_policy() {
    let mesh = mesh_with_one_receipted_peer().await;
    let ctx = IngressRouteContext {
        requested_by_node_id: Some(hex::encode(mesh.receipted.as_bytes())),
        ..mesh.ctx()
    };
    let request = request("/v1/embeddings", Some("receipts"));
    let restricted = restrict(&ctx, &request, MODEL, &mut Vec::new()).await;
    assert!(matches!(restricted, Ok(None)));
}
