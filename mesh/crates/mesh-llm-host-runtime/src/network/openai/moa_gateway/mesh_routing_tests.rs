//! Regression (CodeRabbit, PR #1671 round 2 follow-up): `x-mesh-target`/
//! `x-mesh-exclude` must not be rejected before `try_handle_moa` knows
//! whether a committee will actually be convened for `model: "mesh"`.
//!
//! Before this fix, the ingress-level pre-check
//! (`enforce_mesh_routing_headers_before_dispatch` /
//! `mesh_routing_unsupported_dispatch_kind`) rejected every `model: "mesh"`
//! request carrying either header with 409, before `try_handle_moa` ever
//! ran -- including the case where `try_handle_moa` was about to degrade
//! `model: "mesh"` to a single concrete model it could have routed with the
//! headers honored. These tests exercise `try_handle_moa` directly: a real
//! committee (fabricated via `fleet_sim_tests::node_with_fleet`, no sockets)
//! must still reject, but a degrade (zero admitted workers) must not.

use super::fleet_sim_tests::{BIG_MODELS, node_with_fleet};
use super::{MoaRoutingContext, try_handle_moa};
use crate::inference::election;
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::transport as proxy;
use crate::network::openai::transport::ResponseAdapter;
use mesh_llm_events::logging::identifiers::RequestId;

fn moa_request_with_target(target_hex: &str) -> proxy::BufferedHttpRequest {
    let body = serde_json::json!({
        "model": "mesh",
        "messages": [{ "role": "user", "content": "hello" }],
    });
    let body_bytes = serde_json::to_vec(&body).expect("serialize body");
    let mut raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         x-mesh-target: {target_hex}\r\nContent-Length: {}\r\n\r\n",
        body_bytes.len()
    )
    .into_bytes();
    raw.extend_from_slice(&body_bytes);
    proxy::BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body_bytes.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some("mesh".to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
    }
}

async fn test_stream_pair() -> (ClientStream, tokio::net::TcpStream) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let client = tokio::net::TcpStream::connect(addr);
    let server = async { listener.accept().await.map(|(stream, _)| stream) };
    let (client_side, server_side) = tokio::join!(client, server);
    let client_side = client_side.expect("connect");
    let tcp_stream: ClientStream = server_side.expect("accept").into();
    (tcp_stream, client_side)
}

/// A real committee (>=1 admitted worker) must still refuse to honor
/// `x-mesh-target`/`x-mesh-exclude` -- a fan-out across every admitted
/// worker cannot single out or exclude one peer. Deleting the
/// `mesh_routing_requested` guard in `try_handle_moa` (or hardcoding it to
/// `false`) turns this red: the request would instead run `run_moa_turn`
/// and hang/fail on a real dial to the fabricated peer.
#[tokio::test]
async fn try_handle_moa_rejects_routing_headers_once_a_committee_is_convened() {
    use tokio::io::AsyncReadExt;

    let node = node_with_fleet(&[(BIG_MODELS[0], 1), (BIG_MODELS[1], 1)]).await;
    let targets = election::ModelTargets::default();
    let affinity = crate::network::affinity::AffinityRouter::default();
    let mut request = moa_request_with_target("aabbccdd");
    let (tcp_stream, mut client_side) = test_stream_pair().await;

    let result = try_handle_moa(
        &node,
        tcp_stream,
        &mut request,
        Some("mesh"),
        MoaRoutingContext {
            targets: Some(&targets),
            required_tokens: None,
            affinity: &affinity,
            mesh_routing_requested: true,
        },
        crate::logging::OpenAiRouteObserver::default(),
    )
    .await;

    assert!(
        matches!(result, super::MoaDispatchResult::Responded(409)),
        "expected a 409 once a committee is convened, got a different MoaDispatchResult variant"
    );

    let mut response = Vec::new();
    client_side
        .read_to_end(&mut response)
        .await
        .expect("read response");
    let response_text = String::from_utf8_lossy(&response);
    assert!(
        response_text.starts_with("HTTP/1.1 409"),
        "expected 409 on the wire, got: {response_text}"
    );
    assert!(
        response_text.contains("committee"),
        "expected the committee-specific rejection message, got: {response_text}"
    );
}

/// Zero admitted workers must degrade `model: "mesh"` to a single concrete
/// model and hand the stream back as `Passthrough` regardless of
/// `mesh_routing_requested` -- the caller (`route_request`) re-parses and
/// honors the headers against the rewritten model. Restoring the old
/// eager rejection (checking `mesh_routing_requested` before knowing
/// whether a committee will form) turns this red: the request would be
/// rejected with 409 even though no committee was ever going to run.
#[tokio::test]
async fn try_handle_moa_degrades_and_continues_when_no_committee_can_form() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node should start");
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "solo/only-model:Q4_K_M".to_string(),
        vec![election::InferenceTarget::Remote(iroh::EndpointId::from(
            iroh::SecretKey::from_bytes(&[7u8; 32]).public(),
        ))],
    );
    let affinity = crate::network::affinity::AffinityRouter::default();
    let mut request = moa_request_with_target("aabbccdd");
    let (tcp_stream, _client_side) = test_stream_pair().await;

    let result = try_handle_moa(
        &node,
        tcp_stream,
        &mut request,
        Some("mesh"),
        MoaRoutingContext {
            targets: Some(&targets),
            required_tokens: None,
            affinity: &affinity,
            mesh_routing_requested: true, // must NOT block the degrade
        },
        crate::logging::OpenAiRouteObserver::default(),
    )
    .await;

    assert!(
        matches!(result, super::MoaDispatchResult::Passthrough(_)),
        "expected Passthrough on degrade even with routing headers present, got a different \
         MoaDispatchResult variant"
    );
    assert_eq!(
        request.model_name.as_deref(),
        Some("solo/only-model:Q4_K_M"),
        "degrade must rewrite the virtual model so route_request can honor the routing headers \
         against the real target"
    );
}
