//! Automatic (`mesh`) routing widens the host draw to every placement of the
//! chosen model. These tests pin what it must still respect: the client's
//! routing headers and each placement's own media capability.

use super::*;

/// An automatic request carrying the given routing headers, as ingress leaves
/// it after resolving the directive to `MODEL`.
fn automatic_request_with_headers(headers: &[(&str, String)]) -> BufferedHttpRequest {
    let mut request = request_with_model("/v1/chat/completions", Some("mesh"));
    crate::network::openai::ingress::maybe_enable_auto_route_hooks(&mut request, Some(MODEL));
    let lines = headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect::<String>();
    let raw = String::from_utf8(request.raw).expect("utf-8 request");
    request.raw = raw
        .replacen(
            "Host: localhost\r\n",
            &format!("Host: localhost\r\n{lines}"),
            1,
        )
        .into_bytes();
    request
}

#[tokio::test]
/// Automatic routing widens the host draw to every peer serving the model,
/// but never past the client's `x-mesh-exclude` / `x-mesh-target`. Ingress
/// already applied them to the targets it hands over: an excluded peer must
/// not be re-added, and a target allows no widening at all, so the serving
/// host is the one `x-mesh-served-by` names.
async fn automatic_candidates_keep_mesh_routing_headers() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    let excluded = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    let other = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    node.set_served_model_descriptors(vec![descriptor(
        Some(ModelWorkloadClass::CausalGeneration),
        false,
    )])
    .await;
    let local = election::InferenceTarget::Local(9337);
    let mut local_targets = election::ModelTargets::default();
    local_targets
        .targets
        .insert(MODEL.into(), vec![local.clone()]);
    let hex_of = |id: iroh::EndpointId| hex::encode(id.as_bytes());
    let candidates = |request: BufferedHttpRequest, targets: election::ModelTargets| {
        let node = node.clone();
        async move {
            super::super::super::route_model::execution_candidates(
                &node, MODEL, &request, &targets, false,
            )
            .await
        }
    };
    // Exclude a peer: this node still serves, and the draw covers the others.
    let request = automatic_request_with_headers(&[("x-mesh-exclude", hex_of(excluded))]);
    assert_eq!(
        candidates(request, local_targets.clone()).await,
        vec![local.clone(), election::InferenceTarget::Remote(other)],
        "an excluded peer must never be re-added to the draw"
    );
    // Exclude this node and a peer: ingress forces the remote pool without them.
    let request = automatic_request_with_headers(&[(
        "x-mesh-exclude",
        format!("{},{}", hex_of(node.id()), hex_of(excluded)),
    )]);
    let mut remote_pool = election::ModelTargets::default();
    remote_pool
        .targets
        .insert(MODEL.into(), vec![election::InferenceTarget::Remote(other)]);
    assert_eq!(
        candidates(request, remote_pool).await,
        vec![election::InferenceTarget::Remote(other)],
        "the forced remote pool must not regain an excluded peer"
    );
    // Target a peer: ingress hands over exactly that peer.
    let request = automatic_request_with_headers(&[("x-mesh-target", hex_of(other))]);
    let mut forced = election::ModelTargets::default();
    forced
        .targets
        .insert(MODEL.into(), vec![election::InferenceTarget::Remote(other)]);
    assert_eq!(
        candidates(request, forced).await,
        vec![election::InferenceTarget::Remote(other)],
        "a targeted request is served only by its target"
    );
    // Target this node: it serves locally and names itself in x-mesh-served-by.
    let request = automatic_request_with_headers(&[("x-mesh-target", hex_of(node.id()))]);
    assert_eq!(
        candidates(request, local_targets).await,
        vec![local],
        "a request targeting this node must not be drawn to a peer"
    );
}

#[tokio::test]
/// The automatic draw spans every placement of the chosen model, and the model
/// was chosen on fleet-wide capabilities. An image request must not land on a
/// copy of that model whose own descriptor cannot see images; a text request
/// still draws over both.
async fn automatic_image_request_draws_only_vision_placements() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    let text_only = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    let vision = iroh::SecretKey::generate().public();
    let mut vision_peer = test_peer_serving_model(vision, MODEL);
    let mut vision_descriptor = descriptor(Some(ModelWorkloadClass::CausalGeneration), false);
    vision_descriptor.capabilities.vision = CapabilityLevel::Supported;
    vision_descriptor.capabilities.multimodal = true;
    vision_peer.served_model_descriptors = vec![vision_descriptor];
    node.insert_test_peer(vision_peer).await;
    let mut pool = election::ModelTargets::default();
    pool.targets.insert(
        MODEL.into(),
        vec![
            election::InferenceTarget::Remote(text_only),
            election::InferenceTarget::Remote(vision),
        ],
    );
    let automatic = |content: serde_json::Value| {
        let mut request = request_with_model("/v1/chat/completions", Some("mesh"));
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [{"role": "user", "content": content}],
        });
        let bytes = serde_json::to_vec(&body).expect("serialize body");
        request.raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            bytes.len()
        )
        .into_bytes();
        request.body_len_bytes = bytes.len();
        request.body_bytes = Some(bytes);
        request.body_json = Some(body);
        // Ingress writes the resolved model back into the body it forwards.
        crate::network::openai::ingress::maybe_enable_auto_route_hooks(&mut request, Some(MODEL));
        request
    };
    let image = automatic(serde_json::json!([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]));
    let mut drawn =
        super::super::super::route_model::execution_candidates(&node, MODEL, &image, &pool, false)
            .await;
    drawn.sort_by_key(|target| format!("{target:?}"));
    assert_eq!(
        drawn,
        vec![election::InferenceTarget::Remote(vision)],
        "an image request draws only the vision-capable placement"
    );
    let text = automatic(serde_json::json!("hello"));
    let drawn =
        super::super::super::route_model::execution_candidates(&node, MODEL, &text, &pool, false)
            .await;
    assert_eq!(drawn.len(), 2, "a text request draws over both placements");
}

#[tokio::test]
/// The automatic draw widens past the pool ingress narrowed to the hosts the
/// request's trust policy permits, so it must apply the policy itself: under
/// the `receipts` tier only a peer that advertises receipts for the model is
/// drawn, never this node's runtime that issues none, nor a peer without them.
async fn automatic_draw_under_the_receipts_tier_keeps_only_qualifying_hosts() {
    // A name this node can tie to a Hugging Face file, so it can audit peers.
    const TIED: &str = "acme/receipted-chat:Q4_K_M";
    async fn draw(
        node: &mesh::Node,
        request: &BufferedHttpRequest,
        targets: &election::ModelTargets,
    ) -> Vec<election::InferenceTarget> {
        super::super::super::route_model::execution_candidates(node, TIED, request, targets, false)
            .await
    }
    let chat = |receipts: bool| {
        let mut descriptor = descriptor(Some(ModelWorkloadClass::CausalGeneration), false);
        descriptor.identity.model_name = TIED.into();
        descriptor.receipts = receipts;
        descriptor
    };
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    node.set_served_model_descriptors(vec![chat(false)]).await;
    let peer_with = |receipts: bool| {
        let id = iroh::SecretKey::generate().public();
        let mut peer = test_peer_serving_model(id, TIED);
        peer.served_model_descriptors = vec![chat(receipts)];
        (id, peer)
    };
    let (receipted, receipted_peer) = peer_with(true);
    let (plain, plain_peer) = peer_with(false);
    node.insert_test_peer(receipted_peer).await;
    node.insert_test_peer(plain_peer).await;
    let mut local_targets = election::ModelTargets::default();
    local_targets
        .targets
        .insert(TIED.into(), vec![election::InferenceTarget::Local(9337)]);
    let mut request = request_with_model("/v1/chat/completions", Some("mesh"));
    crate::network::openai::ingress::maybe_enable_auto_route_hooks(&mut request, Some(TIED));
    assert_eq!(
        draw(&node, &request, &local_targets).await.len(),
        3,
        "with no policy the draw spans this node and both peers"
    );
    request.trust = crate::network::request_trust::RequestTrust::Receipts { audit: true };
    assert_eq!(
        draw(&node, &request, &local_targets).await,
        vec![election::InferenceTarget::Remote(receipted)],
        "under the receipts tier only the peer with receipts is drawn, not {plain:?} or this node"
    );
}
