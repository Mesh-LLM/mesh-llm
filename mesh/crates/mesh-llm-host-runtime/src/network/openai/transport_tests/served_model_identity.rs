//! The served-model identity `backend_selected` and the peer response headers
//! report: which descriptor, on which runtime instance, for which request.
use super::routing::test_peer_serving_model;
use super::*;

/// `backend_selected` reads the model identity from the target that routing
/// actually selected: this node's own descriptor for a local target, and the
/// named peer's gossiped descriptor for a remote one, never another node's
/// same-named model.
#[tokio::test]
async fn backend_selected_identity_comes_from_the_selected_target() -> Result<()> {
    use crate::network::openai::exchange_admission::selected_model_identity;

    let model = "shared-model";
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker).await?;
    *node.served_model_descriptors.lock().await = vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model.to_owned(),
            canonical_ref: Some("local-ref".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    }];
    node.load_model_for_tests(model, 1, "local-digest").await;
    let peer_a = iroh::SecretKey::generate().public();
    let peer_b = iroh::SecretKey::generate().public();
    for (peer_id, canonical_ref) in [(peer_a, "peer-a-ref"), (peer_b, "peer-b-ref")] {
        let mut peer = test_peer_serving_model(peer_id, model);
        peer.served_model_descriptors[0].identity.canonical_ref = Some(canonical_ref.to_owned());
        node.insert_test_peer(peer).await;
    }

    let local = selected_model_identity(&node, &election::InferenceTarget::Local(1), model, CHAT)
        .await
        .expect("local descriptor");
    assert_eq!(local.canonical_ref.as_deref(), Some("local-ref"));
    assert_eq!(local.weights_digest.as_deref(), Some("local-digest"));

    let remote = selected_model_identity(
        &node,
        &election::InferenceTarget::Remote(peer_b),
        model,
        CHAT,
    )
    .await
    .expect("peer B descriptor");
    assert_eq!(remote.canonical_ref.as_deref(), Some("peer-b-ref"));
    assert_eq!(remote.weights_digest, None, "gossip never carries a digest");

    let unknown_peer = iroh::SecretKey::generate().public();
    assert!(
        selected_model_identity(
            &node,
            &election::InferenceTarget::Remote(unknown_peer),
            model,
            CHAT
        )
        .await
        .is_none()
    );
    assert!(
        selected_model_identity(
            &node,
            &election::InferenceTarget::Remote(peer_a),
            "other-model",
            CHAT
        )
        .await
        .is_none()
    );
    assert!(
        selected_model_identity(&node, &election::InferenceTarget::None, model, CHAT)
            .await
            .is_none()
    );
    node.endpoint.close().await;
    Ok(())
}

const CHAT: &str = "/v1/chat/completions";

/// A node can advertise a native embedding model and a chat model under one
/// name. A chat request is attributed to the chat descriptor only, and an
/// endpoint that matches both (or neither) reports no identity at all.
#[tokio::test]
async fn backend_selected_identity_matches_the_requests_workload_or_reports_none() -> Result<()> {
    use crate::network::openai::exchange_admission::selected_model_identity;

    let model = "shared-name";
    let descriptor = |canonical_ref: &str, workload_class: Option<mesh::ModelWorkloadClass>| {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model.to_owned(),
                canonical_ref: Some(canonical_ref.to_owned()),
                ..Default::default()
            },
            metadata: workload_class.map(|workload_class| mesh::ServedModelMetadata {
                workload_class: Some(workload_class),
                ..Default::default()
            }),
            ..Default::default()
        }
    };
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker).await?;
    let peer_id = iroh::SecretKey::generate().public();
    let mut peer = test_peer_serving_model(peer_id, model);
    peer.served_model_descriptors = vec![
        descriptor("embedding-ref", Some(mesh::ModelWorkloadClass::Embedding)),
        descriptor("chat-ref", None),
    ];
    node.insert_test_peer(peer).await;
    let target = election::InferenceTarget::Remote(peer_id);

    let chat = selected_model_identity(&node, &target, model, CHAT)
        .await
        .expect("the chat descriptor is the only one serving chat");
    assert_eq!(chat.canonical_ref.as_deref(), Some("chat-ref"));
    let embedding = selected_model_identity(&node, &target, model, "/v1/embeddings")
        .await
        .expect("the embedding descriptor is the only one serving embeddings");
    assert_eq!(embedding.canonical_ref.as_deref(), Some("embedding-ref"));

    let mut ambiguous = test_peer_serving_model(peer_id, model);
    ambiguous.served_model_descriptors = vec![
        descriptor("first-chat", None),
        descriptor("second-chat", None),
    ];
    node.insert_test_peer(ambiguous).await;
    assert!(
        selected_model_identity(&node, &target, model, CHAT)
            .await
            .is_none(),
        "two chat descriptors under one name are ambiguous"
    );
    node.endpoint.close().await;
    Ok(())
}

/// The node-wide descriptor keeps one digest per model name. A request that
/// reaches an older runtime instance of that name (its file replaced between
/// loads) is not told the newer file's digest, in the exchange event or in
/// the headers returned to a peer.
#[tokio::test]
async fn a_weights_digest_is_reported_only_for_the_instance_it_was_computed_for() -> Result<()> {
    use crate::network::openai::ServedModelHeaders;
    use crate::network::openai::exchange_admission::selected_model_identity;

    let model = "reloaded";
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker).await?;
    *node.served_model_descriptors.lock().await = vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model.to_owned(),
            canonical_ref: Some("org/repo@main:Q4_K_M".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    }];
    let digest_on = |port| {
        let node = &node;
        async move {
            selected_model_identity(node, &election::InferenceTarget::Local(port), model, CHAT)
                .await
                .and_then(|identity| identity.weights_digest)
        }
    };

    node.load_model_for_tests(model, 1, "old-bytes").await;
    assert_eq!(digest_on(1).await.as_deref(), Some("old-bytes"));

    node.load_model_for_tests(model, 2, "new-bytes").await;
    assert_eq!(digest_on(2).await.as_deref(), Some("new-bytes"));
    assert_eq!(
        digest_on(1).await,
        None,
        "the older instance's bytes are unknown"
    );
    assert_eq!(
        ServedModelHeaders::for_served_model(&node, model, Some(1)).await,
        Some(ServedModelHeaders {
            weights_digest: None,
            canonical_ref: Some("org/repo@main:Q4_K_M".to_owned()),
        })
    );
    node.endpoint.close().await;
    Ok(())
}

/// A request may name a model by its public ID, which routing accepts as
/// well as the runtime `model_name`. `backend_selected` and the served-model
/// headers resolve that alias to the same descriptor routing selected.
#[tokio::test]
async fn backend_selected_identity_resolves_the_public_model_id() -> Result<()> {
    use crate::network::openai::ServedModelHeaders;
    use crate::network::openai::exchange_admission::selected_model_identity;

    let runtime_name = "Qwen3-8B-Q4_K_M";
    let public_id = "Qwen/Qwen3-8B-GGUF@revabc:Q4_K_M";
    let identity = |canonical_ref: &str, weights_digest: Option<&str>| mesh::ServedModelIdentity {
        model_name: runtime_name.to_owned(),
        source_kind: mesh::ModelSourceKind::HuggingFace,
        repository: Some("Qwen/Qwen3-8B-GGUF".to_owned()),
        revision: Some("revabc".to_owned()),
        artifact: Some("Qwen3-8B-Q4_K_M.gguf".to_owned()),
        canonical_ref: Some(canonical_ref.to_owned()),
        weights_digest: weights_digest.map(str::to_owned),
        ..Default::default()
    };
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker).await?;
    *node.served_model_descriptors.lock().await = vec![mesh::ServedModelDescriptor {
        identity: identity("local-ref", None),
        ..Default::default()
    }];
    node.load_model_for_tests(runtime_name, 1, "local-digest")
        .await;
    let peer_id = iroh::SecretKey::generate().public();
    let mut peer = test_peer_serving_model(peer_id, runtime_name);
    peer.served_model_descriptors[0].identity = identity("peer-ref", None);
    assert!(
        crate::mesh::routes_model(&peer, public_id),
        "routing accepts the public ID"
    );
    node.insert_test_peer(peer).await;

    let local =
        selected_model_identity(&node, &election::InferenceTarget::Local(1), public_id, CHAT)
            .await
            .expect("local descriptor by public ID");
    assert_eq!(local.canonical_ref.as_deref(), Some("local-ref"));
    assert_eq!(local.weights_digest.as_deref(), Some("local-digest"));

    let remote = selected_model_identity(
        &node,
        &election::InferenceTarget::Remote(peer_id),
        public_id,
        CHAT,
    )
    .await
    .expect("peer descriptor by public ID");
    assert_eq!(remote.canonical_ref.as_deref(), Some("peer-ref"));
    assert_eq!(remote.weights_digest, None, "gossip never carries a digest");

    assert_eq!(
        ServedModelHeaders::for_served_model(&node, public_id, Some(1)).await,
        Some(ServedModelHeaders {
            weights_digest: Some("local-digest".to_owned()),
            canonical_ref: Some("local-ref".to_owned()),
        })
    );
    node.endpoint.close().await;
    Ok(())
}
