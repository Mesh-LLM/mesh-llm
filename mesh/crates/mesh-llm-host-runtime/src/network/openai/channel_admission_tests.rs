//! Hard channel admission must survive automatic readiness and committee fallbacks.

use super::{auto_route, request_parse, virtual_model, workload_routing};
use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::{self, Node, NodeRole};
use crate::network::{affinity::AffinityRouter, router::RoutingCandidate};
use mesh_llm_types::mesh::ChannelRequirement;

async fn insert_provider(node: &Node, models: &[(&str, bool)]) -> iroh::EndpointId {
    let id = iroh::SecretKey::generate().public();
    let mut ann = node.build_local_announcement(node.snapshot_local_announcement_data().await);
    ann.role = NodeRole::Host { http_port: 9337 };
    ann.serving_models = models.iter().map(|(model, _)| (*model).into()).collect();
    ann.hosted_models = Some(ann.serving_models.clone());
    ann.served_model_descriptors = models
        .iter()
        .map(|(model, gated)| mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: (*model).into(),
                ..Default::default()
            },
            requires: if *gated {
                vec![ChannelRequirement {
                    channel: "fixture.v1".into(),
                    terms: vec![1],
                }]
            } else {
                Vec::new()
            },
            ..Default::default()
        })
        .collect();
    let mut peer = mesh::PeerInfo::from_announcement(
        id,
        iroh::EndpointAddr::new(id),
        &ann,
        crate::crypto::OwnershipSummary::default(),
    );
    peer.admitted = true;
    node.insert_test_peer(peer).await;
    id
}

async fn chat_request() -> request_parse::BufferedHttpRequest {
    use tokio::io::AsyncWriteExt;

    let (mut sender, mut reader) = tokio::io::duplex(1024);
    sender.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}").await.unwrap();
    request_parse::read_http_request(&mut reader).await.unwrap()
}

#[tokio::test]
async fn automatic_admission_excludes_channel_gates_before_readiness_fallback() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let peer = insert_provider(&node, &[("gated", true), ("free", false)]).await;
    let mut targets = ModelTargets::default();
    // An elected table can outlive the announcement that added the gate.
    for model in ["gated", "free"] {
        targets
            .targets
            .insert(model.into(), vec![InferenceTarget::Remote(peer)]);
    }
    let available =
        ["gated", "free"].map(|model| RoutingCandidate::unscored(model, Default::default()));
    let request = chat_request().await;
    let affinity = AffinityRouter::new();
    for targets in [Some(&targets), None] {
        let (admitted, ready) = auto_route::readiness::ready_models(
            &node, targets, &request, &available, None, &affinity,
        )
        .await;
        assert_eq!(
            admitted
                .iter()
                .map(|candidate| candidate.name)
                .collect::<Vec<_>>(),
            vec!["free"],
            "hard exclusions must happen before the soft readiness fallback"
        );
        let pool = auto_route::pool_for_ready_models(&admitted, &ready);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].name, "free");

        let (admitted, ready) = auto_route::readiness::ready_models(
            &node,
            targets,
            &request,
            &available[..1],
            None,
            &affinity,
        )
        .await;
        assert!(auto_route::pool_for_ready_models(&admitted, &ready).is_empty());
    }
    assert_eq!(
        node.peers().await[0].served_model_descriptors[0]
            .requires
            .len(),
        1
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn a_free_replica_does_not_authorize_a_channel_gated_target() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let gated = insert_provider(&node, &[("shared", true)]).await;
    let free = insert_provider(&node, &[("shared", false)]).await;
    let eligible = workload_routing::eligible_targets(
        &node,
        "shared",
        "/v1/chat/completions",
        &[
            InferenceTarget::Remote(gated),
            InferenceTarget::Remote(free),
        ],
    )
    .await;
    assert_eq!(eligible, vec![InferenceTarget::Remote(free)]);
    let candidates = virtual_model::virtual_model_candidates(
        &node,
        vec!["shared".into()],
        &Default::default(),
        None,
    )
    .await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].target_node_id,
        Some(hex::encode(free.as_bytes()))
    );
    node.state.lock().await.peers.remove(&free);
    assert!(
        virtual_model::virtual_model_candidates(
            &node,
            vec!["shared".into()],
            &Default::default(),
            None,
        )
        .await
        .is_empty(),
        "a gated-only model must not return as an untargeted committee candidate"
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn stale_remote_targets_cannot_bypass_peer_admission() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let departed = insert_provider(&node, &[("shared", false)]).await;
    let quarantined = insert_provider(&node, &[("shared", false)]).await;
    let legacy = insert_provider(&node, &[("shared", false)]).await;
    {
        let mut state = node.state.lock().await;
        state.peers.remove(&departed);
        state.peers.get_mut(&quarantined).unwrap().admitted = false;
        state
            .peers
            .get_mut(&legacy)
            .unwrap()
            .served_model_descriptors
            .clear();
    }
    let eligible = workload_routing::eligible_targets(
        &node,
        "shared",
        "/v1/chat/completions",
        &[departed, quarantined, legacy].map(InferenceTarget::Remote),
    )
    .await;
    assert_eq!(eligible, vec![InferenceTarget::Remote(legacy)]);

    let mut targets = ModelTargets::default();
    targets
        .targets
        .insert("shared".into(), vec![InferenceTarget::Remote(departed)]);
    assert_eq!(
        workload_routing::ingress_candidates(
            &node,
            "shared",
            "/v1/chat/completions",
            &targets,
            crate::network::request_trust::RequestTrust::UNRESTRICTED
        )
        .await,
        vec![InferenceTarget::Remote(legacy)],
        "a stale elected target must not shadow an admitted legacy replica"
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn a_channel_gated_remote_does_not_hide_local_native_or_plugin_execution() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let peer = insert_provider(&node, &[("shared", true)]).await;
    node.set_hosted_models(vec!["shared".into()]).await;
    let eligible = workload_routing::eligible_targets(
        &node,
        "shared",
        "/v1/chat/completions",
        &[InferenceTarget::Remote(peer), InferenceTarget::Local(9000)],
    )
    .await;
    assert_eq!(eligible, vec![InferenceTarget::Local(9000)]);
    let candidates = virtual_model::virtual_model_candidates(
        &node,
        vec!["shared".into()],
        &Default::default(),
        None,
    )
    .await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].target_node_id,
        Some(hex::encode(node.id().as_bytes()))
    );

    node.set_hosted_models(Vec::new()).await;
    let plugins = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    plugins
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "chat".into(),
            endpoint_id: "chat".into(),
            protocol: Some("openai_compatible".into()),
            address: "http://127.0.0.1:1".into(),
            models: vec!["shared".into()],
        }])
        .await;
    node.set_plugin_manager(plugins).await;
    let candidates = virtual_model::virtual_model_candidates(
        &node,
        vec!["shared".into()],
        &Default::default(),
        None,
    )
    .await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].target_node_id, None);
    node.endpoint.close().await;
}
