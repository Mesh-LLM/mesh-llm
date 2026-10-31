//! Workload and alias admission regressions for virtual chat candidates.

use super::*;

#[tokio::test]
async fn public_alias_candidates_preserve_tool_capability() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node must start");
    let public_name = "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M";
    let mut descriptor = mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: "internal-runtime".into(),
            source_kind: mesh::ModelSourceKind::Catalog,
            canonical_ref: Some(public_name.into()),
            ..Default::default()
        },
        ..Default::default()
    };
    for level in [
        crate::models::CapabilityLevel::Supported,
        crate::models::CapabilityLevel::None,
    ] {
        descriptor.capabilities.tool_use = level;
        node.set_served_model_descriptors(vec![descriptor.clone()])
            .await;
        let candidates =
            virtual_model_candidates(&node, vec![public_name.into()], &Default::default(), None)
                .await;
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].model_id, public_name);
        assert_eq!(
            candidates[0].supports_tools,
            level != crate::models::CapabilityLevel::None
        );
    }
    node.endpoint.close().await;
}

fn workload_descriptor(
    model: &str,
    workload: mesh::ModelWorkloadClass,
) -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model.into(),
            ..Default::default()
        },
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(workload),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn chat_candidates_exclude_non_generation_workloads() {
    use mesh::ModelWorkloadClass::{CausalGeneration, Decision, Unknown};
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    node.set_served_model_descriptors(vec![
        workload_descriptor("qwen", CausalGeneration),
        workload_descriptor("laya", Decision),
        workload_descriptor("unknown", Unknown),
    ])
    .await;
    let candidates = virtual_model_candidates(
        &node,
        vec!["qwen".into(), "laya".into(), "unknown".into()],
        &Default::default(),
        None,
    )
    .await;
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.model_id.as_str())
            .collect::<Vec<_>>(),
        vec!["qwen"]
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn chat_candidates_check_each_replica_without_untargeted_fallback() {
    use mesh::ModelWorkloadClass::{CausalGeneration, Decision};
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let remote = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    node.set_hosted_models(vec!["shared".into()]).await;
    node.set_served_model_descriptors(vec![workload_descriptor("shared", Decision)])
        .await;
    remote
        .set_served_model_descriptors(vec![workload_descriptor("shared", CausalGeneration)])
        .await;
    let mut announcement =
        remote.build_local_announcement(remote.snapshot_local_announcement_data().await);
    announcement.role = mesh::NodeRole::Host { http_port: 9337 };
    announcement.serving_models = vec!["shared".into()];
    announcement.hosted_models = Some(vec!["shared".into()]);
    let mut peer = mesh::PeerInfo::from_announcement(
        remote.id(),
        remote.endpoint.addr(),
        &announcement,
        crate::crypto::OwnershipSummary::default(),
    );
    peer.admitted = true;
    node.insert_test_peer(peer).await;
    let candidates =
        virtual_model_candidates(&node, vec!["shared".into()], &Default::default(), None).await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].target_node_id,
        Some(hex::encode(remote.id().as_bytes()))
    );

    // Reverse the capabilities: a compatible local copy must not authorize
    // the incompatible remote replica via the model-level admission check.
    node.set_served_model_descriptors(vec![workload_descriptor("shared", CausalGeneration)])
        .await;
    announcement.served_model_descriptors = vec![workload_descriptor("shared", Decision)];
    let mut peer = mesh::PeerInfo::from_announcement(
        remote.id(),
        remote.endpoint.addr(),
        &announcement,
        crate::crypto::OwnershipSummary::default(),
    );
    peer.admitted = true;
    node.insert_test_peer(peer).await;
    let candidates =
        virtual_model_candidates(&node, vec!["shared".into()], &Default::default(), None).await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].target_node_id,
        Some(hex::encode(node.id().as_bytes()))
    );

    // Once every concrete replica is incompatible, do not reintroduce it via None.
    node.set_served_model_descriptors(vec![workload_descriptor("shared", Decision)])
        .await;
    assert!(
        virtual_model_candidates(&node, vec!["shared".into()], &Default::default(), None)
            .await
            .is_empty()
    );
    remote.endpoint.close().await;
    node.endpoint.close().await;
}

/// A peer that advertises a shared id for another workload class must not
/// suppress the untargeted local plugin candidate that serves chat. ndizazzo
/// [P2], PR #2560: the plugin candidate was suppressed on any *discovered*
/// remote host, even after `eligible_remote_hosts` filtered that host out on
/// workload class, leaving the committee with no candidate for a model the
/// local plugin serves.
#[tokio::test]
async fn chat_candidates_keep_a_local_plugin_when_an_ocr_peer_shares_the_id() {
    let model = "acme/shared-model";
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let remote = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();

    // The local plugin serves the id for chat.
    let plugin_manager = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    plugin_manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "chat-plugin".into(),
            endpoint_id: "chat".into(),
            protocol: Some("openai_compatible".into()),
            address: "http://127.0.0.1:1".into(),
            models: vec![model.into()],
        }])
        .await;
    node.set_plugin_manager(plugin_manager).await;

    // A peer advertises the same id as document OCR only, so nothing it serves
    // admits a chat request.
    remote
        .set_served_model_descriptors(vec![workload_descriptor(
            model,
            mesh::ModelWorkloadClass::DocumentOcr,
        )])
        .await;
    let mut announcement =
        remote.build_local_announcement(remote.snapshot_local_announcement_data().await);
    announcement.role = mesh::NodeRole::Host { http_port: 9337 };
    announcement.serving_models = vec![model.into()];
    announcement.hosted_models = Some(vec![model.into()]);
    announcement.served_model_descriptors = vec![workload_descriptor(
        model,
        mesh::ModelWorkloadClass::DocumentOcr,
    )];
    let mut peer = mesh::PeerInfo::from_announcement(
        remote.id(),
        remote.endpoint.addr(),
        &announcement,
        crate::crypto::OwnershipSummary::default(),
    );
    peer.admitted = true;
    node.insert_test_peer(peer).await;
    assert!(
        !node.hosts_for_model(model).await.is_empty(),
        "the OCR peer must be discovered by name for this to be the reported shape"
    );

    let candidates =
        virtual_model_candidates(&node, vec![model.into()], &Default::default(), None).await;
    assert_eq!(
        candidates.len(),
        1,
        "the local chat plugin must stay a candidate: {candidates:?}"
    );
    assert_eq!(
        candidates[0].target_node_id, None,
        "the plugin candidate is untargeted"
    );
    node.endpoint.close().await;
    remote.endpoint.close().await;
}
