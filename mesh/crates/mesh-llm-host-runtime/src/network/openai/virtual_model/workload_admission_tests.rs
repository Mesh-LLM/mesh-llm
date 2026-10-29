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
