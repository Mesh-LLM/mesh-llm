use super::super::*;
use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
use std::{collections::BTreeMap, sync::Arc};

struct NoRpc;
impl PluginRpcBridge for NoRpc {
    fn handle_request(
        &self,
        _: String,
        _: String,
        _: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        Box::pin(async { unreachable!("channel declarations need no RPC") })
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}
async fn manager() -> PluginManager {
    let manager = PluginManager::for_test_bridge(&["gate"], Arc::new(NoRpc));
    manager
        .set_test_manifests(BTreeMap::from([(
            "gate".into(),
            mesh_llm_plugin::plugin_manifest![mesh_llm_plugin::mesh_channel("fixture.v1")],
        )]))
        .await;
    manager
}
fn request(terms: Vec<u8>) -> proto::ModelRequirementRequest {
    proto::ModelRequirementRequest {
        model: "gated".into(),
        requires: vec![proto::ChannelRequirement {
            channel: "fixture.v1".into(),
            terms,
        }],
    }
}

#[tokio::test]
async fn requirements_round_trip_announce_gate_withdraw_and_local_bypass() {
    let seller = super::make_test_node(NodeRole::Worker).await.unwrap();
    *seller.plugin_manager.lock().await = Some(manager().await);
    seller
        .set_hosted_models(vec!["gated".into(), "free".into()])
        .await;
    seller
        .apply_model_requirement("gate", request(vec![0, 255, 13]))
        .await
        .unwrap();
    let ann = seller.build_local_announcement(seller.snapshot_local_announcement_data().await);
    let descriptors = ann.served_model_descriptors;
    assert_eq!(
        descriptors
            .iter()
            .find(|d| d.identity.model_name == "gated")
            .unwrap()
            .requires[0]
            .terms,
        vec![0, 255, 13]
    );
    // Exercise the actual protobuf codec and host descriptor conversions.
    let ann = seller.build_local_announcement(seller.snapshot_local_announcement_data().await);
    let encoded = crate::protocol::local_ann_to_proto_ann(&ann);
    let bytes = prost::Message::encode_to_vec(&encoded);
    let decoded =
        <crate::proto::node::PeerAnnouncement as prost::Message>::decode(bytes.as_slice()).unwrap();
    let (_, ann) = crate::protocol::proto_ann_to_local(&decoded).unwrap();
    let buyer = super::make_test_node(NodeRole::Worker).await.unwrap();
    let mut peer = super::make_test_peer_info(seller.id());
    peer.role = NodeRole::Host { http_port: 9337 };
    peer.hosted_models_known = true;
    peer.hosted_models = vec!["gated".into(), "free".into()];
    peer.serving_models = peer.hosted_models.clone();
    peer.served_model_descriptors = ann.served_model_descriptors;
    let gated = peer
        .served_model_descriptors
        .iter_mut()
        .find(|d| d.identity.model_name == "gated")
        .unwrap();
    gated.capabilities.vision = crate::models::capabilities::CapabilityLevel::Supported;
    gated.capabilities.audio = crate::models::capabilities::CapabilityLevel::Supported;
    peer.rtt_ms = Some(1);
    buyer.state.lock().await.test_peer_liveness.insert(peer.id);
    buyer.state.lock().await.peers.insert(peer.id, peer.clone());
    assert!(!buyer.models_being_served().await.contains(&"gated".into()));
    assert!(buyer.hosts_for_model("gated").await.is_empty());
    assert!(
        buyer
            .routable_peers()
            .await
            .iter()
            .all(|p| !routes_model(p, "gated"))
    );
    assert!(
        buyer
            .routing_table()
            .await
            .hosts
            .iter()
            .all(|r| r.model != "gated")
    );
    assert!(buyer.models_being_served().await.contains(&"free".into()));
    assert!(
        buyer.peers().await[0]
            .hosted_models
            .contains(&"gated".into()),
        "raw advertisement retained"
    );
    assert!(
        crate::inference::consult::find_vision_peer(&buyer, "")
            .await
            .is_none()
    );
    assert!(
        crate::inference::consult::find_audio_peer(&buyer, "")
            .await
            .is_none()
    );
    assert!(
        buyer.routable_peers().await[0]
            .served_model_descriptors
            .iter()
            .all(|d| d.identity.model_name != "gated")
    );
    buyer.set_hosted_models(vec!["gated".into()]).await;
    assert!(
        buyer.models_being_served().await.contains(&"gated".into()),
        "local target unaffected"
    );
    buyer.set_hosted_models(Vec::new()).await;
    *buyer.plugin_manager.lock().await = Some(manager().await);
    assert!(buyer.models_being_served().await.contains(&"gated".into()));
    assert_eq!(buyer.hosts_for_model("gated").await, vec![seller.id()]);
    assert_eq!(
        crate::inference::consult::find_vision_peer(&buyer, "").await,
        Some(seller.id())
    );
    assert_eq!(
        crate::inference::consult::find_audio_peer(&buyer, "").await,
        Some(seller.id())
    );
    *buyer.plugin_manager.lock().await = None;
    assert!(
        !buyer.models_being_served().await.contains(&"gated".into()),
        "plugin removal is immediately reflected"
    );
    seller
        .apply_model_requirement(
            "gate",
            proto::ModelRequirementRequest {
                model: "gated".into(),
                requires: Vec::new(),
            },
        )
        .await
        .unwrap();
    assert!(
        seller
            .build_local_announcement(seller.snapshot_local_announcement_data().await)
            .served_model_descriptors
            .iter()
            .all(|d| d.requires.is_empty())
    );
    seller.close_endpoint().await;
    buyer.close_endpoint().await;
}

#[tokio::test]
async fn requirement_owner_validation_size_limit_and_exit() {
    let node = super::make_test_node(NodeRole::Worker).await.unwrap();
    *node.plugin_manager.lock().await = Some(manager().await);
    assert!(
        node.apply_model_requirement("impostor", request(Vec::new()))
            .await
            .is_err()
    );
    assert!(
        node.apply_model_requirement("gate", request(vec![0; 1025]))
            .await
            .is_err()
    );
    let mut oversized = request(vec![0; 512]);
    oversized.requires = vec![oversized.requires[0].clone(); 9];
    assert!(
        node.apply_model_requirement("gate", oversized.clone())
            .await
            .is_err()
    );
    oversized.requires.truncate(8);
    oversized.requires[0].terms.push(0);
    assert!(
        node.apply_model_requirement("gate", oversized)
            .await
            .is_err()
    );
    node.apply_model_requirement("gate", request(vec![0; 1024]))
        .await
        .unwrap();
    let mut invalid = request(Vec::new());
    invalid.requires[0].channel = "undeclared.v1".into();
    assert!(node.apply_model_requirement("gate", invalid).await.is_err());
    assert_eq!(
        node.model_requirements.lock().await["gate"]["gated"][0]
            .terms
            .len(),
        1024
    );
    node.forward_plugin_event(crate::plugin::PluginMeshEvent::ModelRequirementsWithdraw {
        plugin_id: "gate".into(),
    })
    .await
    .unwrap();
    assert!(node.model_requirements.lock().await.is_empty());
    node.close_endpoint().await;
}

#[tokio::test]
async fn requirement_attaches_by_public_model_id() {
    let node = super::make_test_node(NodeRole::Worker).await.unwrap();
    *node.plugin_manager.lock().await = Some(manager().await);
    let descriptor = ServedModelDescriptor {
        identity: ServedModelIdentity {
            model_name: "local-gguf/sha256-adc287a9".into(),
            source_kind: ModelSourceKind::HuggingFace,
            repository: Some("unsloth/Qwen3-0.6B-GGUF".into()),
            artifact: Some("Qwen3-0.6B-Q4_K_M.gguf".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let public_id = public_model_id_from_identity(&descriptor.identity).unwrap();
    node.upsert_served_model_descriptor(descriptor).await;
    node.set_hosted_models(vec!["local-gguf/sha256-adc287a9".into()])
        .await;
    let mut requirement = request(vec![42]);
    assert_eq!(public_id, "unsloth/Qwen3-0.6B-GGUF:Q4_K_M");
    requirement.model = public_id.clone();
    node.apply_model_requirement("gate", requirement)
        .await
        .unwrap();
    let descriptors = node
        .descriptors_with_requirements(&["local-gguf/sha256-adc287a9".into()])
        .await;
    assert_eq!(descriptors[0].requires.len(), 1);
    assert_eq!(descriptors[0].requires[0].terms, vec![42]);
    let mut peer = super::make_test_peer_info(node.id());
    peer.hosted_models = vec!["local-gguf/sha256-adc287a9".into()];
    peer.serving_models = peer.hosted_models.clone();
    peer.served_model_descriptors = descriptors;
    let mut projected = vec![peer.clone()];
    *node.plugin_manager.lock().await = None;
    assert!(node.model_channels_unavailable(&peer, &public_id).await);
    node.filter_model_requirements(&mut projected).await;
    assert!(projected[0].served_model_descriptors.is_empty());
    assert!(projected[0].hosted_models.is_empty());
    node.close_endpoint().await;
}
