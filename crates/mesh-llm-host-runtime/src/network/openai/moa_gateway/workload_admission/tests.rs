use super::super::fleet_sim_tests::{BIG_MODELS, SMALL_MODELS, fleet_peer};
use super::super::pool::{assemble_worker_pool, compute_actor_candidates};
use super::*;
use crate::inference::election::ModelTargets;
use crate::network::affinity::AffinityRouter;

fn classified_peer(seed: u32, class: ModelWorkloadClass) -> mesh::PeerInfo {
    let mut peer = fleet_peer(seed, BIG_MODELS[0]);
    peer.served_model_descriptors[0]
        .metadata
        .as_mut()
        .unwrap()
        .workload_class = Some(class);
    peer
}

#[tokio::test]
async fn mixed_workload_fleet_only_admits_chat_models_to_worker_and_actor_roles() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let classes = [
        ModelWorkloadClass::CausalGeneration,
        ModelWorkloadClass::Embedding,
        ModelWorkloadClass::Rerank,
        ModelWorkloadClass::EncoderDecoder,
        ModelWorkloadClass::SpeechSynthesis,
    ];
    for (index, class) in classes.into_iter().enumerate() {
        let model = if index < 3 {
            BIG_MODELS[index]
        } else {
            SMALL_MODELS[index - 3]
        };
        let mut peer = fleet_peer(index as u32 + 1, model);
        peer.served_model_descriptors[0]
            .metadata
            .as_mut()
            .unwrap()
            .workload_class = Some(class);
        node.insert_test_peer(peer).await;
    }
    let (backends, models) =
        assemble_worker_pool(&node, None, None, &reqwest::Client::new(), None).await;
    assert_eq!(backends.len(), 1);
    assert_eq!(models.len(), 1);
    assert_eq!(
        super::super::pool::canonical_base_name(&models[0].name),
        super::super::pool::canonical_base_name(BIG_MODELS[0].name)
    );
    assert_eq!(compute_actor_candidates(&node, &models).await, vec![0]);
}

#[tokio::test]
async fn target_admission_filters_standbys_and_same_model_clones_before_reservations() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let model = BIG_MODELS[0].name;
    let local = classified_peer(1, ModelWorkloadClass::EncoderDecoder);
    node.set_served_model_descriptors(local.served_model_descriptors)
        .await;
    let mut targets = ModelTargets::default();
    targets
        .targets
        .insert(model.into(), vec![InferenceTarget::Local(9337)]);
    let mut candidates = vec![InferenceTarget::Local(9337)];
    for (seed, class) in [
        (2, ModelWorkloadClass::Embedding),
        (3, ModelWorkloadClass::CausalGeneration),
    ] {
        let peer = classified_peer(seed, class);
        candidates.push(InferenceTarget::Remote(peer.id));
        node.insert_test_peer(peer).await;
    }
    let eligible = eligible_targets(&node, model, &candidates).await;
    assert_eq!(eligible, vec![candidates[2].clone()]);
    let remote = match candidates[1] {
        InferenceTarget::Remote(id) => id,
        _ => unreachable!(),
    };
    assert!(
        super::super::context_selection::eligible_remote_hosts(&node, model, None, vec![remote])
            .await
            .is_empty()
    );
    let affinity = AffinityRouter::new();
    let (backends, models) = assemble_worker_pool(
        &node,
        Some(&targets),
        None,
        &reqwest::Client::new(),
        Some(&affinity),
    )
    .await;
    assert_eq!(
        models.len(),
        1,
        "incompatible local/remote copies must not fabricate a committee"
    );
    assert_eq!(backends.len(), 1);
}

#[test]
fn legacy_chat_is_preserved_but_encoder_decoder_never_inherits_a_committee_role() {
    assert!(model_supports_committee("legacy", &[]));
    let mut descriptor = ServedModelDescriptor::default();
    descriptor.identity.model_name = "model".into();
    assert!(model_supports_committee(
        "model",
        std::slice::from_ref(&descriptor)
    ));
    descriptor.metadata = Some(mesh::ServedModelMetadata {
        workload_class: Some(ModelWorkloadClass::EncoderDecoder),
        ..Default::default()
    });
    assert!(!model_supports_committee("model", &[descriptor]));
}

#[test]
fn public_alias_does_not_bypass_non_chat_admission() {
    let mut peer = classified_peer(9, ModelWorkloadClass::Embedding);
    let descriptor = &mut peer.served_model_descriptors[0];
    descriptor.identity.source_kind = mesh::ModelSourceKind::HuggingFace;
    descriptor.identity.repository = Some("fixture/embedding-GGUF".into());
    descriptor.identity.artifact = Some("embedding.Q8_0.gguf".into());
    let alias = peer.public_model_id_for_routable_model(BIG_MODELS[0].name);
    assert_ne!(alias, BIG_MODELS[0].name);
    assert!(!model_supports_committee(
        &alias,
        &peer.served_model_descriptors
    ));
    assert!(
        crate::network::openai::workload_routing::model_satisfies_workload_class(
            &alias,
            ModelWorkloadClass::Embedding,
            &peer.served_model_descriptors
        )
    );
}
