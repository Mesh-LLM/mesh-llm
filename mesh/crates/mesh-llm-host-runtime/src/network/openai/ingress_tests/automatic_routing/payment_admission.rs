//! Payment admission at automatic serving ingress.

use super::*;

async fn add_paid_auto_model(node: &mesh::Node, model: &str) -> iroh::EndpointId {
    let id = iroh::SecretKey::generate().public();
    let mut peer = peer_serving(id, model, false);
    peer.lightning_offers.insert(
        model.to_owned(),
        mesh_llm_payments_types::pricing::Pricing {
            input_msat_per_million: 1,
            output_msat_per_million: 1,
            cache_discount_percent: 0,
        },
    );
    node.insert_test_peer(peer).await;
    id
}

#[tokio::test]
async fn auto_payment_admission_preserves_fresh_local_alternate() {
    let paid = "paid-model-70B";
    let local = "local-model-7B";
    let (node, _) = node_serving(&[local]).await;
    add_paid_auto_model(&node, paid).await;
    // Local serving is known before the target table catches up.
    let targets = election::ModelTargets::default();
    let body = serde_json::json!({"model":"auto", "messages":[{"role":"user","content":"hello"}], "stream":true});
    let descriptors = vec![
        descriptor(paid, false, false),
        descriptor(local, false, false),
    ];
    let resolution = resolve(Some("auto"), &body, &node, &targets, &descriptors).await;
    assert!(
        matches!(resolution, AutoRouteResolution::Continue { effective_model: Some(model), .. } if model == local)
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn auto_payment_admission_rejects_paid_only_ingress() {
    let paid = "paid-model-70B";
    let (node, targets) = node_serving(&[]).await;
    add_paid_auto_model(&node, paid).await;
    let body = serde_json::json!({"model":"auto", "messages":[{"role":"user","content":"hello"}], "stream":true});
    let resolution = resolve(
        Some("auto"),
        &body,
        &node,
        &targets,
        &[descriptor(paid, false, false)],
    )
    .await;
    assert!(matches!(
        resolution,
        AutoRouteResolution::PaymentRequired(_)
    ));
    node.endpoint.close().await;
}

#[tokio::test]
async fn auto_payment_admission_selects_free_remote_even_when_cooling() {
    let paid = "paid-model-70B";
    let free = "free-model-7B";
    let (node, targets) = node_serving(&[]).await;
    add_paid_auto_model(&node, paid).await;
    let free_id = iroh::SecretKey::generate().public();
    node.insert_test_peer(peer_serving(free_id, free, false))
        .await;
    let body = serde_json::json!({"model":"auto", "messages":[{"role":"user","content":"hello"}], "stream":true});
    let mut request = request_with_body(Some("auto"), &body);
    let affinity = affinity::AffinityRouter::new();
    affinity.record_target_outcome(
        Some(free),
        &election::InferenceTarget::Remote(free_id),
        crate::network::target_health::TargetHealthOutcome::Unavailable,
    );
    let descriptors = vec![
        descriptor(paid, false, false),
        descriptor(free, false, false),
    ];
    let resolution = resolve_auto_routed_model(
        &node,
        &mut request,
        &targets,
        None,
        false,
        &descriptors,
        None,
        &affinity,
    )
    .await;
    assert!(
        matches!(resolution, AutoRouteResolution::Continue { effective_model: Some(model), .. } if model == free)
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn auto_payment_admission_keeps_local_copy_of_paid_remote_model() {
    let model = "shared-model-7B";
    let (node, targets) = node_serving(&[model]).await;
    add_paid_auto_model(&node, model).await;
    let body = serde_json::json!({"model":"auto", "messages":[{"role":"user","content":"hello"}], "stream":true, "mesh_payment":{"mode":"free_only"}});
    let resolution = resolve(
        Some("auto"),
        &body,
        &node,
        &targets,
        &[descriptor(model, false, false)],
    )
    .await;
    assert!(
        matches!(resolution, AutoRouteResolution::Continue { effective_model: Some(selected), .. } if selected == model)
    );
    node.endpoint.close().await;
}
