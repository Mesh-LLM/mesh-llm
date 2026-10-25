//! Hard payment admission precedes the soft auto-routing readiness preference.

use super::{model_has_eligible_target, model_is_locally_served};
use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::Node;
use crate::network::affinity::AffinityRouter;
use crate::network::openai::{
    payment_routing, request_parse::BufferedHttpRequest, routing_rank::RankedCandidates,
    workload_routing,
};
use crate::network::router::RoutingCandidate;

/// Retain only models with a payment-admitted route, then measure readiness on
/// those routes alone. An admitted cooling replica must not borrow readiness
/// from a healthy replica which this request is not authorized to pay.
///
/// `None` targets denotes passive ingress (remote hosts only). Serving ingress
/// retains its local-startup exception before the election table is populated.
/// Empty target lists retain the existing availability behavior; only an actual
/// payment denial produces 402. No policy, wallet, or request intent is changed.
pub(crate) async fn admitted_models<'a>(
    node: &Node,
    targets: Option<&ModelTargets>,
    request: &BufferedHttpRequest,
    available: &[RoutingCandidate<'a>],
    required_tokens: Option<u32>,
    affinity: &AffinityRouter,
) -> Result<(Vec<RoutingCandidate<'a>>, Vec<&'a str>), &'static str> {
    let mut admitted = Vec::new();
    let mut ready = Vec::new();
    let mut denial = None;
    for candidate in available {
        let model = candidate.name;
        let candidates = match targets {
            Some(targets) => {
                workload_routing::ingress_candidates(node, model, &request.client_path, targets)
                    .await
            }
            None => {
                let hosts = node
                    .hosts_for_model(model)
                    .await
                    .into_iter()
                    .map(InferenceTarget::Remote)
                    .collect::<Vec<_>>();
                workload_routing::eligible_targets(node, model, &request.client_path, &hosts).await
            }
        };
        let locally_starting = targets.is_some()
            && !candidates
                .iter()
                .any(|target| !matches!(target, InferenceTarget::None))
            && model_is_locally_served(node, model).await;
        let mut ranked = RankedCandidates {
            ordered: candidates,
            equivalent_prefix: 0,
        };
        if let Err(reason) = payment_routing::rank(
            node,
            model,
            u64::from(request.completion_tokens.unwrap_or(256)),
            &mut ranked,
            request.body_json.as_ref(),
        )
        .await
        {
            denial = Some(reason);
            continue;
        }
        admitted.push(candidate.clone());
        if locally_starting
            || model_has_eligible_target(
                node,
                model,
                required_tokens,
                &request.client_path,
                &ranked.ordered,
                affinity,
            )
            .await
        {
            ready.push(model);
        }
    }
    if admitted.is_empty()
        && let Some(reason) = denial
    {
        return Err(reason);
    }
    Ok((admitted, ready))
}

#[cfg(all(test, feature = "payments"))]
mod tests {
    use super::*;
    use crate::mesh::NodeRole;
    use crate::network::target_health::TargetHealthOutcome;

    #[tokio::test]
    async fn mixed_replicas_measure_readiness_only_on_admitted_targets() -> anyhow::Result<()> {
        let node = Node::new_for_tests(NodeRole::Client).await?;
        let seller = Node::new_for_tests(NodeRole::Host { http_port: 9337 }).await?;
        let free = Node::new_for_tests(NodeRole::Host { http_port: 9338 }).await?;
        let model = "mixed-model-7B";
        for host in [&seller, &free] {
            host.set_hosted_models(vec![model.to_owned()]).await;
            let mut announcement =
                host.build_local_announcement(host.snapshot_local_announcement_data().await);
            if host.id() == seller.id() {
                announcement.lightning_offers.insert(
                    model.to_owned(),
                    mesh_llm_payments_types::pricing::Pricing {
                        input_msat_per_million: 1,
                        output_msat_per_million: 1,
                        cache_discount_percent: 0,
                    },
                );
            }
            node.add_peer_after_direct_requirements_validated(
                host.id(),
                host.endpoint.addr(),
                &announcement,
                Some(1),
            )
            .await;
            // The announcement helper does not establish a transport connection.
            // Mark this synthetic peer live, as ordinary routing fixtures do.
            let peer = node.state.lock().await.peers[&host.id()].clone();
            node.insert_test_peer(peer).await;
        }
        let body = r#"{"model":"auto","messages":[],"max_tokens":123,"mesh_payment":{"mode":"free_only"}}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        use tokio::io::AsyncWriteExt;
        let (mut writer, mut reader) = tokio::io::duplex(4096);
        writer.write_all(raw.as_bytes()).await?;
        let mut request =
            crate::network::openai::request_parse::read_http_request(&mut reader).await?;
        request.ensure_body_json();
        let available = vec![RoutingCandidate::unscored(model, Default::default())];
        let affinity = AffinityRouter::new();
        let (admitted, ready) = admitted_models(&node, None, &request, &available, None, &affinity)
            .await
            .unwrap();
        assert_eq!(admitted.len(), 1);
        assert_eq!(ready, vec![model]);
        affinity.record_target_outcome(
            Some(model),
            &InferenceTarget::Remote(free.id()),
            TargetHealthOutcome::Unavailable,
        );
        let (admitted, ready) = admitted_models(&node, None, &request, &available, None, &affinity)
            .await
            .unwrap();
        assert_eq!(
            admitted.len(),
            1,
            "free cooling replica keeps model admitted"
        );
        assert!(
            ready.is_empty(),
            "healthy paid replica cannot lend readiness"
        );
        let pool = super::super::pool_for_ready_models(&admitted, &ready);
        assert_eq!(
            pool[0].name, model,
            "soft fallback remains inside admission"
        );
        node.endpoint.close().await;
        seller.endpoint.close().await;
        free.endpoint.close().await;
        Ok(())
    }
}
