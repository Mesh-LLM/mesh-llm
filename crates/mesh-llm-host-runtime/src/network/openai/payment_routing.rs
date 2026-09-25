use super::routing_rank::RankedCandidates;
use crate::{inference::election::InferenceTarget, mesh::Node};
#[cfg(feature = "payments")]
use mesh_llm_payments_types::contract::{RoutingBudgetRequest, RoutingBudgetResponse, ops};

/// Apply economics after capability/context/health eligibility. Stable sorting
/// preserves observed performance ordering for equal-price offers.
#[cfg(feature = "payments")]
pub(super) async fn rank(
    node: &Node,
    model: &str,
    input_estimate: u64,
    max_output: u64,
    candidates: &mut RankedCandidates<InferenceTarget>,
    request_body: Option<&serde_json::Value>,
) -> Result<bool, &'static str> {
    let mut prices = std::collections::HashMap::new();
    for target in &candidates.ordered {
        if let InferenceTarget::Remote(peer) = target
            && let Some(price) = node.peer_payment_offer(*peer, model).await
        {
            prices.insert(*peer, price);
        }
    }
    if prices.is_empty() {
        return Ok(false);
    }
    // The provider never provisions an empty wallet merely because a paid
    // peer appeared, and reads the balance only when paid is permitted.
    let budget: RoutingBudgetResponse = crate::network::payments::client::call_node(
        node,
        ops::ROUTING_BUDGET,
        &RoutingBudgetRequest {
            request_intent: request_body
                .and_then(|body| body.get("mesh_payment"))
                .cloned(),
        },
    )
    .await
    .unwrap_or_default();
    let (intent, available) = (budget.intent, budget.available_msat);
    let key = |target: &InferenceTarget| -> Option<(u8, u64)> {
        match target {
            InferenceTarget::Local(_) => Some((0, 0)),
            InferenceTarget::Remote(peer) => match prices.get(peer) {
                Some(price) => {
                    let input_amount = price.input_charge(input_estimate).ok()?;
                    let cost = input_amount.checked_add(price.output_charge(max_output).ok()?)?;
                    let total = price.request_cap_msat(input_amount, max_output).ok()?;
                    (intent.permits(price, total) && total <= available).then_some((1, cost))
                }
                None => Some((2, 0)),
            },
            InferenceTarget::None => None,
        }
    };
    let equivalent = candidates.ordered[..candidates.equivalent_prefix].to_vec();
    candidates.ordered.retain(|target| key(target).is_some());
    if candidates.ordered.is_empty() {
        return Err(
            "paid providers are unavailable under the current payment policy, wallet balance or daily budget",
        );
    }
    // No paid provider survived the policy filter, so every remaining target
    // is free and price has nothing to order. Leave the free route exactly as
    // it is without payments: context/throughput order, its equivalent run
    // (minus any removed paid peers), and ordinary cache, session and load
    // selection. Treating it as a price tier would flatten that run and force
    // its first target as a pseudo cache hit.
    if !candidates
        .ordered
        .iter()
        .any(|target| matches!(key(target), Some((1, _))))
    {
        candidates.equivalent_prefix = candidates
            .ordered
            .iter()
            .take_while(|target| equivalent.contains(target))
            .count();
        return Ok(false);
    }
    candidates.ordered.sort_by_key(|target| key(target));
    let first = candidates.ordered.first().and_then(&key);
    candidates.equivalent_prefix = candidates
        .ordered
        .iter()
        .take_while(|target| key(target) == first)
        .count();
    Ok(true)
}

/// Wallets compiled out: no payment tiers exist, so candidate ordering is left
/// exactly as capability/context/health eligibility produced it.
#[cfg(not(feature = "payments"))]
pub(super) async fn rank(
    _node: &Node,
    _model: &str,
    _input_estimate: u64,
    _max_output: u64,
    _candidates: &mut RankedCandidates<InferenceTarget>,
    _request_body: Option<&serde_json::Value>,
) -> Result<bool, &'static str> {
    Ok(false)
}

pub(super) fn cache_candidates<'a>(
    payment_ranked: bool,
    ranked: &'a RankedCandidates<InferenceTarget>,
    ordered: &'a [InferenceTarget],
) -> &'a [InferenceTarget] {
    if payment_ranked {
        &ranked.ordered[..ranked.equivalent_prefix]
    } else {
        ordered
    }
}

/// Keep cache affinity inside the selected price tier.
pub(super) fn prefer_price_tier(
    payment_ranked: bool,
    ranked: &RankedCandidates<InferenceTarget>,
    cached: Option<InferenceTarget>,
) -> Option<InferenceTarget> {
    if payment_ranked {
        cached.or_else(|| ranked.ordered.first().cloned())
    } else {
        cached
    }
}

#[cfg(all(test, feature = "payments"))]
mod tests {
    use super::*;
    use mesh_llm_payments::{intent::PaymentIntent, pricing::Pricing, service::PaymentService};

    #[tokio::test]
    async fn free_only_excludes_paid_without_provisioning_a_wallet() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let node = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let seller = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let service = std::sync::Arc::new(PaymentService::open(directory.path())?);
        node.payments
            .set(service.clone())
            .map_err(|_| anyhow::anyhow!("already set"))?;
        let mut announcement =
            seller.build_local_announcement(seller.snapshot_local_announcement_data().await);
        announcement.lightning_offers.insert(
            "test".into(),
            Pricing {
                input_msat_per_million: 1,
                output_msat_per_million: 1,
                minimum_invoice_msat: 1,
            },
        );
        node.add_peer_after_direct_requirements_validated(
            seller.id(),
            seller.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
        let free = iroh::SecretKey::generate().public();
        let mut candidates = RankedCandidates {
            ordered: vec![
                InferenceTarget::Remote(seller.id()),
                InferenceTarget::Remote(free),
            ],
            equivalent_prefix: 2,
        };
        // Only free targets remain, so the route is left unranked by price.
        assert!(
            !rank(&node, "test", 1, 1, &mut candidates, None)
                .await
                .unwrap()
        );
        assert_eq!(candidates.ordered, vec![InferenceTarget::Remote(free)]);
        assert_eq!(candidates.equivalent_prefix, 1);
        // The free route keeps its context/throughput tiering: a tied run of
        // two free targets ahead of a slower one survives the paid peer's
        // removal, instead of being flattened into one price tier.
        let fast = iroh::SecretKey::generate().public();
        let slow = iroh::SecretKey::generate().public();
        let mut tiered = RankedCandidates {
            ordered: vec![
                InferenceTarget::Remote(seller.id()),
                InferenceTarget::Remote(free),
                InferenceTarget::Remote(fast),
                InferenceTarget::Remote(slow),
            ],
            equivalent_prefix: 3,
        };
        assert!(!rank(&node, "test", 1, 1, &mut tiered, None).await.unwrap());
        assert_eq!(
            tiered.ordered,
            vec![
                InferenceTarget::Remote(free),
                InferenceTarget::Remote(fast),
                InferenceTarget::Remote(slow),
            ]
        );
        assert_eq!(tiered.equivalent_prefix, 2);
        let mut paid_only = RankedCandidates {
            ordered: vec![InferenceTarget::Remote(seller.id())],
            equivalent_prefix: 1,
        };
        assert!(
            rank(&node, "test", 1, 1, &mut paid_only, None)
                .await
                .is_err()
        );
        let mut empty = RankedCandidates {
            ordered: vec![],
            equivalent_prefix: 0,
        };
        assert!(!rank(&node, "test", 1, 1, &mut empty, None).await.unwrap());
        assert!(!service.has_wallet());
        assert!(service.ledger.requests()?.is_empty());
        assert!(matches!(
            service.ledger.payment_intent()?,
            PaymentIntent::FreeOnly
        ));
        node.endpoint.close().await;
        seller.endpoint.close().await;
        Ok(())
    }
}
