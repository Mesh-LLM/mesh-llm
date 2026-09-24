use super::routing_rank::RankedCandidates;
use crate::{inference::election::InferenceTarget, mesh::Node};

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
    // Do not provision an empty wallet merely because a paid peer appeared.
    let service = node.payment_service().await.ok();
    let intent = service
        .as_ref()
        .and_then(|service| service.ledger.payment_intent().ok())
        .unwrap_or_default();
    let intent = match request_body.and_then(|body| body.get("mesh_payment")) {
        Some(value) => {
            match serde_json::from_value::<mesh_llm_payments::intent::PaymentIntent>(value.clone())
            {
                Ok(request) if request.validate().is_ok() => intent.restrict(&request),
                _ => mesh_llm_payments::intent::PaymentIntent::FreeOnly,
            }
        }
        None => intent,
    };
    let available = match service {
        Some(service)
            if !matches!(intent, mesh_llm_payments::intent::PaymentIntent::FreeOnly)
                && service.has_wallet() =>
        {
            match service.wallet().await {
                Ok(wallet) => wallet
                    .balance()
                    .await
                    .ok()
                    .and_then(|balance| {
                        service
                            .ledger
                            .available_budget(balance.spendable_msat, mesh_llm_payments::now_ms())
                            .ok()
                    })
                    .unwrap_or(0),
                Err(_) => 0,
            }
        }
        _ => 0,
    };
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

/// Payments compiled out: this node cannot pay, so it behaves like the
/// free-only policy. Peers that charge for the model are dropped (they would
/// only answer 402, which is not retried); the remaining free route keeps its
/// capability/context/health order and is never price-ranked.
#[cfg(not(feature = "payments"))]
pub(super) async fn rank(
    node: &Node,
    model: &str,
    _input_estimate: u64,
    _max_output: u64,
    candidates: &mut RankedCandidates<InferenceTarget>,
    _request_body: Option<&serde_json::Value>,
) -> Result<bool, &'static str> {
    let mut paid = std::collections::HashSet::new();
    for target in &candidates.ordered {
        if let InferenceTarget::Remote(peer) = target
            && node.peer_charges_for(*peer, model).await
        {
            paid.insert(*peer);
        }
    }
    if paid.is_empty() {
        return Ok(false);
    }
    let is_paid = |target: &InferenceTarget| matches!(target, InferenceTarget::Remote(peer) if paid.contains(peer));
    let equivalent = candidates.ordered[..candidates.equivalent_prefix].to_vec();
    candidates.ordered.retain(|target| !is_paid(target));
    if candidates.ordered.is_empty() {
        return Err("only paid providers serve this model and this build cannot pay");
    }
    candidates.equivalent_prefix = candidates
        .ordered
        .iter()
        .take_while(|target| equivalent.contains(target))
        .count();
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

#[cfg(all(test, not(feature = "payments")))]
mod no_payments_tests {
    use super::*;

    #[tokio::test]
    async fn build_without_payments_skips_paid_peers() -> anyhow::Result<()> {
        let node = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let seller = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let mut announcement =
            seller.build_local_announcement(seller.snapshot_local_announcement_data().await);
        announcement.lightning_offers.insert("test".into());
        node.add_peer_after_direct_requirements_validated(
            seller.id(),
            seller.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
        let free = iroh::SecretKey::generate().public();
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
        // A model the seller does not price is untouched.
        let mut other = RankedCandidates {
            ordered: vec![InferenceTarget::Remote(seller.id())],
            equivalent_prefix: 1,
        };
        assert!(!rank(&node, "other", 1, 1, &mut other, None).await.unwrap());
        assert_eq!(other.ordered, vec![InferenceTarget::Remote(seller.id())]);
        let mut paid_only = RankedCandidates {
            ordered: vec![InferenceTarget::Remote(seller.id())],
            equivalent_prefix: 1,
        };
        assert!(
            rank(&node, "test", 1, 1, &mut paid_only, None)
                .await
                .is_err()
        );
        node.endpoint.close().await;
        seller.endpoint.close().await;
        Ok(())
    }
}
