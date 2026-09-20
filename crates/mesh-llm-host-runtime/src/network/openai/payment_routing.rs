use super::routing_rank::RankedCandidates;
use crate::{inference::election::InferenceTarget, mesh::Node};

/// Apply economics after capability/context/health eligibility. Stable sorting
/// preserves observed performance ordering for equal-price offers.
pub(super) async fn rank(
    node: &Node,
    model: &str,
    input_estimate: u64,
    max_output: u64,
    candidates: &mut RankedCandidates<InferenceTarget>,
) -> bool {
    let mut prices = std::collections::HashMap::new();
    for target in &candidates.ordered {
        if let InferenceTarget::Remote(peer) = target
            && let Some(price) = node.peer_payment_offer(*peer, model).await
        {
            prices.insert(*peer, price);
        }
    }
    if prices.is_empty() {
        return false;
    }
    // Do not provision an empty wallet merely because a paid peer appeared.
    let service = node.payment_service().await.ok();
    let intent = service
        .as_ref()
        .and_then(|service| service.ledger.payment_intent().ok())
        .unwrap_or_default();
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
                    let cost = price
                        .input_charge(input_estimate)
                        .ok()?
                        .checked_add(price.output_charge(max_output).ok()?)?;
                    let total =
                        cost.checked_add(2 * mesh_llm_payments::pricing::FEE_ALLOWANCE_MSAT)?;
                    (intent.permits(price, total) && total <= available).then_some((1, cost))
                }
                None => Some((2, 0)),
            },
            InferenceTarget::None => None,
        }
    };
    candidates.ordered.retain(|target| key(target).is_some());
    candidates.ordered.sort_by_key(|target| key(target));
    let first = candidates.ordered.first().and_then(&key);
    candidates.equivalent_prefix = candidates
        .ordered
        .iter()
        .take_while(|target| key(target) == first)
        .count();
    true
}

#[cfg(test)]
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
        assert!(rank(&node, "test", 1, 1, &mut candidates).await);
        assert_eq!(candidates.ordered, vec![InferenceTarget::Remote(free)]);
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
