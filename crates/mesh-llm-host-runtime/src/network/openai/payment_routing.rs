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
    let available = match node.payment_service().await {
        Ok(service) if service.has_wallet() => match service.wallet().await {
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
        },
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
                    (cost.checked_add(2 * mesh_llm_payments::pricing::FEE_ALLOWANCE_MSAT)?
                        <= available)
                        .then_some((1, cost))
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
