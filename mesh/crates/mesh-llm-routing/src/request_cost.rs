//! Service-cost estimate for the request being routed, cache hits and cold
//! candidates compared on the same scale.
//!
//! Every component is optional. A missing value makes the candidate
//! incomparable; it never reads as zero, so absent evidence cannot make a
//! candidate look cheaper than one that reported its costs.

use crate::InferenceTarget;
use crate::affinity::{prompt_bytes_from_body, scaffold_prefix_from_body};
use serde_json::Value;

/// Typical bytes per token for prompt text, used to size the whole prompt.
pub const PROMPT_BYTES_PER_TOKEN: usize = 4;
/// Deliberately high bytes per token for the shared prefix: dividing by it
/// gives a count below what a tokenizer would report for ordinary text, so an
/// estimated credit errs toward too small, never toward too large.
pub const SHARED_PREFIX_BYTES_PER_TOKEN: usize = 8;

/// Size this request from its parsed body. Without a body nothing is known;
/// with one, the provably shared prefix is the hashed scaffold, even when an
/// explicit cache key chose the prefix identity (a key says "same prefix",
/// not "this deep").
pub fn request_cost_input_from_body(
    body: Option<&Value>,
    prefix_fallback_to_first_user: bool,
) -> RequestCostInput {
    let Some(body) = body else {
        return RequestCostInput {
            prompt_tokens: None,
            reusable_prefix_tokens: None,
        };
    };
    let prompt_bytes = prompt_bytes_from_body(body);
    let shared_bytes = scaffold_prefix_from_body(body, prefix_fallback_to_first_user)
        .map_or(0, |scaffold| scaffold.bytes);
    RequestCostInput {
        prompt_tokens: Some(TokenEstimate {
            tokens: u32::try_from(prompt_bytes.div_ceil(PROMPT_BYTES_PER_TOKEN))
                .unwrap_or(u32::MAX),
            source: TokenSource::Heuristic,
        }),
        // Rounded down: a partial token is not credited.
        reusable_prefix_tokens: Some(
            u32::try_from(shared_bytes / SHARED_PREFIX_BYTES_PER_TOKEN).unwrap_or(u32::MAX),
        ),
    }
}

/// How a token count was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenSource {
    /// Counted by a tokenizer.
    Measured,
    /// Derived from request size with a conservative ratio.
    Heuristic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenEstimate {
    pub tokens: u32,
    pub source: TokenSource,
}

/// What the router knows about the request itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestCostInput {
    /// Prompt length of this request.
    pub prompt_tokens: Option<TokenEstimate>,
    /// Lower bound of the prefix this request provably shares with the cached
    /// one (the hashed scaffold). A cache hit is never credited beyond it.
    pub reusable_prefix_tokens: Option<u32>,
}

/// Positive cache evidence a candidate advertised for this prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachedPrefix {
    pub matched_tokens: u32,
    pub restore_micros: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateCostInput {
    pub target: InferenceTarget,
    /// Current wait before this candidate could start, not a past sample.
    pub queue_micros: Option<u64>,
    pub prefill_micros_per_token: Option<u64>,
    pub cached: Option<CachedPrefix>,
}

/// The first component that kept an estimate from being made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingComponent {
    PromptTokens,
    ReusablePrefix,
    QueueDelay,
    PrefillRate,
    RestoreCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceCost {
    pub micros: u64,
    /// Prompt tokens this estimate does not have to prefill.
    pub credited_tokens: u32,
}

/// Cost of serving the request on `candidate` without its cache.
pub fn cold_cost(
    request: RequestCostInput,
    candidate: &CandidateCostInput,
) -> Result<ServiceCost, MissingComponent> {
    let prompt = request
        .prompt_tokens
        .ok_or(MissingComponent::PromptTokens)?;
    let queue = candidate.queue_micros.ok_or(MissingComponent::QueueDelay)?;
    let rate = candidate
        .prefill_micros_per_token
        .ok_or(MissingComponent::PrefillRate)?;
    Ok(ServiceCost {
        micros: queue.saturating_add(u64::from(prompt.tokens).saturating_mul(rate)),
        credited_tokens: 0,
    })
}

/// Cost of serving the request on `candidate` from its cached prefix, or
/// `None` when the candidate advertised no cache for it.
pub fn warm_cost(
    request: RequestCostInput,
    candidate: &CandidateCostInput,
) -> Option<Result<ServiceCost, MissingComponent>> {
    let cached = candidate.cached?;
    Some(cached_cost(request, candidate, cached))
}

fn cached_cost(
    request: RequestCostInput,
    candidate: &CandidateCostInput,
    cached: CachedPrefix,
) -> Result<ServiceCost, MissingComponent> {
    let prompt = request
        .prompt_tokens
        .ok_or(MissingComponent::PromptTokens)?;
    let reusable = request
        .reusable_prefix_tokens
        .ok_or(MissingComponent::ReusablePrefix)?;
    let queue = candidate.queue_micros.ok_or(MissingComponent::QueueDelay)?;
    let rate = candidate
        .prefill_micros_per_token
        .ok_or(MissingComponent::PrefillRate)?;
    let restore = cached.restore_micros.ok_or(MissingComponent::RestoreCost)?;
    let credited = cached.matched_tokens.min(reusable).min(prompt.tokens);
    let suffix = prompt.tokens - credited;
    Ok(ServiceCost {
        micros: queue
            .saturating_add(restore)
            .saturating_add(u64::from(suffix).saturating_mul(rate)),
        credited_tokens: credited,
    })
}

/// Why cache affinity did or did not take over routing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheRouteDecision {
    /// The cheapest comparable cache hit beats every comparable cold
    /// candidate by the configured margin.
    CacheHitWins {
        target: InferenceTarget,
        cost: ServiceCost,
        best_cold_micros: Option<u64>,
    },
    /// A cold candidate is at least as cheap; keep the normal order.
    ColdWins,
    /// Cache evidence exists but no hit could be costed.
    Incomparable(MissingComponent),
    /// No candidate advertised a cache hit worth considering.
    NoEvidence,
}

/// Pick a cache target only when it is cheaper for this request than serving
/// it cold anywhere, including on the cached candidate itself. A hit must also
/// credit at least `min_saved_tokens`, and beat the best cold estimate by the
/// prefill time of those tokens at its own rate, so a marginal estimate never
/// overrides the established order. Candidate order breaks ties.
pub fn decide_cache_route(
    request: RequestCostInput,
    candidates: &[CandidateCostInput],
    min_saved_tokens: u32,
) -> CacheRouteDecision {
    let best_cold = candidates
        .iter()
        .filter_map(|candidate| cold_cost(request, candidate).ok())
        .map(|cost| cost.micros)
        .min();

    let mut best_warm: Option<(&CandidateCostInput, ServiceCost)> = None;
    let mut first_missing = None;
    for candidate in candidates {
        match warm_cost(request, candidate) {
            None => {}
            Some(Err(missing)) => {
                first_missing.get_or_insert(missing);
            }
            Some(Ok(cost))
                if cost.credited_tokens >= min_saved_tokens
                    && best_warm.is_none_or(|(_, best)| cost.micros < best.micros) =>
            {
                best_warm = Some((candidate, cost));
            }
            Some(Ok(_)) => {}
        }
    }

    let Some((winner, cost)) = best_warm else {
        return match first_missing {
            Some(missing) => CacheRouteDecision::Incomparable(missing),
            None => CacheRouteDecision::NoEvidence,
        };
    };
    let margin = winner.prefill_micros_per_token.map_or(u64::MAX, |rate| {
        u64::from(min_saved_tokens).saturating_mul(rate)
    });
    match best_cold {
        Some(cold) if cost.micros.saturating_add(margin) > cold => CacheRouteDecision::ColdWins,
        _ => CacheRouteDecision::CacheHitWins {
            target: winner.target.clone(),
            cost,
            best_cold_micros: best_cold,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    const MIN_SAVED: u32 = 256;

    fn remote(seed: u8) -> InferenceTarget {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        InferenceTarget::Remote(SecretKey::from_bytes(&bytes).public())
    }

    fn request(prompt: u32, reusable: u32) -> RequestCostInput {
        RequestCostInput {
            prompt_tokens: Some(TokenEstimate {
                tokens: prompt,
                source: TokenSource::Heuristic,
            }),
            reusable_prefix_tokens: Some(reusable),
        }
    }

    fn cold(seed: u8, queue: u64, rate: u64) -> CandidateCostInput {
        CandidateCostInput {
            target: remote(seed),
            queue_micros: Some(queue),
            prefill_micros_per_token: Some(rate),
            cached: None,
        }
    }

    fn warm(seed: u8, queue: u64, rate: u64, matched: u32, restore: u64) -> CandidateCostInput {
        CandidateCostInput {
            cached: Some(CachedPrefix {
                matched_tokens: matched,
                restore_micros: Some(restore),
            }),
            ..cold(seed, queue, rate)
        }
    }

    fn winner(decision: &CacheRouteDecision) -> Option<InferenceTarget> {
        match decision {
            CacheRouteDecision::CacheHitWins { target, .. } => Some(target.clone()),
            _ => None,
        }
    }

    #[test]
    fn a_long_queue_on_the_hit_loses_to_an_idle_cold_node() {
        // The issue's example: A holds the prefix but waits 6 s, B is idle.
        let decision = decide_cache_route(
            request(4_000, 3_000),
            &[warm(1, 6_000_000, 250, 3_000, 0), cold(2, 0, 250)],
            MIN_SAVED,
        );
        assert_eq!(decision, CacheRouteDecision::ColdWins);
    }

    #[test]
    fn the_crossover_follows_the_current_request_length() {
        // Same evidence, same nodes: a hit that saves 2 000 tokens wins while
        // the suffix is short, and loses once the suffix dominates and the
        // hit's slower rate costs more than the saved prefix.
        let candidates = [warm(1, 0, 1_000, 2_000, 0), cold(2, 0, 400)];
        let short = decide_cache_route(request(2_200, 2_000), &candidates, MIN_SAVED);
        assert_eq!(winner(&short), Some(remote(1)));
        let long = decide_cache_route(request(20_000, 2_000), &candidates, MIN_SAVED);
        assert_eq!(long, CacheRouteDecision::ColdWins);
    }

    #[test]
    fn credit_never_exceeds_the_reusable_prefix() {
        // 8 000 cached tokens, but only a 300-token scaffold is provably shared.
        let candidate = warm(1, 0, 1_000, 8_000, 0);
        let cost = warm_cost(request(9_000, 300), &candidate).unwrap().unwrap();
        assert_eq!(cost.credited_tokens, 300);
        assert_eq!(cost.micros, 8_700 * 1_000);
    }

    #[test]
    fn credit_never_exceeds_the_prompt() {
        let candidate = warm(1, 0, 1_000, 8_000, 0);
        let cost = warm_cost(request(500, 8_000), &candidate).unwrap().unwrap();
        assert_eq!(cost.credited_tokens, 500);
        assert_eq!(cost.micros, 0);
    }

    #[test]
    fn small_hits_keep_the_normal_order() {
        let decision = decide_cache_route(
            request(1_000, 1_000),
            &[warm(1, 0, 1_000, MIN_SAVED - 1, 0)],
            MIN_SAVED,
        );
        assert_eq!(decision, CacheRouteDecision::NoEvidence);
    }

    #[test]
    fn a_marginal_saving_does_not_override_the_order() {
        // The hit saves 300 tokens of its own time but is barely cheaper than
        // a faster cold node: less than the margin, so the order stands.
        let decision = decide_cache_route(
            request(1_300, 300),
            &[warm(1, 0, 1_000, 300, 0), cold(2, 0, 790)],
            MIN_SAVED,
        );
        assert_eq!(decision, CacheRouteDecision::ColdWins);
    }

    #[test]
    fn restore_cost_counts_against_the_hit() {
        let decision = decide_cache_route(
            request(2_000, 1_000),
            &[warm(1, 0, 1_000, 1_000, 1_500_000)],
            MIN_SAVED,
        );
        assert_eq!(decision, CacheRouteDecision::ColdWins);
    }

    #[test]
    fn missing_components_make_a_hit_incomparable_not_cheaper() {
        let mut no_queue = warm(1, 0, 1_000, 2_000, 0);
        no_queue.queue_micros = None;
        let mut no_rate = warm(1, 0, 1_000, 2_000, 0);
        no_rate.prefill_micros_per_token = None;
        let mut no_restore = warm(1, 0, 1_000, 2_000, 0);
        no_restore.cached.as_mut().unwrap().restore_micros = None;
        let cases = [
            (no_queue, MissingComponent::QueueDelay),
            (no_rate, MissingComponent::PrefillRate),
            (no_restore, MissingComponent::RestoreCost),
        ];
        for (candidate, missing) in cases {
            assert_eq!(
                decide_cache_route(
                    request(3_000, 2_000),
                    &[candidate, cold(2, 0, 1_000)],
                    MIN_SAVED
                ),
                CacheRouteDecision::Incomparable(missing)
            );
        }
        let unknown_prompt = RequestCostInput {
            prompt_tokens: None,
            reusable_prefix_tokens: Some(2_000),
        };
        assert_eq!(
            decide_cache_route(unknown_prompt, &[warm(1, 0, 1_000, 2_000, 0)], MIN_SAVED),
            CacheRouteDecision::Incomparable(MissingComponent::PromptTokens)
        );
        let unknown_prefix = RequestCostInput {
            reusable_prefix_tokens: None,
            ..request(3_000, 0)
        };
        assert_eq!(
            decide_cache_route(unknown_prefix, &[warm(1, 0, 1_000, 2_000, 0)], MIN_SAVED),
            CacheRouteDecision::Incomparable(MissingComponent::ReusablePrefix)
        );
    }

    #[test]
    fn an_uncosted_cold_node_cannot_veto_a_costed_hit() {
        let mut unknown = cold(2, 0, 1);
        unknown.queue_micros = None;
        let decision = decide_cache_route(
            request(3_000, 2_000),
            &[warm(1, 0, 1_000, 2_000, 0), unknown],
            MIN_SAVED,
        );
        assert_eq!(winner(&decision), Some(remote(1)));
    }

    #[test]
    fn the_cheaper_hit_wins_and_order_breaks_ties() {
        let decision = decide_cache_route(
            request(3_000, 2_000),
            &[warm(1, 0, 1_000, 1_500, 0), warm(2, 0, 1_000, 2_000, 0)],
            MIN_SAVED,
        );
        assert_eq!(winner(&decision), Some(remote(2)));
        let tie = decide_cache_route(
            request(3_000, 2_000),
            &[warm(2, 0, 1_000, 2_000, 0), warm(1, 0, 1_000, 2_000, 0)],
            MIN_SAVED,
        );
        assert_eq!(winner(&tie), Some(remote(2)));
    }

    #[test]
    fn saturating_inputs_do_not_overflow() {
        // A saturated hit must still lose to a finite cold estimate.
        let decision = decide_cache_route(
            request(u32::MAX, u32::MAX),
            &[
                warm(1, u64::MAX, u64::MAX, u32::MAX, u64::MAX),
                cold(2, 0, 1),
            ],
            MIN_SAVED,
        );
        assert_eq!(decision, CacheRouteDecision::ColdWins);
    }

    fn chat_body(system: usize, user: usize) -> Value {
        serde_json::json!({
            "messages": [
                {"role": "system", "content": "s".repeat(system)},
                {"role": "user", "content": "u".repeat(user)},
            ]
        })
    }

    #[test]
    fn the_request_is_sized_from_its_body() {
        let input = request_cost_input_from_body(Some(&chat_body(800, 400)), true);
        assert_eq!(input.prompt_tokens.map(|p| p.tokens), Some(300));
        assert_eq!(input.reusable_prefix_tokens, Some(100));
    }

    #[test]
    fn later_turns_are_never_part_of_the_shared_prefix() {
        let mut body = chat_body(800, 400);
        let messages = body["messages"].as_array_mut().unwrap();
        messages.push(serde_json::json!({"role": "assistant", "content": "a".repeat(4_000)}));
        messages.push(serde_json::json!({"role": "user", "content": "u".repeat(400)}));
        let input = request_cost_input_from_body(Some(&body), true);
        assert_eq!(input.prompt_tokens.map(|p| p.tokens), Some(1_400));
        assert_eq!(input.reusable_prefix_tokens, Some(100));
    }

    #[test]
    fn an_explicit_cache_key_does_not_deepen_the_credit() {
        let mut body = chat_body(800, 400);
        body["prompt_cache_key"] = Value::from("k".repeat(4_000));
        let input = request_cost_input_from_body(Some(&body), true);
        assert_eq!(input.reusable_prefix_tokens, Some(100));
    }

    #[test]
    fn without_a_scaffold_only_the_fallback_policy_shares_the_first_user_turn() {
        let body = serde_json::json!({"messages": [{"role": "user", "content": "u".repeat(800)}]});
        assert_eq!(
            request_cost_input_from_body(Some(&body), true).reusable_prefix_tokens,
            Some(100)
        );
        assert_eq!(
            request_cost_input_from_body(Some(&body), false).reusable_prefix_tokens,
            Some(0)
        );
    }

    #[test]
    fn an_unparsed_body_leaves_the_request_unknown() {
        let input = request_cost_input_from_body(None, true);
        assert_eq!(input.prompt_tokens, None);
        assert_eq!(input.reusable_prefix_tokens, None);
    }

    #[test]
    fn the_sized_scaffold_is_the_hashed_scaffold() {
        let body = chat_body(800, 400);
        assert_eq!(
            crate::affinity::scaffold_prefix_from_body(&body, true).map(|s| s.hash),
            crate::affinity::scaffold_prefix_hash_from_body(&body, true)
        );
    }

    #[test]
    fn a_zero_token_prompt_costs_only_the_wait() {
        let cost = cold_cost(request(0, 0), &cold(1, 5_000, 1_000)).unwrap();
        assert_eq!(cost.micros, 5_000);
    }
}
