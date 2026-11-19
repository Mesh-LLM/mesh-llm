//! First-token latency targets (#2774), observe-only first slice.
//!
//! Given a target and the current request's per-candidate service estimate
//! (#2773, [`crate::request_cost`]) plus the measured path cost to each host,
//! work out which candidates would meet the target and which are equivalent
//! to the best one. Pure: nothing here changes routing, reservations or
//! retries. "First token" is router-observed (dispatch to the first upstream
//! response byte), never end-user latency.

use crate::InferenceTarget;
use crate::request_cost::{
    CandidateCostInput, MissingComponent, RequestCostInput, cold_cost, warm_cost,
};

/// What the caller and operator asked for. Percentages are whole percent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyPolicy {
    /// First-token target, router-observed, in milliseconds.
    pub first_token_ms: u32,
    /// Safety margin added to every estimate before it is compared.
    pub margin_pct: u16,
    /// A qualifying candidate whose estimate is within this much of the best
    /// one is objective-equivalent to it.
    pub equivalence_pct: u16,
}

/// One candidate as the assessment sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateLatencyInput {
    pub cost: CandidateCostInput,
    /// Measured round trip to the host, `None` when there is no path sample.
    /// A local target has no path and passes `Some(0)`.
    pub path_rtt_ms: Option<u32>,
}

/// Why a candidate could not be judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    Service(MissingComponent),
    PathCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateVerdict {
    /// The margin-adjusted estimate meets the target.
    Qualifies { predicted_micros: u64 },
    /// The margin-adjusted estimate exceeds the target.
    Misses { predicted_micros: u64 },
    /// Missing or incomparable evidence: never counted as qualifying.
    Unknown(UnknownReason),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatencyAssessment {
    /// One verdict per candidate, in the order they were given.
    pub verdicts: Vec<(InferenceTarget, CandidateVerdict)>,
    /// Candidates that meet the target, in the order they were given.
    pub qualifying: Vec<InferenceTarget>,
    /// Qualifying candidates within `equivalence_pct` of the best estimate.
    pub equivalent: Vec<InferenceTarget>,
}

/// Judge every candidate against `policy`. The service estimate is the
/// candidate's warm cost when it has a usable cache for this request, its
/// cold cost otherwise; the path cost is the round trip, counted once.
pub fn assess_first_token(
    request: RequestCostInput,
    candidates: &[CandidateLatencyInput],
    policy: LatencyPolicy,
) -> LatencyAssessment {
    let target_micros = u64::from(policy.first_token_ms).saturating_mul(1_000);
    let verdicts: Vec<(InferenceTarget, CandidateVerdict)> = candidates
        .iter()
        .map(|candidate| {
            let verdict = match predicted_micros(request, candidate, policy.margin_pct) {
                Ok(predicted) if predicted <= target_micros => CandidateVerdict::Qualifies {
                    predicted_micros: predicted,
                },
                Ok(predicted) => CandidateVerdict::Misses {
                    predicted_micros: predicted,
                },
                Err(reason) => CandidateVerdict::Unknown(reason),
            };
            (candidate.cost.target.clone(), verdict)
        })
        .collect();

    let qualifying_estimates: Vec<(&InferenceTarget, u64)> = verdicts
        .iter()
        .filter_map(|(target, verdict)| match verdict {
            CandidateVerdict::Qualifies { predicted_micros } => Some((target, *predicted_micros)),
            _ => None,
        })
        .collect();
    let best = qualifying_estimates.iter().map(|(_, micros)| *micros).min();
    let equivalent = match best {
        Some(best) => {
            let bound = scale_pct(best, policy.equivalence_pct);
            qualifying_estimates
                .iter()
                .filter(|(_, micros)| *micros <= bound)
                .map(|(target, _)| (*target).clone())
                .collect()
        }
        None => Vec::new(),
    };
    LatencyAssessment {
        qualifying: qualifying_estimates
            .iter()
            .map(|(target, _)| (*target).clone())
            .collect(),
        equivalent,
        verdicts,
    }
}

fn predicted_micros(
    request: RequestCostInput,
    candidate: &CandidateLatencyInput,
    margin_pct: u16,
) -> Result<u64, UnknownReason> {
    let cold = cold_cost(request, &candidate.cost);
    // A warm estimate only helps when it is usable and cheaper; a cache entry
    // with missing restore evidence falls back to the cold estimate.
    let service = match (warm_cost(request, &candidate.cost), cold) {
        (Some(Ok(warm)), Ok(cold)) => warm.micros.min(cold.micros),
        (Some(Ok(warm)), Err(_)) => warm.micros,
        (_, Ok(cold)) => cold.micros,
        (_, Err(missing)) => return Err(UnknownReason::Service(missing)),
    };
    let path = candidate.path_rtt_ms.ok_or(UnknownReason::PathCost)?;
    let raw = service.saturating_add(u64::from(path).saturating_mul(1_000));
    Ok(scale_pct(raw, margin_pct))
}

fn scale_pct(value: u64, pct: u16) -> u64 {
    value.saturating_mul(100 + u64::from(pct)) / 100
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_cost::{CachedPrefix, TokenEstimate, TokenSource};

    fn request(prompt_tokens: u32, reusable: u32) -> RequestCostInput {
        RequestCostInput {
            prompt_tokens: Some(TokenEstimate {
                tokens: prompt_tokens,
                source: TokenSource::Measured,
            }),
            reusable_prefix_tokens: Some(reusable),
        }
    }

    fn candidate(
        port: u16,
        micros_per_token: Option<u64>,
        cached: Option<CachedPrefix>,
        path_rtt_ms: Option<u32>,
    ) -> CandidateLatencyInput {
        CandidateLatencyInput {
            cost: CandidateCostInput {
                target: InferenceTarget::Local(port),
                queue_micros: Some(0),
                prefill_micros_per_token: micros_per_token,
                cached,
            },
            path_rtt_ms,
        }
    }

    const POLICY: LatencyPolicy = LatencyPolicy {
        first_token_ms: 900,
        margin_pct: 0,
        equivalence_pct: 10,
    };

    #[test]
    fn the_issue_example_lets_both_hosts_serve_a_short_prompt() {
        // 1,000 tokens: A at 0.7 ms/token, B at 0.85 ms/token, no path cost,
        // so 700 ms and 850 ms against a 900 ms target.
        let candidates = [
            candidate(1, Some(700), None, Some(0)),
            candidate(2, Some(850), None, Some(0)),
        ];
        let assessment = assess_first_token(request(1_000, 0), &candidates, POLICY);
        assert_eq!(
            assessment.qualifying,
            [InferenceTarget::Local(1), InferenceTarget::Local(2)]
        );
        // 850 is not within 10 % of 700, so only A is equivalent to the best.
        assert_eq!(assessment.equivalent, [InferenceTarget::Local(1)]);
    }

    #[test]
    fn every_qualifying_candidate_within_the_equivalence_band_is_kept() {
        // 700, 760 and 850 ms all meet 900 ms; 760 is within 10 % of 700
        // (up to 770), 850 is not.
        let candidates = [
            candidate(1, Some(700), None, Some(0)),
            candidate(2, Some(760), None, Some(0)),
            candidate(3, Some(850), None, Some(0)),
        ];
        let assessment = assess_first_token(request(1_000, 0), &candidates, POLICY);
        assert_eq!(assessment.qualifying.len(), 3);
        assert_eq!(
            assessment.equivalent,
            [InferenceTarget::Local(1), InferenceTarget::Local(2)]
        );
    }

    #[test]
    fn a_longer_prompt_leaves_only_the_host_that_holds_it_in_cache() {
        // 3,000 tokens: A cold at 0.7 ms/token is 2,100 ms; B has 2,900 of
        // them cached, so it prefills 100 tokens plus a 30 ms restore.
        let cached = CachedPrefix {
            matched_tokens: 2_900,
            restore_micros: Some(30_000),
        };
        let candidates = [
            candidate(1, Some(700), None, Some(0)),
            candidate(2, Some(850), Some(cached), Some(0)),
        ];
        let assessment = assess_first_token(request(3_000, 2_900), &candidates, POLICY);
        assert_eq!(assessment.qualifying, [InferenceTarget::Local(2)]);
        assert!(matches!(
            assessment.verdicts[0].1,
            CandidateVerdict::Misses {
                predicted_micros: 2_100_000
            }
        ));
    }

    #[test]
    fn the_margin_and_the_path_count_against_the_target() {
        // 800 ms of service plus a 60 ms round trip is 860 ms: it meets 900 ms
        // without a margin and misses it with 10 %.
        let candidates = [candidate(1, Some(800), None, Some(60))];
        let bare = assess_first_token(request(1_000, 0), &candidates, POLICY);
        assert_eq!(bare.qualifying, [InferenceTarget::Local(1)]);
        let margined = assess_first_token(
            request(1_000, 0),
            &candidates,
            LatencyPolicy {
                margin_pct: 10,
                ..POLICY
            },
        );
        assert!(margined.qualifying.is_empty());
    }

    #[test]
    fn missing_evidence_never_qualifies() {
        let candidates = [
            // No measured prefill rate.
            candidate(1, None, None, Some(0)),
            // No path sample to a remote host.
            candidate(2, Some(100), None, None),
        ];
        let assessment = assess_first_token(request(1_000, 0), &candidates, POLICY);
        assert!(assessment.qualifying.is_empty());
        assert!(assessment.equivalent.is_empty());
        assert_eq!(
            assessment.verdicts[0].1,
            CandidateVerdict::Unknown(UnknownReason::Service(MissingComponent::PrefillRate))
        );
        assert_eq!(
            assessment.verdicts[1].1,
            CandidateVerdict::Unknown(UnknownReason::PathCost)
        );
    }

    #[test]
    fn a_cache_entry_without_restore_evidence_falls_back_to_the_cold_estimate() {
        let cached = CachedPrefix {
            matched_tokens: 900,
            restore_micros: None,
        };
        let candidates = [candidate(1, Some(700), Some(cached), Some(0))];
        let assessment = assess_first_token(request(1_000, 900), &candidates, POLICY);
        assert_eq!(
            assessment.verdicts[0].1,
            CandidateVerdict::Qualifies {
                predicted_micros: 700_000
            }
        );
    }
}
