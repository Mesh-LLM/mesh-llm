//! Cross-peer refinement round (Together's `layers`) for text turns.
//!
//! Round 1 workers answer independently. In the refinement round every worker
//! sees *all* round-1 drafts and rewrites its own answer, after which the
//! reducer synthesizes the refined set.
//!
//! Why it exists: measured over 40 preregistered reasoning prompts x 3 draws
//! (`evals/moa-openrouter/RESULTS.md`), a pool of four 8B-class models beat its
//! own best member only when the refinement round was present —
//! single-round synthesis alone was indistinguishable from the aggregator
//! acting solo (p=0.37), while refine-then-synthesize won (p=5.2e-05). For a
//! strong aggregator the extra round adds much less, so the round is gated.
//!
//! Mesh flavour: the round is best-effort. It refines with whichever drafts
//! arrived, bounds its own wait, and on any shortfall returns the round-1
//! outputs unchanged rather than failing the turn.

use crate::backend::{SamplingParams, call_backend};
use crate::context;
use crate::normalize::{self, WorkerOutput};
use crate::session::Session;
use crate::worker;
use crate::{GatewayConfig, RefinementPolicy, WorkerSummary};
use std::time::Instant;

/// Minimum round-1 drafts needed for refinement to be meaningful.
pub(crate) const MIN_DRAFTS: usize = 2;

/// Share of the worker budget the refinement round may spend.
///
/// Refinement sits between round 1 and the reducer, so an unbounded round would
/// let one turn pay three sequential worker budgets. Half a budget is enough
/// for a pool that answered round 1 promptly, and caps the worst case at ~2.5x
/// a plain turn instead of 3x.
const REFINEMENT_BUDGET_NUMERATOR: u32 = 1;
const REFINEMENT_BUDGET_DENOMINATOR: u32 = 2;

fn refinement_budget(worker_timeout: std::time::Duration) -> std::time::Duration {
    worker_timeout / REFINEMENT_BUDGET_DENOMINATOR * REFINEMENT_BUDGET_NUMERATOR
}

/// Should this text turn run a refinement round?
///
/// `Auto` follows the evidence: refine when the pool is dominated by
/// small-tier models (where the round is what makes the collective beat its
/// best member) and skip it when a big-tier model is present to synthesize
/// directly, since there the extra round buys much less than it costs.
pub(crate) fn should_refine(config: &GatewayConfig, drafts: usize) -> bool {
    drafts >= MIN_DRAFTS && refinement_expected(config)
}

/// Will this config refine, given enough drafts?
///
/// Depends only on policy and pool shape, so it can be answered *before*
/// dispatch — which the text path needs in order to decide whether the answer
/// grace may pre-empt the round.
pub(crate) fn refinement_expected(config: &GatewayConfig) -> bool {
    match config.refinement_policy {
        RefinementPolicy::Never => false,
        RefinementPolicy::Always => config.models.len() >= MIN_DRAFTS,
        RefinementPolicy::Auto => {
            if config.models.len() < MIN_DRAFTS {
                return false;
            }
            // Refine when the drafts will be correlated — that is where the
            // cross-peer round measurably pays:
            //   * homogeneous pool (same model, incl. repeated instances or
            //     quant variants): same-model 32B ×2 wins 48/2 with refinement
            //     vs 35/10 without;
            //   * all-small pool: 8B needs the round to beat its best member.
            // A diverse pool that already has a big-tier synthesizer gains
            // ~nothing from the extra fan-out (mid diverse: 49/6 layered vs
            // 47/4 single-round), so skip it there to save the round-trip.
            // See `evals/moa-openrouter/RESULTS.md`.
            if worker::pool_is_homogeneous(&config.models) {
                return true;
            }
            config
                .models
                .iter()
                .all(|m| worker::model_name_is_small_tier(&m.name))
        }
    }
}

/// Run one refinement round over `drafts`.
///
/// Returns the refined outputs plus a summary per refining worker. Any worker
/// that fails or times out simply doesn't contribute; if fewer than
/// [`MIN_DRAFTS`] refinements land we return `None` so the caller keeps the
/// round-1 outputs.
pub(crate) async fn refine_round(
    config: &GatewayConfig,
    session: &Session,
    drafts: &[WorkerOutput],
) -> Option<(Vec<WorkerOutput>, Vec<WorkerSummary>)> {
    let assignments = worker::assign_roles(&config.models);
    let texts: Vec<String> = drafts.iter().map(|d| d.payload.clone()).collect();

    let mut join_set = tokio::task::JoinSet::new();
    for a in &assignments {
        let packed = context::pack_for_refinement(session, &texts);
        let model = a.model_name.clone();
        let role = a.role;
        let backend = config.backends[a.backend_index].clone();
        let timeout = config.worker_timeout;
        let thinking = config.enable_thinking;
        join_set.spawn(async move {
            let t0 = Instant::now();
            let result = call_backend(
                &*backend,
                &model,
                &packed.messages,
                None, // text path: refinement never carries tools
                packed.max_tokens,
                timeout,
                SamplingParams::worker().with_thinking(thinking),
            )
            .await;
            (model, role, result, t0.elapsed().as_millis() as u64)
        });
    }

    // Bounded well inside the worker budget. Refinement is an *optional*
    // improvement inserted between round 1 and the reducer, so at full
    // `worker_timeout` a slow pool could pay three sequential budgets for one
    // turn — the worst outcome on exactly the high-latency meshes this feature
    // targets. Give the round a fraction of the budget and fall back to the
    // round-1 drafts if the pool can't refine in that time.
    let deadline = tokio::time::sleep(refinement_budget(config.worker_timeout));
    tokio::pin!(deadline);

    let mut refined = Vec::new();
    let mut summaries = Vec::new();
    loop {
        tokio::select! {
            biased;
            joined = join_set.join_next() => {
                let Some(joined) = joined else { break };
                match joined {
                    Ok((model, role, Ok(reply), elapsed)) => {
                        if reply.text.trim().is_empty() {
                            continue;
                        }
                        let mut out = normalize::normalize_worker_output(
                            &reply.text, &model, role, elapsed,
                        );
                        out.truncated = reply.truncated;
                        summaries.push(WorkerSummary {
                            model,
                            role,
                            succeeded: true,
                            elapsed_ms: elapsed,
                            output_kind: Some(out.kind),
                            confidence: Some(out.confidence),
                        });
                        refined.push(out);
                    }
                    Ok((model, role, Err(e), elapsed)) => {
                        tracing::warn!("moa: refinement worker {model} failed: {e}");
                        summaries.push(WorkerSummary {
                            model,
                            role,
                            succeeded: false,
                            elapsed_ms: elapsed,
                            output_kind: None,
                            confidence: None,
                        });
                    }
                    Err(e) => tracing::warn!("moa: refinement task cancelled: {e}"),
                }
            }
            _ = &mut deadline => {
                tracing::info!(
                    "moa: refinement deadline reached with {} refined draft(s)",
                    refined.len(),
                );
                break;
            }
        }
    }
    join_set.abort_all();

    if refined.len() < MIN_DRAFTS {
        tracing::info!(
            "moa: refinement produced {} draft(s), keeping round-1 outputs",
            refined.len()
        );
        return None;
    }
    Some((refined, summaries))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ModelEntry;
    use std::time::Duration;

    fn config(models: &[&str], policy: RefinementPolicy) -> GatewayConfig {
        GatewayConfig {
            backends: Vec::new(),
            models: models
                .iter()
                .map(|n| ModelEntry {
                    name: (*n).to_string(),
                    backend_index: 0,
                })
                .collect(),
            worker_timeout: Duration::from_secs(60),
            hedge_delay: Duration::from_secs(5),
            reducer_timeout: Duration::from_secs(60),
            first_answer_grace: Duration::ZERO,
            strong_patience: Duration::ZERO,
            enable_thinking: Some(false),
            actor_candidates: Vec::new(),
            reference_policy: Default::default(),
            refinement_policy: policy,
        }
    }

    /// The measured case: an all-small pool beats its best member only with
    /// the refinement round, so Auto must run it.
    #[test]
    fn auto_refines_an_all_small_pool() {
        let c = config(
            &["Qwen3-8B", "Llama-3.1-8B", "Ministral-8B"],
            RefinementPolicy::Auto,
        );
        assert!(should_refine(&c, 3));
    }

    /// A *diverse* pool with a big-tier synthesizer gains ~nothing from the
    /// extra round (measured 49/6 layered vs 47/4 single-round), so Auto skips
    /// it to save the round-trip.
    #[test]
    fn auto_skips_for_a_diverse_big_tier_pool() {
        let c = config(&["Qwen3-32B", "Qwen3-8B"], RefinementPolicy::Auto);
        assert!(!should_refine(&c, 2));
    }

    /// A *homogeneous* big-tier pool (same model, incl. repeated instances)
    /// produces correlated drafts, and the round is what pulls them apart —
    /// same-model 32B ×2 wins 48/2 with refinement vs 35/10 without. Auto must
    /// refine here even though the members are big-tier.
    #[test]
    fn auto_refines_a_homogeneous_big_tier_pool() {
        let c = config(&["Qwen3-32B", "Qwen3-32B"], RefinementPolicy::Auto);
        assert!(should_refine(&c, 2));
    }

    /// Repeated instances of one model (same alias) are homogeneous.
    #[test]
    fn auto_refines_repeated_instances_of_one_model() {
        let c = config(
            &["Qwen3-32B", "Qwen3-32B", "Qwen3-32B"],
            RefinementPolicy::Auto,
        );
        assert!(should_refine(&c, 3));
    }

    #[test]
    fn refinement_needs_at_least_two_drafts() {
        let c = config(&["Qwen3-8B", "Llama-3.1-8B"], RefinementPolicy::Always);
        assert!(!should_refine(&c, 1));
        assert!(should_refine(&c, 2));
    }

    #[test]
    fn explicit_policies_override_pool_shape() {
        let never = config(&["Qwen3-8B", "Llama-3.1-8B"], RefinementPolicy::Never);
        assert!(!should_refine(&never, 3));
        let always = config(&["Qwen3-32B", "Qwen3-8B"], RefinementPolicy::Always);
        assert!(should_refine(&always, 2));
    }

    #[test]
    fn auto_is_the_default() {
        assert_eq!(RefinementPolicy::default(), RefinementPolicy::Auto);
    }

    /// Refinement must not spend a full worker budget: it sits between round 1
    /// and the reducer, so an unbounded round would make one turn pay three
    /// sequential budgets on exactly the slow meshes this feature targets.
    #[test]
    fn refinement_budget_is_a_fraction_of_the_worker_timeout() {
        let budget = refinement_budget(Duration::from_secs(60));
        assert_eq!(budget, Duration::from_secs(30));
        assert!(budget < Duration::from_secs(60));
    }
}
