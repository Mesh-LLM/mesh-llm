//! Optional prefix preparation owns its failures and completed-work accounting.

use super::*;
use futures_util::Stream;

pub(super) struct PreparedPrefixes {
    pub usage: DecisionReadoutUsage,
    pub reuse_cache: bool,
}

impl StageOpenAiBackend {
    /// Prepare shared prefixes without making cache optimization a scoring prerequisite.
    pub(super) async fn warm_decision_prefixes(
        &self,
        plan: &DecisionPlan,
        cancellation: &CancellationToken,
    ) -> InferenceResult<PreparedPrefixes> {
        let mut usage = DecisionReadoutUsage::default();
        if !self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.should_record() && kv.should_lookup())
        {
            return Ok(PreparedPrefixes {
                usage,
                reuse_cache: true,
            });
        }
        if plan.questions.len() > 1 && plan.shared_prefix_tokens > 0 {
            let tokens = plan.questions[0].prompt[..plan.shared_prefix_tokens].to_vec();
            let result = self.run_readout_job(warmup_job(tokens), cancellation).await;
            if !accumulate_preparation(stream::iter([result]), cancellation, &mut usage).await? {
                return Ok(PreparedPrefixes {
                    usage,
                    reuse_cache: false,
                });
            }
        }
        let width = self.generation_limit.current_limit().clamp(1, 64);
        let prompts = plan
            .questions
            .iter()
            .filter(|q| q.method == ReadoutMethod::CandidateLogLikelihood && q.prompt.len() > 1)
            .map(|q| warmup_job(q.prompt[..q.prompt.len() - 1].to_vec()))
            .collect::<Vec<_>>();
        let results = stream::iter(
            prompts
                .into_iter()
                .map(|job| self.run_readout_job(job, cancellation)),
        )
        .buffer_unordered(width);
        let reuse_cache = accumulate_preparation(results, cancellation, &mut usage).await?;
        Ok(PreparedPrefixes { usage, reuse_cache })
    }
}

/// Drain preparation, retaining completed work and propagating only cancellation.
async fn accumulate_preparation(
    results: impl Stream<Item = Result<ReadoutResult, ReadoutFailure>>,
    cancellation: &CancellationToken,
    usage: &mut DecisionReadoutUsage,
) -> InferenceResult<bool> {
    let mut reuse_cache = true;
    futures_util::pin_mut!(results);
    while let Some(result) = results.next().await {
        let (evaluated, cached) = match result {
            Ok(result) => (result.evaluated_tokens, result.cached_tokens),
            Err(failure) => {
                reuse_cache = false;
                if failure.error.body().error.code.as_deref() == Some("request_cancelled") {
                    cancellation.cancel();
                } else {
                    tracing::warn!(error = %failure.error, "decision prefix preparation failed; scoring will recompute without prefix caching");
                }
                // A failed warmup owns an isolated session, which execute() drops
                // before yielding this result. Never reuse its unfinished lane.
                (failure.evaluated_tokens, failure.cached_tokens)
            }
        };
        usage.evaluated_tokens += evaluated;
        usage.cached_tokens += cached;
        usage.cache_preparation_tokens += evaluated;
    }
    if cancellation.is_cancelled() {
        return Err(InferenceError::cancelled("decision preparation cancelled"));
    }
    Ok(reuse_cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failed optional warmup neither cancels scoring nor discards completed work.
    #[tokio::test]
    async fn failed_warmup_preserves_accounting_and_allows_scoring() {
        let cancellation = CancellationToken::new();
        let mut usage = DecisionReadoutUsage {
            evaluated_tokens: 17,
            cached_tokens: 3,
            cache_preparation_tokens: 0,
        };
        let results = stream::iter([
            Ok(ReadoutResult {
                scores: vec![],
                evaluated_tokens: 4,
                cached_tokens: 2,
            }),
            Err(ReadoutFailure {
                error: InferenceError::backend("transient downstream checkout failure").into(),
                evaluated_tokens: 0,
                cached_tokens: 0,
            }),
            Err(ReadoutFailure {
                error: InferenceError::backend("record failed after one completed prefill").into(),
                evaluated_tokens: 5,
                cached_tokens: 1,
            }),
            Ok(ReadoutResult {
                scores: vec![],
                evaluated_tokens: 6,
                cached_tokens: 2,
            }),
        ]);
        let reuse_cache = accumulate_preparation(results, &cancellation, &mut usage)
            .await
            .unwrap();
        assert!(
            !reuse_cache,
            "failed cache preparation requires fresh uncached scoring"
        );
        assert!(!cancellation.is_cancelled());
        assert_eq!(
            usage,
            DecisionReadoutUsage {
                evaluated_tokens: 32,
                cached_tokens: 8,
                cache_preparation_tokens: 15,
            }
        );
    }

    /// Cancellation remains fatal and does not skip draining other admitted warmups.
    #[tokio::test]
    async fn cancelled_warmup_drains_before_returning_error() {
        let cancellation = CancellationToken::new();
        let mut usage = DecisionReadoutUsage::default();
        let results = stream::iter([
            Err(InferenceError::cancelled("client disconnected").into()),
            Ok(ReadoutResult {
                scores: vec![],
                evaluated_tokens: 9,
                cached_tokens: 2,
            }),
        ]);
        let error = accumulate_preparation(results, &cancellation, &mut usage)
            .await
            .unwrap_err();
        assert_eq!(
            error.body().error.code.as_deref(),
            Some("request_cancelled")
        );
        assert!(cancellation.is_cancelled());
        assert_eq!(usage.cache_preparation_tokens, 9);
        assert_eq!(usage.cached_tokens, 2);
    }
}
