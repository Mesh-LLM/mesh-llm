//! Reducer candidate selection + hedged call ladder.
//!
//! The reducer is invoked when arbitration can't reach a decision from
//! worker outputs alone. Rather than picking a single reducer model and
//! eating its timeout when it's slow/broken, we keep an ordered ladder
//! of candidates (big-tier first, small-tier as last resort) and call
//! them with hedging: start the first, hedge to the next after
//! `hedge_delay`, race for the first OK. On fast errors, jump to the
//! next candidate immediately.

use crate::GatewayConfig;
use crate::backend::{ModelBackend, SamplingParams, call_backend};
use crate::worker;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Pick the reducer — prefers first model (typically local, zero RTT).
/// Reducer candidates in priority order: big-tier models first (multi-
/// digit B, or names with no size like MiniMax), then small-tier models
/// as last-resort fallback. Callers should try each in order and stop
/// on the first that succeeds, so a broken big-tier peer (e.g. a peer
/// running a stale binary that 502s on tool calls) doesn't take down
/// the whole reducer step.
pub(crate) fn reducer_candidates(config: &GatewayConfig) -> Vec<(String, usize)> {
    // Host-provided actor priority wins. In the asymmetric tool path the actor
    // is the model that actually emits the tool call, so it must be the best
    // available tool-caller — a judgement the host makes from gossiped
    // `tool_use` capability, model size, and peer health, none of which this
    // crate can see. `actor_candidates` are indices into `config.models`,
    // best-first; we translate them to `(name, backend_index)` and skip any
    // stale/out-of-range index defensively.
    if !config.actor_candidates.is_empty() {
        let ordered: Vec<(String, usize)> = config
            .actor_candidates
            .iter()
            .filter_map(|&i| config.models.get(i))
            .map(|m| (m.name.clone(), m.backend_index))
            .collect();
        if !ordered.is_empty() {
            return ordered;
        }
        // Every provided index was stale — fall through to the size heuristic
        // rather than return empty and fail the turn.
    }

    // No host guidance (or all indices stale): fall back to capacity order,
    // strongest first. Uses the same ordering as role assignment — big tier
    // before small, and by verified size within each tier — so the reducer
    // ladder starts at the largest model the host has verified rather than
    // whichever big-tier model happened to arrive first.
    let mut sorted = config.models.clone();
    worker::sort_by_capacity_desc(&mut sorted);
    // Intentionally allow returning empty: hedged_reducer_call's empty-input
    // path surfaces the right error. A fake ("unknown", 0) entry would call
    // backend_index=0 with a bogus model name and mask real bugs.
    sorted
        .iter()
        .map(|m| (m.name.clone(), m.backend_index))
        .collect()
}

/// Successful hedged-reducer outcome.
///
/// `attempts` reports how many candidates were actually spawned:
/// `1` = clean happy path (cand 0 returned before hedge fired),
/// `≥2` = hedge fired or a fast-fail cascaded to the next candidate.
#[derive(Debug)]
pub(crate) struct HedgedReducerOk {
    pub winner: String,
    pub text: String,
    pub attempts: u32,
}

/// Failure outcome from the hedged-reducer ladder.
///
/// Carries `attempts` so observability ("we tried N times and all failed")
/// stays accurate even on the all-fail path. Without this the caller
/// reports `attempts=0` even when 2+ candidates actually ran, which is
/// what bit us in the live goose test.
#[derive(Debug)]
pub(crate) struct HedgedReducerErr {
    pub err: String,
    pub attempts: u32,
}

/// Call the ordered reducer candidates with hedging.
///
/// Starts the first candidate immediately. If it hasn't returned within
/// `hedge_delay`, the next candidate is started in parallel without
/// cancelling the in-flight one — we race for the first OK. If a candidate
/// errors, the next one is started immediately (no hedge wait).
///
/// Returns the first successful [`HedgedReducerOk`]. If every candidate
/// fails, returns the last error encountered.
///
/// Cost shape:
/// - Happy path (cand 0 OK in <hedge_delay): exactly 1 backend call.
/// - Slow happy path (cand 0 OK in hedge_delay..reducer_timeout): up to 2
///   overlapping calls, accept whichever wins, cancel the loser.
/// - Fast-fail (cand 0 errors quickly): immediate move to cand 1, 1 call.
/// - All fail: at most N calls, capped at reducer_timeout + (N-1)·hedge_delay
///   end-to-end (vs N·reducer_timeout sequentially).
pub(crate) async fn hedged_reducer_call(
    backends: &[Arc<dyn ModelBackend>],
    candidates: Vec<(String, usize)>,
    messages: Vec<Value>,
    tools: Option<Value>,
    timeout: Duration,
    hedge_delay: Duration,
    enable_thinking: Option<bool>,
) -> Result<HedgedReducerOk, HedgedReducerErr> {
    use tokio::task::JoinSet;

    if candidates.is_empty() {
        return Err(HedgedReducerErr {
            err: "no reducer candidates".into(),
            attempts: 0,
        });
    }

    let mut join_set: JoinSet<(String, Result<String, String>)> = JoinSet::new();
    let mut remaining = candidates.into_iter();
    let mut last_err: Option<String> = None;
    let mut attempts: u32 = 0;

    // Spawn a single candidate. Captures `enable_thinking` from the
    // outer scope so each candidate honours the caller's reasoning
    // override consistently.
    let spawn = |join_set: &mut JoinSet<(String, Result<String, String>)>,
                 backends: &[Arc<dyn ModelBackend>],
                 name: String,
                 backend_idx: usize,
                 messages: Vec<Value>,
                 tools: Option<Value>,
                 timeout: Duration| {
        let backend = backends[backend_idx].clone();
        tracing::info!("moa: reducer hedge → {name}");
        join_set.spawn(async move {
            let result = call_backend(
                &*backend,
                &name,
                &messages,
                tools.as_ref(),
                2048,
                timeout,
                SamplingParams::reducer().with_thinking(enable_thinking),
            )
            .await;
            // The reducer's output *is* the final response, so there is no
            // later arbitration step that could act on truncation. Keep the
            // text; the 2048-token reducer budget is well clear of the
            // worker budget where truncation actually bites.
            (name, result.map(|reply| reply.text))
        });
    };

    // Start candidate 0.
    if let Some((name, idx)) = remaining.next() {
        spawn(
            &mut join_set,
            backends,
            name,
            idx,
            messages.clone(),
            tools.clone(),
            timeout,
        );
        attempts += 1;
    }

    // Try to spawn the next candidate. Returns true while there was one to
    // spawn; once it returns false the caller should stop arming the hedge
    // timer.
    let try_spawn_next = |join_set: &mut JoinSet<(String, Result<String, String>)>,
                          remaining: &mut std::vec::IntoIter<(String, usize)>,
                          attempts: &mut u32| {
        if let Some((next_name, next_idx)) = remaining.next() {
            spawn(
                join_set,
                backends,
                next_name,
                next_idx,
                messages.clone(),
                tools.clone(),
                timeout,
            );
            *attempts += 1;
            true
        } else {
            false
        }
    };

    // Race in-flight calls against a hedge timer. Once `remaining` is
    // exhausted we drop the timer entirely and just await join_next() so
    // we don't wake every hedge_delay just to no-op.
    let mut remaining_exhausted = false;
    while !join_set.is_empty() {
        // Without the hedge timer: just await whichever candidate finishes.
        if remaining_exhausted {
            match join_set.join_next().await {
                Some(Ok((name, Ok(text)))) => {
                    join_set.abort_all();
                    while join_set.join_next().await.is_some() {}
                    return Ok(HedgedReducerOk {
                        winner: name,
                        text,
                        attempts,
                    });
                }
                Some(Ok((name, Err(e)))) => {
                    tracing::warn!("moa: reducer {name} failed: {e} (no more candidates)");
                    last_err = Some(e);
                }
                Some(Err(join_err)) => {
                    tracing::warn!("moa: reducer task join error: {join_err}");
                }
                None => break,
            }
            continue;
        }

        let hedge_sleep = tokio::time::sleep(hedge_delay);
        tokio::pin!(hedge_sleep);

        tokio::select! {
            // A candidate finished.
            joined = join_set.join_next() => {
                match joined {
                    Some(Ok((name, Ok(text)))) => {
                        // First success wins. Cancel the rest.
                        join_set.abort_all();
                        // Drain so cancellations complete cleanly.
                        while join_set.join_next().await.is_some() {}
                        return Ok(HedgedReducerOk {
                            winner: name,
                            text,
                            attempts,
                        });
                    }
                    Some(Ok((name, Err(e)))) => {
                        tracing::warn!(
                            "moa: reducer {name} failed: {e}, trying next candidate"
                        );
                        last_err = Some(e);
                        // Start the next candidate immediately on failure.
                        if !try_spawn_next(&mut join_set, &mut remaining, &mut attempts) {
                            remaining_exhausted = true;
                        }
                    }
                    Some(Err(join_err)) => {
                        tracing::warn!("moa: reducer task join error: {join_err}");
                        if !try_spawn_next(&mut join_set, &mut remaining, &mut attempts) {
                            remaining_exhausted = true;
                        }
                    }
                    None => break,
                }
            }
            // Hedge timer fires: start another candidate alongside in-flight ones.
            _ = &mut hedge_sleep => {
                if !try_spawn_next(&mut join_set, &mut remaining, &mut attempts) {
                    remaining_exhausted = true;
                }
            }
        }
    }

    Err(HedgedReducerErr {
        err: last_err.unwrap_or_else(|| "all reducer candidates failed".into()),
        attempts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    enum FakeBehavior {
        Text(Duration, String),
        Raw(Duration, Value),
        Error(Duration, String),
    }

    struct FakeBackend {
        behaviors: std::sync::Mutex<std::collections::HashMap<String, FakeBehavior>>,
        calls: AtomicUsize,
    }

    impl FakeBackend {
        fn new(behaviors: Vec<(&str, FakeBehavior)>) -> Arc<Self> {
            let mut map = std::collections::HashMap::new();
            for (n, b) in behaviors {
                map.insert(n.to_string(), b);
            }
            Arc::new(FakeBackend {
                behaviors: std::sync::Mutex::new(map),
                calls: AtomicUsize::new(0),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ModelBackend for FakeBackend {
        async fn chat_completion(
            &self,
            model: &str,
            _messages: &[Value],
            _tools: Option<&Value>,
            _max_tokens: u32,
            _timeout: Duration,
            _sampling: SamplingParams,
        ) -> Result<Value, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let behavior = self.behaviors.lock().unwrap().get(model).cloned();
            match behavior {
                Some(FakeBehavior::Text(d, body)) => {
                    tokio::time::sleep(d).await;
                    Ok(json!({
                        "choices": [{"message": {"content": body}}],
                    }))
                }
                Some(FakeBehavior::Raw(d, body)) => {
                    tokio::time::sleep(d).await;
                    Ok(body)
                }
                Some(FakeBehavior::Error(d, msg)) => {
                    tokio::time::sleep(d).await;
                    Err(msg)
                }
                None => Err(format!("unconfigured model: {model}")),
            }
        }
    }

    /// A config carrying only what `reducer_candidates` reads.
    fn config_with_models(models: Vec<crate::backend::ModelEntry>) -> GatewayConfig {
        GatewayConfig {
            backends: Vec::new(),
            models,
            worker_timeout: Duration::from_secs(30),
            hedge_delay: Duration::from_millis(200),
            reducer_timeout: Duration::from_secs(2),
            first_answer_grace: Duration::ZERO,
            strong_patience: Duration::ZERO,
            enable_thinking: None,
            actor_candidates: Vec::new(),
            reference_policy: Default::default(),
            refinement_policy: Default::default(),
        }
    }

    fn sized(name: &str, index: usize, size: Option<f64>) -> crate::backend::ModelEntry {
        crate::backend::ModelEntry::new(name.to_string(), index).with_parameter_count_b(size)
    }

    #[test]
    fn reducer_ladder_starts_at_the_largest_verified_model() {
        // Partitioning by tier alone left big-tier models in input order, so a
        // pool arriving as [70B, 8B, 32B] could hedge to the 32B before the
        // verified 70B. The ladder must be strongest-first.
        let config = config_with_models(vec![
            sized("big-70b", 0, Some(70.6)),
            sized("small-8b", 1, Some(8.2)),
            sized("mid-32b", 2, Some(32.8)),
        ]);

        let candidates = reducer_candidates(&config);
        let order: Vec<&str> = candidates.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(order, vec!["big-70b", "mid-32b", "small-8b"]);
    }

    #[test]
    fn reducer_ladder_prefers_a_verified_size_over_an_unknown_one() {
        // Unknown size counts as big-tier, but must not lead the ladder ahead
        // of a model the host verified to be large.
        let config = config_with_models(vec![
            sized("unknown-size", 0, None),
            sized("verified-70b", 1, Some(70.0)),
        ]);

        let candidates = reducer_candidates(&config);
        assert_eq!(
            candidates.first().map(|(name, _)| name.as_str()),
            Some("verified-70b")
        );
        // The unsized model stays on the ladder as a fallback rather than being
        // dropped — a broken primary must still have somewhere to hedge.
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn reducer_ladder_keeps_input_order_for_equally_ranked_models() {
        // A pool of same-tier, unsized models has no capacity signal to sort on,
        // so it must come back in the order the host supplied. Producing the
        // strongest-first order by reversing an ascending sort flipped these
        // instead, which changed the primary reducer pick and broke tool-call
        // survival in fan-out (`sim_real_traces`).
        let config = config_with_models(vec![
            sized("first", 0, None),
            sized("second", 1, None),
            sized("third", 2, None),
        ]);

        let candidates = reducer_candidates(&config);
        let order: Vec<&str> = candidates.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(order, vec!["first", "second", "third"]);
    }

    #[test]
    fn reducer_ladder_keeps_host_actor_priority() {
        // Host guidance wins over local size ordering: the host sees tool_use
        // capability and peer health, which this crate cannot.
        let mut config = config_with_models(vec![
            sized("big-70b", 0, Some(70.6)),
            sized("small-8b", 1, Some(8.2)),
        ]);
        config.actor_candidates = vec![1];

        let candidates = reducer_candidates(&config);
        assert_eq!(candidates, vec![("small-8b".to_string(), 1)]);
    }

    #[tokio::test]
    async fn hedged_reducer_happy_path_calls_only_first() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::Text(Duration::from_millis(50), "alpha-resp".into()),
            ),
            (
                "beta",
                FakeBehavior::Text(Duration::from_millis(50), "beta-resp".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_secs(5),
            None,
        )
        .await;

        let ok = res.expect("happy path returns Ok");
        assert_eq!(ok.winner, "alpha", "first candidate should win");
        assert_eq!(ok.attempts, 1, "happy path spawns exactly one candidate");
        assert_eq!(fake.calls(), 1, "only one backend call on happy path");
    }

    #[tokio::test]
    async fn hedged_reducer_slow_first_hedges_to_second() {
        let fake = FakeBackend::new(vec![
            // alpha takes longer than hedge_delay; beta is fast.
            (
                "alpha",
                FakeBehavior::Text(Duration::from_millis(800), "alpha-late".into()),
            ),
            (
                "beta",
                FakeBehavior::Text(Duration::from_millis(100), "beta-fast".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_millis(100),
            None,
        )
        .await;

        let ok = res.expect("hedge returns Ok");
        assert_eq!(
            ok.winner, "beta",
            "hedge winner should be the faster second candidate"
        );
        assert_eq!(ok.text, "beta-fast");
        assert_eq!(ok.attempts, 2, "hedge fires the second candidate");
        assert_eq!(fake.calls(), 2, "both candidates should have been issued");
    }

    #[tokio::test]
    async fn hedged_reducer_reasoning_only_primary_falls_through_to_next_candidate() {
        let reasoning_only = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "I'll start by inspecting the project structure."
                }
            }]
        });
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::Raw(Duration::from_millis(10), reasoning_only),
            ),
            (
                "beta",
                FakeBehavior::Text(Duration::from_millis(10), "beta-action".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let result = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_secs(60),
            Some(false),
        )
        .await
        .expect("reasoning-only output must not win the reducer hedge");

        assert_eq!(result.winner, "beta");
        assert_eq!(result.text, "beta-action");
        assert_eq!(result.attempts, 2);
        assert_eq!(fake.calls(), 2);
    }

    #[tokio::test]
    async fn hedged_reducer_fast_fail_starts_next_immediately() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::Error(Duration::from_millis(50), "boom".into()),
            ),
            (
                "beta",
                FakeBehavior::Text(Duration::from_millis(100), "beta-ok".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let start = tokio::time::Instant::now();
        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            // Large hedge_delay — the fast-fail path must not wait for it.
            Duration::from_secs(60),
            None,
        )
        .await;

        let ok = res.expect("fail-then-recover returns Ok");
        let elapsed = start.elapsed();
        assert_eq!(ok.winner, "beta");
        assert_eq!(ok.text, "beta-ok");
        assert_eq!(
            ok.attempts, 2,
            "fast-fail should cascade to a second attempt"
        );
        assert_eq!(fake.calls(), 2);
        assert!(
            elapsed < Duration::from_secs(10),
            "fast-fail should not wait for hedge_delay; took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn hedged_reducer_all_fail_returns_last_err() {
        let fake = FakeBackend::new(vec![
            (
                "alpha",
                FakeBehavior::Error(Duration::from_millis(10), "alpha-boom".into()),
            ),
            (
                "beta",
                FakeBehavior::Error(Duration::from_millis(10), "beta-boom".into()),
            ),
        ]);
        let backends: Vec<Arc<dyn ModelBackend>> = vec![fake.clone(), fake.clone()];
        let candidates = vec![("alpha".into(), 0), ("beta".into(), 1)];

        let res = hedged_reducer_call(
            &backends,
            candidates,
            vec![],
            None,
            Duration::from_secs(15),
            Duration::from_millis(200),
            None,
        )
        .await;

        let err = res.expect_err("all-fail returns Err");
        assert!(
            err.err.contains("boom"),
            "should surface a backend error: {}",
            err.err
        );
        assert_eq!(err.attempts, 2, "all-fail still reports attempts spawned");
        assert_eq!(fake.calls(), 2);
    }
}
