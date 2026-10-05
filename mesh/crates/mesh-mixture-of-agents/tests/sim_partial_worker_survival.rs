//! Pin the partial-survival robustness contract: when *some* dispatched
//! workers fail mid-turn but at least one answers, the turn must still complete
//! with the survivor(s) — never hang, never fail.
//!
//! `sim_all_workers_fail` covers total failure (clean structured error). This
//! covers the far more common mesh reality: nodes flicker, so a subset of the
//! committee dies mid-turn while the rest answer. MoA must degrade to the
//! survivors, not the error path.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

/// Answers with fixed text after a delay.
struct AnswerBackend {
    text: String,
    delay: Duration,
}

/// Always fails, simulating a peer that dropped mid-turn.
struct DeadBackend;

impl AnswerBackend {
    fn new(text: impl Into<String>, delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            delay,
        })
    }
}

#[async_trait]
impl moa::ModelBackend for AnswerBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        tokio::time::sleep(self.delay).await;
        Ok(json!({"choices": [{"message": {"content": self.text}, "finish_reason": "stop"}]}))
    }
}

#[async_trait]
impl moa::ModelBackend for DeadBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        Err("peer dropped mid-turn".into())
    }
}

fn config(backends: Vec<Arc<dyn moa::ModelBackend>>, names: &[&str]) -> moa::GatewayConfig {
    let models = names
        .iter()
        .enumerate()
        .map(|(i, n)| moa::ModelEntry::new((*n).to_string(), i))
        .collect();
    moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(5),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(5),
        first_answer_grace: Duration::ZERO,
        strong_patience: Duration::ZERO,
        enable_thinking: Some(false),
        actor_candidates: Vec::new(),
        reference_policy: Default::default(),
        refinement_policy: Default::default(),
    }
}

fn request() -> Value {
    json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "explain backpressure"}],
        "max_tokens": 256,
    })
}

/// Half the committee dies, the rest answer: the turn completes with the
/// survivors, and their failure is accounted rather than hanging.
#[tokio::test(flavor = "multi_thread")]
async fn turn_completes_when_some_workers_die() {
    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![
        AnswerBackend::new("queues fill up when the consumer is slow", Duration::ZERO),
        Arc::new(DeadBackend),
        AnswerBackend::new("it is a flow-control signal upstream", Duration::ZERO),
        Arc::new(DeadBackend),
    ];
    let cfg = config(
        backends,
        &["Qwen3-8B", "Llama-3.1-8B", "Ministral-8B", "Granite-4.1-8B"],
    );

    let result = moa::handle_turn(&cfg, &request()).await;

    assert_ne!(
        result.turn_kind,
        moa::TurnKind::Failed,
        "turn must complete with survivors when only some workers fail"
    );
    let body = serde_json::to_string(&result.response_body).unwrap();
    assert!(
        body.contains("choices"),
        "a well-formed response must be returned, got: {body}"
    );
    // Every dispatched worker is accounted for, dead ones included — none is
    // silently dropped. We deliberately do NOT assert exact succeeded/failed
    // counts: early-exit consensus may abort a live worker once a usable answer
    // is in hand, so the split between "succeeded" and "aborted" is timing
    // dependent. The contract is that all four appear and the two dead ones are
    // recorded as not-succeeded.
    assert_eq!(
        result.worker_summaries.len(),
        4,
        "all four dispatched workers must appear in summaries"
    );
    let failed = result
        .worker_summaries
        .iter()
        .filter(|s| !s.succeeded)
        .count();
    assert!(
        failed >= 2,
        "the two dead workers must be recorded as failed, not dropped (got {failed})"
    );
}

/// A single survivor is still enough: three of four die, the turn answers.
#[tokio::test(flavor = "multi_thread")]
async fn turn_completes_with_a_lone_survivor() {
    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![
        Arc::new(DeadBackend),
        Arc::new(DeadBackend),
        AnswerBackend::new("the one worker that lived", Duration::ZERO),
        Arc::new(DeadBackend),
    ];
    let cfg = config(
        backends,
        &["Qwen3-8B", "Llama-3.1-8B", "Ministral-8B", "Granite-4.1-8B"],
    );

    let result = moa::handle_turn(&cfg, &request()).await;

    assert_ne!(result.turn_kind, moa::TurnKind::Failed);
    let body = serde_json::to_string(&result.response_body).unwrap();
    assert!(body.contains("choices"), "got: {body}");
}

/// A slow-dying worker must not hold the turn hostage: the survivor answers
/// immediately and the turn does not wait out a long tail on the failing one.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_failing_worker_does_not_stall_the_turn() {
    struct SlowDead;
    #[async_trait]
    impl moa::ModelBackend for SlowDead {
        async fn chat_completion(
            &self,
            _m: &str,
            _msg: &[Value],
            _t: Option<&Value>,
            _mt: u32,
            timeout: Duration,
            _s: moa::SamplingParams,
        ) -> Result<Value, String> {
            // Fail near the worker timeout, not instantly.
            tokio::time::sleep(timeout.min(Duration::from_millis(400))).await;
            Err("slow drop".into())
        }
    }
    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![
        AnswerBackend::new("fast survivor A", Duration::ZERO),
        AnswerBackend::new("fast survivor B", Duration::ZERO),
        Arc::new(SlowDead),
    ];
    let cfg = config(backends, &["Qwen3-8B", "Llama-3.1-8B", "Ministral-8B"]);

    let started = std::time::Instant::now();
    let result = moa::handle_turn(&cfg, &request()).await;
    let elapsed = started.elapsed();

    assert_ne!(result.turn_kind, moa::TurnKind::Failed);
    assert!(
        elapsed < Duration::from_secs(4),
        "a slow-failing worker must not stall the turn (took {elapsed:?})"
    );
}
