//! Pin the tier-gate (strong patience) contract.
//!
//! Background: with a mixed pool (big-tier Strong worker + small-tier
//! workers, e.g. MiniMax + small Qwens), two fast small workers agreeing
//! could finalize the turn before the strong worker produced anything —
//! "dumbing down" the mesh answer to small-model consensus. Self-MoA
//! (arXiv:2502.00674) shows MoA output quality tracks proposer quality
//! far more than diversity, so the strong worker's answer must get a
//! bounded chance to land.
//!
//! Three contracts pinned here:
//!
//! 1. **Held consensus**: small-tier consensus does not finalize while
//!    the strong worker is still running inside the patience window;
//!    when the strong worker lands with a usable answer, it wins.
//! 2. **Hard bound (anti-#820)**: when the strong worker hangs past the
//!    patience window, the turn still completes promptly with the held
//!    small-tier consensus — never blocked until worker_timeout.
//! 3. **Same-tier pools unaffected**: an all-small pool keeps the
//!    pre-gate early-exit behavior (gate disabled by tier analysis).

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

/// Backend that returns a fixed answer after a configurable delay.
struct DelayedAnswerBackend {
    text: String,
    delay: Duration,
}

impl DelayedAnswerBackend {
    fn new(text: impl Into<String>, delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            delay,
        })
    }
}

#[async_trait]
impl moa::ModelBackend for DelayedAnswerBackend {
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
        Ok(json!({
            "choices": [{"message": {"content": self.text}}],
        }))
    }
}

fn user_turn(content: &str) -> Value {
    json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 32,
    })
}

fn response_text(result: &moa::TurnResult) -> String {
    result
        .response_body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Mixed-tier pool: two fast small workers agree on a (wrong) answer,
/// the big-tier strong worker lands later with a different one.
fn mixed_pool(
    small_delay: Duration,
    strong_delay: Duration,
    strong_patience: Duration,
) -> moa::GatewayConfig {
    let small_a = DelayedAnswerBackend::new("Sydney", small_delay);
    let small_b = DelayedAnswerBackend::new("Sydney", small_delay + Duration::from_millis(20));
    let strong = DelayedAnswerBackend::new("Canberra", strong_delay);

    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![small_a, small_b, strong];
    let models = vec![
        moa::ModelEntry {
            name: "qwen-3b".into(),
            backend_index: 0,
        },
        moa::ModelEntry {
            name: "qwen-8b".into(),
            backend_index: 1,
        },
        moa::ModelEntry {
            name: "MiniMax-M2.5".into(),
            backend_index: 2,
        },
    ];

    moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(30),
        hedge_delay: Duration::from_millis(200),
        reducer_timeout: Duration::from_secs(2),
        // Grace disabled so this test isolates the consensus path.
        first_answer_grace: Duration::ZERO,
        strong_patience,
        enable_thinking: None,
    }
}

#[tokio::test]
async fn small_consensus_is_held_until_strong_lands() {
    // Small workers agree at ~50ms; strong lands at 300ms — well inside
    // the patience window. The strong answer must get to participate
    // instead of the turn finalizing at 50ms on small consensus alone.
    let config = mixed_pool(
        Duration::from_millis(50),
        Duration::from_millis(300),
        Duration::from_secs(10),
    );
    let result = moa::handle_turn(&config, &user_turn("Capital of Australia? One word.")).await;

    let strong_summary = result
        .worker_summaries
        .iter()
        .find(|w| w.model == "MiniMax-M2.5")
        .expect("strong worker must be accounted for");
    assert!(
        strong_summary.succeeded,
        "strong worker must have been allowed to finish (not aborted by \
         small-tier consensus); summaries: {:?}",
        result
            .worker_summaries
            .iter()
            .map(|w| (&w.model, w.succeeded))
            .collect::<Vec<_>>()
    );
    // The contract is "when the strong worker lands with a usable answer,
    // it wins" — not merely "it was allowed to finish". The strong worker
    // disagrees with the small-tier consensus ("Sydney"), so its answer
    // ("Canberra") must be the one that ships.
    assert!(
        response_text(&result).contains("Canberra"),
        "strong worker's answer must win over small-tier consensus once it lands; \
         got {:?}",
        response_text(&result)
    );
}

#[tokio::test]
async fn patience_expiry_releases_held_consensus() {
    // Anti-#820: the strong worker effectively hangs (10s) while patience
    // is 500ms. The held small-tier consensus must ship at patience
    // expiry — NOT wait for the strong worker or worker_timeout.
    let config = mixed_pool(
        Duration::from_millis(50),
        Duration::from_secs(10),
        Duration::from_millis(500),
    );
    let started = std::time::Instant::now();
    let result = moa::handle_turn(&config, &user_turn("Capital of Australia? One word.")).await;
    let elapsed = started.elapsed();

    // Pin the bound near the configured patience window (500ms) with
    // scheduler slack, so a regression that pushes release out toward
    // worker_timeout (30s) is actually caught — a loose 5s bound would
    // silently pass several-second regressions.
    assert!(
        elapsed < Duration::from_millis(1500),
        "held consensus must be released promptly at patience expiry (~500ms), \
         not block on the hung strong worker; took {elapsed:?}"
    );
    assert!(
        response_text(&result).contains("Sydney"),
        "after patience expiry the small-tier consensus answer ships; got {:?}",
        response_text(&result)
    );
}

#[tokio::test]
async fn same_tier_pool_keeps_early_exit() {
    // All-small pool: gate must be disabled by tier analysis, so two
    // agreeing fast workers early-exit without waiting for the third.
    let fast_a = DelayedAnswerBackend::new("Tokyo", Duration::from_millis(20));
    let fast_b = DelayedAnswerBackend::new("Tokyo", Duration::from_millis(40));
    let slow = DelayedAnswerBackend::new("Tokyo", Duration::from_secs(10));

    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![fast_a, fast_b, slow];
    let models = vec![
        moa::ModelEntry {
            name: "qwen-3b".into(),
            backend_index: 0,
        },
        moa::ModelEntry {
            name: "llama-7b".into(),
            backend_index: 1,
        },
        moa::ModelEntry {
            name: "qwen-8b".into(),
            backend_index: 2,
        },
    ];
    let config = moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(30),
        hedge_delay: Duration::from_millis(200),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
        // Patience configured ON — but the pool is same-tier, so the
        // tier analysis must disable the gate anyway.
        strong_patience: Duration::from_secs(10),
        enable_thinking: None,
    };

    let started = std::time::Instant::now();
    let result = moa::handle_turn(&config, &user_turn("Capital of Japan? One word.")).await;
    let elapsed = started.elapsed();

    assert_eq!(
        result.turn_kind,
        moa::TurnKind::EarlyExit,
        "same-tier consensus must keep the early-exit path"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "same-tier pool must not wait for the slow worker; took {elapsed:?}"
    );
}
