//! Pin the cross-peer refinement round's behaviour under mesh conditions.
//!
//! Refinement is what makes a pool of small models beat its best member
//! (`evals/moa-openrouter/RESULTS.md`: 42/66/12, p=5.2e-05 for an all-8B pool),
//! so it runs on exactly the hardware where peers are slowest and least
//! reliable. A second fan-out is a second chance to hang the turn, so the round
//! must be strictly best-effort.
//!
//! Contracts pinned here:
//!
//! 1. **It runs and its output is used** — on an all-small pool the refined
//!    drafts, not the round-1 drafts, are what reach the reducer.
//! 2. **A hanging peer cannot cost the turn** — refinement is bounded by
//!    `worker_timeout`; the turn completes with whoever answered.
//! 3. **Total failure degrades, never breaks** — if every refiner errors, the
//!    round-1 outputs are used and the turn still answers.
//! 4. **Pool shape gates it** — a big-tier model present means `Auto` skips the
//!    extra fan-out.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Marker text so we can prove which round's text reached the reducer.
const ROUND1: &str = "ROUND1-DRAFT";
const REFINED: &str = "REFINED-DRAFT";

/// What a peer does when asked to refine.
#[derive(Clone, Copy)]
enum RefineBehavior {
    /// Return the refined marker after a delay.
    Ok(Duration),
    /// Hang past any deadline.
    Hang,
    /// Fail outright.
    Fail,
}

/// A mesh peer: answers round 1, then behaves per `on_refine`.
///
/// Call kind is inferred from the system prompt, which is how the three phases
/// are distinguishable without touching engine internals.
struct MeshPeer {
    /// Distinct per peer so round-1 answers do NOT cluster into consensus —
    /// otherwise early-exit short-circuits the turn and synthesis (the only
    /// path refinement feeds) never runs.
    round1_text: String,
    round1_delay: Duration,
    on_refine: RefineBehavior,
    refine_calls: Arc<AtomicUsize>,
}

impl MeshPeer {
    fn new(
        round1_text: &str,
        round1_delay: Duration,
        on_refine: RefineBehavior,
    ) -> (Arc<Self>, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(Self {
                round1_text: format!("{ROUND1} {round1_text}"),
                round1_delay,
                on_refine,
                refine_calls: counter.clone(),
            }),
            counter,
        )
    }
}

fn system_text(messages: &[Value]) -> String {
    messages
        .iter()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|m| m.get("content").and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

fn reply(text: &str) -> Value {
    json!({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})
}

#[async_trait]
impl moa::ModelBackend for MeshPeer {
    async fn chat_completion(
        &self,
        _model: &str,
        messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        let sys = system_text(messages);

        // Reducer synthesis: echo back whichever drafts it was given, so the
        // test can assert which round's text made it through.
        if sys.contains("## Worker outputs") {
            let saw_refined = sys.contains(REFINED);
            return Ok(reply(if saw_refined {
                "FINAL-FROM-REFINED"
            } else {
                "FINAL-FROM-ROUND1"
            }));
        }

        // Refinement round.
        // Refinement round: peers are shown each other's drafts under a
        // "Candidate responses:" header. Checked after the reducer branch
        // above, which is identified by its "## Worker outputs" section.
        if sys.contains("Candidate responses:") {
            self.refine_calls.fetch_add(1, Ordering::SeqCst);
            return match self.on_refine {
                RefineBehavior::Ok(d) => {
                    tokio::time::sleep(d).await;
                    Ok(reply(REFINED))
                }
                RefineBehavior::Hang => {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    Ok(reply(REFINED))
                }
                RefineBehavior::Fail => Err("peer unavailable".into()),
            };
        }

        // Round 1.
        tokio::time::sleep(self.round1_delay).await;
        Ok(reply(&self.round1_text))
    }
}

fn config_with_grace(
    models: &[(&str, Arc<MeshPeer>)],
    policy: moa::RefinementPolicy,
    worker_timeout: Duration,
    first_answer_grace: Duration,
) -> moa::GatewayConfig {
    let mut cfg = config(models, policy, worker_timeout);
    cfg.first_answer_grace = first_answer_grace;
    cfg
}

fn config(
    models: &[(&str, Arc<MeshPeer>)],
    policy: moa::RefinementPolicy,
    worker_timeout: Duration,
) -> moa::GatewayConfig {
    let mut backends: Vec<Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut entries = Vec::new();
    for (name, backend) in models {
        entries.push(moa::ModelEntry::new((*name).to_string(), backends.len()));
        backends.push(backend.clone());
    }
    moa::GatewayConfig {
        backends,
        models: entries,
        worker_timeout,
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(5),
        // Disable the cheap paths so every turn reaches synthesis, which is
        // the only path refinement participates in.
        first_answer_grace: Duration::ZERO,
        strong_patience: Duration::ZERO,
        enable_thinking: Some(false),
        actor_candidates: Vec::new(),
        reference_policy: Default::default(),
        refinement_policy: policy,
    }
}

/// Distinct prompts per worker so round-1 answers disagree and the turn is
/// forced into synthesis rather than early-exit consensus.
fn request() -> Value {
    json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "explain backpressure"}],
        "max_tokens": 256,
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn refined_drafts_are_what_reach_the_reducer() {
    let (a, ca) = MeshPeer::new(
        "queues fill up",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (b, cb) = MeshPeer::new(
        "latency grows unbounded",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (c, cc) = MeshPeer::new(
        "memory exhausts eventually",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let cfg = config(
        &[("Qwen3-8B", a), ("Llama-3.1-8B", b), ("Ministral-8B", c)],
        moa::RefinementPolicy::Always,
        Duration::from_secs(5),
    );

    let result = moa::handle_turn(&cfg, &request()).await;

    assert!(
        ca.load(Ordering::SeqCst) + cb.load(Ordering::SeqCst) + cc.load(Ordering::SeqCst) >= 2,
        "an all-small pool must run the refinement round"
    );
    // The contract is that the *refined* text is what reaches the client. It
    // may arrive either via the reducer ("FINAL-FROM-REFINED") or directly,
    // when the refined drafts agree and the arbiter takes consensus — both are
    // correct; what must never happen is round-1 text being returned.
    let body = serde_json::to_string(&result.response_body).unwrap();
    assert!(
        body.contains(REFINED) || body.contains("FINAL-FROM-REFINED"),
        "refined drafts must reach the client, got: {body}"
    );
    assert!(
        !body.contains(ROUND1),
        "round-1 drafts must not survive a successful refinement round, got: {body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hanging_refiner_cannot_hold_the_turn() {
    let (a, _) = MeshPeer::new(
        "queues fill up",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (b, _) = MeshPeer::new(
        "latency grows unbounded",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    // One peer never returns from refinement — the mesh reality this guards.
    let (c, _) = MeshPeer::new(
        "memory exhausts eventually",
        Duration::ZERO,
        RefineBehavior::Hang,
    );
    let worker_timeout = Duration::from_millis(400);
    let cfg = config(
        &[("Qwen3-8B", a), ("Llama-3.1-8B", b), ("Ministral-8B", c)],
        moa::RefinementPolicy::Always,
        worker_timeout,
    );

    let started = std::time::Instant::now();
    let result = moa::handle_turn(&cfg, &request()).await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < worker_timeout * 4,
        "a hanging refiner must not extend the turn (took {elapsed:?})"
    );
    assert_ne!(
        result.turn_kind,
        moa::TurnKind::Failed,
        "the turn must still answer with the refiners that did return"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn total_refinement_failure_falls_back_to_round_one() {
    let (a, _) = MeshPeer::new("queues fill up", Duration::ZERO, RefineBehavior::Fail);
    let (b, _) = MeshPeer::new(
        "latency grows unbounded",
        Duration::ZERO,
        RefineBehavior::Fail,
    );
    let (c, _) = MeshPeer::new(
        "memory exhausts eventually",
        Duration::ZERO,
        RefineBehavior::Fail,
    );
    let cfg = config(
        &[("Qwen3-8B", a), ("Llama-3.1-8B", b), ("Ministral-8B", c)],
        moa::RefinementPolicy::Always,
        Duration::from_secs(5),
    );

    let result = moa::handle_turn(&cfg, &request()).await;

    assert_ne!(
        result.turn_kind,
        moa::TurnKind::Failed,
        "every refiner failing must degrade to round-1, not fail the turn"
    );
    let body = serde_json::to_string(&result.response_body).unwrap();
    assert!(
        body.contains("FINAL-FROM-ROUND1"),
        "round-1 drafts must be used when refinement produces nothing, got: {body}"
    );
}

/// The production-settings check: refinement must still run when
/// `first_answer_grace` is the real 3s default rather than the ZERO used by the
/// other tests here.
///
/// Grace produces an `early_decision`, and refinement is skipped whenever one
/// exists — so if grace fired on a fast all-small pool, this feature would be
/// dead code on real traffic. Grace only arms once the window has *elapsed*, so
/// a pool that answers promptly reaches synthesis (and refinement) first; this
/// pins that ordering.
#[tokio::test(flavor = "multi_thread")]
async fn refinement_still_runs_under_production_grace() {
    let (a, ca) = MeshPeer::new(
        "queues fill up",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (b, cb) = MeshPeer::new(
        "latency grows unbounded",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (c, cc) = MeshPeer::new(
        "memory exhausts eventually",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let cfg = config_with_grace(
        &[("Qwen3-8B", a), ("Llama-3.1-8B", b), ("Ministral-8B", c)],
        moa::RefinementPolicy::Always,
        Duration::from_secs(5),
        // The production default from `build_moa_config`.
        Duration::from_secs(3),
    );

    let result = moa::handle_turn(&cfg, &request()).await;

    assert!(
        ca.load(Ordering::SeqCst) + cb.load(Ordering::SeqCst) + cc.load(Ordering::SeqCst) >= 2,
        "refinement must still run at the production grace setting, or the \
         feature is dead code on real traffic"
    );
    let body = serde_json::to_string(&result.response_body).unwrap();
    assert!(
        body.contains(REFINED) || body.contains("FINAL-FROM-REFINED"),
        "refined drafts must reach the client under production grace, got: {body}"
    );
}

/// Straggling peers must NOT cost the refinement round.
///
/// This is the mesh case: one peer answers instantly, the others lag. The
/// answer grace would normally ship the fast lone answer and skip refinement —
/// but on an all-small pool refinement is the only step that beats the best
/// member (26/75/19, p=0.37 without it vs 42/66/12 with). Since variable peer
/// latency is the norm on consumer hardware, letting grace win here would
/// silently disable the feature exactly where it matters, while still paying
/// for the fan-out.
///
/// So when refinement is expected, grace is disabled for the turn. Round 1 is
/// still bounded by `worker_timeout` and refinement by its own half-budget, so
/// this costs bounded latency, never an unbounded wait.
#[tokio::test(flavor = "multi_thread")]
async fn straggling_peers_do_not_cost_the_refinement_round() {
    let grace = Duration::from_millis(150);
    // First peer answers immediately; the others straggle well past the grace.
    let (a, ca) = MeshPeer::new(
        "queues fill up",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (b, cb) = MeshPeer::new(
        "latency grows unbounded",
        Duration::from_millis(600),
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (c, cc) = MeshPeer::new(
        "memory exhausts eventually",
        Duration::from_millis(600),
        RefineBehavior::Ok(Duration::ZERO),
    );
    let cfg = config_with_grace(
        &[("Qwen3-8B", a), ("Llama-3.1-8B", b), ("Ministral-8B", c)],
        moa::RefinementPolicy::Always,
        Duration::from_secs(5),
        grace,
    );

    let result = moa::handle_turn(&cfg, &request()).await;

    assert!(
        ca.load(Ordering::SeqCst) + cb.load(Ordering::SeqCst) + cc.load(Ordering::SeqCst) >= 2,
        "a fast peer answering first must not skip refinement on an all-small pool"
    );
    let body = serde_json::to_string(&result.response_body).unwrap();
    assert!(
        body.contains(REFINED) || body.contains("FINAL-FROM-REFINED"),
        "refined drafts must reach the client despite stragglers, got: {body}"
    );
    assert_ne!(result.turn_kind, moa::TurnKind::Failed);
}

/// The grace must still work normally when refinement is NOT in play — a
/// big-tier pool keeps its fast chat path.
#[tokio::test(flavor = "multi_thread")]
async fn grace_still_short_circuits_when_refinement_is_not_expected() {
    let grace = Duration::from_millis(100);
    let (a, ca) = MeshPeer::new(
        "queues fill up",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (b, cb) = MeshPeer::new(
        "latency grows unbounded",
        Duration::from_secs(30), // would stall the turn if grace didn't fire
        RefineBehavior::Ok(Duration::ZERO),
    );
    let cfg = config_with_grace(
        // Big-tier present => Auto does not refine => grace stays enabled.
        &[("Qwen3-32B", a), ("Qwen3-8B", b)],
        moa::RefinementPolicy::Auto,
        Duration::from_secs(60),
        grace,
    );

    let started = std::time::Instant::now();
    let result = moa::handle_turn(&cfg, &request()).await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "grace must still short-circuit a slow peer when refinement is off (took {elapsed:?})"
    );
    assert_eq!(
        ca.load(Ordering::SeqCst) + cb.load(Ordering::SeqCst),
        0,
        "a big-tier pool must not refine"
    );
    assert_ne!(result.turn_kind, moa::TurnKind::Failed);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_big_tier_pool_skips_the_extra_fanout() {
    let (a, ca) = MeshPeer::new(
        "queues fill up",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let (b, cb) = MeshPeer::new(
        "latency grows unbounded",
        Duration::ZERO,
        RefineBehavior::Ok(Duration::ZERO),
    );
    let cfg = config(
        // A big-tier model is present, so Auto should not pay for a 2nd round.
        &[("Qwen3-32B", a), ("Qwen3-8B", b)],
        moa::RefinementPolicy::Auto,
        Duration::from_secs(5),
    );

    let result = moa::handle_turn(&cfg, &request()).await;

    assert_eq!(
        ca.load(Ordering::SeqCst) + cb.load(Ordering::SeqCst),
        0,
        "Auto must skip refinement when a big-tier model can synthesize directly"
    );
    assert_ne!(result.turn_kind, moa::TurnKind::Failed);
}
