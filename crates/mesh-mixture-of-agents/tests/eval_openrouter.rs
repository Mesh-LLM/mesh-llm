//! Live MoA eval against real open-weight models via OpenRouter.
//!
//! # Why OpenRouter is a valid stand-in for a mesh
//!
//! From the MoA engine's point of view, a worker is a `ModelBackend` — a thing
//! that takes `(model, messages, tools, max_tokens, timeout, sampling)` and
//! returns an OpenAI-shaped body. The two shipped backends,
//! `LocalModelBackend` and `RemoteModelBackend`, build a *byte-identical*
//! request body and differ only in transport (a local HTTP port vs. a framed
//! QUIC stream). Everything above transport — arbiter, reducer, fan-out,
//! synthesis, `SamplingParams`, the thinking policy — is transport-agnostic.
//!
//! So a third backend that hits OpenRouter with the same body plus an auth
//! header is indistinguishable to the engine from a mesh peer solo-serving
//! that model. We are not approximating the MoA logic here; we run the real
//! `handle_turn`. What OpenRouter stands in for is only the *worker*: a
//! similarly-sized model from the same family a mesh node would solo-serve.
//! Exact quant/backend parity is explicitly out of scope — tier and family
//! are what matter.
//!
//! # What this is NOT
//!
//! * Not a QUIC-transport test (CI's two-node smokes cover that).
//! * Not a `build_moa_config` dedup test (unit tests cover name normalization).
//! * Not deterministic — workers run at temperature 0.8. Treat single-run
//!   numbers as directional; record k≥3 for anything load-bearing.
//!
//! # Running
//!
//! ```text
//! OPENROUTER_API_KEY=... cargo test -p mesh-mixture-of-agents --test eval_openrouter -- --ignored --nocapture
//! ```
//!
//! Every test is `#[ignore]` (never runs in CI / normal `cargo test`) and also
//! no-ops with a printed notice if the key is absent, so an accidental
//! `--ignored` run without a key still passes.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use moa::{GatewayConfig, ModelBackend, ModelEntry, SamplingParams};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ─── OpenRouter backend ──────────────────────────────────────────────

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// A `ModelBackend` that reaches a real open-weight model through OpenRouter.
///
/// The request body is constructed to match `LocalModelBackend` /
/// `RemoteModelBackend` exactly (same keys, same `apply_enable_thinking`
/// injection), so the model sees what a mesh worker would. The only additions
/// are the bearer header OpenRouter requires and the same `HTTP 400
/// reasoning-mandatory` retry the in-tree `HttpBackend` carries.
struct OpenRouterBackend {
    http: reqwest::Client,
    api_key: String,
}

impl OpenRouterBackend {
    fn new(api_key: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .unwrap_or_default();
        Self { http, api_key }
    }

    fn build_body(
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        sampling: SamplingParams,
    ) -> Value {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        // Identical thinking-flag injection to the shipped mesh backends.
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        body
    }

    async fn post(&self, body: &Value, timeout: Duration) -> Result<reqwest::Response, String> {
        self.http
            .post(OPENROUTER_URL)
            .bearer_auth(&self.api_key)
            .json(body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("openrouter request failed: {e}"))
    }

    /// Call with a predeclared retry on transient upstream/infra errors.
    ///
    /// OpenRouter surfaces provider rate-limits and gateway hiccups as
    /// 429/502/503/504. The eval must NOT score these as *capability*
    /// failures (the expert review flagged this), so we retry with backoff and,
    /// if still failing, return a distinguishable `INFRA:` error the caller
    /// treats as "excluded / missing", never as a wrong answer.
    async fn chat_completion_retrying(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let mut last = String::new();
        for attempt in 0..4 {
            match self
                .chat_completion(
                    model,
                    messages,
                    tools,
                    max_tokens,
                    Duration::from_secs(90),
                    sampling,
                )
                .await
            {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let transient = ["429", "502", "503", "504", "temporarily", "aborted"]
                        .iter()
                        .any(|c| e.contains(c));
                    last = e;
                    if !transient {
                        return Err(last); // genuine error (e.g. 400) — surface it
                    }
                    // backoff: 0.5s, 1s, 2s
                    tokio::time::sleep(Duration::from_millis(500 << attempt)).await;
                }
            }
        }
        Err(format!("INFRA: {last}"))
    }
}

#[async_trait]
impl ModelBackend for OpenRouterBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let body = Self::build_body(model, messages, tools, max_tokens, sampling);
        let resp = self.post(&body, timeout).await?;
        let status = resp.status();

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();

            // Same failure mode the in-tree HttpBackend handles: some
            // endpoints reject our thinking-disable flags with HTTP 400.
            // Verified against minimax — it failed 12/12 until the flags
            // were dropped. A strict endpoint should cost a slightly slower
            // worker, not the worker entirely.
            if status.as_u16() == 400
                && text.to_ascii_lowercase().contains("reasoning")
                && sampling.enable_thinking == Some(false)
            {
                let mut retry = body.clone();
                if let Some(obj) = retry.as_object_mut() {
                    obj.remove("reasoning_effort");
                    obj.remove("chat_template_kwargs");
                }
                let r = self.post(&retry, timeout).await?;
                let retry_status = r.status();
                if !retry_status.is_success() {
                    let t = r.text().await.unwrap_or_default();
                    return Err(format!("HTTP {retry_status}: {}", truncate(&t, 200)));
                }
                return r.json::<Value>().await.map_err(|e| format!("parse: {e}"));
            }
            return Err(format!("HTTP {status}: {}", truncate(&text, 200)));
        }

        resp.json::<Value>()
            .await
            .map_err(|e| format!("parse: {e}"))
    }
}

// ─── Mesh-realism wrapper ────────────────────────────────────────────

/// Per-worker fault profile simulating a consumer-hardware mesh node.
///
/// OpenRouter is faster and far more reliable than a laptop on home wifi, so
/// this wrapper adds the two things a real mesh has that a cloud API does not:
/// slowness and flakiness. This is exactly the surface Together's MoA has no
/// answer for, so it is the part worth stressing hardest.
#[derive(Clone, Copy)]
struct MeshFault {
    /// Fixed slowdown added before delegating (models cold-loading, weak GPUs).
    extra_latency_ms: u64,
    /// Random 0..jitter added on top, sampled per call.
    jitter_ms: u64,
    /// Probability in [0,1] the worker hard-fails instead of answering
    /// (peer reset, OOM, timeout).
    failure_rate: f64,
}

impl MeshFault {
    const RELIABLE_FAST: Self = Self {
        extra_latency_ms: 150,
        jitter_ms: 400,
        failure_rate: 0.0,
    };
    const TYPICAL: Self = Self {
        extra_latency_ms: 600,
        jitter_ms: 1500,
        failure_rate: 0.05,
    };
    /// A big-tier node that is powerful but slow to first token — the case
    /// `strong_patience` exists for.
    const SLOW_STRONG: Self = Self {
        extra_latency_ms: 3000,
        jitter_ms: 3000,
        failure_rate: 0.05,
    };
    /// A genuinely unreliable peer.
    const FLAKY: Self = Self {
        extra_latency_ms: 400,
        jitter_ms: 2000,
        failure_rate: 0.25,
    };
}

/// Wraps any backend, injecting deterministic-per-call latency and failures.
struct MeshRealismBackend {
    inner: Arc<dyn ModelBackend>,
    fault: MeshFault,
    /// Seed mixed with a per-call counter so faults are reproducible within a
    /// run but differ across workers and calls.
    seed: u64,
    calls: AtomicU64,
}

impl MeshRealismBackend {
    fn wrap(inner: Arc<dyn ModelBackend>, fault: MeshFault, seed: u64) -> Arc<dyn ModelBackend> {
        Arc::new(Self {
            inner,
            fault,
            seed,
            calls: AtomicU64::new(0),
        })
    }
}

#[async_trait]
impl ModelBackend for MeshRealismBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        let mut rng = SmallRng::new(self.seed ^ (n.wrapping_mul(0x9E37_79B9_7F4A_7C15)));

        let jitter = if self.fault.jitter_ms > 0 {
            rng.next_u64() % self.fault.jitter_ms
        } else {
            0
        };
        tokio::time::sleep(Duration::from_millis(self.fault.extra_latency_ms + jitter)).await;

        if self.fault.failure_rate > 0.0 && rng.next_f64() < self.fault.failure_rate {
            return Err(format!("simulated mesh fault: {model} peer unreachable"));
        }

        self.inner
            .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
            .await
    }
}

/// Minimal splitmix64 — avoids adding a `rand` dependency for fault injection.
struct SmallRng(u64);
impl SmallRng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─── Mesh-likely model pool ──────────────────────────────────────────

#[derive(Clone, Copy)]
enum Tier {
    Small,
    Big,
}

#[derive(Clone, Copy)]
struct PoolModel {
    id: &'static str,
    tier: Tier,
    fault: MeshFault,
}

/// A smorgasbord of tool-capable open-weight models a mesh would plausibly
/// solo-serve: small models on laptops/minis, bigger ones on a single good
/// GPU. Nothing that would require splitting. All verified tool-capable
/// during corpus recording. Fault profiles spread across the realism space so
/// one turn exercises fast, typical, slow-strong, and flaky peers together.
fn mesh_pool() -> Vec<PoolModel> {
    vec![
        PoolModel {
            id: "qwen/qwen3-8b",
            tier: Tier::Small,
            fault: MeshFault::RELIABLE_FAST,
        },
        PoolModel {
            id: "mistralai/ministral-8b-2512",
            tier: Tier::Small,
            fault: MeshFault::TYPICAL,
        },
        PoolModel {
            id: "meta-llama/llama-3.2-3b-instruct",
            tier: Tier::Small,
            fault: MeshFault::FLAKY,
        },
        PoolModel {
            id: "qwen/qwen3-14b",
            tier: Tier::Big,
            fault: MeshFault::TYPICAL,
        },
        PoolModel {
            id: "mistralai/mistral-small-3.2-24b-instruct",
            tier: Tier::Big,
            fault: MeshFault::TYPICAL,
        },
        PoolModel {
            id: "qwen/qwen3-32b",
            tier: Tier::Big,
            fault: MeshFault::SLOW_STRONG,
        },
    ]
}

/// The pool's declared tiers must match the tier MoA *derives* from each model
/// name. This is the footgun from the design discussion: role assignment keys
/// off the name (single-digit-B ⇒ small), so if a name doesn't classify the
/// way you assumed, the Strong role — which also seeds the reducer — lands on
/// the wrong model and tier-aware patience misfires.
///
/// Pure and network-free, so it runs in normal `cargo test` / CI and guards
/// the pool against silent tier drift when models are added or renamed.
#[test]
fn declared_tiers_match_moa_role_assignment() {
    let pool = mesh_pool();
    let models: Vec<ModelEntry> = pool
        .iter()
        .enumerate()
        .map(|(i, m)| ModelEntry {
            name: m.id.to_string(),
            backend_index: i,
        })
        .collect();
    let assignments = moa::worker::assign_roles(&models);

    let tier_of = |name: &str| pool.iter().find(|m| m.id == name).map(|m| m.tier).unwrap();

    // Fast goes to the smallest model, Strong (and the reducer) to the biggest.
    // Those two extremes must agree with how we labelled them.
    for a in &assignments {
        match a.role {
            moa::worker::WorkerRole::Fast => assert!(
                matches!(tier_of(&a.model_name), Tier::Small),
                "Fast role landed on {} which we declared Big — name-derived tier disagrees",
                a.model_name
            ),
            moa::worker::WorkerRole::Strong => assert!(
                matches!(tier_of(&a.model_name), Tier::Big),
                "Strong role landed on {} which we declared Small — name-derived tier disagrees",
                a.model_name
            ),
            _ => {}
        }
    }
}

fn moa_config(pool: &[PoolModel], api_key: &str, realism: bool) -> GatewayConfig {
    let mut backends: Vec<Arc<dyn ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for (i, m) in pool.iter().enumerate() {
        let base: Arc<dyn ModelBackend> = Arc::new(OpenRouterBackend::new(api_key.to_string()));
        let backend = if realism {
            MeshRealismBackend::wrap(base, m.fault, 0xE7A1_u64.wrapping_add(i as u64))
        } else {
            base
        };
        models.push(ModelEntry {
            name: m.id.to_string(),
            backend_index: backends.len(),
        });
        backends.push(backend);
    }
    GatewayConfig {
        backends,
        models,
        // Generous worker timeout: realism latency + real model latency can
        // stack, and we want to see slow workers land, not time out.
        worker_timeout: Duration::from_secs(90),
        hedge_delay: Duration::from_secs(5),
        reducer_timeout: Duration::from_secs(60),
        first_answer_grace: Duration::from_secs(3),
        strong_patience: Duration::from_secs(20),
        // MoA policy: thinking always off (matches effective_enable_thinking_for_moa).
        enable_thinking: Some(false),
        actor_candidates: Vec::new(),
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

fn api_key_or_skip(test: &str) -> Option<String> {
    match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.trim().is_empty() => Some(k),
        _ => {
            eprintln!("[{test}] OPENROUTER_API_KEY not set — skipping live eval");
            None
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n])
    }
}

fn tool_schema(name: &str, params: &[(&str, &str)]) -> Value {
    let props: serde_json::Map<String, Value> = params
        .iter()
        .map(|(p, ty)| (p.to_string(), json!({"type": ty})))
        .collect();
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": format!("{name} tool"),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": params.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            }
        }
    })
}

fn agent_tools() -> Value {
    json!([
        tool_schema("list_dir", &[("path", "string")]),
        tool_schema("read_file", &[("path", "string")]),
        tool_schema("search", &[("pattern", "string"), ("path", "string")]),
        tool_schema("run_command", &[("cmd", "string")]),
    ])
}

fn user_turn(prompt: &str, tools: Option<Value>) -> Value {
    let mut body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    });
    if let Some(t) = tools {
        body.as_object_mut().unwrap().insert("tools".into(), t);
    }
    body
}

fn response_text(body: &Value) -> String {
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn response_tool_calls(body: &Value) -> Vec<(String, String)> {
    body.pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|tcs| {
            tcs.iter()
                .filter_map(|tc| {
                    Some((
                        tc.pointer("/function/name")?.as_str()?.to_string(),
                        tc.pointer("/function/arguments")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

// ─── Test 1: pool liveness ───────────────────────────────────────────

/// Confirm every pool model resolves and answers before spending money on the
/// real evals. Run this first after any pool edit.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn pool_is_live() {
    let Some(key) = api_key_or_skip("pool_is_live") else {
        return;
    };
    let backend = OpenRouterBackend::new(key);
    let msg = vec![json!({"role": "user", "content": "Reply with the single word: ok"})];

    let mut dead = Vec::new();
    for m in mesh_pool() {
        let t0 = Instant::now();
        let result = backend
            .chat_completion(
                m.id,
                &msg,
                None,
                32,
                Duration::from_secs(60),
                SamplingParams::worker().with_thinking(Some(false)),
            )
            .await;
        match result {
            Ok(body) => {
                let txt = response_text(&body);
                eprintln!(
                    "  ok   {:44} {:>6}ms  {:?}",
                    m.id,
                    t0.elapsed().as_millis(),
                    truncate(txt.trim(), 40)
                );
            }
            Err(e) => {
                eprintln!("  DEAD {:44} {e}", m.id);
                dead.push(m.id);
            }
        }
    }
    assert!(dead.is_empty(), "dead pool models: {dead:?}");
}

// ─── Test 2: tool coherence, MoA vs best single ──────────────────────

struct ToolTask {
    name: &'static str,
    prompt: &'static str,
    /// Expected tool name, or None if no tool should be called.
    expect_tool: Option<&'static str>,
    /// Optional substring the winning arguments must contain (majority-correct
    /// answer), used to catch hallucinated arguments.
    expect_arg_contains: Option<&'static str>,
    /// Optional substring the winning arguments must NOT contain (a known
    /// hallucination some single models emit).
    reject_arg_contains: Option<&'static str>,
}

fn tool_tasks() -> Vec<ToolTask> {
    vec![
        ToolTask {
            name: "explore_src",
            prompt: "I need to understand this Rust project's layout. Start by looking at what is in the src directory.",
            expect_tool: Some("list_dir"),
            expect_arg_contains: Some("src"),
            reject_arg_contains: None,
        },
        ToolTask {
            name: "find_symbol",
            prompt: "Find every place MeshError::Timeout is constructed in this repo.",
            expect_tool: Some("search"),
            expect_arg_contains: Some("Timeout"),
            reject_arg_contains: None,
        },
        ToolTask {
            name: "triage_failing_test",
            prompt: "The test suite is failing. Find out which test fails and why.",
            expect_tool: Some("run_command"),
            expect_arg_contains: None,
            reject_arg_contains: None,
        },
        ToolTask {
            name: "no_tool_concept",
            prompt: "What does the Rust `?` operator do? Just explain it, do not look at any files.",
            expect_tool: None,
            expect_arg_contains: None,
            reject_arg_contains: None,
        },
    ]
}

/// A single model's verdict on one task.
async fn solo_tool_result(
    backend: &OpenRouterBackend,
    model: &str,
    task: &ToolTask,
) -> Result<Vec<(String, String)>, String> {
    let msg = vec![json!({"role": "user", "content": task.prompt})];
    let body = backend
        .chat_completion(
            model,
            &msg,
            Some(&agent_tools()),
            512,
            Duration::from_secs(60),
            // Match MoA worker conditions: thinking off. Sampling can stay at
            // worker defaults so the comparison is aggregation-vs-not, holding
            // the thinking policy constant.
            SamplingParams::worker().with_thinking(Some(false)),
        )
        .await?;
    Ok(response_tool_calls(&body))
}

fn scores_task(tools: &[(String, String)], task: &ToolTask) -> bool {
    match task.expect_tool {
        None => tools.is_empty(),
        Some(expected) => {
            let Some((name, args)) = tools.first() else {
                return false;
            };
            if name != expected {
                return false;
            }
            if let Some(needle) = task.expect_arg_contains
                && !args.contains(needle)
            {
                return false;
            }
            if let Some(bad) = task.reject_arg_contains
                && args.contains(bad)
            {
                return false;
            }
            true
        }
    }
}

// ─── Ablation: does the advice actually help the actor? ──────────────
//
// The controlled experiment the eval review demanded. In the asymmetric
// design the actor is already the best tool-caller in the pool — so the only
// question that matters is whether the *references* add anything the actor
// could not do alone. "MoA beats the best single model" is the wrong bar,
// because the actor IS one of the models.
//
// Everything is held fixed except the one variable of interest — the
// references — so a difference cannot be blamed on sampling, token budget,
// system prompt, or a different actor winning a latency race:
//
//   * ONE pinned actor model, at the actor's own sampling and token budget,
//     with the real `pack_for_actor` system prompt.
//   * Three arms differing ONLY in the `references` passed to the actor:
//       A. none      — actor alone (the true comparator)
//       B. real      — advice from the other pool models (the production path)
//       C. shuffled  — advice generated for a DIFFERENT task (length-similar)
//
// Metric: rescue = (A fails, B passes); harm = (A passes, B fails);
//         net uplift = rescue - harm. Arm C separates "useful information"
//         from "extra tokens + a think-carefully prompt": if B beats A but C
//         beats A just as much, the gain is not the advice.
//
// Robustness to imperfect labels: the same scorer is applied to all three
// arms of the *same* actor, so a systematically over/under-specified label
// affects every arm equally and largely cancels in the rescue-minus-harm
// delta. That is why this paired design is sound where "beat the best single
// model" (a max over models on the eval tasks) was not.

/// The pinned actor for the ablation, overridable via `MOA_ABLATION_ACTOR`.
///
/// Default is the strongest tool-caller in the pool. But a strong actor tends
/// to ace easy single-tool tasks alone, leaving no headroom for references to
/// rescue — so the ablation is only informative when the actor fails a
/// meaningful fraction alone. Pinning a *weaker* actor (a realistic case: a
/// mesh of only small models) creates that headroom and directly tests whether
/// advice from the pool rescues an actor that would otherwise fail.
fn ablation_actor() -> String {
    std::env::var("MOA_ABLATION_ACTOR").unwrap_or_else(|_| "qwen/qwen3-32b".to_string())
}

fn ablation_draws() -> usize {
    std::env::var("MOA_ABLATION_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// Run every non-actor pool model tool-free and collect its prose advice,
/// exactly as the production reference phase does.
async fn gather_advice(
    backend: &OpenRouterBackend,
    pool: &[PoolModel],
    session: &moa::session::Session,
    actor: &str,
) -> Vec<moa::normalize::WorkerOutput> {
    let mut refs = Vec::new();
    for m in pool {
        if m.id == actor {
            continue; // the actor advises itself implicitly when it acts
        }
        let packed = moa::context::pack_for_worker_selected(
            session,
            moa::worker::WorkerRole::Generalist,
            false, // tool-free: references only advise
            &[],
        );
        match backend
            .chat_completion_retrying(
                m.id,
                &packed.messages,
                packed.tools.as_ref(),
                packed.max_tokens,
                SamplingParams::worker().with_thinking(Some(false)),
            )
            .await
        {
            Ok(body) => {
                let text = response_text(&body);
                if !text.trim().is_empty() {
                    refs.push(moa::normalize::normalize_worker_output(
                        &text,
                        m.id,
                        moa::worker::WorkerRole::Generalist,
                        0,
                    ));
                }
            }
            Err(e) => eprintln!("      advisor {} unavailable: {}", m.id, truncate(&e, 70)),
        }
    }
    refs
}

/// Outcome of one actor arm on one task/draw.
enum ArmOutcome {
    Pass,
    Fail,
    /// Excluded from the capability analysis (transient infra error).
    Infra,
}

async fn run_actor_arm(
    backend: &OpenRouterBackend,
    session: &moa::session::Session,
    references: &[moa::normalize::WorkerOutput],
    selected: &[String],
    task: &ToolTask,
    actor: &str,
) -> ArmOutcome {
    let (messages, tools) = moa::context::pack_for_actor(session, references, true, selected);
    match backend
        .chat_completion_retrying(
            actor,
            &messages,
            tools.as_ref(),
            2048,
            SamplingParams::reducer().with_thinking(Some(false)),
        )
        .await
    {
        Ok(body) => {
            if scores_task(&response_tool_calls(&body), task) {
                ArmOutcome::Pass
            } else {
                ArmOutcome::Fail
            }
        }
        Err(e) if e.starts_with("INFRA:") => ArmOutcome::Infra,
        Err(_) => ArmOutcome::Fail,
    }
}

/// The falsification test for the asymmetric design: do references rescue the
/// actor more than they harm it? Reports rescue / harm / net uplift and a
/// shuffled-advice control. Pilot scale by default — the printed protocol note
/// states what a merge-blocking result would require.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn ablation_do_references_help_the_actor() {
    let Some(key) = api_key_or_skip("ablation") else {
        return;
    };
    let pool = mesh_pool();
    let tasks = tool_tasks();
    let draws = ablation_draws();
    let backend = OpenRouterBackend::new(key);
    let selected = agent_tools_names();

    // Per-task tallies across draws.
    let n = tasks.len();
    let mut a_pass = vec![0usize; n];
    let mut b_pass = vec![0usize; n];
    let mut c_pass = vec![0usize; n];
    let mut a_scored = vec![0usize; n]; // non-infra A trials
    let mut b_scored = vec![0usize; n];
    let mut c_scored = vec![0usize; n];
    let mut rescue = 0usize; // A fail, B pass (both scored)
    let mut harm = 0usize; // A pass, B fail (both scored)
    let mut paired = 0usize; // trials where both A and B were scored

    let actor = ablation_actor();
    eprintln!("\n=== ablation: do references help the actor? ===");
    eprintln!("actor = {actor}, {n} tasks x {draws} draws, 3 arms\n");

    for draw in 0..draws {
        // Generate advice for every task first, so arm C can borrow a
        // different task's advice within the same draw.
        let sessions: Vec<moa::session::Session> = tasks.iter().map(session_for_task).collect();
        let mut advice: Vec<Vec<moa::normalize::WorkerOutput>> = Vec::with_capacity(n);
        for s in &sessions {
            advice.push(gather_advice(&backend, &pool, s, &actor).await);
        }

        for (i, task) in tasks.iter().enumerate() {
            let real = &advice[i];
            let shuffled = &advice[(i + 1) % n]; // different task's advice
            let s = &sessions[i];

            let a = run_actor_arm(&backend, s, &[], &selected, task, &actor).await;
            let b = run_actor_arm(&backend, s, real, &selected, task, &actor).await;
            let c = run_actor_arm(&backend, s, shuffled, &selected, task, &actor).await;

            let tally = |o: &ArmOutcome, pass: &mut usize, scored: &mut usize| match o {
                ArmOutcome::Pass => {
                    *pass += 1;
                    *scored += 1;
                }
                ArmOutcome::Fail => *scored += 1,
                ArmOutcome::Infra => {}
            };
            tally(&a, &mut a_pass[i], &mut a_scored[i]);
            tally(&b, &mut b_pass[i], &mut b_scored[i]);
            tally(&c, &mut c_pass[i], &mut c_scored[i]);

            // Paired rescue/harm only when both A and B produced a real score.
            if let (ArmOutcome::Pass | ArmOutcome::Fail, ArmOutcome::Pass | ArmOutcome::Fail) =
                (&a, &b)
            {
                paired += 1;
                match (matches!(a, ArmOutcome::Pass), matches!(b, ArmOutcome::Pass)) {
                    (false, true) => rescue += 1,
                    (true, false) => harm += 1,
                    _ => {}
                }
            }

            eprintln!(
                "  draw {draw} {:22} A={} B={} C={}  ({} advisors)",
                task.name,
                outcome_mark(&a),
                outcome_mark(&b),
                outcome_mark(&c),
                real.len(),
            );
        }
    }

    eprintln!("\n  per-task pass rates (actor-alone A / +real B / +shuffled C):");
    for (i, task) in tasks.iter().enumerate() {
        eprintln!(
            "    {:22} A {}/{}  B {}/{}  C {}/{}",
            task.name, a_pass[i], a_scored[i], b_pass[i], b_scored[i], c_pass[i], c_scored[i],
        );
    }
    let sum = |v: &[usize]| v.iter().sum::<usize>();
    eprintln!(
        "\n  aggregate: A {}/{}  B {}/{}  C {}/{}",
        sum(&a_pass),
        sum(&a_scored),
        sum(&b_pass),
        sum(&b_scored),
        sum(&c_pass),
        sum(&c_scored),
    );
    eprintln!(
        "  paired A↔B trials: {paired}   rescue (A✗→B✓): {rescue}   harm (A✓→B✗): {harm}   net uplift: {}",
        rescue as i64 - harm as i64,
    );
    eprintln!(
        "\n  PILOT — directional only. A defensible / merge-blocking result needs\n  \
         ~40 preregistered stratified tasks x k>=10 draws, a paired hierarchical\n  \
         bootstrap CI on net uplift, and the production-selected actor. This run\n  \
         builds the instrument and shows direction; it is not that study.\n"
    );

    // Guard: fail loudly if the API was down (so we never report a silent 0/0
    // as if it were a real null result).
    assert!(
        paired >= (n * draws) / 2,
        "too few scored A↔B pairs ({paired}) — API likely unavailable, results not interpretable"
    );
}

/// Tool names offered to every arm (held constant across arms).
fn agent_tools_names() -> Vec<String> {
    vec![
        "list_dir".into(),
        "read_file".into(),
        "search".into(),
        "run_command".into(),
    ]
}

fn session_for_task(task: &ToolTask) -> moa::session::Session {
    let mut s = moa::session::Session::new();
    s.ingest(
        &[json!({"role": "user", "content": task.prompt})],
        &Some(agent_tools()),
    );
    s
}

fn outcome_mark(o: &ArmOutcome) -> &'static str {
    match o {
        ArmOutcome::Pass => "✓",
        ArmOutcome::Fail => "✗",
        ArmOutcome::Infra => "∅",
    }
}

/// The headline claim: MoA over the pool is at least as tool-coherent as the
/// best single model in the pool, measured on the same agentic tasks under the
/// same thinking policy.
///
/// NOTE: superseded by `ablation_do_references_help_the_actor`. This test
/// compares MoA against a max over single models at *different* sampling and
/// prompt than the actor, so a win here is confounded by sampling/prompt and
/// must NOT be cited as evidence the design helps. Kept only as a smoke test
/// that the full `handle_turn` path produces valid tool calls end to end.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn tool_coherence_moa_at_least_best_single() {
    let Some(key) = api_key_or_skip("tool_coherence") else {
        return;
    };
    let pool = mesh_pool();
    let tasks = tool_tasks();
    let solo_backend = OpenRouterBackend::new(key.clone());
    // Realism off for this eval: we are measuring correctness, and injected
    // failures would add noise to a small task set. Durability under faults is
    // its own test below.
    let config = moa_config(&pool, &key, false);

    // Per-model solo pass counts.
    let mut solo_pass: Vec<(String, usize)> = pool.iter().map(|m| (m.id.to_string(), 0)).collect();
    let mut moa_pass = 0usize;

    eprintln!("\n=== tool coherence: {} tasks ===\n", tasks.len());
    for task in &tasks {
        // Each single model.
        let mut solo_line = String::new();
        for (i, m) in pool.iter().enumerate() {
            let ok = match solo_tool_result(&solo_backend, m.id, task).await {
                Ok(tools) => scores_task(&tools, task),
                Err(e) => {
                    eprintln!("  solo {} {} ERR {e}", m.id, task.name);
                    false
                }
            };
            if ok {
                solo_pass[i].1 += 1;
            }
            solo_line.push_str(if ok { "+" } else { "." });
        }

        // MoA over the pool.
        let result = moa::handle_turn(&config, &user_turn(task.prompt, Some(agent_tools()))).await;
        let moa_tools = response_tool_calls(&result.response_body);
        let moa_ok = scores_task(&moa_tools, task);
        if moa_ok {
            moa_pass += 1;
        }

        eprintln!(
            "  {:22} solo[{}]  MoA={}  ({:?}{})",
            task.name,
            solo_line,
            if moa_ok { "PASS" } else { "FAIL" },
            result.turn_kind,
            if result.reducer_used { " +reducer" } else { "" },
        );
        if !moa_ok {
            eprintln!("        MoA emitted: {moa_tools:?}");
        }
    }

    let best_single = solo_pass.iter().map(|(_, n)| *n).max().unwrap_or(0);
    let n = tasks.len();
    eprintln!("\n  per-model solo scores:");
    for (name, pass) in &solo_pass {
        eprintln!("    {pass}/{n}  {name}");
    }
    eprintln!("\n  best single: {best_single}/{n}");
    eprintln!("  MoA:         {moa_pass}/{n}\n");

    assert!(
        moa_pass >= best_single,
        "MoA ({moa_pass}/{n}) should be at least as tool-coherent as the best single model ({best_single}/{n})"
    );
}

// ─── Test 3: durability under mesh faults ────────────────────────────

/// With realism on — slow strong worker, a 25%-flaky peer, typical jitter —
/// MoA must still return a usable turn. This is the property Together's
/// implementation cannot hold: one flaky peer takes down its whole turn.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn survives_mesh_faults() {
    let Some(key) = api_key_or_skip("survives_mesh_faults") else {
        return;
    };
    let pool = mesh_pool();
    let config = moa_config(&pool, &key, true); // realism ON

    let prompts = [
        "Explain what a Merkle tree is in two sentences.",
        "I need to understand this project's error handling. Look in the src directory first.",
        "What is the capital of Australia?",
    ];

    for prompt in prompts {
        let has_tools = prompt.contains("directory");
        let tools = has_tools.then(agent_tools);
        let result = moa::handle_turn(&config, &user_turn(prompt, tools)).await;

        let text = response_text(&result.response_body);
        let calls = response_tool_calls(&result.response_body);
        let failed = result
            .worker_summaries
            .iter()
            .filter(|w| !w.succeeded)
            .count();

        eprintln!(
            "  {:?} kind={:?} failed_workers={}/{} reducer={} -> {}",
            truncate(prompt, 40),
            result.turn_kind,
            failed,
            result.worker_summaries.len(),
            result.reducer_used,
            if !text.trim().is_empty() {
                format!("text[{}]", text.len())
            } else if !calls.is_empty() {
                format!("tool={}", calls[0].0)
            } else {
                "EMPTY".to_string()
            },
        );

        assert_ne!(
            result.turn_kind,
            moa::TurnKind::Failed,
            "a mesh with some flaky workers must still complete the turn"
        );
        assert!(
            !text.trim().is_empty() || !calls.is_empty(),
            "turn produced neither text nor a tool call under faults"
        );
    }
}
