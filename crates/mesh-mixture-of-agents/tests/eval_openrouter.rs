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
        .map(|(i, m)| ModelEntry::new(m.id.to_string(), i))
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
        models.push(ModelEntry::new(m.id.to_string(), backends.len()));
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
        reference_policy: Default::default(),
        refinement_policy: Default::default(),
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

/// Assistant text, mirroring the in-tree backend's fallback chain.
///
/// Reasoning models routinely spend their whole budget in `reasoning` and
/// return `content: null` (the failure Hermes' troubleshooting doc also
/// documents). Reading `content` alone silently dropped 20/30 committee trials,
/// so fall back to `reasoning` before declaring the response empty.
fn response_text(body: &Value) -> String {
    let msg = body.pointer("/choices/0/message");
    let content = msg
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !content.is_empty() {
        return content.to_string();
    }
    msg.and_then(|m| m.get("reasoning"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
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

/// Which reference-packing style to test: `hermes` strips the agent system
/// prompt and tool transcript and never asks for a tool call; default is the
/// original worker packing (full system prompt + role-varied history).
fn reference_packing_is_hermes() -> bool {
    std::env::var("MOA_REFERENCE_PACKING")
        .map(|v| v.eq_ignore_ascii_case("hermes"))
        .unwrap_or(false)
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
        // Packing style under test. MOA_REFERENCE_PACKING=hermes strips the
        // agent system prompt + tool transcript and drops the "or tool call"
        // instruction; default reproduces the original worker packing.
        let packed = if reference_packing_is_hermes() {
            moa::context::pack_for_reference(session, 6)
        } else {
            moa::context::pack_for_worker_selected(
                session,
                moa::worker::WorkerRole::Generalist,
                false, // tool-free: references only advise
                &[],
            )
        };
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

// ─── Scaled ablation study (preregistered) ───────────────────────────
//
// The defensible version of the pilot: loads a preregistered 40-task fixture
// (4 strata x 10), runs the same A/B/C arms at k>=10 draws with bounded
// concurrency, and writes one JSONL trial line per (draw, task, arm). The
// statistics (paired hierarchical bootstrap CI on net uplift) are computed by
// `evals/moa-openrouter/analyze_ablation.py` from that JSONL — the live
// measurement and the analysis are separate so the analysis is deterministic
// and re-runnable.

#[derive(Clone, serde::Deserialize)]
struct AblationTask {
    id: String,
    category: String,
    prompt: String,
    /// Acceptable tools (a SET). Empty ⇒ pass iff NO tool call is emitted.
    accept_tools: Vec<String>,
    arg_must_contain: Option<String>,
    #[serde(default)]
    arg_must_not_contain: Option<String>,
}

#[derive(serde::Deserialize)]
struct AblationFixture {
    tasks: Vec<AblationTask>,
}

fn load_ablation_tasks() -> Vec<AblationTask> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ablation_tasks.json"
    );
    let data = std::fs::read_to_string(path).expect("read ablation fixture");
    let f: AblationFixture = serde_json::from_str(&data).expect("parse ablation fixture");
    f.tasks
}

/// Set-valued scorer: the emitted tool must be in `accept_tools` (or none, when
/// the set is empty), and argument substring constraints must hold.
fn scores_ablation(tools: &[(String, String)], task: &AblationTask) -> bool {
    if task.accept_tools.is_empty() {
        return tools.is_empty();
    }
    let Some((name, args)) = tools.first() else {
        return false;
    };
    if !task.accept_tools.iter().any(|t| t == name) {
        return false;
    }
    if let Some(needle) = &task.arg_must_contain
        && !args.contains(needle.as_str())
    {
        return false;
    }
    if let Some(bad) = &task.arg_must_not_contain
        && args.contains(bad.as_str())
    {
        return false;
    }
    true
}

fn session_for_prompt(prompt: &str) -> moa::session::Session {
    let mut s = moa::session::Session::new();
    s.ingest(
        &[json!({"role": "user", "content": prompt})],
        &Some(agent_tools()),
    );
    s
}

/// Run one actor arm against a fixture task, scoring set-valued acceptance.
async fn actor_outcome(
    backend: &OpenRouterBackend,
    session: &moa::session::Session,
    references: &[moa::normalize::WorkerOutput],
    selected: &[String],
    actor: &str,
    task: &AblationTask,
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
            if scores_ablation(&response_tool_calls(&body), task) {
                ArmOutcome::Pass
            } else {
                ArmOutcome::Fail
            }
        }
        Err(e) if e.starts_with("INFRA:") => ArmOutcome::Infra,
        Err(_) => ArmOutcome::Fail,
    }
}

#[derive(serde::Serialize)]
struct TrialLine {
    draw: usize,
    task_id: String,
    category: String,
    arm: &'static str,
    outcome: &'static str,
    n_advisors: usize,
}

impl TrialLine {
    fn new(
        draw: usize,
        task: &AblationTask,
        arm: &'static str,
        o: &ArmOutcome,
        n_adv: usize,
    ) -> Self {
        Self {
            draw,
            task_id: task.id.clone(),
            category: task.category.clone(),
            arm,
            outcome: match o {
                ArmOutcome::Pass => "pass",
                ArmOutcome::Fail => "fail",
                ArmOutcome::Infra => "infra",
            },
            n_advisors: n_adv,
        }
    }
}

/// Gather tool-free advice for every task, concurrently (bounded).
async fn gather_all_advice(
    backend: &Arc<OpenRouterBackend>,
    pool: &Arc<Vec<PoolModel>>,
    tasks: &Arc<Vec<AblationTask>>,
    actor: &str,
    concurrency: usize,
) -> Vec<Vec<moa::normalize::WorkerOutput>> {
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut js: tokio::task::JoinSet<(usize, Vec<moa::normalize::WorkerOutput>)> =
        tokio::task::JoinSet::new();
    for i in 0..tasks.len() {
        let backend = backend.clone();
        let pool = pool.clone();
        let tasks = tasks.clone();
        let actor = actor.to_string();
        let sem = sem.clone();
        js.spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let session = session_for_prompt(&tasks[i].prompt);
            (i, gather_advice(&backend, &pool, &session, &actor).await)
        });
    }
    let mut out: Vec<Vec<moa::normalize::WorkerOutput>> =
        (0..tasks.len()).map(|_| Vec::new()).collect();
    while let Some(r) = js.join_next().await {
        let (i, adv) = r.expect("advice task panicked");
        out[i] = adv;
    }
    out
}

/// Run all three arms for every task in one draw, concurrently (bounded).
async fn run_draw_arms(
    backend: &Arc<OpenRouterBackend>,
    tasks: &Arc<Vec<AblationTask>>,
    advice: &Arc<Vec<Vec<moa::normalize::WorkerOutput>>>,
    actor: &str,
    selected: &Arc<Vec<String>>,
    draw: usize,
    concurrency: usize,
) -> Vec<TrialLine> {
    let n = tasks.len();
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut js: tokio::task::JoinSet<Vec<TrialLine>> = tokio::task::JoinSet::new();
    for i in 0..n {
        let backend = backend.clone();
        let tasks = tasks.clone();
        let advice = advice.clone();
        let actor = actor.to_string();
        let selected = selected.clone();
        let sem = sem.clone();
        js.spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let task = &tasks[i];
            let session = session_for_prompt(&task.prompt);
            let real = &advice[i];
            let shuffled = &advice[(i + 1) % n]; // advice generated for a different task
            let a = actor_outcome(&backend, &session, &[], &selected, &actor, task).await;
            let b = actor_outcome(&backend, &session, real, &selected, &actor, task).await;
            let c = actor_outcome(&backend, &session, shuffled, &selected, &actor, task).await;
            vec![
                TrialLine::new(draw, task, "A", &a, 0),
                TrialLine::new(draw, task, "B", &b, real.len()),
                TrialLine::new(draw, task, "C", &c, shuffled.len()),
            ]
        });
    }
    let mut out = Vec::new();
    while let Some(r) = js.join_next().await {
        out.extend(r.expect("arm task panicked"));
    }
    out
}

/// The scaled, preregistered study. Writes per-trial JSONL for offline
/// analysis; prints a compact summary. Defaults: 10 draws, concurrency 6,
/// actor = strongest pool tool-caller (override with MOA_ABLATION_ACTOR to pin
/// a weaker actor and probe rescue headroom).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn ablation_scaled_study() {
    let Some(key) = api_key_or_skip("ablation_scaled") else {
        return;
    };
    let draws = std::env::var("MOA_ABLATION_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let concurrency = std::env::var("MOA_ABLATION_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let out_path =
        std::env::var("MOA_ABLATION_OUT").unwrap_or_else(|_| "/tmp/moa_ablation.jsonl".to_string());
    let actor = ablation_actor();

    let backend = Arc::new(OpenRouterBackend::new(key));
    let pool = Arc::new(mesh_pool());
    let tasks = Arc::new(load_ablation_tasks());
    let selected = Arc::new(agent_tools_names());

    eprintln!("\n=== scaled ablation study ===");
    eprintln!(
        "actor={actor}  tasks={}  draws={draws}  concurrency={concurrency}  -> {out_path}\n",
        tasks.len()
    );

    let mut lines: Vec<TrialLine> = Vec::new();
    for draw in 0..draws {
        let advice =
            Arc::new(gather_all_advice(&backend, &pool, &tasks, &actor, concurrency).await);
        let draw_lines = run_draw_arms(
            &backend,
            &tasks,
            &advice,
            &actor,
            &selected,
            draw,
            concurrency,
        )
        .await;
        let passes = |arm: &str| {
            draw_lines
                .iter()
                .filter(|l| l.arm == arm && l.outcome == "pass")
                .count()
        };
        eprintln!(
            "  draw {draw}: A {}/{} B {}/{} C {}/{}",
            passes("A"),
            tasks.len(),
            passes("B"),
            tasks.len(),
            passes("C"),
            tasks.len(),
        );
        lines.extend(draw_lines);
    }

    let mut buf = String::new();
    for l in &lines {
        buf.push_str(&serde_json::to_string(l).expect("serialize trial"));
        buf.push('\n');
    }
    std::fs::write(&out_path, buf).expect("write jsonl");

    let total = lines.len();
    let count = |arm: &str, oc: &str| {
        lines
            .iter()
            .filter(|l| l.arm == arm && l.outcome == oc)
            .count()
    };
    eprintln!("\n  wrote {total} trials to {out_path}");
    eprintln!(
        "  aggregate pass:  A {}  B {}  C {}  (of {} each)",
        count("A", "pass"),
        count("B", "pass"),
        count("C", "pass"),
        total / 3,
    );
    eprintln!("  run: python3 evals/moa-openrouter/analyze_ablation.py {out_path}\n");

    let infra = lines.iter().filter(|l| l.outcome == "infra").count();
    assert!(
        infra * 4 < total.max(1),
        "too many infra errors ({infra}/{total}) — API degraded, results not interpretable"
    );
}

// ─── Matched-peer structured-proposal study ──────────────────────────
//
// Tests the one design cell the ablation did NOT: do *similar-strength,
// different-family* peers make a fixed finalizer better at tool selection when
// they contribute STRUCTURED candidate tool calls (not prose)?
//
// Arms share ONE fixed finalizer; only the candidate proposals it sees vary:
//   A solo        — finalizer alone, no candidates
//   B diverse     — 2 candidates, one each from 2 different-family peers
//   C homogeneous — 2 candidates, both resampled from the finalizer's OWN model
//
// The key comparison is B − C (diverse minus homogeneous): it isolates
// cross-family diversity from "just more samples of the same model". Reuses
// the A/B/C JSONL schema so analyze_ablation.py computes it as `differential
// B-C`. Also records oracle-union (did ANY candidate score): if oracle is high
// but the final is not, the selector is the bug, not the pool.
//
// Candidates are structured `tool(args)`, never prose — prose advice is the
// part the ablation already showed harms a strong actor.

fn matched_finalizer() -> String {
    std::env::var("MOA_MATCHED_FINALIZER").unwrap_or_else(|_| "qwen/qwen3-14b".to_string())
}

/// Different-family peers, similar-ish strength to the finalizer. Strength is
/// approximate (no calibration set yet) — configurable so a matched trio can
/// be pinned once one is defined.
fn matched_diverse_peers() -> Vec<String> {
    std::env::var("MOA_MATCHED_DIVERSE")
        .unwrap_or_else(|_| {
            "mistralai/mistral-small-3.2-24b-instruct,minimax/minimax-m2.5".to_string()
        })
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Ask one model for a single structured tool-call proposal for the task.
async fn structured_proposal(
    backend: &OpenRouterBackend,
    model: &str,
    task: &AblationTask,
) -> Option<(String, String)> {
    let msgs = vec![json!({"role": "user", "content": task.prompt})];
    let body = backend
        .chat_completion_retrying(
            model,
            &msgs,
            Some(&agent_tools()),
            512,
            SamplingParams::worker().with_thinking(Some(false)),
        )
        .await
        .ok()?;
    response_tool_calls(&body).into_iter().next()
}

/// Present anonymized candidate tool calls to the finalizer and let it emit the
/// final call. Empty candidates ⇒ solo (finalizer acts with no suggestions).
/// The scaffold is identical across arms except the candidate list.
fn pack_finalizer(task: &AblationTask, candidates: &[(String, String)]) -> Vec<Value> {
    let mut sys = String::from(
        "You must choose the single best action for the user's request: emit exactly \
         one tool call, or answer directly if no tool fits.",
    );
    if !candidates.is_empty() {
        sys.push_str(
            "\n\nOther models proposed these candidate tool calls (anonymized). Evaluate \
             them critically — some may be wrong — then emit the single best tool call:\n",
        );
        for (i, (name, args)) in candidates.iter().enumerate() {
            sys.push_str(&format!("  {}. {name}({args})\n", i + 1));
        }
    }
    vec![
        json!({"role": "system", "content": sys}),
        json!({"role": "user", "content": task.prompt}),
    ]
}

async fn finalizer_outcome(
    backend: &OpenRouterBackend,
    finalizer: &str,
    task: &AblationTask,
    candidates: &[(String, String)],
) -> ArmOutcome {
    let msgs = pack_finalizer(task, candidates);
    match backend
        .chat_completion_retrying(
            finalizer,
            &msgs,
            Some(&agent_tools()),
            2048,
            SamplingParams::reducer().with_thinking(Some(false)),
        )
        .await
    {
        Ok(body) => {
            if scores_ablation(&response_tool_calls(&body), task) {
                ArmOutcome::Pass
            } else {
                ArmOutcome::Fail
            }
        }
        Err(e) if e.starts_with("INFRA:") => ArmOutcome::Infra,
        Err(_) => ArmOutcome::Fail,
    }
}

#[derive(serde::Serialize)]
struct MatchedTrial {
    draw: usize,
    task_id: String,
    category: String,
    arm: &'static str,
    outcome: &'static str,
    /// Did any candidate proposal in this arm score the task on its own?
    oracle: bool,
    n_candidates: usize,
}

/// Does cross-family structured diversity beat same-model resampling on tool
/// turns? Writes A/B/C JSONL (A=solo, B=diverse, C=homogeneous) for
/// analyze_ablation.py; the `differential B-C` it prints is diverse−homogeneous.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn matched_peer_structured_study() {
    let Some(key) = api_key_or_skip("matched_peer") else {
        return;
    };
    let draws = std::env::var("MOA_MATCHED_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let out_path =
        std::env::var("MOA_MATCHED_OUT").unwrap_or_else(|_| "/tmp/moa_matched.jsonl".to_string());
    let finalizer = matched_finalizer();
    let peers = matched_diverse_peers();
    let backend = OpenRouterBackend::new(key);
    let tasks = load_ablation_tasks();

    eprintln!("\n=== matched-peer structured-proposal study ===");
    eprintln!("finalizer={finalizer}");
    eprintln!("diverse peers={peers:?}");
    eprintln!("tasks={}  draws={draws}  -> {out_path}\n", tasks.len());

    let score_candidate =
        |c: &(String, String), t: &AblationTask| scores_ablation(std::slice::from_ref(c), t);
    let mut lines: Vec<MatchedTrial> = Vec::new();

    for draw in 0..draws {
        for task in &tasks {
            // Diverse candidates: one per different-family peer.
            let mut diverse = Vec::new();
            for p in &peers {
                if let Some(c) = structured_proposal(&backend, p, task).await {
                    diverse.push(c);
                }
            }
            // Homogeneous candidates: same count, resampled from finalizer's model.
            let mut homo = Vec::new();
            for _ in 0..diverse.len().max(1) {
                if let Some(c) = structured_proposal(&backend, &finalizer, task).await {
                    homo.push(c);
                }
            }

            let a = finalizer_outcome(&backend, &finalizer, task, &[]).await;
            let b = finalizer_outcome(&backend, &finalizer, task, &diverse).await;
            let c = finalizer_outcome(&backend, &finalizer, task, &homo).await;

            let mark = |o: &ArmOutcome| match o {
                ArmOutcome::Pass => "pass",
                ArmOutcome::Fail => "fail",
                ArmOutcome::Infra => "infra",
            };
            let div_oracle = diverse.iter().any(|c| score_candidate(c, task));
            let homo_oracle = homo.iter().any(|c| score_candidate(c, task));

            lines.push(MatchedTrial {
                draw,
                task_id: task.id.clone(),
                category: task.category.clone(),
                arm: "A",
                outcome: mark(&a),
                oracle: false,
                n_candidates: 0,
            });
            lines.push(MatchedTrial {
                draw,
                task_id: task.id.clone(),
                category: task.category.clone(),
                arm: "B",
                outcome: mark(&b),
                oracle: div_oracle,
                n_candidates: diverse.len(),
            });
            lines.push(MatchedTrial {
                draw,
                task_id: task.id.clone(),
                category: task.category.clone(),
                arm: "C",
                outcome: mark(&c),
                oracle: homo_oracle,
                n_candidates: homo.len(),
            });
        }
        let dp = |arm: &str| {
            lines
                .iter()
                .filter(|l| l.draw == draw && l.arm == arm && l.outcome == "pass")
                .count()
        };
        eprintln!(
            "  draw {draw}: A(solo) {}/{} B(diverse) {}/{} C(homo) {}/{}",
            dp("A"),
            tasks.len(),
            dp("B"),
            tasks.len(),
            dp("C"),
            tasks.len(),
        );
    }

    let mut buf = String::new();
    for l in &lines {
        buf.push_str(&serde_json::to_string(l).expect("serialize"));
        buf.push('\n');
    }
    std::fs::write(&out_path, buf).expect("write jsonl");

    let count = |arm: &str, oc: &str| {
        lines
            .iter()
            .filter(|l| l.arm == arm && l.outcome == oc)
            .count()
    };
    let oracle = |arm: &str| lines.iter().filter(|l| l.arm == arm && l.oracle).count();
    eprintln!("\n  wrote {} trials to {out_path}", lines.len());
    eprintln!(
        "  pass: A(solo) {}  B(diverse) {}  C(homo) {}  (of {} each)",
        count("A", "pass"),
        count("B", "pass"),
        count("C", "pass"),
        lines.len() / 3,
    );
    eprintln!(
        "  oracle-union (any candidate correct): B(diverse) {}  C(homo) {}",
        oracle("B"),
        oracle("C"),
    );
    eprintln!(
        "  run: python3 evals/moa-openrouter/analyze_ablation.py {out_path}  (differential B-C = diverse - homogeneous)\n"
    );

    let infra = lines.iter().filter(|l| l.outcome == "infra").count();
    assert!(
        infra * 4 < lines.len().max(1),
        "too many infra errors ({infra}/{}) — API degraded",
        lines.len()
    );
}

// ─── Post-hoc tool-call correction study ─────────────────────────────
//
// The mesh scenario neither Hermes nor Together handles: a WEAK tool-caller
// with no strong peer to defer to. Pre-hoc advice was inert-to-harmful (both
// prose and structured). This tests correction of the *concrete drafted call*
// instead — there's a real artifact to judge, not open-ended dilution.
//
// One weak finalizer drafts a call, then three arms (reuse A/B/C JSONL):
//   A draft-alone         — baseline, no correction
//   B deterministic       — schema-validate the draft; on structural failure
//                           re-prompt the finalizer with the specific error
//   C semantic            — a different-family peer critiques the concrete
//                           drafted call; the finalizer revises once
//
// analyze_ablation.py then gives B-vs-A (deterministic rescue), C-vs-A
// (semantic rescue), and differential B-C.

fn correction_finalizer() -> String {
    std::env::var("MOA_CORRECTION_FINALIZER").unwrap_or_else(|_| "qwen/qwen3-8b".to_string())
}

fn correction_critic() -> String {
    std::env::var("MOA_CORRECTION_CRITIC")
        .unwrap_or_else(|_| "mistralai/mistral-small-3.2-24b-instruct".to_string())
}

/// Structural validity of a drafted call: unknown tool, unparseable args, or a
/// missing/empty required field. Returns an error message, or None if valid.
/// This is the deterministic check a mesh can do with zero extra model calls.
fn validate_tool_call(name: &str, args_json: &str) -> Option<String> {
    let required: &[&str] = match name {
        "list_dir" | "read_file" => &["path"],
        "search" => &["pattern", "path"],
        "run_command" => &["cmd"],
        other => return Some(format!("unknown tool '{other}'")),
    };
    let parsed: Value = match serde_json::from_str(args_json) {
        Ok(v) => v,
        Err(e) => return Some(format!("arguments are not valid JSON: {e}")),
    };
    let Some(obj) = parsed.as_object() else {
        return Some("arguments must be a JSON object".to_string());
    };
    for field in required {
        match obj.get(*field).and_then(Value::as_str) {
            Some(s) if !s.trim().is_empty() => {}
            _ => return Some(format!("missing or empty required field '{field}'")),
        }
    }
    None
}

/// One tool-drafting call against the finalizer with the given extra messages
/// appended after the task prompt (for correction re-prompts).
async fn draft_call(
    backend: &OpenRouterBackend,
    finalizer: &str,
    task: &AblationTask,
    extra: &[Value],
) -> Result<Option<(String, String)>, String> {
    let mut msgs = vec![json!({"role": "user", "content": task.prompt})];
    msgs.extend_from_slice(extra);
    let body = backend
        .chat_completion_retrying(
            finalizer,
            &msgs,
            Some(&agent_tools()),
            2048,
            SamplingParams::reducer().with_thinking(Some(false)),
        )
        .await?;
    Ok(response_tool_calls(&body).into_iter().next())
}

/// Deterministic arm: draft, validate, and on structural failure re-prompt with
/// the concrete error (up to 2 corrections). No extra model beyond the retries.
async fn deterministic_correction(
    backend: &OpenRouterBackend,
    finalizer: &str,
    task: &AblationTask,
) -> ArmOutcome {
    let mut extra: Vec<Value> = Vec::new();
    for _ in 0..3 {
        let draft = match draft_call(backend, finalizer, task, &extra).await {
            Ok(d) => d,
            Err(e) if e.starts_with("INFRA:") => return ArmOutcome::Infra,
            Err(_) => return ArmOutcome::Fail,
        };
        let Some((name, args)) = draft else {
            // No tool call — acceptable only if the task wanted none.
            return if task.accept_tools.is_empty() {
                ArmOutcome::Pass
            } else {
                ArmOutcome::Fail
            };
        };
        match validate_tool_call(&name, &args) {
            None => return outcome_for(&[(name, args)], task),
            Some(err) => {
                extra.push(json!({"role": "assistant", "content": Value::Null,
                    "tool_calls": [{"id": "c", "type": "function",
                        "function": {"name": name, "arguments": args}}]}));
                extra.push(json!({"role": "user",
                    "content": format!("That tool call is invalid: {err}. Emit a corrected tool call.")}));
            }
        }
    }
    ArmOutcome::Fail
}

/// Semantic arm: draft, have a different-family critic review the CONCRETE call,
/// then let the finalizer revise once given the critique.
async fn semantic_correction(
    backend: &OpenRouterBackend,
    finalizer: &str,
    critic: &str,
    task: &AblationTask,
) -> ArmOutcome {
    let draft = match draft_call(backend, finalizer, task, &[]).await {
        Ok(Some(d)) => d,
        Ok(None) => {
            return if task.accept_tools.is_empty() {
                ArmOutcome::Pass
            } else {
                ArmOutcome::Fail
            };
        }
        Err(e) if e.starts_with("INFRA:") => return ArmOutcome::Infra,
        Err(_) => return ArmOutcome::Fail,
    };

    let critique_prompt = format!(
        "Task: {}\n\nA model proposed this tool call:\n  {}({})\n\nTools available: \
         list_dir(path), read_file(path), search(pattern,path), run_command(cmd).\n\
         Is this the right tool and arguments for the task? If good, reply exactly OK. \
         If not, briefly say what's wrong.",
        task.prompt, draft.0, draft.1
    );
    let critique = match backend
        .chat_completion_retrying(
            critic,
            &[json!({"role": "user", "content": critique_prompt})],
            None,
            512,
            SamplingParams::worker().with_thinking(Some(false)),
        )
        .await
    {
        Ok(body) => response_text(&body),
        Err(e) if e.starts_with("INFRA:") => return ArmOutcome::Infra,
        Err(_) => return outcome_for(&[draft], task), // critic down → keep draft
    };

    if critique.trim().eq_ignore_ascii_case("OK") || critique.trim().is_empty() {
        return outcome_for(&[draft], task);
    }

    // Revise once given the concrete critique.
    let extra = vec![
        json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": "c", "type": "function",
                "function": {"name": draft.0, "arguments": draft.1}}]}),
        json!({"role": "user",
            "content": format!("A reviewer said about your tool call: {}. Emit the corrected tool call.", critique.trim())}),
    ];
    match draft_call(backend, finalizer, task, &extra).await {
        Ok(Some(revised)) => outcome_for(&[revised], task),
        Ok(None) => ArmOutcome::Fail,
        Err(e) if e.starts_with("INFRA:") => ArmOutcome::Infra,
        Err(_) => ArmOutcome::Fail,
    }
}

fn outcome_for(tools: &[(String, String)], task: &AblationTask) -> ArmOutcome {
    if scores_ablation(tools, task) {
        ArmOutcome::Pass
    } else {
        ArmOutcome::Fail
    }
}

/// Does post-hoc correction of a concrete drafted call rescue a weak
/// tool-caller? Writes A/B/C JSONL (A=draft-alone, B=deterministic,
/// C=semantic) for analyze_ablation.py.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn correction_rescues_weak_tool_caller() {
    let Some(key) = api_key_or_skip("correction") else {
        return;
    };
    let draws = std::env::var("MOA_CORRECTION_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let out_path = std::env::var("MOA_CORRECTION_OUT")
        .unwrap_or_else(|_| "/tmp/moa_correction.jsonl".to_string());
    let finalizer = correction_finalizer();
    let critic = correction_critic();
    let backend = OpenRouterBackend::new(key);
    let tasks = load_ablation_tasks();

    eprintln!("\n=== post-hoc correction study ===");
    eprintln!("finalizer(weak)={finalizer}  critic={critic}");
    eprintln!("tasks={}  draws={draws}  -> {out_path}\n", tasks.len());

    let mark = |o: &ArmOutcome| match o {
        ArmOutcome::Pass => "pass",
        ArmOutcome::Fail => "fail",
        ArmOutcome::Infra => "infra",
    };
    let mut lines: Vec<TrialLine> = Vec::new();

    for draw in 0..draws {
        for task in &tasks {
            // A: draft alone (baseline).
            let a = match draft_call(&backend, &finalizer, task, &[]).await {
                Ok(d) => match d {
                    Some(call) => outcome_for(&[call], task),
                    None if task.accept_tools.is_empty() => ArmOutcome::Pass,
                    None => ArmOutcome::Fail,
                },
                Err(e) if e.starts_with("INFRA:") => ArmOutcome::Infra,
                Err(_) => ArmOutcome::Fail,
            };
            let b = deterministic_correction(&backend, &finalizer, task).await;
            let c = semantic_correction(&backend, &finalizer, &critic, task).await;

            for (arm, o) in [("A", &a), ("B", &b), ("C", &c)] {
                lines.push(TrialLine {
                    draw,
                    task_id: task.id.clone(),
                    category: task.category.clone(),
                    arm,
                    outcome: mark(o),
                    n_advisors: 0,
                });
            }
        }
        let dp = |arm: &str| {
            lines
                .iter()
                .filter(|l| l.draw == draw && l.arm == arm && l.outcome == "pass")
                .count()
        };
        eprintln!(
            "  draw {draw}: A(draft) {}/{} B(determ) {}/{} C(semantic) {}/{}",
            dp("A"),
            tasks.len(),
            dp("B"),
            tasks.len(),
            dp("C"),
            tasks.len(),
        );
    }

    let mut buf = String::new();
    for l in &lines {
        buf.push_str(&serde_json::to_string(l).expect("serialize"));
        buf.push('\n');
    }
    std::fs::write(&out_path, buf).expect("write jsonl");

    let count = |arm: &str, oc: &str| {
        lines
            .iter()
            .filter(|l| l.arm == arm && l.outcome == oc)
            .count()
    };
    eprintln!("\n  wrote {} trials to {out_path}", lines.len());
    eprintln!(
        "  pass: A(draft) {}  B(determ) {}  C(semantic) {}  (of {} each)",
        count("A", "pass"),
        count("B", "pass"),
        count("C", "pass"),
        lines.len() / 3,
    );
    eprintln!("  run: python3 evals/moa-openrouter/analyze_ablation.py {out_path}\n");

    let infra = lines.iter().filter(|l| l.outcome == "infra").count();
    assert!(
        infra * 4 < lines.len().max(1),
        "too many infra errors ({infra}/{}) — API degraded",
        lines.len()
    );
}

// ─── Committee study: reasoning/answer turns ─────────────────────────
//
// Tests where MoA's value actually lives per Together's validated claim:
// open-ended ANSWER QUALITY on realistic agent-session turns (reasoning over
// tool output, planning, explanation) — NOT tool selection.
//
// Fixed aggregator throughout; only its input varies (mirrors Together's own
// ablation and keeps the comparison clean):
//   A alone     — aggregator answers with no peer input
//   B committee  — aggregator synthesizes 3 diverse-family peer drafts (1 round)
//   C layered    — same, but peers first refine seeing each other's drafts
//                  (Together's `layers`), then aggregator synthesizes
//
// Judged pairwise by an out-of-pool, different-family judge (gpt-4o-mini),
// POSITION-SWAPPED: a win counts only if it survives both orders, else tie.
// Output lengths are logged (not truncated) so a length-vs-preference check is
// possible — the expert's required control against verbosity bias.

fn committee_aggregator() -> String {
    std::env::var("MOA_COMMITTEE_AGGREGATOR").unwrap_or_else(|_| "qwen/qwen3-32b".to_string())
}

fn committee_peers() -> Vec<String> {
    std::env::var("MOA_COMMITTEE_PEERS")
        .unwrap_or_else(|_| {
            "qwen/qwen3-14b,mistralai/mistral-small-3.2-24b-instruct,minimax/minimax-m2.5"
                .to_string()
        })
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn committee_judge() -> String {
    std::env::var("MOA_COMMITTEE_JUDGE").unwrap_or_else(|_| "openai/gpt-4o-mini".to_string())
}

#[derive(serde::Deserialize)]
struct CommitteeTask {
    id: String,
    category: String,
    prompt: String,
}

#[derive(serde::Deserialize)]
struct CommitteeFixture {
    tasks: Vec<CommitteeTask>,
}

fn load_committee_tasks() -> Vec<CommitteeTask> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/committee_tasks.json"
    );
    let data = std::fs::read_to_string(path).expect("read committee fixture");
    serde_json::from_str::<CommitteeFixture>(&data)
        .expect("parse committee fixture")
        .tasks
}

const COMMITTEE_SYNTH_PROMPT: &str = "You have been given a user request and several candidate responses from other \
     models. Synthesize them into one high-quality response. Critically evaluate them — \
     some may be biased or incorrect, and agreement is not proof of correctness. Do not \
     merely copy the longest or most confident; produce the most accurate, well-structured \
     reply. Be direct.";

async fn plain_answer(backend: &OpenRouterBackend, model: &str, prompt: &str) -> Option<String> {
    let msgs = vec![json!({"role": "user", "content": prompt})];
    let body = backend
        .chat_completion_retrying(
            model,
            &msgs,
            None,
            1024,
            SamplingParams::worker().with_thinking(Some(false)),
        )
        .await
        .ok()?;
    let t = response_text(&body);
    (!t.trim().is_empty()).then_some(t)
}

fn synth_messages(prompt: &str, drafts: &[String]) -> Vec<Value> {
    let mut refs = String::new();
    for (i, d) in drafts.iter().enumerate() {
        refs.push_str(&format!("\n[Response {}]:\n{}\n", i + 1, d));
    }
    vec![
        json!({"role": "system", "content": format!("{COMMITTEE_SYNTH_PROMPT}\n\nCandidate responses:{refs}")}),
        json!({"role": "user", "content": prompt}),
    ]
}

async fn synthesize(
    backend: &OpenRouterBackend,
    aggregator: &str,
    prompt: &str,
    drafts: &[String],
) -> Option<String> {
    if drafts.is_empty() {
        return None;
    }
    let body = backend
        .chat_completion_retrying(
            aggregator,
            &synth_messages(prompt, drafts),
            None,
            1024,
            SamplingParams::reducer().with_thinking(Some(false)),
        )
        .await
        .ok()?;
    let t = response_text(&body);
    (!t.trim().is_empty()).then_some(t)
}

/// One peer refines its own draft after seeing all peers' drafts (Together's
/// layer 2+): the peer, not the aggregator, does the cross-pollination.
async fn refine(
    backend: &OpenRouterBackend,
    model: &str,
    prompt: &str,
    all_drafts: &[String],
) -> Option<String> {
    let mut refs = String::new();
    for (i, d) in all_drafts.iter().enumerate() {
        refs.push_str(&format!("\n[Response {}]:\n{}\n", i + 1, d));
    }
    let msgs = vec![
        json!({"role": "system", "content": format!("{COMMITTEE_SYNTH_PROMPT}\n\nCandidate responses:{refs}")}),
        json!({"role": "user", "content": prompt}),
    ];
    let body = backend
        .chat_completion_retrying(
            model,
            &msgs,
            None,
            1024,
            SamplingParams::worker().with_thinking(Some(false)),
        )
        .await
        .ok()?;
    let t = response_text(&body);
    (!t.trim().is_empty()).then_some(t)
}

/// Judge verdict for one ordered pair: which response better answers the prompt.
/// Returns 1 (first better), 2 (second better), or 0 (tie/uncertain).
async fn judge_once(
    backend: &OpenRouterBackend,
    judge: &str,
    prompt: &str,
    first: &str,
    second: &str,
) -> Option<u8> {
    // Length control. The unguarded wording ("more accurate, complete, and
    // useful") produced a judge that scored length, not quality: over 80 e2e
    // trials the longer answer won 13-0 and the shorter one lost 24-4,
    // point-biserial r=+0.68 between length delta and verdict. "Complete"
    // in particular reads as "longer". The instruction below is the standard
    // mitigation — name the bias and forbid it explicitly.
    let j = format!(
        "User request:\n{prompt}\n\n--- Response A ---\n{first}\n\n--- Response B ---\n{second}\n\n\
         Which response better answers the request? Judge only on correctness, relevance, \
         and whether it actually addresses what was asked. Length is NOT quality: do not \
         prefer a response for being longer, more detailed, or more thorough-looking. A \
         shorter response that answers correctly beats a longer one that pads, repeats, or \
         drifts. Reply with exactly one token: A, B, or TIE.",
    );
    let body = backend
        .chat_completion_retrying(
            judge,
            &[json!({"role": "user", "content": j})],
            None,
            8,
            SamplingParams::reducer().with_thinking(Some(false)),
        )
        .await
        .ok()?;
    let v = response_text(&body).trim().to_ascii_uppercase();
    if v.starts_with('A') {
        Some(1)
    } else if v.starts_with('B') {
        Some(2)
    } else {
        Some(0)
    }
}

/// Position-swapped pairwise judgment of x vs y. Returns +1 (x wins), -1 (y
/// wins), 0 (tie) — a win only if it survives BOTH orderings.
async fn judge_pair(
    backend: &OpenRouterBackend,
    judge: &str,
    prompt: &str,
    x: &str,
    y: &str,
) -> i8 {
    let fwd = judge_once(backend, judge, prompt, x, y).await; // A=x B=y
    let rev = judge_once(backend, judge, prompt, y, x).await; // A=y B=x
    match (fwd, rev) {
        (Some(1), Some(2)) => 1,  // x preferred both times
        (Some(2), Some(1)) => -1, // y preferred both times
        _ => 0,                   // disagreement or tie => tie
    }
}

#[derive(serde::Serialize)]
struct CommitteeTrial {
    draw: usize,
    task_id: String,
    category: String,
    /// +1 committee(B) beats alone(A), -1 A beats B, 0 tie.
    b_vs_a: i8,
    /// +1 layered(C) beats alone(A).
    c_vs_a: i8,
    /// +1 layered(C) beats committee(B).
    c_vs_b: i8,
    len_a: usize,
    len_b: usize,
    len_c: usize,
}

/// One committee trial: build all three arms for a prompt, then judge them.
/// Returns `None` when an arm produced no usable text (counted as skipped
/// rather than scored, so an empty output never masquerades as a preference).
#[allow(clippy::too_many_arguments)]
async fn run_committee_trial(
    backend: &Arc<OpenRouterBackend>,
    aggregator: &str,
    peers: &[String],
    judge: &str,
    draw: usize,
    task_id: &str,
    category: &str,
    prompt: &str,
) -> Option<CommitteeTrial> {
    // A: aggregator alone.
    let a = plain_answer(backend, aggregator, prompt).await?;

    // Peer drafts (round 1), gathered concurrently.
    let mut drafts = Vec::new();
    let mut js: tokio::task::JoinSet<Option<String>> = tokio::task::JoinSet::new();
    for peer in peers {
        let (b, p, q) = (backend.clone(), peer.clone(), prompt.to_string());
        js.spawn(async move { plain_answer(&b, &p, &q).await });
    }
    while let Some(r) = js.join_next().await {
        if let Some(d) = r.ok().flatten() {
            drafts.push(d);
        }
    }
    if drafts.len() < 2 {
        return None; // not enough peers to form a committee
    }

    // B: synthesize the round-1 drafts.
    let b = synthesize(backend, aggregator, prompt, &drafts).await?;

    // C: peers refine seeing each other's drafts, then synthesize.
    let mut refined = Vec::new();
    let mut js: tokio::task::JoinSet<Option<String>> = tokio::task::JoinSet::new();
    for peer in peers {
        let (bk, p, q, d) = (
            backend.clone(),
            peer.clone(),
            prompt.to_string(),
            drafts.clone(),
        );
        js.spawn(async move { refine(&bk, &p, &q, &d).await });
    }
    while let Some(r) = js.join_next().await {
        if let Some(x) = r.ok().flatten() {
            refined.push(x);
        }
    }
    let c = synthesize(backend, aggregator, prompt, &refined)
        .await
        .unwrap_or_else(|| b.clone());

    // The three comparisons are independent — judge them concurrently rather
    // than three serial awaits (~3x faster on the judging phase).
    let (b_vs_a, c_vs_a, c_vs_b) = tokio::join!(
        judge_pair(backend, judge, prompt, &b, &a),
        judge_pair(backend, judge, prompt, &c, &a),
        judge_pair(backend, judge, prompt, &c, &b),
    );

    Some(CommitteeTrial {
        draw,
        task_id: task_id.to_string(),
        category: category.to_string(),
        b_vs_a,
        c_vs_a,
        c_vs_b,
        len_a: a.len(),
        len_b: b.len(),
        len_c: c.len(),
    })
}

// ─── End-to-end: the shipped path, not the harness ───────────────────
//
// The committee study above measures MoA's *mechanism* using this file's own
// helpers. This one measures `moa::handle_turn` itself — the code that actually
// runs in production, including round-1 fan-out, the refinement round, grace,
// arbitration and the reducer.
//
// It exists because "the harness and production do the same thing" has been
// wrong three times in this branch (reference packing, draft truncation,
// refinement prompt). This converts the headline claim from an expectation
// about the shipped path into an observation of it.

/// An all-small, family-diverse pool: what a few laptops actually look like.
fn small_mesh_pool() -> Vec<String> {
    std::env::var("MOA_E2E_POOL")
        .unwrap_or_else(|_| {
            "qwen/qwen3-8b,meta-llama/llama-3.1-8b-instruct,\
             ibm-granite/granite-4.1-8b,mistralai/ministral-8b-2512"
                .to_string()
        })
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build a `GatewayConfig` over live OpenRouter backends with production-shaped
/// timings, so the turn exercises the real policies (`ReferencePolicy::Auto`,
/// `RefinementPolicy::Auto`) rather than test overrides.
fn e2e_config(pool: &[String], api_key: &str) -> GatewayConfig {
    let mut backends: Vec<Arc<dyn ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for id in pool {
        models.push(ModelEntry::new(id.clone(), backends.len()));
        backends.push(Arc::new(OpenRouterBackend::new(api_key.to_string())));
    }
    GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(90),
        hedge_delay: Duration::from_secs(5),
        reducer_timeout: Duration::from_secs(60),
        // Production private-mesh default is now 10s (widened from 3s so the
        // committee completes before grace arms on fast infra).
        // `MOA_E2E_GRACE_SECS=0` disables grace entirely for isolation tests.
        first_answer_grace: Duration::from_secs(
            std::env::var("MOA_E2E_GRACE_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
        ),
        strong_patience: Duration::from_secs(20),
        enable_thinking: Some(false),
        actor_candidates: Vec::new(),
        reference_policy: Default::default(),
        refinement_policy: Default::default(),
    }
}

/// The claim, measured through the shipped entrypoint: does `handle_turn` over
/// a small diverse pool beat the single best member of that pool?
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn e2e_handle_turn_beats_best_single_small_model() {
    let Some(key) = api_key_or_skip("e2e") else {
        return;
    };
    let draws = std::env::var("MOA_E2E_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let concurrency = std::env::var("MOA_E2E_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let out_path =
        std::env::var("MOA_E2E_OUT").unwrap_or_else(|_| "/tmp/moa_e2e.jsonl".to_string());

    let pool = small_mesh_pool();
    // Solo baseline is the pool's strongest member, so this is "mesh vs the
    // best you could have run alone", not "mesh vs an average member".
    let solo = pool[0].clone();
    let judge = committee_judge();
    let tasks = load_committee_tasks();
    let backend = Arc::new(OpenRouterBackend::new(key.clone()));
    let cfg = Arc::new(e2e_config(&pool, &key));

    eprintln!("\n=== e2e: handle_turn vs best single small model ===");
    eprintln!("pool={pool:?}");
    eprintln!(
        "solo baseline={solo}  judge={judge}  tasks={}  draws={draws}  -> {out_path}\n",
        tasks.len()
    );

    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut js: tokio::task::JoinSet<Option<CommitteeTrial>> = tokio::task::JoinSet::new();
    for draw in 0..draws {
        for task in &tasks {
            let (backend, cfg, sem) = (backend.clone(), cfg.clone(), sem.clone());
            let (solo, judge) = (solo.clone(), judge.clone());
            let (id, category, prompt) =
                (task.id.clone(), task.category.clone(), task.prompt.clone());
            js.spawn(async move {
                let _permit = sem.acquire_owned().await.ok()?;
                // A: the best single model, alone.
                let a = plain_answer(&backend, &solo, &prompt).await?;
                // B: the shipped MoA path over the whole pool.
                let turn = moa::handle_turn(&cfg, &user_turn(&prompt, None)).await;
                let b = response_text(&turn.response_body);
                if b.trim().is_empty() {
                    eprintln!("  draw {draw} {id:22} SKIPPED (empty MoA turn)");
                    return None;
                }
                let verdict = judge_pair(&backend, &judge, &prompt, &b, &a).await;
                eprintln!(
                    "  draw {draw} {id:22} MoA/solo={verdict:+}  kind={:?} reducer={}  len A{} B{}",
                    turn.turn_kind,
                    turn.reducer_used,
                    a.len(),
                    b.len(),
                );
                Some(CommitteeTrial {
                    draw,
                    task_id: id,
                    category,
                    b_vs_a: verdict,
                    c_vs_a: 0,
                    c_vs_b: 0,
                    len_a: a.len(),
                    len_b: b.len(),
                    len_c: 0,
                })
            });
        }
    }

    let mut lines: Vec<CommitteeTrial> = Vec::new();
    while let Some(r) = js.join_next().await {
        if let Some(t) = r.expect("e2e trial panicked") {
            lines.push(t);
        }
    }
    lines.sort_by(|x, y| (x.draw, &x.task_id).cmp(&(y.draw, &y.task_id)));

    let mut buf = String::new();
    for l in &lines {
        buf.push_str(&serde_json::to_string(l).expect("serialize"));
        buf.push('\n');
    }
    std::fs::write(&out_path, buf).expect("write jsonl");

    let w = lines.iter().filter(|l| l.b_vs_a == 1).count();
    let t = lines.iter().filter(|l| l.b_vs_a == 0).count();
    let l_ = lines.iter().filter(|l| l.b_vs_a == -1).count();
    let mean = |f: fn(&CommitteeTrial) -> usize| {
        if lines.is_empty() {
            0
        } else {
            lines.iter().map(f).sum::<usize>() / lines.len()
        }
    };
    eprintln!("\n  trials={}  (position-swapped)", lines.len());
    eprintln!("  handle_turn vs best single small model:  win {w}  tie {t}  loss {l_}");
    eprintln!(
        "  mean output chars: solo {}  MoA {}",
        mean(|x| x.len_a),
        mean(|x| x.len_b)
    );
    eprintln!(
        "  analyze: python3 evals/moa-openrouter/analyze_ablation.py is for the ablation; use the printed w/t/l here\n"
    );

    assert!(
        !lines.is_empty(),
        "no e2e trials completed — API likely unavailable"
    );
}

/// Does a committee of diverse peers beat the aggregator alone on realistic
/// reasoning/answer turns? Writes per-trial JSONL; prints win/tie/loss and mean
/// lengths. Pilot scale by default.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn committee_beats_solo_on_reasoning() {
    let Some(key) = api_key_or_skip("committee") else {
        return;
    };
    let draws = std::env::var("MOA_COMMITTEE_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let out_path = std::env::var("MOA_COMMITTEE_OUT")
        .unwrap_or_else(|_| "/tmp/moa_committee.jsonl".to_string());
    let aggregator = committee_aggregator();
    let peers = committee_peers();
    let judge = committee_judge();
    let backend = OpenRouterBackend::new(key);
    let tasks = load_committee_tasks();

    eprintln!("\n=== committee study (reasoning/answer turns) ===");
    eprintln!("aggregator={aggregator}");
    eprintln!("peers={peers:?}");
    eprintln!(
        "judge={judge}  tasks={}  draws={draws}  -> {out_path}\n",
        tasks.len()
    );

    // Trials are independent, so run them with bounded concurrency: serial
    // execution was ~18 calls x 30 trials and far too slow to scale to the
    // prompt count a defensible claim needs.
    let concurrency = std::env::var("MOA_COMMITTEE_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let backend = Arc::new(backend);
    let peers = Arc::new(peers);
    let aggregator = Arc::new(aggregator);
    let judge = Arc::new(judge);
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let skipped = Arc::new(AtomicU64::new(0));

    let mut js: tokio::task::JoinSet<Option<CommitteeTrial>> = tokio::task::JoinSet::new();
    for draw in 0..draws {
        for task in &tasks {
            let (backend, peers, aggregator, judge, sem, skipped) = (
                backend.clone(),
                peers.clone(),
                aggregator.clone(),
                judge.clone(),
                sem.clone(),
                skipped.clone(),
            );
            let (id, category, prompt) =
                (task.id.clone(), task.category.clone(), task.prompt.clone());
            js.spawn(async move {
                let _permit = sem.acquire_owned().await.ok()?;
                let t = run_committee_trial(
                    &backend,
                    &aggregator,
                    &peers,
                    &judge,
                    draw,
                    &id,
                    &category,
                    &prompt,
                )
                .await;
                if t.is_none() {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    eprintln!("  draw {draw} {id:22} SKIPPED (empty output)");
                }
                t
            });
        }
    }

    let mut lines: Vec<CommitteeTrial> = Vec::new();
    while let Some(res) = js.join_next().await {
        if let Some(t) = res.expect("committee trial panicked") {
            eprintln!(
                "  draw {} {:22} B/A={:+} C/A={:+} C/B={:+}  len A{} B{} C{}",
                t.draw, t.task_id, t.b_vs_a, t.c_vs_a, t.c_vs_b, t.len_a, t.len_b, t.len_c
            );
            lines.push(t);
        }
    }
    lines.sort_by(|x, y| (x.draw, &x.task_id).cmp(&(y.draw, &y.task_id)));
    let skipped = skipped.load(Ordering::Relaxed);

    let mut buf = String::new();
    for l in &lines {
        buf.push_str(&serde_json::to_string(l).expect("serialize"));
        buf.push('\n');
    }
    std::fs::write(&out_path, buf).expect("write jsonl");

    let tally = |sel: fn(&CommitteeTrial) -> i8| {
        let (mut w, mut t, mut l) = (0, 0, 0);
        for x in &lines {
            match sel(x) {
                1 => w += 1,
                0 => t += 1,
                _ => l += 1,
            }
        }
        (w, t, l)
    };
    let (bw, bt, bl) = tally(|x| x.b_vs_a);
    let (cw, ct, cl) = tally(|x| x.c_vs_a);
    let (cbw, cbt, cbl) = tally(|x| x.c_vs_b);
    let mean = |f: fn(&CommitteeTrial) -> usize| {
        if lines.is_empty() {
            0
        } else {
            lines.iter().map(f).sum::<usize>() / lines.len()
        }
    };
    eprintln!(
        "\n  trials={}  skipped={skipped} (empty output)  \
         (position-swapped; win only if consistent both orders)",
        lines.len()
    );
    eprintln!("  committee(B) vs alone(A):  win {bw}  tie {bt}  loss {bl}");
    eprintln!("  layered(C)   vs alone(A):  win {cw}  tie {ct}  loss {cl}");
    eprintln!("  layered(C)   vs committee(B): win {cbw}  tie {cbt}  loss {cbl}");
    eprintln!(
        "  mean output chars: A {}  B {}  C {}  (length-vs-preference control)",
        mean(|x| x.len_a),
        mean(|x| x.len_b),
        mean(|x| x.len_c),
    );
    eprintln!(
        "\n  PILOT — directional. A defensible claim needs ~100 preregistered prompts, k>=3, bootstrap CI, and a length-controlled preference check.\n"
    );

    assert!(
        !lines.is_empty(),
        "no committee trials completed — API likely unavailable"
    );
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

/// Does a tool turn stay healthy when a real multi-model pool is present?
///
/// The tool path is deliberately asymmetric: it routes to the single best
/// tool-caller rather than fanning out and voting, because voting on tool calls
/// measured null-to-harmful. That design decision is only safe if the structured
/// `tool_calls` still come back intact when several models are available — the
/// actor has to be selected and its call must survive arbitration rather than
/// the turn collapsing or the arguments getting mangled.
///
/// So this asserts the *contract*, not a quality delta: with a multi-model pool
/// on the wire, a tool prompt produces a well-formed tool call, and a "no tool
/// needed" prompt does not invent one. It also records `x-moa`-equivalent turn
/// facts (workers dispatched, actor count) so the single-actor behaviour is
/// visible rather than assumed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live network + cost; run explicitly with --ignored"]
async fn e2e_tool_turns_stay_healthy_with_a_multi_model_pool() {
    let Some(key) = api_key_or_skip("e2e_tool_turns_stay_healthy_with_a_multi_model_pool") else {
        return;
    };
    let pool = small_mesh_pool();
    assert!(
        pool.len() >= 2,
        "this test is about MULTI-model pools; got {pool:?}"
    );
    let config = e2e_config(&pool, &key);
    let tools = agent_tools();

    println!(
        "\n=== tool turns through handle_turn, pool of {} ===",
        pool.len()
    );
    for m in &pool {
        println!("  peer: {m}");
    }

    let mut failures: Vec<String> = Vec::new();

    for task in tool_tasks() {
        let body = user_turn(task.prompt, Some(tools.clone()));
        let started = Instant::now();
        let result = moa::handle_turn(&config, &body).await;
        let elapsed = started.elapsed();

        let calls = response_tool_calls(&result.response_body);
        let text = response_text(&result.response_body);
        let dispatched = result.worker_summaries.len();
        let ok = result
            .worker_summaries
            .iter()
            .filter(|w| w.succeeded)
            .count();

        println!(
            "\n[{}] kind={:?} dispatched={} ok={} reducer={} {:.1}s",
            task.name,
            result.turn_kind,
            dispatched,
            ok,
            result.reducer_used,
            elapsed.as_secs_f64(),
        );
        for (n, a) in &calls {
            println!("   tool_call: {n}({a})");
        }
        if calls.is_empty() {
            println!("   text: {:?}", truncate(&text, 100));
        }

        match task.expect_tool {
            // A tool is expected: exactly one well-formed call, correct name,
            // arguments must be valid JSON (a mangled call is the failure mode
            // this test exists to catch).
            Some(expected) => {
                if calls.is_empty() {
                    failures.push(format!("{}: expected a tool call, got none", task.name));
                    continue;
                }
                if calls.len() > 1 {
                    failures.push(format!(
                        "{}: expected one tool call, got {}",
                        task.name,
                        calls.len()
                    ));
                }
                let (name, args) = &calls[0];
                if serde_json::from_str::<Value>(args).is_err() {
                    failures.push(format!(
                        "{}: arguments are not valid JSON: {args:?}",
                        task.name
                    ));
                }
                // The tool chosen can legitimately differ from the single-model
                // expectation (any of the offered tools is a defensible first
                // move), so only flag a name that is not an offered tool at all.
                if !agent_tools_names().iter().any(|t| t == name) {
                    failures.push(format!(
                        "{}: called {name:?}, which is not an offered tool",
                        task.name
                    ));
                }
                if name != expected {
                    println!("   note: chose {name:?}, single-model baseline expects {expected:?}");
                }
            }
            // No tool needed: must not invent one.
            None => {
                if !calls.is_empty() {
                    failures.push(format!(
                        "{}: invented a tool call on a no-tool prompt: {:?}",
                        task.name, calls[0]
                    ));
                }
                if text.trim().is_empty() {
                    failures.push(format!("{}: no answer text produced", task.name));
                }
            }
        }

        if result.turn_kind == moa::TurnKind::Failed {
            failures.push(format!("{}: turn failed outright", task.name));
        }
    }

    assert!(
        failures.is_empty(),
        "tool-turn contract violations with a {}-model pool:\n  {}",
        pool.len(),
        failures.join("\n  ")
    );
    println!(
        "\nOK: tool turns healthy across a {}-model pool",
        pool.len()
    );
}
