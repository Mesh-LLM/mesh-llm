use super::*;

// ─── Test 2: tool coherence, MoA vs best single ──────────────────────

pub(crate) struct ToolTask {
    pub(crate) name: &'static str,
    pub(crate) prompt: &'static str,
    /// Expected tool name, or None if no tool should be called.
    pub(crate) expect_tool: Option<&'static str>,
    /// Optional substring the winning arguments must contain (majority-correct
    /// answer), used to catch hallucinated arguments.
    pub(crate) expect_arg_contains: Option<&'static str>,
    /// Optional substring the winning arguments must NOT contain (a known
    /// hallucination some single models emit).
    pub(crate) reject_arg_contains: Option<&'static str>,
}

pub(crate) fn tool_tasks() -> Vec<ToolTask> {
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
pub(crate) async fn solo_tool_result(
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

pub(crate) fn scores_task(tools: &[(String, String)], task: &ToolTask) -> bool {
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
pub(crate) fn ablation_actor() -> String {
    std::env::var("MOA_ABLATION_ACTOR").unwrap_or_else(|_| "qwen/qwen3-32b".to_string())
}

pub(crate) fn ablation_draws() -> usize {
    std::env::var("MOA_ABLATION_DRAWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// Which reference-packing style to test: `hermes` strips the agent system
/// prompt and tool transcript and never asks for a tool call; default is the
/// original worker packing (full system prompt + role-varied history).
pub(crate) fn reference_packing_is_hermes() -> bool {
    std::env::var("MOA_REFERENCE_PACKING")
        .map(|v| v.eq_ignore_ascii_case("hermes"))
        .unwrap_or(false)
}

/// Run every non-actor pool model tool-free and collect its prose advice,
/// exactly as the production reference phase does.
pub(crate) async fn gather_advice(
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
pub(crate) enum ArmOutcome {
    Pass,
    Fail,
    /// Excluded from the capability analysis (transient infra error).
    Infra,
}

pub(crate) async fn run_actor_arm(
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
pub(crate) struct AblationTask {
    pub(crate) id: String,
    pub(crate) category: String,
    pub(crate) prompt: String,
    /// Acceptable tools (a SET). Empty ⇒ pass iff NO tool call is emitted.
    pub(crate) accept_tools: Vec<String>,
    pub(crate) arg_must_contain: Option<String>,
    #[serde(default)]
    pub(crate) arg_must_not_contain: Option<String>,
}

#[derive(serde::Deserialize)]
pub(crate) struct AblationFixture {
    pub(crate) tasks: Vec<AblationTask>,
}

pub(crate) fn load_ablation_tasks() -> Vec<AblationTask> {
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
pub(crate) fn scores_ablation(tools: &[(String, String)], task: &AblationTask) -> bool {
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

pub(crate) fn session_for_prompt(prompt: &str) -> moa::session::Session {
    let mut s = moa::session::Session::new();
    s.ingest(
        &[json!({"role": "user", "content": prompt})],
        &Some(agent_tools()),
    );
    s
}

/// Run one actor arm against a fixture task, scoring set-valued acceptance.
pub(crate) async fn actor_outcome(
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
pub(crate) struct TrialLine {
    pub(crate) draw: usize,
    pub(crate) task_id: String,
    pub(crate) category: String,
    pub(crate) arm: &'static str,
    pub(crate) outcome: &'static str,
    pub(crate) n_advisors: usize,
}

impl TrialLine {
    pub(crate) fn new(
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
pub(crate) async fn gather_all_advice(
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
pub(crate) async fn run_draw_arms(
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
