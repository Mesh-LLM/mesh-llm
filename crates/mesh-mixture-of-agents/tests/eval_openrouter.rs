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

#[path = "eval_openrouter/ablation.rs"]
mod ablation;
#[path = "eval_openrouter/backend.rs"]
mod backend;
#[path = "eval_openrouter/committee.rs"]
mod committee;
#[path = "eval_openrouter/correction.rs"]
mod correction;
#[path = "eval_openrouter/matched.rs"]
mod matched;
#[path = "eval_openrouter/pool.rs"]
mod pool;
#[path = "eval_openrouter/request_helpers.rs"]
mod request_helpers;

use ablation::*;
use backend::*;
use committee::*;
use correction::*;
use matched::*;
use pool::*;
use request_helpers::*;

// ─── Mesh-likely model pool ──────────────────────────────────────────

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
