use super::*;

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

pub(crate) fn committee_aggregator() -> String {
    std::env::var("MOA_COMMITTEE_AGGREGATOR").unwrap_or_else(|_| "qwen/qwen3-32b".to_string())
}

pub(crate) fn committee_peers() -> Vec<String> {
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

pub(crate) fn committee_judge() -> String {
    std::env::var("MOA_COMMITTEE_JUDGE").unwrap_or_else(|_| "openai/gpt-4o-mini".to_string())
}

#[derive(serde::Deserialize)]
pub(crate) struct CommitteeTask {
    pub(crate) id: String,
    pub(crate) category: String,
    pub(crate) prompt: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct CommitteeFixture {
    pub(crate) tasks: Vec<CommitteeTask>,
}

pub(crate) fn load_committee_tasks() -> Vec<CommitteeTask> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/committee_tasks.json"
    );
    let data = std::fs::read_to_string(path).expect("read committee fixture");
    serde_json::from_str::<CommitteeFixture>(&data)
        .expect("parse committee fixture")
        .tasks
}

pub(crate) const COMMITTEE_SYNTH_PROMPT: &str = "You have been given a user request and several candidate responses from other \
     models. Synthesize them into one high-quality response. Critically evaluate them — \
     some may be biased or incorrect, and agreement is not proof of correctness. Do not \
     merely copy the longest or most confident; produce the most accurate, well-structured \
     reply. Be direct.";

pub(crate) async fn plain_answer(
    backend: &OpenRouterBackend,
    model: &str,
    prompt: &str,
) -> Option<String> {
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

pub(crate) fn synth_messages(prompt: &str, drafts: &[String]) -> Vec<Value> {
    let mut refs = String::new();
    for (i, d) in drafts.iter().enumerate() {
        refs.push_str(&format!("\n[Response {}]:\n{}\n", i + 1, d));
    }
    vec![
        json!({"role": "system", "content": format!("{COMMITTEE_SYNTH_PROMPT}\n\nCandidate responses:{refs}")}),
        json!({"role": "user", "content": prompt}),
    ]
}

pub(crate) async fn synthesize(
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
pub(crate) async fn refine(
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
pub(crate) async fn judge_once(
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
    match v.as_str() {
        "A" => Some(1),
        "B" => Some(2),
        _ => Some(0),
    }
}

/// Position-swapped pairwise judgment of x vs y. Returns +1 (x wins), -1 (y
/// wins), 0 (tie) — a win only if it survives BOTH orderings.
pub(crate) async fn judge_pair(
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
pub(crate) struct CommitteeTrial {
    pub(crate) draw: usize,
    pub(crate) task_id: String,
    pub(crate) category: String,
    /// +1 committee(B) beats alone(A), -1 A beats B, 0 tie.
    pub(crate) b_vs_a: i8,
    /// +1 layered(C) beats alone(A).
    pub(crate) c_vs_a: i8,
    /// +1 layered(C) beats committee(B).
    pub(crate) c_vs_b: i8,
    pub(crate) len_a: usize,
    pub(crate) len_b: usize,
    pub(crate) len_c: usize,
}

/// One committee trial: build all three arms for a prompt, then judge them.
/// Returns `None` when an arm produced no usable text (counted as skipped
/// rather than scored, so an empty output never masquerades as a preference).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_committee_trial(
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
pub(crate) fn small_mesh_pool() -> Vec<String> {
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
pub(crate) fn e2e_config(pool: &[String], api_key: &str) -> GatewayConfig {
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
