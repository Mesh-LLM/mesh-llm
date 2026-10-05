use super::*;

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

pub(crate) fn matched_finalizer() -> String {
    std::env::var("MOA_MATCHED_FINALIZER").unwrap_or_else(|_| "qwen/qwen3-14b".to_string())
}

/// Different-family peers, similar-ish strength to the finalizer. Strength is
/// approximate (no calibration set yet) — configurable so a matched trio can
/// be pinned once one is defined.
pub(crate) fn matched_diverse_peers() -> Vec<String> {
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
pub(crate) async fn structured_proposal(
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
pub(crate) fn pack_finalizer(task: &AblationTask, candidates: &[(String, String)]) -> Vec<Value> {
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

pub(crate) async fn finalizer_outcome(
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
pub(crate) struct MatchedTrial {
    pub(crate) draw: usize,
    pub(crate) task_id: String,
    pub(crate) category: String,
    pub(crate) arm: &'static str,
    pub(crate) outcome: &'static str,
    /// Did any candidate proposal in this arm score the task on its own?
    pub(crate) oracle: bool,
    pub(crate) n_candidates: usize,
}
