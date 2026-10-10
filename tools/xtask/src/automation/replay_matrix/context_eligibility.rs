use super::session_evidence::{Request, Role, Trajectory, complete};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub(super) struct Budget {
    pub context_tokens: u64,
    pub maximum_output_tokens: u64,
    pub minimum_session_prompt_tokens: u64,
}

#[derive(Serialize)]
pub(super) struct Turn {
    request_id: String,
    prompt_tokens: Option<u64>,
    output_budget: u64,
    fits: bool,
}

#[derive(Serialize)]
pub(super) struct Eligibility {
    pub passed: bool,
    pub problems: Vec<String>,
    context_tokens: u64,
    minimum_session_prompt_tokens: u64,
    turns: Vec<Turn>,
}

pub(super) fn evaluate(
    trajectories: &[Trajectory],
    probes: &[Request],
    budget: &Budget,
) -> Eligibility {
    let completeness = complete(trajectories, probes);
    let by_id: BTreeMap<_, _> = probes
        .iter()
        .map(|probe| (probe.request_id.as_str(), probe))
        .collect();
    let mut evidence = Eligibility {
        passed: completeness.passed,
        problems: completeness.problems,
        context_tokens: budget.context_tokens,
        minimum_session_prompt_tokens: budget.minimum_session_prompt_tokens,
        turns: Vec::new(),
    };
    for trajectory in trajectories {
        let mut longest = None::<u64>;
        for (turn, message) in trajectory
            .messages
            .iter()
            .filter(|message| matches!(message.role, Role::Assistant))
            .enumerate()
        {
            let request_id = format!("{}:{turn}", trajectory.session_id);
            let tokens = by_id
                .get(request_id.as_str())
                .and_then(|probe| probe.prompt_tokens);
            let output = message.output_budget(budget.maximum_output_tokens);
            let fits = tokens.is_some_and(|tokens| {
                tokens > 0
                    && tokens
                        .checked_add(output)
                        .is_some_and(|total| total <= budget.context_tokens)
            });
            if !fits {
                evidence.problems.push(format!(
                    "{request_id}: formatted prompt + output does not fit {}",
                    budget.context_tokens
                ));
            }
            if let Some(tokens) = tokens {
                longest = Some(longest.unwrap_or(0).max(tokens));
            }
            evidence.turns.push(Turn {
                request_id,
                prompt_tokens: tokens,
                output_budget: output,
                fits,
            });
        }
        if longest.is_none_or(|tokens| tokens < budget.minimum_session_prompt_tokens) {
            evidence.problems.push(format!(
                "{}: no prompt reaches {} tokens",
                trajectory.session_id, budget.minimum_session_prompt_tokens
            ));
        }
    }
    evidence.passed &= evidence.problems.is_empty();
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_and_recorded_output_must_fit_without_short_session_substitution() {
        let trajectories: Vec<Trajectory> = serde_json::from_value(serde_json::json!([
            {"session_id":"s","messages":[{"role":"user"},{"role":"assistant","content":"recorded answer"}]}
        ])).unwrap();
        let mut probes: Vec<Request> = serde_json::from_value(serde_json::json!([
            {"session_id":"s","request_id":"s:0","prompt_tokens":40000}
        ]))
        .unwrap();
        let budget = Budget {
            context_tokens: 131072,
            maximum_output_tokens: 2048,
            minimum_session_prompt_tokens: 32768,
        };
        assert!(evaluate(&trajectories, &probes, &budget).passed);
        probes[0].prompt_tokens = Some(131070);
        assert!(!evaluate(&trajectories, &probes, &budget).passed);
        probes[0].prompt_tokens = Some(8192);
        assert!(!evaluate(&trajectories, &probes, &budget).passed);
        probes[0].prompt_tokens = None;
        assert!(!evaluate(&trajectories, &probes, &budget).passed);
    }
}
