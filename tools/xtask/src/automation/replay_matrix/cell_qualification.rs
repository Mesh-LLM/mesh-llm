use super::{context_eligibility::Budget, recorded_requests::Trajectory, session_evidence};
use crate::command::DynResult;

pub(super) fn apply(
    evidence: &mut serde_json::Value,
    input: (&[Trajectory], &[serde_json::Value]),
    budget: &Budget,
) -> DynResult<()> {
    let trajectories = input
        .0
        .iter()
        .map(|trajectory| {
            serde_json::from_value::<session_evidence::Trajectory>(serde_json::json!({
                "session_id": trajectory.session_id, "messages": trajectory.messages
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let requests = input
        .1
        .iter()
        .map(|record| serde_json::from_value::<session_evidence::Request>(record.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let eligibility = super::context_eligibility::evaluate(&trajectories, &requests, budget);
    let passed = evidence["acceptance"]["passed"] == true && eligibility.passed;
    let problems = evidence["acceptance"]["problems"]
        .as_array_mut()
        .ok_or("missing cell acceptance problems")?;
    problems.extend(
        eligibility
            .problems
            .iter()
            .cloned()
            .map(serde_json::Value::String),
    );
    evidence["acceptance"]["passed"] = passed.into();
    evidence["eligibility"] = serde_json::to_value(eligibility)?;
    Ok(())
}
