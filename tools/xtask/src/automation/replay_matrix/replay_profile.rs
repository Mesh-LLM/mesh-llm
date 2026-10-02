//! Manual measured-request selection. Prefix construction remains recorded_requests-owned.
use super::recorded_requests::{Selection, Trajectory, Turn};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Mode {
    Checkpoint,
    Final,
    #[default]
    All,
}
impl Mode {
    pub(super) fn is_all(&self) -> bool {
        *self == Self::All
    }

    pub(super) fn parse(value: &str) -> DynResult<Self> {
        match value {
            "checkpoint" => Ok(Self::Checkpoint),
            "final" => Ok(Self::Final),
            "all" => Ok(Self::All),
            _ => Err("replay mode must be checkpoint, final or all".into()),
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Stage {
    Early,
    Middle,
    Late,
    Final,
}
impl Stage {
    fn index(self, count: usize) -> usize {
        let last = count.saturating_sub(1);
        match self {
            Self::Early => 0,
            Self::Middle => last / 3,
            Self::Late => last.saturating_mul(2) / 3,
            Self::Final => last,
        }
    }
}
fn rank(session: &str) -> ([u8; 32], &str) {
    (Sha256::digest(session.as_bytes()).into(), session)
}
fn stages(trajectories: &[Trajectory]) -> DynResult<BTreeMap<&str, Stage>> {
    let mut groups = BTreeMap::<&str, Vec<&str>>::new();
    let mut seen = BTreeSet::new();
    for trajectory in trajectories {
        if !seen.insert(trajectory.session_id.as_str()) {
            return Err("duplicate checkpoint session identity".into());
        }
        groups
            .entry(&trajectory.agent_framework)
            .or_default()
            .push(&trajectory.session_id);
    }
    let mut stages = BTreeMap::new();
    for sessions in groups.values_mut() {
        sessions.sort_by(|left, right| rank(left).cmp(&rank(right)));
        for (index, session) in sessions.iter().enumerate() {
            stages.insert(
                *session,
                [Stage::Early, Stage::Middle, Stage::Late, Stage::Final][index % 4],
            );
        }
    }
    Ok(stages)
}
pub(super) fn build(
    trajectories: &[Trajectory],
    selection: &Selection<'_>,
    mode: Mode,
) -> DynResult<Vec<Vec<Turn>>> {
    if trajectories.is_empty() {
        return Err("empty measured replay cohort".into());
    }
    if trajectories
        .iter()
        .map(|trajectory| trajectory.session_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        != trajectories.len()
    {
        return Err("duplicate checkpoint session identity".into());
    }
    let stages = if mode == Mode::Checkpoint {
        stages(trajectories)?
    } else {
        BTreeMap::new()
    };

    trajectories
        .iter()
        .map(|trajectory| {
            let turns = super::recorded_requests::build(trajectory, selection)?;
            if mode == Mode::All {
                return Ok(turns);
            }
            let stage = if mode == Mode::Final {
                Stage::Final
            } else {
                stages[trajectory.session_id.as_str()]
            };
            let index = stage.index(turns.len());
            Ok(turns
                .into_iter()
                .enumerate()
                .filter_map(|(current, turn)| (current == index).then_some(turn))
                .collect())
        })
        .collect()
}
pub(super) fn expected(trajectories: &[Trajectory], mode: Mode) -> DynResult<Vec<String>> {
    Ok(build(
        trajectories,
        &Selection {
            model: "identity",
            maximum_output_tokens: 1,
            turn_limit: None,
            qualification_probe: true,
        },
        mode,
    )?
    .into_iter()
    .flatten()
    .map(|turn| turn.request_id)
    .collect())
}
/// Completion checks use the selected recorded IDs, without renumbering checkpoints.
pub(super) fn completeness(
    trajectories: &[Trajectory],
    records: &[serde_json::Value],
    mode: Mode,
) -> DynResult<serde_json::Value> {
    let expected = expected(trajectories, mode)?;
    let mut wanted = BTreeMap::<&str, Vec<&str>>::new();
    for id in &expected {
        let (session, _) = id
            .rsplit_once(':')
            .ok_or("invalid selected request identity")?;
        wanted.entry(session).or_default().push(id);
    }
    let mut observed = BTreeMap::<&str, Vec<&str>>::new();
    let mut problems = Vec::new();
    for record in records {
        let session = record["session_id"]
            .as_str()
            .ok_or("missing measured session identity")?;
        let id = record["request_id"]
            .as_str()
            .ok_or("missing measured request identity")?;
        observed.entry(session).or_default().push(id);
        if let Some(error) = record.get("error") {
            problems.push(format!("{id}: {error}"));
        }
    }
    let complete = wanted
        .iter()
        .filter(|(session, ids)| {
            observed.get(*session) == Some(*ids)
                && records
                    .iter()
                    .filter(|record| record["session_id"] == **session)
                    .all(|record| record.get("error").is_none())
        })
        .count();
    for (session, ids) in &wanted {
        if observed.get(session) != Some(ids) {
            problems.push(format!(
                "{session}: missing, duplicate or out-of-order selected requests"
            ));
        }
    }
    if observed.keys().any(|session| !wanted.contains_key(session)) {
        problems.push("unexpected measured sessions".into());
    }
    Ok(
        serde_json::json!({"passed":problems.is_empty(),"problems":problems,"expected_request_ids":expected,"expected_turns":expected.len(),"complete_sessions":complete}),
    )
}
/// Numerical metrics are still cell_summary-owned; only measured-profile completeness changes.
pub(super) fn summarize(
    trajectories: &[Trajectory],
    records: &[serde_json::Value],
    concurrency: usize,
    mode: Mode,
) -> DynResult<serde_json::Value> {
    let mut summary = super::cell_summary::summarize(trajectories, records, concurrency)?;
    if mode == Mode::All {
        return Ok(summary);
    }
    let complete = completeness(trajectories, records, mode)?;
    summary["replay_mode"] = serde_json::to_value(mode)?;
    summary["successful_trajectories"] = complete["complete_sessions"].clone();
    summary["failed_trajectories"] = (trajectories.len()
        - usize::try_from(
            complete["complete_sessions"]
                .as_u64()
                .ok_or("missing complete session count")?,
        )?)
    .into();
    summary["measured_selected_requests"] = complete["expected_turns"].clone();
    summary["acceptance"] =
        serde_json::json!({"passed":complete["passed"],"problems":complete["problems"]});
    summary["completeness"] = complete;
    let selected_ids = summary["completeness"]["expected_request_ids"]
        .as_array()
        .ok_or("missing selected IDs")?
        .clone();
    if let Some(sessions) = summary["sessions"].as_array_mut() {
        for session in sessions {
            let id = session["session_id"]
                .as_str()
                .ok_or("missing session summary identity")?;
            let selected = records
                .iter()
                .filter(|record| record["session_id"] == id)
                .cloned()
                .collect::<Vec<_>>();

            // The cohort-level stage assignment must remain authoritative, rather than re-ranking one session.
            let expected = selected_ids
                .iter()
                .filter(|value| {
                    value
                        .as_str()
                        .and_then(|v| v.rsplit_once(':'))
                        .is_some_and(|(session, _)| session == id)
                })
                .cloned()
                .collect::<Vec<_>>();
            let observed = selected
                .iter()
                .map(|record| record["request_id"].clone())
                .collect::<Vec<_>>();
            session["complete"] = (observed == expected
                && selected.iter().all(|record| record.get("error").is_none()))
            .into();
            session["measured_selected_requests"] = expected.len().into();
            session["expected_request_ids"] = expected.into();
        }
    }
    Ok(summary)
}
