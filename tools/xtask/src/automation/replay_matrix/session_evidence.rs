use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(in crate::automation) use super::session_evidence_command::run;

#[derive(Deserialize)]
pub(super) struct Runtime {
    pub models: Vec<Model>,
}

#[derive(Deserialize)]
pub(super) struct Model {
    pub context_length: u64,
}

impl Runtime {
    pub(super) fn context(&self, required: u64) -> Result<u64, &'static str> {
        match self.models.as_slice() {
            [model] if required > 0 && model.context_length >= required => Ok(model.context_length),
            [_] => Err("effective runtime context is below required context"),
            _ => Err("context preflight requires one identified local runtime model"),
        }
    }
}

#[derive(Deserialize)]
pub(super) struct Trajectory {
    pub session_id: String,
    pub messages: Vec<Message>,
}

#[derive(Deserialize)]
pub(super) struct Message {
    pub role: Role,
    pub content: Option<String>,
    pub tool_calls_json: Option<String>,
    pub tool_calls: Option<serde_json::Value>,
}

impl Message {
    pub(super) fn output_budget(&self, maximum: u64) -> u64 {
        let content = self
            .content
            .as_deref()
            .map_or(0, |text| text.chars().count());
        let encoded = self
            .tool_calls_json
            .as_deref()
            .map_or(0, |text| text.chars().count());
        let tools = self
            .tool_calls
            .as_ref()
            .map_or(0, |value| value.to_string().chars().count());
        u64::try_from(
            content
                .saturating_add(encoded)
                .saturating_add(tools)
                .div_ceil(4),
        )
        .unwrap_or(u64::MAX)
        .max(8)
        .min(maximum)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Deserialize)]
pub(super) struct Request {
    pub session_id: String,
    pub request_id: String,
    pub error: Option<String>,
    #[serde(default)]
    pub assistant_turn: u64,
    pub prompt_tokens: Option<u64>,
}

#[derive(Serialize)]
pub(super) struct Completeness {
    pub passed: bool,
    pub problems: Vec<String>,
    pub expected_request_ids: Vec<String>,
    pub expected_turns: usize,
}

pub(super) fn complete(trajectories: &[Trajectory], requests: &[Request]) -> Completeness {
    let mut expected = BTreeMap::new();
    let mut session_order = Vec::new();
    let mut problems = Vec::new();
    if trajectories.is_empty() {
        problems.push("empty trajectory workload".into());
    }
    for trajectory in trajectories {
        let count = trajectory
            .messages
            .iter()
            .filter(|message| matches!(message.role, Role::Assistant))
            .count();
        let ids = (0..count)
            .map(|turn| format!("{}:{turn}", trajectory.session_id))
            .collect::<Vec<_>>();
        if ids.is_empty() {
            problems.push(format!("{}: no assistant turns", trajectory.session_id));
        }
        if expected
            .insert(trajectory.session_id.as_str(), ids)
            .is_some()
        {
            problems.push("duplicate session IDs".into());
        } else {
            session_order.push(trajectory.session_id.as_str());
        }
    }
    let mut observed = BTreeMap::<&str, Vec<&str>>::new();
    for request in requests {
        observed
            .entry(&request.session_id)
            .or_default()
            .push(&request.request_id);
        if let Some(error) = &request.error {
            problems.push(format!("{}: {error}", request.request_id));
        }
    }
    for (session, ids) in &expected {
        if observed.get(session).map(|values| values.as_slice())
            != Some(
                ids.iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        {
            problems.push(format!(
                "{session}: missing, duplicate, or out-of-order turns"
            ));
        }
    }
    if observed
        .keys()
        .any(|session| !expected.contains_key(session))
    {
        problems.push("unexpected sessions".into());
    }
    let expected_request_ids = session_order
        .into_iter()
        .filter_map(|session| expected.remove(session))
        .flatten()
        .collect::<Vec<_>>();
    Completeness {
        passed: problems.is_empty(),
        problems,
        expected_turns: expected_request_ids.len(),
        expected_request_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_requests_keep_manifest_session_order() {
        let trajectories: Vec<Trajectory> = serde_json::from_value(serde_json::json!([
            {"session_id":"z", "messages":[{"role":"assistant"}]},
            {"session_id":"a", "messages":[{"role":"assistant"}]}
        ]))
        .unwrap();
        let requests: Vec<Request> = serde_json::from_value(serde_json::json!([
            {"session_id":"z","request_id":"z:0"},
            {"session_id":"a","request_id":"a:0"}
        ]))
        .unwrap();
        let report = complete(&trajectories, &requests);
        assert_eq!(report.expected_request_ids, ["z:0", "a:0"]);
    }

    #[test]
    fn context_uses_one_model_window_not_stage_capacity() {
        let runtime: Runtime = serde_json::from_value(
            serde_json::json!({"models":[{"context_length":131072}], "stages":[{"ctx_size":1024}]}),
        )
        .unwrap();
        assert_eq!(runtime.context(131072), Ok(131072));
        assert!(runtime.context(131073).is_err());
        let empty = Runtime { models: Vec::new() };
        assert!(empty.context(131072).is_err());
    }

    #[test]
    fn complete_sessions_require_order_unique_turns_and_success() {
        let trajectories: Vec<Trajectory> = serde_json::from_value(serde_json::json!([{"session_id":"s", "messages":[{"role":"user"},{"role":"assistant"},{"role":"tool"},{"role":"assistant"}]}])).unwrap();
        let mut requests: Vec<Request> = serde_json::from_value(serde_json::json!([{"session_id":"s","request_id":"s:0"},{"session_id":"s","request_id":"s:1"}])).unwrap();
        assert!(complete(&trajectories, &requests).passed);
        requests.push(
            serde_json::from_value(serde_json::json!({"session_id":"s","request_id":"s:0"}))
                .unwrap(),
        );
        assert!(!complete(&trajectories, &requests).passed);
        requests.pop();
        requests.reverse();
        assert!(!complete(&trajectories, &requests).passed);
        requests.reverse();
        requests[1].error = Some("HTTP 500".into());
        assert!(!complete(&trajectories, &requests).passed);
        assert!(!complete(&trajectories, &requests[..1]).passed);
    }
}
