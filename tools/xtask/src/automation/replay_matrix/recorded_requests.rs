use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::ops::Range;

#[derive(Serialize)]
pub(super) struct Trajectory {
    pub session_id: String,
    pub source_dataset: String,
    pub agent_framework: String,
    pub recorded_model: Option<String>,
    pub messages: Vec<Map<String, Value>>,
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_action_protocol: Option<super::text_action_protocol::Protocol>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_window: Option<super::replay_profile::Window>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_sampling: Option<super::replay_sampling::Sampling>,
    #[serde(skip)]
    pub original: Value,
}

impl<'de> Deserialize<'de> for Trajectory {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Recorded {
            session_id: String,
            source_dataset: String,
            agent_framework: String,
            recorded_model: Option<String>,
            messages: Vec<Map<String, Value>>,
            tools: Option<Vec<Value>>,
            text_action_protocol: Option<super::text_action_protocol::Protocol>,
            replay_window: Option<super::replay_profile::Window>,
            replay_sampling: Option<super::replay_sampling::Sampling>,
        }
        let original = Value::deserialize(deserializer)?;
        let recorded: Recorded =
            serde_json::from_value(original.clone()).map_err(serde::de::Error::custom)?;
        Ok(Self {
            session_id: recorded.session_id,
            source_dataset: recorded.source_dataset,
            agent_framework: recorded.agent_framework,
            recorded_model: recorded.recorded_model,
            messages: recorded.messages,
            tools: recorded.tools,
            text_action_protocol: recorded.text_action_protocol,
            replay_window: recorded.replay_window,
            replay_sampling: recorded.replay_sampling,
            original,
        })
    }
}

#[derive(Serialize)]
pub(super) struct Turn {
    pub request_id: String,
    pub session_id: String,
    pub source_dataset: String,
    pub agent_framework: String,
    pub recorded_model: Option<String>,
    pub assistant_turn: usize,
    pub recorded_message_index: usize,
    pub history_message_count: usize,
    pub requested_output_tokens: Option<u64>,
    pub recorded_output_characters: usize,
    pub available_tools: usize,
    pub qualification_probe: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_action_protocol: Option<super::text_action_protocol::Protocol>,
    pub body: Value,
}

pub(super) struct Selection<'a> {
    pub model: &'a str,
    pub maximum_output_tokens: Option<u64>,
    pub turn_limit: Option<usize>,
    pub qualification_probe: bool,
}

fn message(recorded: &Map<String, Value>) -> DynResult<Map<String, Value>> {
    let mut message: Map<_, _> = recorded
        .iter()
        .filter(|(key, value)| key.as_str() != "tool_calls_json" && !value.is_null())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if let Some(encoded) = recorded.get("tool_calls_json").and_then(Value::as_str)
        && !encoded.is_empty()
    {
        let calls: Vec<Value> = serde_json::from_str(encoded)?;
        message.insert("tool_calls".into(), Value::Array(calls));
    }
    Ok(message)
}

fn tools(trajectory: &Trajectory) -> DynResult<Vec<Value>> {
    if let Some(tools) = &trajectory.tools {
        return Ok(tools.clone());
    }
    let mut names = BTreeSet::new();
    for message in &trajectory.messages {
        if let Some(encoded) = message.get("tool_calls_json").and_then(Value::as_str)
            && !encoded.is_empty()
        {
            let calls: Vec<Value> = serde_json::from_str(encoded)?;
            for call in calls {
                if let Some(name) = call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                {
                    names.insert(name.to_owned());
                }
            }
        }
    }
    Ok(names
        .into_iter()
        .map(|name| {
            serde_json::json!({
                "type":"function", "function":{
                    "name":name, "description":"Tool available in the recorded agent trajectory.",
                    "parameters":{"type":"object","additionalProperties":true}
                }
            })
        })
        .collect())
}

/// Assistant turns a build can materialise for this selection, without building any body.
pub(super) fn assistant_turn_count(trajectory: &Trajectory, selection: &Selection<'_>) -> usize {
    let mut count = 0_usize;
    for recorded in &trajectory.messages {
        if recorded.get("role").and_then(Value::as_str) == Some("assistant") {
            if selection.turn_limit.is_some_and(|limit| count >= limit) {
                break;
            }
            count += 1;
        }
    }
    count
}

pub(super) fn build(trajectory: &Trajectory, selection: &Selection<'_>) -> DynResult<Vec<Turn>> {
    build_selected(trajectory, selection, None)
}

/// Builds only the assistant turns inside `selected` (all of them when `None`).
/// The recorded message walk is shared, so a selected turn still carries its complete
/// recorded prefix while unselected turns never materialise a request body.
pub(super) fn build_selected(
    trajectory: &Trajectory,
    selection: &Selection<'_>,
    selected: Option<Range<usize>>,
) -> DynResult<Vec<Turn>> {
    if trajectory.session_id.is_empty() || selection.maximum_output_tokens == Some(0) {
        return Err("session identity and output budget must be nonempty".into());
    }
    let tools = tools(trajectory)?;
    if trajectory.text_action_protocol.is_some() && !tools.is_empty() {
        return Err("text action protocol cannot be combined with API tools".into());
    }
    if let Some(sampling) = &trajectory.replay_sampling {
        sampling.validate()?;
    }
    let mut history = Vec::new();
    let mut turns = Vec::new();
    let mut assistant_turn = 0_usize;
    for (index, recorded) in trajectory.messages.iter().enumerate() {
        let role = recorded
            .get("role")
            .and_then(Value::as_str)
            .ok_or("missing message role")?;
        if !["system", "developer", "user", "assistant", "tool"].contains(&role) {
            return Err("unsupported recorded message role".into());
        }
        if role == "assistant" {
            if selection
                .turn_limit
                .is_some_and(|limit| assistant_turn >= limit)
            {
                break;
            }
            if selected
                .as_ref()
                .is_none_or(|range| range.contains(&assistant_turn))
            {
                // A recorded answer's length is not a safe bound for another model's
                // completion. Omit the limit unless the caller explicitly supplies it.
                let output = if selection.qualification_probe {
                    Some(1)
                } else {
                    selection.maximum_output_tokens
                };
                let mut body = serde_json::json!({
                    "prompt_cache_key":trajectory.session_id, "model":selection.model,
                    "messages":history, "temperature":0,
                    "seed":42, "stream":true, "stream_options":{"include_usage":true}
                });
                if let Some(output) = output {
                    body["max_tokens"] = output.into();
                }
                if let Some(sampling) = &trajectory.replay_sampling {
                    sampling.apply(&mut body)?;
                }
                if !tools.is_empty() {
                    body["tools"] = Value::Array(tools.clone());
                }
                if let Some(protocol) = trajectory.text_action_protocol {
                    protocol.apply(&mut body)?;
                }
                turns.push(Turn {
                    request_id: format!("{}:{}", trajectory.session_id, assistant_turn),
                    session_id: trajectory.session_id.clone(),
                    source_dataset: trajectory.source_dataset.clone(),
                    agent_framework: trajectory.agent_framework.clone(),
                    recorded_model: trajectory.recorded_model.clone(),
                    assistant_turn,
                    recorded_message_index: index,
                    history_message_count: history.len(),
                    requested_output_tokens: output,
                    recorded_output_characters: recorded
                        .get("content")
                        .and_then(Value::as_str)
                        .map_or(0, |content| content.chars().count()),
                    available_tools: tools.len(),
                    qualification_probe: selection.qualification_probe,
                    text_action_protocol: trajectory.text_action_protocol,
                    body,
                });
            }
            assistant_turn += 1;
        }
        history.push(Value::Object(message(recorded)?));
    }
    if turns.is_empty() {
        return Err("trajectory has no selected assistant turns".into());
    }
    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_limits_follow_the_caller_and_never_the_recorded_answer_length() {
        let trajectory: Trajectory = serde_json::from_value(serde_json::json!({
            "session_id":"s","source_dataset":"fixture","agent_framework":"fixture",
            "messages":[{"role":"user","content":"task"},
                {"role":"assistant","content":"ok","max_tokens":8},
                {"role":"user","content":"continue"},
                {"role":"assistant","content":"a much longer recorded answer"}]
        }))
        .unwrap();
        for limit in [None, Some(2048)] {
            let mut selection = Selection {
                model: "target",
                maximum_output_tokens: limit,
                turn_limit: None,
                qualification_probe: false,
            };
            for turn in build(&trajectory, &selection).unwrap() {
                assert_eq!(turn.body["temperature"], 0);
                assert_eq!(turn.requested_output_tokens, limit);
                assert_eq!(turn.body.get("max_tokens"), limit.map(Value::from).as_ref());
                assert!(turn.body.get("max_completion_tokens").is_none());
            }
            selection.qualification_probe = true;
            for turn in build(&trajectory, &selection).unwrap() {
                assert_eq!(turn.body["max_tokens"], 1);
                assert_eq!(turn.requested_output_tokens, Some(1));
            }
        }
        assert!(
            build(
                &trajectory,
                &Selection {
                    model: "target",
                    maximum_output_tokens: Some(0),
                    turn_limit: None,
                    qualification_probe: false,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn captured_assistant_openai_fields_and_direct_tool_calls_are_preserved() {
        let recorded = serde_json::json!({"role":"assistant","content":"","name":"agent",
            "reasoning_content":"private reasoning","provider_extension":{"cache_control":"ephemeral"},
            "tool_calls":[{"id":"call-1","type":"function"}]});
        let actual = message(recorded.as_object().unwrap()).unwrap();
        assert_eq!(serde_json::Value::Object(actual), recorded);
    }

    #[test]
    fn every_turn_uses_the_recorded_prefix_and_exact_captured_tools() {
        let trajectory: Trajectory = serde_json::from_value(serde_json::json!({
            "session_id":"s", "source_dataset":"capture", "agent_framework":"goose", "recorded_model":null,
            "tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object","required":["query"]}}}],
            "messages":[{"role":"user","content":"task","extension":{"private":false}},
                {"role":"assistant","content":"recorded","name":"recorded-agent","reasoning_content":"recorded private reasoning","provider_extension":{"opaque":"unchanged"},"tool_calls_json":"[{\"id\":\"call-1\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]"},
                {"role":"tool","tool_call_id":"call-1","content":"observation","optional":null},
                {"role":"assistant","content":"final"}]
        })).unwrap();
        let turns = build(
            &trajectory,
            &Selection {
                model: "target",
                maximum_output_tokens: Some(2048),
                turn_limit: None,
                qualification_probe: false,
            },
        )
        .unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(turns[1].body["messages"][1]["content"], "recorded");
        assert_eq!(
            turns[1].body["messages"][1],
            serde_json::json!({
                "role":"assistant", "content":"recorded", "name":"recorded-agent",
                "reasoning_content":"recorded private reasoning", "provider_extension":{"opaque":"unchanged"},
                "tool_calls":[{"id":"call-1","function":{"name":"search","arguments":"{}"}}]
            })
        );
        for message in turns[1].body["messages"].as_array().unwrap() {
            assert!(message.get("tool_calls_json").is_none());
            assert!(message.get("reasoning").is_none());
        }
        assert_eq!(
            turns[1].body["messages"][1]["tool_calls"][0]["id"],
            "call-1"
        );
        assert!(turns[1].body["messages"][2].get("optional").is_none());
        assert_eq!(turns[1].body["tools"], serde_json::json!(trajectory.tools));
        assert_eq!(
            turns[1].body["messages"][0]["extension"],
            serde_json::json!({"private":false})
        );
        assert_eq!(turns[1].request_id, "s:1");
    }

    #[test]
    fn warmup_limit_selects_only_ordered_initial_turns() {
        let trajectory: Trajectory = serde_json::from_value(serde_json::json!({
            "session_id":"s","source_dataset":"fixture","agent_framework":"fixture","recorded_model":null,
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"},{"role":"user","content":"next"},{"role":"assistant","content":"second"}]
        })).unwrap();
        let turns = build(
            &trajectory,
            &Selection {
                model: "target",
                maximum_output_tokens: Some(1),
                turn_limit: Some(1),
                qualification_probe: true,
            },
        )
        .unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].body["max_tokens"], 1);
        assert!(turns[0].qualification_probe);
    }

    fn counting_trajectory() -> Trajectory {
        serde_json::from_value(serde_json::json!({
            "session_id":"s","source_dataset":"fixture","agent_framework":"fixture","recorded_model":null,
            "messages":[{"role":"user","content":"task"},
                {"role":"assistant","content":"first"},
                {"role":"tool","tool_call_id":"call-0","content":"observation"},
                {"role":"assistant","content":"second"},
                {"role":"user","content":"more"},
                {"role":"assistant","content":"third"},
                {"role":"assistant","content":"fourth"}]
        }))
        .unwrap()
    }

    #[test]
    fn turn_count_without_bodies_matches_the_materialised_turns() {
        let trajectory = counting_trajectory();
        for turn_limit in [None, Some(1), Some(2), Some(4), Some(9)] {
            let selection = Selection {
                model: "target",
                maximum_output_tokens: Some(2048),
                turn_limit,
                qualification_probe: false,
            };
            assert_eq!(
                assistant_turn_count(&trajectory, &selection),
                build(&trajectory, &selection).unwrap().len()
            );
        }
        let exhausted = Selection {
            model: "target",
            maximum_output_tokens: Some(2048),
            turn_limit: Some(0),
            qualification_probe: false,
        };
        assert_eq!(assistant_turn_count(&trajectory, &exhausted), 0);
        assert!(build(&trajectory, &exhausted).is_err());
    }

    #[test]
    fn selecting_a_range_materialises_only_those_turns_with_intact_prefixes() {
        let trajectory = counting_trajectory();
        let selection = Selection {
            model: "target",
            maximum_output_tokens: Some(2048),
            turn_limit: None,
            qualification_probe: false,
        };
        let all = build(&trajectory, &selection).unwrap();
        assert_eq!(all.len(), 4);
        for (start, end) in [(0, 1), (1, 3), (3, 4)] {
            let selected = build_selected(&trajectory, &selection, Some(start..end)).unwrap();
            assert_eq!(selected.len(), end - start);
            for (actual, expected) in selected.iter().zip(&all[start..end]) {
                assert_eq!(actual.request_id, expected.request_id);
                assert_eq!(actual.assistant_turn, expected.assistant_turn);
                assert_eq!(actual.history_message_count, expected.history_message_count);
                assert_eq!(actual.body["messages"], expected.body["messages"]);
            }
        }
        assert!(build_selected(&trajectory, &selection, Some(4..5)).is_err());
    }
}
