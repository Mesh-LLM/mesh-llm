use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeSet;

#[derive(Serialize)]
pub(super) struct Trajectory {
    pub session_id: String,
    pub source_dataset: String,
    pub agent_framework: String,
    pub recorded_model: Option<String>,
    pub messages: Vec<Map<String, Value>>,
    pub tools: Option<Vec<Value>>,
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
    pub requested_output_tokens: u64,
    pub recorded_output_characters: usize,
    pub available_tools: usize,
    pub qualification_probe: bool,
    pub body: Value,
}

pub(super) struct Selection<'a> {
    pub model: &'a str,
    pub maximum_output_tokens: u64,
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

pub(super) fn build(trajectory: &Trajectory, selection: &Selection<'_>) -> DynResult<Vec<Turn>> {
    if trajectory.session_id.is_empty() || selection.maximum_output_tokens == 0 {
        return Err("session identity and output budget must be nonempty".into());
    }
    let tools = tools(trajectory)?;
    let mut history = Vec::new();
    let mut turns = Vec::new();
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
                .is_some_and(|limit| turns.len() >= limit)
            {
                break;
            }
            let budget_message: super::session_evidence::Message =
                serde_json::from_value(Value::Object(recorded.clone()))?;
            let output = budget_message.output_budget(selection.maximum_output_tokens);
            let mut body = serde_json::json!({
                "prompt_cache_key":trajectory.session_id, "model":selection.model,
                "messages":history, "max_tokens":output, "temperature":0,
                "seed":42, "stream":true, "stream_options":{"include_usage":true}
            });
            if !tools.is_empty() {
                body["tools"] = Value::Array(tools.clone());
            }
            turns.push(Turn {
                request_id: format!("{}:{}", trajectory.session_id, turns.len()),
                session_id: trajectory.session_id.clone(),
                source_dataset: trajectory.source_dataset.clone(),
                agent_framework: trajectory.agent_framework.clone(),
                recorded_model: trajectory.recorded_model.clone(),
                assistant_turn: turns.len(),
                recorded_message_index: index,
                history_message_count: history.len(),
                requested_output_tokens: output,
                recorded_output_characters: recorded
                    .get("content")
                    .and_then(Value::as_str)
                    .map_or(0, |content| content.chars().count()),
                available_tools: tools.len(),
                qualification_probe: selection.qualification_probe,
                body,
            });
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
                maximum_output_tokens: 2048,
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
                maximum_output_tokens: 1,
                turn_limit: Some(1),
                qualification_probe: true,
            },
        )
        .unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].body["max_tokens"], 1);
        assert!(turns[0].qualification_probe);
    }
}
