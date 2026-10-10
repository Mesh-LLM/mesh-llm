use super::responses::{first_choice, message};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) const TOOL_NAME: &str = "lookup_fixture_fact";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum FixtureKey {
    Codeword,
    Checksum,
}

impl FixtureKey {
    pub fn name(self) -> &'static str {
        match self {
            Self::Codeword => "codeword",
            Self::Checksum => "checksum",
        }
    }

    pub fn fact(self) -> &'static str {
        match self {
            Self::Codeword => "signal-7429",
            Self::Checksum => "FS-319-DELTA",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    key: FixtureKey,
}

#[derive(Debug)]
pub(super) struct Call {
    pub id: String,
    pub key: FixtureKey,
}

pub(super) fn extract(response: &Value) -> Result<Call, String> {
    if first_choice(response)?["finish_reason"] != "tool_calls" {
        return Err("tool-call turn finish_reason was not tool_calls".into());
    }
    let calls = message(response)?
        .get("tool_calls")
        .and_then(Value::as_array)
        .ok_or("response did not contain tool_calls")?;
    if calls.len() != 1 {
        return Err("tool-call fixture requires exactly one call".into());
    }
    let raw = &calls[0];
    let id = raw
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or("tool call is missing an id")?;
    let function = raw
        .get("function")
        .filter(|value| value.is_object())
        .ok_or("tool call is missing a function object")?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or("tool name missing")?;
    let arguments = function.get("arguments").ok_or("tool arguments missing")?;
    let decoded: Arguments = match arguments {
        Value::String(text) => serde_json::from_str(text),
        Value::Object(_) => serde_json::from_value(arguments.clone()),
        _ => return Err("tool arguments were neither JSON string nor object".into()),
    }
    .map_err(|_| "tool arguments must contain one supported fixture key")?;
    validate(id, name, decoded.key)
}

fn validate(id: &str, name: &str, key: FixtureKey) -> Result<Call, String> {
    if id.is_empty() || id.len() > 65536 {
        return Err("tool call has an invalid id".into());
    }
    if name != TOOL_NAME {
        return Err("unexpected tool name".into());
    }
    Ok(Call { id: id.into(), key })
}

#[derive(Default)]
struct Parts {
    id: String,
    name: String,
    arguments: String,
}

pub(super) fn extract_stream(events: &[Value]) -> Result<Call, String> {
    let mut parts = BTreeMap::<u64, Parts>::new();
    let mut finished = false;
    for event in events {
        let Some(choices) = event.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for choice in choices {
            finished |= choice
                .get("finish_reason")
                .is_some_and(|value| value == "tool_calls");
            collect_choice(&mut parts, choice)?;
        }
    }
    if !finished {
        return Err("stream did not finish with tool_calls".into());
    }
    let first = parts
        .into_values()
        .next()
        .ok_or("stream did not contain tool_call deltas")?;
    let arguments: Arguments = serde_json::from_str(&first.arguments)
        .map_err(|_| "stream tool arguments must contain one supported fixture key")?;
    validate(&first.id, &first.name, arguments.key)
}

fn collect_choice(parts: &mut BTreeMap<u64, Parts>, choice: &Value) -> Result<(), String> {
    let Some(calls) = choice
        .get("delta")
        .and_then(|delta| delta.get("tool_calls"))
        .and_then(Value::as_array)
    else {
        return Ok(());
    };
    for raw in calls {
        let index = match raw.get("index") {
            None => 0,
            Some(value) => value
                .as_u64()
                .ok_or("tool-call delta index is not a nonnegative integer")?,
        };
        if !parts.contains_key(&index) && !parts.is_empty() {
            return Err("tool-call fixture received multiple streamed calls".into());
        }
        merge(parts.entry(index).or_default(), raw)?;
    }
    Ok(())
}

fn merge(parts: &mut Parts, raw: &Value) -> Result<(), String> {
    if let Some(id) = raw
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        if !parts.id.is_empty() && parts.id != id {
            return Err("streamed tool call changed its identity".into());
        }
        parts.id = id.into();
    }
    if let Some(function) = raw.get("function") {
        for (key, target) in [
            ("name", &mut parts.name),
            ("arguments", &mut parts.arguments),
        ] {
            if let Some(fragment) = function.get(key).and_then(Value::as_str) {
                if fragment.len() > 65536usize.saturating_sub(target.len()) {
                    return Err("streamed tool field exceeds 64 KiB".into());
                }
                target.push_str(fragment);
            }
        }
    }
    Ok(())
}

pub(super) fn assistant(call: &Call, original: Option<&Value>) -> Value {
    match original {
        Some(message) => {
            let mut result = json!({"role":"assistant","tool_calls":message["tool_calls"]});
            if !message["content"].is_null() {
                result["content"] = message["content"].clone();
            }
            result
        }
        None => json!({"role":"assistant","content":null,"tool_calls":[{
            "id":call.id,"type":"function","function":{
                "name":TOOL_NAME,"arguments":json!({"key":call.key}).to_string()
            }
        }]}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(arguments: Value) -> Value {
        json!({"choices":[{"finish_reason":"tool_calls","message":{
            "role":"assistant","reasoning_content":"private scratch",
            "tool_calls":[{"id":"fixture-call","type":"function","function":{
                "name":TOOL_NAME,"arguments":arguments
            }}]
        }}]})
    }

    fn fragments() -> Vec<Value> {
        vec![
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"fixture-call",
                "function":{"name":"lookup_fixture_","arguments":"{\"key\":\"check"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"",
                "function":{"name":"fact","arguments":"sum\"}"}}]}}]}),
            json!({"choices":[{"finish_reason":"tool_calls","delta":{}}]}),
            json!({"choices":[],"usage":{"completion_tokens":3}}),
        ]
    }

    #[test]
    fn arguments_are_typed_and_private_assistant_fields_are_not_forwarded() {
        for arguments in [json!({"key":"codeword"}), json!("{\"key\":\"codeword\"}")] {
            let raw = response(arguments);
            let call = extract(&raw).unwrap();
            assert_eq!(call.key, FixtureKey::Codeword);
            let forwarded = assistant(&call, Some(message(&raw).unwrap()));
            assert!(forwarded.get("reasoning_content").is_none());
            assert_eq!(forwarded["tool_calls"][0]["id"], "fixture-call");
        }
        for arguments in [
            json!({"key":"unknown"}),
            json!({"key":"codeword","extra":true}),
            json!("not json"),
            json!(42),
        ] {
            assert!(extract(&response(arguments)).is_err());
        }
    }

    #[test]
    fn initial_tool_turn_requires_one_identified_call_and_its_finish_reason() {
        let original = response(json!({"key":"codeword"}));
        let mut wrong_finish = original.clone();
        wrong_finish["choices"][0]["finish_reason"] = json!("stop");
        assert!(extract(&wrong_finish).is_err());
        let mut missing_id = original.clone();
        missing_id["choices"][0]["message"]["tool_calls"][0]["id"] = json!("");
        assert!(extract(&missing_id).is_err());
        let mut wrong_name = original.clone();
        wrong_name["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = json!("other");
        assert!(extract(&wrong_name).is_err());
        let mut multiple = original;
        let calls = multiple["choices"][0]["message"]["tool_calls"]
            .as_array_mut()
            .unwrap();
        calls.push(calls[0].clone());
        assert!(extract(&multiple).is_err());
    }

    #[test]
    fn streamed_name_and_arguments_reassemble_but_conflicting_identity_is_rejected() {
        let events = fragments();
        let call = extract_stream(&events).unwrap();
        assert_eq!(call.id, "fixture-call");
        assert_eq!(call.key, FixtureKey::Checksum);
        assert_eq!(
            assistant(&call, None)["tool_calls"][0]["function"]["arguments"],
            "{\"key\":\"checksum\"}"
        );
        let mut conflicting = fragments();
        conflicting[1]["choices"][0]["delta"]["tool_calls"][0]["id"] = json!("another-call");
        assert!(extract_stream(&conflicting).is_err());
        let mut multiple = fragments();
        multiple[1]["choices"][0]["delta"]["tool_calls"][0]["index"] = json!(1);
        assert!(extract_stream(&multiple).is_err());
        assert!(extract_stream(&events[..2]).is_err());
        let mut negative = fragments();
        negative[0]["choices"][0]["delta"]["tool_calls"][0]["index"] = json!(-1);
        assert!(extract_stream(&negative).is_err());
    }
}
