//! Capability evidence from captured OpenAI requests, independent of a Python client.
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
struct Event {
    method: String,
    path: String,
    #[serde(default)]
    body: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
struct Chat {
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    tool_calls: Option<Vec<Value>>,
}

#[derive(Default)]
struct Capabilities {
    models: usize,
    chats: usize,
    streaming: bool,
    nonstreaming: bool,
    tools: bool,
    tool_choice: bool,
    parallel_tool_calls: bool,
    instructions: bool,
    user: bool,
    assistant_calls: bool,
    tool_result: bool,
    history: bool,
    long_request: bool,
}

impl Capabilities {
    fn observe(&mut self, event: Event, minimum_body_bytes: usize) -> DynResult<()> {
        if event.method == "GET" && event.path.trim_end_matches('/') == "/v1/models" {
            self.models += 1;
        }
        if event.method != "POST"
            || event.path.split('?').next().unwrap().trim_end_matches('/') != "/v1/chat/completions"
        {
            return Ok(());
        }
        self.chats += 1;
        let body = event.body.unwrap_or_default();
        // Count compact UTF-8 JSON bytes. This is evidence size, not wire size;
        // capture's parsed body does not preserve the original HTTP whitespace.
        self.long_request |= serde_json::to_vec(&body)?.len() >= minimum_body_bytes;
        self.tool_choice |= body.contains_key("tool_choice");
        self.parallel_tool_calls |= body.contains_key("parallel_tool_calls");
        let chat: Chat = serde_json::from_value(Value::Object(body))?;
        self.streaming |= chat.stream == Some(true);
        self.nonstreaming |= chat.stream == Some(false);
        self.tools |= chat.tools.is_some_and(|tools| !tools.is_empty());
        self.history |= chat.messages.len() >= 4;
        for message in chat.messages {
            self.instructions |= matches!(message.role.as_str(), "system" | "developer");
            self.user |= message.role == "user";
            self.tool_result |= message.role == "tool";
            self.assistant_calls |= message.role == "assistant"
                && message.tool_calls.is_some_and(|calls| !calls.is_empty());
        }
        Ok(())
    }

    fn report(&self, minimum_body_bytes: usize) -> DynResult<String> {
        let mut required = vec![
            ("GET /v1/models", self.models > 0),
            ("POST /v1/chat/completions", self.chats > 0),
            ("streaming chat request", self.streaming),
            ("non-stream chat request", self.nonstreaming),
            ("tools schema", self.tools),
            ("tool_choice field", self.tool_choice),
            ("parallel_tool_calls field", self.parallel_tool_calls),
            ("system/developer instructions", self.instructions),
            ("user message", self.user),
            ("assistant tool-call history", self.assistant_calls),
            ("tool-result message", self.tool_result),
            ("multi-message history", self.history),
        ];
        if minimum_body_bytes > 0 {
            required.push(("long prompt request", self.long_request));
        }
        let missing = required
            .iter()
            .filter_map(|(name, ok)| (!ok).then_some(*name))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!("OpenAI agent surface validation failed: missing {}; captured requests: models={} chat={}", missing.join(", "), self.models, self.chats).into());
        }
        Ok(format!(
            "OpenAI agent surface validation passed\n  captured requests: models={} chat={}\n",
            self.models, self.chats
        ))
    }
}

pub(super) fn validate(bytes: &[u8], minimum_body_bytes: usize) -> DynResult<String> {
    let source = std::str::from_utf8(bytes)?;
    let mut capabilities = Capabilities::default();
    for (index, line) in source.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str(line)
            .map_err(|error| format!("invalid captured request on line {}: {error}", index + 1))?;
        capabilities.observe(event, minimum_body_bytes)?;
    }
    capabilities.report(minimum_body_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn events() -> Vec<Value> {
        let body = json!({"model":"fixture","stream":true,"tools":[{"type":"function"}],"tool_choice":"auto","parallel_tool_calls":true,"messages":[
            {"role":"system","content":"instructions"},
            {"role":"user","content":"request"},
            {"role":"assistant","tool_calls":[{"id":"fixture"}]},
            {"role":"tool","content":"result"}]});
        let mut second = body.clone();
        second["stream"] = json!(false);
        vec![
            json!({"method":"GET","path":"/v1/models/","body":null}),
            json!({"method":"POST","path":"/v1/chat/completions/?trace=1","body":body}),
            json!({"method":"POST","path":"/v1/chat/completions","body":second}),
        ]
    }

    fn check(events: &[Value], minimum: usize) -> DynResult<String> {
        let text = events
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        validate(text.as_bytes(), minimum)
    }

    #[test]
    fn complete_surface_accepts_queries_trailing_slashes_and_developer_instructions() {
        let mut rows = events();
        for row in rows.iter_mut().skip(1) {
            row["body"]["messages"][0]["role"] = json!("developer");
        }
        assert_eq!(
            check(&rows, 0).unwrap(),
            "OpenAI agent surface validation passed\n  captured requests: models=1 chat=2\n"
        );
    }

    #[test]
    fn every_required_surface_capability_has_independent_rejection_evidence() {
        for (field, expected) in [
            ("tools", "tools schema"),
            ("tool_choice", "tool_choice field"),
            ("parallel_tool_calls", "parallel_tool_calls field"),
        ] {
            let mut rows = events();
            for row in rows.iter_mut().skip(1) {
                row["body"].as_object_mut().unwrap().remove(field);
            }
            assert!(check(&rows, 0).unwrap_err().to_string().contains(expected));
        }
        for (stream, expected) in [
            (true, "non-stream chat request"),
            (false, "streaming chat request"),
        ] {
            let mut rows = events();
            for row in rows.iter_mut().skip(1) {
                row["body"]["stream"] = json!(stream);
            }
            assert!(check(&rows, 0).unwrap_err().to_string().contains(expected));
        }
        for (index, expected) in [
            (0, "system/developer instructions"),
            (1, "user message"),
            (2, "assistant tool-call history"),
            (3, "tool-result message"),
        ] {
            let mut rows = events();
            for row in rows.iter_mut().skip(1) {
                row["body"]["messages"][index]["role"] = json!("unknown");
            }
            assert!(check(&rows, 0).unwrap_err().to_string().contains(expected));
        }
        let mut rows = events();
        rows.remove(0);
        assert!(
            check(&rows, 0)
                .unwrap_err()
                .to_string()
                .contains("GET /v1/models")
        );
        assert!(
            check(&events()[..1], 0)
                .unwrap_err()
                .to_string()
                .contains("POST /v1/chat/completions")
        );
        let mut rows = events();
        for row in rows.iter_mut().skip(1) {
            row["body"]["messages"].as_array_mut().unwrap().truncate(3);
        }
        assert!(
            check(&rows, 0)
                .unwrap_err()
                .to_string()
                .contains("multi-message history")
        );
    }

    #[test]
    fn tool_calls_and_tool_schema_require_nonempty_arrays() {
        for target in ["tools", "tool_calls"] {
            let mut rows = events();
            for row in rows.iter_mut().skip(1) {
                if target == "tools" {
                    row["body"]["tools"] = json!([]);
                } else {
                    row["body"]["messages"][2]["tool_calls"] = json!([]);
                }
            }
            assert!(check(&rows, 0).is_err());
        }
    }

    #[test]
    fn long_request_threshold_can_be_disabled_but_cannot_be_satisfied_by_short_requests() {
        let mut rows = events();
        assert!(check(&rows, 0).is_ok());
        assert!(check(&rows, 9000).is_err());
        rows[1]["body"]["messages"][1]["content"] = json!("a".repeat(10000));
        assert!(check(&rows, 9000).is_ok());
        assert!(check(&rows, 11000).is_err());
    }

    #[test]
    fn malformed_capture_and_wrong_request_field_types_are_rejected() {
        for bytes in [
            b"not JSON".as_slice(),
            b"{}",
            b"{\"method\":false,\"path\":\"/v1/models\"}",
            b"\xff",
        ] {
            assert!(validate(bytes, 0).is_err());
        }
        for bad in [json!(true), json!("true")] {
            let mut rows = events();
            rows[1]["body"]["stream"] = bad;
            if rows[1]["body"]["stream"].is_boolean() {
                rows[1]["body"]["messages"] = json!(false);
            }
            assert!(check(&rows, 0).is_err());
        }
    }
}
