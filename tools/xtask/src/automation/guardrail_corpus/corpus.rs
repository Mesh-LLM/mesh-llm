//! Fixed guardrail requests, separate from endpoint execution.
use serde_json::{Value, json};

pub(super) struct Case {
    pub id: &'static str,
    pub category: &'static str,
    pub prompt: &'static str,
    pub expected: &'static str,
    pub declared_retries: u64,
    pub overrides: Value,
}
pub(super) fn cases() -> Vec<Case> {
    let tool = json!({"type":"function","function":{"name":"calculator","description":"Add two integers and return the sum.","parameters":{"type":"object","properties":{"left":{"type":"integer"},"right":{"type":"integer"}},"required":["left","right"],"additionalProperties":false}}});
    let schema = json!({"type":"json_schema","json_schema":{"name":"guardrail_status","strict":true,"schema":{"type":"object","properties":{"status":{"type":"string"},"count":{"type":"integer"},"note":{"type":"string"}},"required":["status","count","note"],"additionalProperties":false}}});
    vec![
        Case {
            id: "streaming-pass-through",
            category: "streaming",
            prompt: "Reply with the word pass-through and nothing else.",
            expected: "pass_through",
            declared_retries: 0,
            overrides: json!({"stream":true,"max_tokens":16}),
        },
        Case {
            id: "tool-call-reliability",
            category: "tools",
            prompt: "Use the calculator tool to add 17 and 25, then stop.",
            expected: "tool_call",
            declared_retries: 0,
            overrides: json!({"tools":[tool.clone()],"tool_choice":"auto","max_tokens":64}),
        },
        Case {
            id: "structured-object",
            category: "structured",
            prompt: "Return a JSON object with status, count, and note.",
            expected: "structured_object",
            declared_retries: 0,
            overrides: json!({"response_format":{"type":"json_object"},"max_tokens":64}),
        },
        Case {
            id: "strict-structured-schema",
            category: "structured",
            prompt: "Return a JSON object that matches the supplied schema exactly.",
            expected: "strict_structured",
            declared_retries: 1,
            overrides: json!({"response_format":schema.clone(),"max_tokens":64}),
        },
        Case {
            id: "unsupported-tools-plus-structured",
            category: "unsupported",
            prompt: "Try to use the calculator tool and satisfy the strict schema together.",
            expected: "unsupported_real_tools_plus_strict_structured",
            declared_retries: 0,
            overrides: json!({"tools":[tool],"tool_choice":"auto","response_format":schema,"max_tokens":64}),
        },
    ]
}
impl Case {
    pub(super) fn request(&self, model: &str, mode: &str) -> Value {
        let mut value = json!({"model":model,"messages":[{"role":"user","content":self.prompt}],"temperature":0,"mesh_guardrails":mode!="off"});
        value
            .as_object_mut()
            .unwrap()
            .extend(self.overrides.as_object().unwrap().clone());
        value
    }
    pub(super) fn supported(&self) -> bool {
        self.category != "unsupported"
    }
}
