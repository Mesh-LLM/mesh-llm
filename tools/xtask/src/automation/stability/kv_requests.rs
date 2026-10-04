use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub(super) const TOOL: &str = "lookup_probe_fact";
pub(super) const PIN: &str = "KV-PIN-8842";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Key {
    Primary,
    Secondary,
}

impl Key {
    pub fn name(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Secondary => "secondary",
        }
    }
    pub fn fact(self) -> &'static str {
        match self {
            Self::Primary => "KV-STABILITY-PRIMARY-7429",
            Self::Secondary => "KV-STABILITY-SECONDARY-319",
        }
    }
}

pub(super) fn prefix() -> String {
    let mut lines = vec![
        "You are a deterministic KV/cache stability certification endpoint.".into(),
        format!("Pinned recall token: {PIN}."),
        "When a tool is requested, call exactly the requested tool and never invent facts.".into(),
    ];
    for index in 0..192 {
        lines.push(format!("stable-prefix-block-{index:03}: preserve tool schema, conversation state, and cached prompt geometry."));
    }
    lines.join("\n")
}

pub(super) fn initial(attempt: u32, key: Key) -> Vec<Value> {
    vec![
        json!({"role":"system","content":prefix()}),
        json!({"role":"user","content":
        format!("Attempt {attempt}: call {TOOL} with key={}. Keep the pinned context value {PIN} in memory for the final answer. Do not answer directly before the tool call.",key.name())}),
    ]
}

pub(super) fn tool(model: &str, messages: &[Value]) -> Value {
    let mut request = text(model, messages, 128);
    request["tools"] = json!([{"type":"function","function":{"name":TOOL,
        "description":"Return one deterministic KV/tool-loop probe fact.",
        "parameters":{"type":"object","properties":{"key":{"type":"string","enum":["primary","secondary"]}},
            "required":["key"],"additionalProperties":false}}}]);
    request["tool_choice"] = json!({"type":"function","function":{"name":TOOL}});
    request["parallel_tool_calls"] = json!(false);
    request
}

pub(super) fn text(model: &str, messages: &[Value], max_tokens: u32) -> Value {
    json!({"model":model,"messages":messages,"stream":false,"temperature":0,"max_tokens":max_tokens,
        "reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}})
}

pub(super) fn cache(model: &str, tail: &str) -> Value {
    text(
        model,
        &[
            json!({"role":"system","content":prefix()}),
            json!({"role":"user","content":
        format!("{tail}\nReturn exactly this pinned value in one sentence: {PIN}.")}),
        ],
        32,
    )
}

pub(super) struct Overlap {
    pub label: String,
    pub payload: Value,
    pub key: Option<Key>,
}

pub(super) fn overlap(model: &str, attempt: u32, count: usize) -> Vec<Overlap> {
    let title = text(
        model,
        &[
            json!({"role":"system","content":prefix()}),
            json!({"role":"user","content":format!("Concurrent title probe attempt {attempt}. Return a short title that includes {PIN}.")}),
        ],
        48,
    );
    let mut rows = vec![Overlap {
        label: "title".into(),
        payload: title,
        key: None,
    }];
    for index in 1..count {
        let key = if index == 1 || index % 2 == 0 {
            Key::Primary
        } else {
            Key::Secondary
        };
        let label = if index == 1 {
            "tool_primary".into()
        } else {
            format!("tool_{index}")
        };
        let messages = vec![
            json!({"role":"system","content":prefix()}),
            json!({"role":"user","content":
            format!("Concurrent tool probe {label} attempt {attempt}: call {TOOL} with key={}. Keep {PIN} in memory. Do not answer directly before the tool call.",key.name())}),
        ];
        rows.push(Overlap {
            label,
            payload: tool(model, &messages),
            key: Some(key),
        });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_and_measured_cache_geometry_preserves_prefix_and_exact_body_semantics() {
        let warm = cache("fixture", "alpha");
        let measured = cache("fixture", "beta");
        assert_eq!(warm["messages"][0], measured["messages"][0]);
        assert_ne!(warm["messages"][1], measured["messages"][1]);
        assert_eq!(warm, cache("fixture", "alpha"));
        assert_eq!(prefix().lines().count(), 195);
        assert_eq!(warm["max_tokens"], 32);
        assert_eq!(warm["reasoning_effort"], "none");
    }

    #[test]
    fn overlap_cohort_has_title_and_distinct_tool_histories_under_one_prefix() {
        let rows = overlap("fixture", 2, 4);
        assert_eq!(rows.len(), 4);
        assert!(rows[0].key.is_none());
        assert_eq!(rows[1].key, Some(Key::Primary));
        assert_eq!(rows[2].key, Some(Key::Primary));
        assert_eq!(rows[3].key, Some(Key::Secondary));
        for row in &rows[1..] {
            assert_eq!(row.payload["messages"][0], rows[0].payload["messages"][0]);
            assert_eq!(row.payload["parallel_tool_calls"], false);
            assert_eq!(row.payload["tool_choice"]["function"]["name"], TOOL);
            assert_ne!(row.payload["messages"][1], rows[0].payload["messages"][1]);
        }
    }
}
