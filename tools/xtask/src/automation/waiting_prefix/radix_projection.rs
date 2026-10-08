//! Typed allowlisted pre-sanitizer summaries; no raw child line or request content is retained.
use serde_json::{Value, json};
#[derive(Default)]
pub(super) struct Projection {
    pub rows: Vec<Value>,
    pub error: Option<String>,
    pub suspect: bool,
}
impl Projection {
    pub fn observe(&mut self, bytes: &[u8]) {
        if bytes.len() > 8192 {
            self.error = Some("radix observation line exceeds owner bound".into());
            return;
        }
        let lower = bytes.iter().map(u8::to_ascii_lowercase).collect::<Vec<_>>();
        self.suspect |= [
            b"resident_error".as_slice(),
            b"failed to find a memory slot",
            b"panic",
        ]
        .iter()
        .any(|term| lower.windows(term.len()).any(|w| w == *term));
        let Ok(event) = serde_json::from_slice::<Value>(bytes) else {
            return;
        };
        if event["event"] != "stage.openai_generation_summary" {
            return;
        }
        if self.rows.len() >= 10000 {
            self.error = Some("radix summary roster exceeds bound".into());
            return;
        }
        let Some(attrs) = event["attributes"].as_object() else {
            self.error = Some("radix generation summary lacks typed attributes".into());
            return;
        };
        let mut selected = json!({"event":"stage.openai_generation_summary","attributes":{}});
        if let Some(status) = attrs.get("skippy.kv.status") {
            if !matches!(status.as_str(), Some("hit" | "miss" | "disabled")) {
                self.error = Some("radix invalid cache status".into());
                return;
            }
            selected["attributes"]["skippy.kv.status"] = status.clone();
        }
        for (key, value) in attrs {
            if matches!(
                key.as_str(),
                "skippy.kv.matched_prefix_tokens"
                    | "skippy.kv.suffix_prefill_tokens"
                    | "llama_stage.prompt_token_count"
            ) || key.starts_with("skippy.kv.radix.")
            {
                if value.as_u64().is_none() {
                    self.error =
                        Some("radix numeric cache summary is not nonnegative integer".into());
                    return;
                }
                if matches!(
                    key.as_str(),
                    "skippy.kv.matched_prefix_tokens"
                        | "skippy.kv.suffix_prefill_tokens"
                        | "llama_stage.prompt_token_count"
                        | "skippy.kv.radix.namespaces"
                        | "skippy.kv.radix.nodes"
                        | "skippy.kv.radix.token_edges"
                        | "skippy.kv.radix.splits"
                        | "skippy.kv.radix.resident_entries"
                        | "skippy.kv.radix.resident_active_refs"
                        | "skippy.kv.radix.recurrent_entries"
                        | "skippy.kv.radix.recurrent_active_refs"
                        | "skippy.kv.radix.resident_evictions"
                        | "skippy.kv.radix.recurrent_evictions"
                ) {
                    selected["attributes"][key] = value.clone();
                }
            }
        }
        self.rows.push(selected);
    }
}
