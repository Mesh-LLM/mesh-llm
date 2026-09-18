//! Agent limits come from the same serving inventory as the model IDs.
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Write;

// Legacy servers without serving metadata use an explicit 8k assumption.
// This is not native-model capacity or a guarantee about an unreported server.
const UNKNOWN_CONTEXT_LIMIT: u32 = 8192;

#[derive(Default)]
pub(super) struct ModelInventory {
    pub names: Vec<String>,
    pub context_lengths: HashMap<String, Option<u32>>,
}

impl ModelInventory {
    pub fn from_response(body: &Value) -> Self {
        let mut inventory = Self::default();
        if let Some(models) = body["data"].as_array() {
            for model in models {
                let Some(id) = model["id"].as_str() else {
                    continue;
                };
                let context = model["metadata"]["context_length"]
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .filter(|value| *value > 0);
                inventory.names.push(id.to_owned());
                inventory.context_lengths.insert(id.to_owned(), context);
            }
        }
        // A virtual route may choose any advertised model. Do not use native
        // capacity or max_context_length (headroom available on only some nodes).
        let mesh_limit = inventory
            .names
            .iter()
            .filter(|id| id.as_str() != "mesh")
            .map(|id| inventory.context_limit(id))
            .min()
            .unwrap_or(UNKNOWN_CONTEXT_LIMIT);
        for (id, limit) in &mut inventory.context_lengths {
            if limit.is_none() {
                let fallback = if id == "mesh" {
                    mesh_limit
                } else {
                    UNKNOWN_CONTEXT_LIMIT
                };
                let _ = writeln!(
                    mesh_llm_events::console_err(),
                    "⚠️  {id} has no served context metadata; using a {fallback}-token launcher fallback."
                );
                *limit = Some(fallback);
            }
        }
        inventory
    }

    pub fn context_limit(&self, id: &str) -> u32 {
        self.context_lengths
            .get(id)
            .copied()
            .flatten()
            .unwrap_or(UNKNOWN_CONTEXT_LIMIT)
    }

    pub fn goose_models(&self) -> Vec<Value> {
        self.names
            .iter()
            .map(|id| json!({"name":id,"context_limit":self.context_limit(id)}))
            .collect()
    }
}

pub(super) fn apply_claude_limits(settings: &mut Value, context: u32) {
    settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = json!(context.to_string());
    settings["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] =
        json!((context / 4).clamp(1, 4096).to_string());
}
