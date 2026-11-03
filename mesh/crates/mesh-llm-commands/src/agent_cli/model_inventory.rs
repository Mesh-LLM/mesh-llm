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
    pub(super) fallbacks: Vec<(String, u32)>,
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
        inventory.apply_fallbacks();
        inventory
    }

    fn apply_fallbacks(&mut self) {
        // A virtual route may choose any advertised model. Do not use native
        // capacity or max_context_length (headroom available on only some nodes).
        let mesh_limit = self
            .names
            .iter()
            .filter(|id| id.as_str() != "mesh")
            .map(|id| self.context_limit(id))
            .min()
            .unwrap_or(UNKNOWN_CONTEXT_LIMIT);
        for (id, limit) in &mut self.context_lengths {
            if limit.is_none() {
                let fallback = if id == "mesh" {
                    mesh_limit
                } else {
                    UNKNOWN_CONTEXT_LIMIT
                };
                self.fallbacks.push((id.clone(), fallback));
                *limit = Some(fallback);
            }
        }
    }

    /// Report served-inventory fallbacks to the operator. Parsing stays pure;
    /// callers call this once after they settle on an inventory.
    pub fn report_fallbacks(&self) {
        for (id, fallback) in &self.fallbacks {
            let _ = writeln!(
                mesh_llm_events::console_err(),
                "⚠️  {id} has no served context metadata; using a {fallback}-token launcher fallback."
            );
        }
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

/// Claude Code (v2.1.193+) honors CLAUDE_CODE_MAX_CONTEXT_TOKENS only for model
/// IDs it does not recognize as first-party. A mesh-served `claude-*` ID keeps
/// Claude's built-in window, so the served limit is a silent no-op there and
/// auto-compact would fire on the built-in window; DISABLE_COMPACT opts out for
/// that case. For unrecognized IDs the served limit applies directly and
/// compaction stays enabled.
pub(super) fn apply_claude_limits(settings: &mut Value, model_id: &str, context: u32) {
    if model_id.starts_with("claude-") {
        settings["env"]["DISABLE_COMPACT"] = json!("1");
    }
    settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = json!(context.to_string());
    settings["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] =
        json!((context / 4).clamp(1, 4096).to_string());
}

/// Enrich the compatibility list without widening its LLM-only identity set.
/// Legacy/third-party servers retain the explicit launcher fallback when they
/// do not expose Mesh discovery. Partial or invalid catalog pages are ignored.
pub(super) async fn enrich_from_mesh(
    client: &reqwest::Client,
    compatibility_url: &str,
    inventory: &mut ModelInventory,
) {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        fetch_catalog(client, compatibility_url),
    )
    .await;
    if let Ok(Ok(models)) = result {
        inventory.apply_catalog(&models);
    }
}

async fn fetch_catalog(
    client: &reqwest::Client,
    compatibility_url: &str,
) -> anyhow::Result<Vec<Value>> {
    const MAX_PAGES: usize = 16;
    const MAX_PAGE_BYTES: usize = 4 * 1024 * 1024;
    let mut url = url::Url::parse(compatibility_url)?;
    url.set_path("/mesh/v1/models");
    url.set_fragment(None);
    let mut cursor = None::<String>;
    let mut seen = std::collections::HashSet::new();
    let mut models = Vec::new();
    for _ in 0..MAX_PAGES {
        url.set_query(None);
        url.query_pairs_mut().append_pair("limit", "500");
        if let Some(cursor) = &cursor {
            url.query_pairs_mut().append_pair("cursor", cursor);
        }
        let mut response = client.get(url.clone()).send().await?.error_for_status()?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            anyhow::ensure!(
                bytes.len().saturating_add(chunk.len()) <= MAX_PAGE_BYTES,
                "Mesh model page too large"
            );
            bytes.extend_from_slice(&chunk);
        }
        let body: Value = serde_json::from_slice(&bytes)?;
        let entries = body["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("invalid Mesh model page"))?;
        models.extend(entries.iter().cloned());
        match body.get("next_cursor") {
            Some(Value::Null) => return Ok(models),
            Some(Value::String(next)) if !next.is_empty() && seen.insert(next.clone()) => {
                cursor = Some(next.clone())
            }
            _ => anyhow::bail!("invalid Mesh model cursor"),
        }
    }
    anyhow::bail!("Mesh model catalog exceeded page limit")
}

impl ModelInventory {
    fn apply_catalog(&mut self, models: &[Value]) {
        // Remove inferred values before replacing them with actual observations.
        for (id, _) in self.fallbacks.drain(..) {
            self.context_lengths.insert(id, None);
        }
        for id in &self.names {
            if let Some(model) = models.iter().find(|model| model["id"].as_str() == Some(id)) {
                self.context_lengths
                    .insert(id.clone(), provider_context_limit(model));
            }
        }
        self.apply_fallbacks();
    }
}

fn provider_context_limit(model: &Value) -> Option<u32> {
    if model["kind"].as_str() == Some("virtual") {
        return None;
    }
    let providers = model["providers"].as_array()?;
    let mut minimum = None::<u32>;
    for provider in providers {
        if provider["available"] != true {
            continue;
        }
        // Legacy replicas without a workload class remain eligible for chat
        // admission. Their missing windows must not disappear from this check.
        let generation = matches!(
            provider["workload_class"].as_str(),
            None | Some("causal_generation" | "encoder_decoder")
        );
        if !generation {
            continue;
        }
        let length = provider["runtime"]["context_length"]
            .as_u64()
            .and_then(|length| u32::try_from(length).ok())
            .filter(|length| *length > 0)?;
        minimum = Some(minimum.map_or(length, |previous| previous.min(length)));
    }
    minimum
}

#[cfg(test)]
mod tests;
