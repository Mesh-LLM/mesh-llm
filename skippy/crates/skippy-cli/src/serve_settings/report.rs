//! Report resolved launch settings without opening a model or cache store.

use serde_json::{Value, json};

use super::ServeSettings;

impl ServeSettings {
    pub fn report(
        &self,
        stage: &skippy_protocol::StageConfig,
        frontend: Option<&skippy_api::serving::OpenAiOptions>,
        tuning: &skippy_serving::settings::ServingTuning,
    ) -> serde_json::Value {
        let command = super::command();
        let mut sources = self.sources.clone();
        let mut options = self.values.clone();
        for argument in command
            .get_arguments()
            .chain(command.find_subcommand("serve").unwrap().get_arguments())
        {
            let Some(name) = argument.get_long() else {
                continue;
            };
            sources
                .entry(name.into())
                .or_insert_with(|| "automatic/default".into());
            options.entry(name.into()).or_insert_with(|| {
                argument
                    .get_default_values()
                    .first()
                    .map(|value| {
                        let value = value.to_string_lossy();
                        serde_json::from_str(&value)
                            .unwrap_or_else(|_| Value::String(value.into_owned()))
                    })
                    .unwrap_or(Value::Null)
            });
        }
        let compaction = tuning
            .guardrails
            .as_ref()
            .and_then(|policy| policy.compaction)
            .map(|config| {
                json!({
                    "enabled": config.enabled,
                    "context_limit_tokens": config.context_limit_tokens,
                    "trigger_ratio_percent": config.trigger_ratio_percent,
                    "target_ratio_percent": config.target_ratio_percent,
                    "allow_reasoning_drop": config.allow_reasoning_drop,
                })
            });
        json!({
            "schema_version": 1,
            "resolution_phase": "before-model-load",
            "stage": stage,
            "frontend": frontend,
            "execution": {"threads": tuning.n_threads, "threads_batch": tuning.n_threads_batch},
            "guardrails": tuning.guardrails.as_ref().map(skippy_serving::OpenAiGuardrailsConfig::status),
            "compaction": compaction,
            "options": options,
            "overrides": self.values,
            "sources": sources,
            "constraints": {
                "unified_kv": "required",
                "ram_cache_requires_disk": true,
                "prefix_budget": "retention target; one indivisible exact-state snapshot may exceed its limit",
                "capabilities": "payload/backend compatibility resolves after model load",
            },
        })
    }
}
