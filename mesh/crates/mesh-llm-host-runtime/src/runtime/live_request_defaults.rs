//! Live application of `request_defaults` config to already loaded models.
//!
//! A model resolves its request defaults when it loads, but the config schema
//! marks every supported `request_defaults` setting as dynamic with no restart.
//! Loads register here what is needed to resolve a model's defaults again, and
//! a config apply re-resolves them against the new config and swaps them into
//! the running backends.

use std::sync::Mutex;

use mesh_llm_config::MeshConfig;
use skippy_runtime::package::PackageGenerationInfo;
use skippy_serving::EmbeddedOpenAiRequestDefaults;

use crate::inference::skippy;

struct TrackedModel {
    /// Model id the serving backend registered its defaults under.
    backend_model_id: String,
    /// `models[].model` selector that picks this model's config entry.
    config_model_id: Option<String>,
    package_generation: Option<PackageGenerationInfo>,
}

static TRACKED: Mutex<Vec<TrackedModel>> = Mutex::new(Vec::new());

/// Remember how to re-resolve the request defaults of the model about to load.
pub(crate) fn track_model(
    backend_model_id: &str,
    config_model_id: Option<&str>,
    package_generation: Option<&PackageGenerationInfo>,
) {
    track_model_in(
        &TRACKED,
        backend_model_id,
        config_model_id,
        package_generation,
    );
}

fn track_model_in(
    registry: &Mutex<Vec<TrackedModel>>,
    backend_model_id: &str,
    config_model_id: Option<&str>,
    package_generation: Option<&PackageGenerationInfo>,
) {
    let mut tracked = registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    tracked.retain(|entry| entry.backend_model_id != backend_model_id);
    tracked.push(TrackedModel {
        backend_model_id: backend_model_id.to_string(),
        config_model_id: config_model_id.map(str::to_string),
        package_generation: package_generation.cloned(),
    });
}

/// Whether any request default differs between two configs.
pub(crate) fn request_defaults_changed(old: &MeshConfig, new: &MeshConfig) -> bool {
    request_defaults_fingerprint(old) != request_defaults_fingerprint(new)
}

fn request_defaults_fingerprint(config: &MeshConfig) -> serde_json::Value {
    let global = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.request_defaults.as_ref());
    let per_model: Vec<_> = config
        .models
        .iter()
        .filter_map(|entry| Some((&entry.model, entry.request_defaults.as_ref()?)))
        .collect();
    serde_json::json!({ "global": global, "models": per_model })
}

/// Re-resolve every tracked model against `config` and swap the result into its
/// running backends. Returns whether every loaded model was updated; a model
/// whose defaults no longer resolve keeps the ones it is serving with.
pub(crate) fn apply_to_loaded_models(config: &MeshConfig) -> bool {
    apply_with(
        &TRACKED,
        config,
        skippy_serving::replace_live_request_defaults,
    )
}

fn apply_with(
    registry: &Mutex<Vec<TrackedModel>>,
    config: &MeshConfig,
    replace: impl Fn(&str, EmbeddedOpenAiRequestDefaults) -> usize,
) -> bool {
    let mut tracked = registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut all_applied = true;
    tracked.retain(|entry| {
        let defaults = match skippy::resolve_embedded_request_defaults(
            config,
            entry.config_model_id.as_deref(),
            entry.package_generation.as_ref(),
        ) {
            Ok(defaults) => defaults,
            Err(error) => {
                tracing::warn!(
                    model = %entry.backend_model_id,
                    %error,
                    "request defaults did not resolve; model keeps its current defaults"
                );
                all_applied = false;
                return true;
            }
        };
        // Zero updated backends means the model has unloaded since it was tracked.
        replace(&entry.backend_model_id, defaults) > 0
    });
    all_applied
}

#[cfg(test)]
mod tests;
