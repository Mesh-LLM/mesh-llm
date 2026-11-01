//! Live application of `request_defaults` config to already loaded models.
//!
//! A model resolves its request defaults when it loads, but the config schema
//! marks every supported `request_defaults` setting as dynamic with no restart.
//! Loads register here what is needed to resolve a model's defaults again, and
//! a config apply re-resolves them against the new config and swaps them into
//! the running backends.
//!
//! A load works from the config copy its lifecycle captured, which can predate
//! an apply, and an apply can land while a model is still loading. So the last
//! applied config is kept, and each load syncs to it once its backend is up.

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
    /// A swap has reached the model's backend. Until then a swap that finds no
    /// backend means the model is still loading, not that it is gone.
    registered: bool,
    /// Distinguishes a reload of the same model id from the entry an in-flight
    /// apply or sync snapshotted.
    generation: u64,
}

/// Tracked models plus the last config applied to them.
pub(crate) struct LiveRequestDefaults {
    tracked: Mutex<Vec<TrackedModel>>,
    latest: Mutex<Option<MeshConfig>>,
    /// Held across an apply or a post-load sync, so a sync that resolved an
    /// older config cannot overwrite what a later apply just swapped in.
    swap: Mutex<()>,
    next_generation: std::sync::atomic::AtomicU64,
}

static LIVE: LiveRequestDefaults = LiveRequestDefaults::new();

/// Remember how to re-resolve the request defaults of the model about to load.
pub(crate) fn track_model(
    backend_model_id: &str,
    config_model_id: Option<&str>,
    package_generation: Option<&PackageGenerationInfo>,
) -> u64 {
    LIVE.track_model(backend_model_id, config_model_id, package_generation)
}

/// Bring a model whose backend just came up in line with the last applied
/// config. Call it once the load has finished, with the generation
/// `track_model` returned for that load so an older load never syncs a newer
/// one's entry.
pub(crate) fn sync_after_load(backend_model_id: &str, generation: u64) {
    LIVE.sync_after_load(
        backend_model_id,
        generation,
        skippy_serving::replace_live_request_defaults,
    );
}

/// Swap the request defaults `config` resolves to into every loaded model.
/// Returns whether every tracked model was updated; a model whose defaults no
/// longer resolve keeps the ones it is serving with.
pub(crate) fn apply_to_loaded_models(config: &MeshConfig) -> bool {
    LIVE.apply(config, skippy_serving::replace_live_request_defaults)
}

/// Whether any request default differs between two configs.
pub(crate) fn request_defaults_changed(old: &MeshConfig, new: &MeshConfig) -> bool {
    let global = |config: &MeshConfig| {
        config
            .defaults
            .as_ref()
            .and_then(|defaults| defaults.request_defaults.as_ref())
            .cloned()
    };
    let per_model = |config: &MeshConfig| {
        config
            .models
            .iter()
            .filter_map(|entry| Some((entry.model.clone(), entry.request_defaults.clone()?)))
            .collect::<Vec<_>>()
    };
    global(old) != global(new) || per_model(old) != per_model(new)
}

impl LiveRequestDefaults {
    const fn new() -> Self {
        Self {
            tracked: Mutex::new(Vec::new()),
            latest: Mutex::new(None),
            swap: Mutex::new(()),
            next_generation: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn track_model(
        &self,
        backend_model_id: &str,
        config_model_id: Option<&str>,
        package_generation: Option<&PackageGenerationInfo>,
    ) -> u64 {
        let mut tracked = lock(&self.tracked);
        tracked.retain(|entry| entry.backend_model_id != backend_model_id);
        let generation = self
            .next_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracked.push(TrackedModel {
            backend_model_id: backend_model_id.to_string(),
            config_model_id: config_model_id.map(str::to_string),
            package_generation: package_generation.cloned(),
            registered: false,
            generation,
        });
        generation
    }

    fn sync_after_load(
        &self,
        backend_model_id: &str,
        generation: u64,
        replace: impl Fn(&str, EmbeddedOpenAiRequestDefaults) -> usize,
    ) {
        let _swap = lock(&self.swap);
        let Some(latest) = lock(&self.latest).clone() else {
            return;
        };
        let Some(entry) = self
            .snapshot(|entry| {
                entry.backend_model_id == backend_model_id && entry.generation == generation
            })
            .pop()
        else {
            return;
        };
        if let Some(defaults) = resolve(&entry, &latest) {
            let updated = replace(&entry.backend_model_id, defaults);
            self.mark_registered(&entry, updated);
        }
    }

    fn apply(
        &self,
        config: &MeshConfig,
        replace: impl Fn(&str, EmbeddedOpenAiRequestDefaults) -> usize,
    ) -> bool {
        let _swap = lock(&self.swap);
        *lock(&self.latest) = Some(config.clone());
        // Resolving can read a template file from disk, so keep it off the lock
        // that every model load takes.
        let entries = self.snapshot(|_| true);
        let mut all_applied = true;
        let mut gone = Vec::new();
        for entry in &entries {
            let Some(defaults) = resolve(entry, config) else {
                all_applied = false;
                continue;
            };
            let updated = replace(&entry.backend_model_id, defaults);
            if updated > 0 {
                self.mark_registered(entry, updated);
            } else if entry.registered {
                gone.push((entry.backend_model_id.clone(), entry.generation));
            }
        }
        lock(&self.tracked)
            .retain(|entry| !gone.contains(&(entry.backend_model_id.clone(), entry.generation)));
        all_applied
    }

    fn snapshot(&self, keep: impl Fn(&TrackedModel) -> bool) -> Vec<TrackedModel> {
        lock(&self.tracked)
            .iter()
            .filter(|entry| keep(entry))
            .map(|entry| TrackedModel {
                backend_model_id: entry.backend_model_id.clone(),
                config_model_id: entry.config_model_id.clone(),
                package_generation: entry.package_generation.clone(),
                registered: entry.registered,
                generation: entry.generation,
            })
            .collect()
    }

    fn mark_registered(&self, snapshotted: &TrackedModel, updated: usize) {
        if updated == 0 {
            return;
        }
        if let Some(entry) = lock(&self.tracked).iter_mut().find(|entry| {
            entry.backend_model_id == snapshotted.backend_model_id
                && entry.generation == snapshotted.generation
        }) {
            entry.registered = true;
        }
    }
}

/// Resolve and validate a tracked model's defaults, or warn and return `None`
/// so the model keeps serving with the ones it has.
fn resolve(entry: &TrackedModel, config: &MeshConfig) -> Option<EmbeddedOpenAiRequestDefaults> {
    let resolved = skippy::resolve_embedded_request_defaults(
        config,
        entry.config_model_id.as_deref(),
        entry.package_generation.as_ref(),
    )
    .and_then(|defaults| defaults.validate().map(|()| defaults));
    match resolved {
        Ok(defaults) => Some(defaults),
        Err(error) => {
            tracing::warn!(
                model = %entry.backend_model_id,
                %error,
                "request defaults did not resolve; model keeps its current defaults"
            );
            None
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
