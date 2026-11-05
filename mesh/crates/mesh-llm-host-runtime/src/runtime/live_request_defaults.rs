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

#[derive(Debug)]
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

/// Where the swapped defaults land: the running backends of one owner.
pub(crate) trait RequestDefaultsBackends {
    /// What a backend was serving with before a swap, to put back on rollback.
    type Previous;

    /// Swap `defaults` into every backend of `backend_model_id`; returns how
    /// many were updated.
    fn replace(&self, backend_model_id: &str, defaults: EmbeddedOpenAiRequestDefaults) -> usize;
    fn capture(&self, backend_model_id: &str) -> Self::Previous;
    fn restore(&self, previous: Self::Previous);
}

/// The backends this owner loaded, reached through the serving registry under
/// the owner's scope.
pub(crate) struct ScopedServingBackends {
    scope: u64,
}

impl RequestDefaultsBackends for ScopedServingBackends {
    type Previous = skippy_serving::PreviousRequestDefaults;

    fn replace(&self, backend_model_id: &str, defaults: EmbeddedOpenAiRequestDefaults) -> usize {
        skippy_serving::replace_live_request_defaults(self.scope, backend_model_id, defaults)
    }

    fn capture(&self, backend_model_id: &str) -> Self::Previous {
        skippy_serving::capture_live_request_defaults(self.scope, backend_model_id)
    }

    fn restore(&self, previous: Self::Previous) {
        skippy_serving::restore_live_request_defaults(previous);
    }
}

/// Scope `0` belongs to backends nobody swaps live, so owners start at 1.
static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One owner's tracked models plus the last config applied to them. Each
/// embedded node has its own, so a config apply on one node never reaches the
/// models of another node in the same process, and a node that starts later
/// never inherits the config an earlier node applied.
#[derive(Debug)]
pub(crate) struct LiveRequestDefaults {
    /// The serving-registry scope this owner's backends register under.
    scope: u64,
    tracked: Mutex<Vec<TrackedModel>>,
    latest: Mutex<Option<MeshConfig>>,
    /// Held across an apply or a post-load sync, so a sync that resolved an
    /// older config cannot overwrite what a later apply just swapped in.
    swap: Mutex<()>,
    next_generation: std::sync::atomic::AtomicU64,
}

/// What an apply left behind, so a failed save can undo exactly that.
#[derive(Debug)]
pub(crate) struct AppliedRequestDefaults<P> {
    /// Every tracked model took the new defaults.
    pub(crate) all_applied: bool,
    undo: Option<RequestDefaultsUndo<P>>,
}

#[derive(Debug)]
struct RequestDefaultsUndo<P> {
    previous_latest: Option<MeshConfig>,
    previous: Vec<P>,
}

impl<P> AppliedRequestDefaults<P> {
    fn nothing_changed() -> Self {
        Self {
            all_applied: false,
            undo: None,
        }
    }
}

impl LiveRequestDefaults {
    pub(crate) fn new() -> Self {
        Self {
            scope: NEXT_SCOPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            tracked: Mutex::new(Vec::new()),
            latest: Mutex::new(None),
            swap: Mutex::new(()),
            next_generation: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// The scope this owner's backends must register their defaults under.
    pub(crate) fn scope(&self) -> u64 {
        self.scope
    }

    pub(crate) fn serving_backends(&self) -> ScopedServingBackends {
        ScopedServingBackends { scope: self.scope }
    }

    /// Remember how to re-resolve the request defaults of the model about to
    /// load; returns the generation of that load.
    pub(crate) fn track_model(
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

    /// Bring a model whose backend just came up in line with the last applied
    /// config. Call it once the load has finished, with the generation
    /// `track_model` returned for that load so an older load never syncs a
    /// newer one's entry.
    pub(crate) fn sync_after_load(
        &self,
        backend_model_id: &str,
        generation: u64,
        backends: &impl RequestDefaultsBackends,
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
            let updated = backends.replace(&entry.backend_model_id, defaults);
            self.mark_registered(&entry, updated);
        }
    }

    /// Swap the request defaults `config` resolves to into every tracked
    /// model, all or nothing: every model is resolved and validated before the
    /// first swap, so one model whose defaults do not resolve leaves every
    /// backend, and the last applied config, as they were.
    pub(crate) fn apply<B: RequestDefaultsBackends>(
        &self,
        config: &MeshConfig,
        backends: &B,
    ) -> AppliedRequestDefaults<B::Previous> {
        let _swap = lock(&self.swap);
        // Resolving can read a template file from disk, so keep it off the lock
        // that every model load takes.
        let entries = self.snapshot(|_| true);
        let mut resolved = Vec::with_capacity(entries.len());
        for entry in &entries {
            match resolve(entry, config) {
                Some(defaults) => resolved.push(defaults),
                None => return AppliedRequestDefaults::nothing_changed(),
            }
        }
        let previous_latest = lock(&self.latest).replace(config.clone());
        let mut previous = Vec::with_capacity(entries.len());
        let mut gone = Vec::new();
        for (entry, defaults) in entries.iter().zip(resolved) {
            previous.push(backends.capture(&entry.backend_model_id));
            let updated = backends.replace(&entry.backend_model_id, defaults);
            if updated > 0 {
                self.mark_registered(entry, updated);
            } else if entry.registered {
                gone.push((entry.backend_model_id.clone(), entry.generation));
            }
        }
        lock(&self.tracked)
            .retain(|entry| !gone.contains(&(entry.backend_model_id.clone(), entry.generation)));
        AppliedRequestDefaults {
            all_applied: true,
            undo: Some(RequestDefaultsUndo {
                previous_latest,
                previous,
            }),
        }
    }

    /// Put back what `applied` replaced: the backends' own previous defaults,
    /// not a re-resolution of the previous config, and the previous last
    /// applied config.
    pub(crate) fn undo<B: RequestDefaultsBackends>(
        &self,
        applied: AppliedRequestDefaults<B::Previous>,
        backends: &B,
    ) {
        let Some(undo) = applied.undo else {
            return;
        };
        let _swap = lock(&self.swap);
        *lock(&self.latest) = undo.previous_latest;
        for previous in undo.previous {
            backends.restore(previous);
        }
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
