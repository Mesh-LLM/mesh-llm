//! Request leases for opt-in native residency. The reconciler owns execution.
use std::{collections::BTreeMap, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use mesh_llm_config::managed_models::{ModelLifecycle, ModelLoadPolicy};
use tokio::sync::{Mutex, OwnedMutexGuard, OwnedRwLockReadGuard, RwLock, mpsc, watch};

use super::{IntentSource, ModelIntent, UnloadOptions, UnloadTarget};
use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::Node,
};

const LOAD_TIMEOUT: Duration = Duration::from_secs(300);

pub(crate) struct NativeModelLifecycle {
    models: BTreeMap<String, Arc<Residency>>,
    targets: watch::Receiver<ModelTargets>,
    intents: mpsc::Sender<ModelIntent>,
}

struct Residency {
    policy: ModelLifecycle,
    spec: String,
    gate: Arc<RwLock<()>>,
    operation: Arc<Mutex<()>>,
    idle_since: std::sync::Mutex<std::time::Instant>,
    instance: std::sync::Mutex<Option<(String, String)>>,
}

pub(crate) struct RequestLease {
    residency: Arc<Residency>,
    _gate: OwnedRwLockReadGuard<()>,
}

impl Drop for RequestLease {
    fn drop(&mut self) {
        *self
            .residency
            .idle_since
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = std::time::Instant::now();
    }
}

impl NativeModelLifecycle {
    /// Resolve a configured identifier or loaded runtime alias to shared state.
    fn residency(&self, model: &str) -> Option<(&String, &Arc<Residency>)> {
        self.models.get_key_value(model).or_else(|| {
            self.models.iter().find(|(_, residency)| {
                residency
                    .instance
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .is_some_and(|(_, loaded)| loaded == model)
            })
        })
    }

    /// Report whether requests for this name need a residency lease.
    pub(crate) fn manages(&self, model: &str) -> bool {
        self.residency(model).is_some()
    }

    /// List configured on-demand model identifiers, including cold models.
    pub(crate) fn model_names(&self) -> impl Iterator<Item = &String> {
        self.models.keys()
    }

    /// Add configured-name aliases to the current runtime routing snapshot.
    pub(crate) fn targets(&self) -> ModelTargets {
        {
            let mut targets = self.targets.borrow().clone();
            for (model, residency) in &self.models {
                if let Some((_, loaded)) = residency
                    .instance
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    && let Some(candidates) = targets.targets.get(loaded).cloned()
                {
                    targets.targets.insert(model.clone(), candidates);
                }
            }
            targets
        }
    }

    /// Keep this lease until dispatch (including streaming) has finished.
    pub(crate) async fn acquire(
        self: &Arc<Self>,
        model: &str,
        local_only: bool,
    ) -> Result<Option<RequestLease>> {
        let Some((model, residency)) = self.residency(model) else {
            return Ok(None);
        };
        let model = model.clone();
        let residency = residency.clone();
        let lifecycle = self.clone();
        let job = tokio::spawn(async move {
            // Operation serialization avoids a load racing an eviction. Request
            // leases share the read gate, so generation remains concurrent.
            let operation = residency.operation.clone().lock_owned().await;
            let gate = residency.gate.clone().read_owned().await;
            if !ready(&lifecycle.targets(), &model, local_only) {
                lifecycle.load(&model, &residency, operation).await?;
            }
            Ok(Some(RequestLease {
                residency: residency.clone(),
                _gate: gate,
            }))
        });
        tokio::time::timeout(LOAD_TIMEOUT, job)
            .await
            .context("native model residency wait timed out")?
            .context("native model residency task failed")?
    }

    /// Use a loaded runtime alias only while its local backend is ready.
    pub(crate) fn execution_model(&self, model: &str) -> String {
        self.residency(model)
            .and_then(|(_, residency)| {
                residency
                    .instance
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .map(|(_, name)| name.clone())
            })
            .filter(|name| ready(&self.targets.borrow(), name, true))
            .unwrap_or_else(|| model.to_owned())
    }

    /// Enqueue a reconciler load and retain its exact instance for idle eviction.
    async fn load(
        &self,
        model: &str,
        residency: &Residency,
        _operation: OwnedMutexGuard<()>,
    ) -> Result<()> {
        let (completion, result) = tokio::sync::oneshot::channel();
        self.intents
            .send(ModelIntent::Load {
                intent_id: None,
                spec: residency.spec.clone(),
                config_model_id: Some(model.to_owned()),
                profile: String::new(),
                source: IntentSource::NativeLifecycle,
                completion: Some(completion),
            })
            .await
            .context("native model lifecycle unavailable")?;
        let loaded = result.await.context("native model load interrupted")??;
        *residency.instance.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((loaded.instance_id, loaded.model));
        Ok(())
    }

    /// Drain idle owned instances without active leases; defer retries by one idle interval.
    async fn evict_idle(&self) {
        for (model, residency) in &self.models {
            let timeout = residency.policy.idle_timeout_secs;
            if timeout == 0 {
                continue;
            }
            let Ok(_operation) = residency.operation.try_lock() else {
                continue;
            };
            let Ok(_gate) = residency.gate.try_write() else {
                continue;
            };
            let elapsed = residency
                .idle_since
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .elapsed();
            if elapsed < Duration::from_secs(timeout) {
                continue;
            }
            let instance = residency
                .instance
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            // Only unload instances created by this service. Startup and split
            // controllers, explicit owner loads and required sessions retain
            // their existing residency policy.
            let Some((instance_id, _)) = instance.as_ref() else {
                continue;
            };
            let (completion, result) = tokio::sync::oneshot::channel();
            let intent = ModelIntent::Unload {
                intent_id: None,
                canonical_model_ref: Some(model.clone()),
                target: UnloadTarget::Instance(instance_id.clone()),
                options: UnloadOptions {
                    force: false,
                    ..Default::default()
                },
                source: IntentSource::NativeLifecycle,
                completion: Some(completion),
            };
            if self.intents.send(intent).await.is_err() {
                return;
            }
            match result.await {
                Ok(Ok(_)) => *residency.instance.lock().unwrap_or_else(|e| e.into_inner()) = None,
                result => {
                    tracing::debug!(model, ?result, "native idle unload deferred");
                    // Retry only after another full idle interval.
                    *residency
                        .idle_since
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = std::time::Instant::now();
                }
            }
        }
    }
}

/// Protect split controllers and higher-priority residency intents from automatic eviction.
pub(super) fn idle_unload_allowed(
    state: &super::model_reconciliation::ModelTargetReconciliationState,
    candidate: Option<&super::run_auto::RuntimeUnloadCandidate>,
    canonical: Option<&str>,
) -> bool {
    candidate.is_some_and(|candidate| {
        candidate.owner == super::run_auto::RuntimeUnloadOwner::Runtime
            && [Some(candidate.model_name.as_str()), canonical]
                .into_iter()
                .flatten()
                .all(|model| {
                    state
                        .effective_intent(model, &candidate.profile)
                        .is_none_or(|intent| {
                            intent.source.precedence() <= IntentSource::NativeLifecycle.precedence()
                        })
                })
    })
}

/// Check for a ready target permitted by the request origin.
fn ready(targets: &ModelTargets, model: &str, local_only: bool) -> bool {
    targets.candidates(model).iter().any(|target| match target {
        InferenceTarget::Local(_) => true,
        InferenceTarget::Remote(_) => !local_only,
        InferenceTarget::None => false,
    })
}

/// Select only uniquely named models for automatic native residency.
/// Entries outside this scope retain the normal startup path.
pub(super) fn is_on_demand_managed(
    config: &crate::plugin::MeshConfig,
    model: &crate::plugin::ModelConfigEntry,
) -> bool {
    model.lifecycle.load == ModelLoadPolicy::OnDemand
        && config
            .models
            .iter()
            .filter(|entry| entry.model == model.model)
            .count()
            == 1
}

/// Construct residency state for exactly the entries excluded from startup.
fn configured_residencies(config: &crate::plugin::MeshConfig) -> BTreeMap<String, Arc<Residency>> {
    config
        .models
        .iter()
        .filter(|model| is_on_demand_managed(config, model))
        .map(|model| {
            (
                model.model.clone(),
                Arc::new(Residency {
                    spec: model
                        .hardware
                        .as_ref()
                        .and_then(|hardware| hardware.model_path.clone())
                        .or_else(|| {
                            config
                                .defaults
                                .as_ref()
                                .and_then(|defaults| defaults.hardware.as_ref())
                                .and_then(|hardware| hardware.model_path.clone())
                        })
                        .unwrap_or_else(|| model.model.clone()),
                    policy: model.lifecycle.clone(),
                    gate: Arc::default(),
                    operation: Arc::default(),
                    idle_since: std::sync::Mutex::new(std::time::Instant::now()),
                    instance: std::sync::Mutex::new(None),
                }),
            )
        })
        .collect()
}

/// Attach host residency and start disk prefetch plus weakly owned idle maintenance.
pub(super) async fn install(
    node: &Node,
    config: &crate::plugin::MeshConfig,
    targets: watch::Receiver<ModelTargets>,
) {
    let Some(intents) = node.model_intent_tx.lock().await.clone() else {
        return;
    };
    let models = configured_residencies(config);
    let lifecycle = Arc::new(NativeModelLifecycle {
        models,
        targets,
        intents,
    });
    if node.native_model_lifecycle.set(lifecycle.clone()).is_err() {
        return;
    }
    for (model, residency) in &lifecycle.models {
        let strict_local = config
            .models
            .iter()
            .find(|entry| entry.model == *model)
            .is_some_and(|entry| {
                super::startup_models::skippy_local_source_required(
                    entry.skippy.as_ref(),
                    config
                        .defaults
                        .as_ref()
                        .and_then(|defaults| defaults.skippy.as_ref()),
                )
            });
        if residency.policy.prefetch && !strict_local {
            let spec = residency.spec.clone();
            let model = model.clone();
            tokio::spawn(async move {
                if let Err(error) =
                    super::startup_models::resolve_model(std::path::Path::new(&spec)).await
                {
                    tracing::warn!(model, %error, "native model prefetch failed; requests may retry");
                }
            });
        }
    }
    // Weak ownership prevents the maintenance task from keeping a stopped node alive.
    let weak = Arc::downgrade(&lifecycle);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let Some(lifecycle) = weak.upgrade() else {
                break;
            };
            if lifecycle.intents.is_closed() {
                break;
            }
            lifecycle.evict_idle().await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        timeout: u64,
    ) -> (
        Arc<NativeModelLifecycle>,
        mpsc::Receiver<ModelIntent>,
        watch::Sender<ModelTargets>,
    ) {
        let (intents, receiver) = mpsc::channel(8);
        let (targets, snapshot) = watch::channel(ModelTargets::default());
        let residency = Arc::new(Residency {
            policy: ModelLifecycle {
                load: ModelLoadPolicy::OnDemand,
                idle_timeout_secs: timeout,
                prefetch: false,
            },
            spec: "test".into(),
            gate: Arc::default(),
            operation: Arc::default(),
            idle_since: std::sync::Mutex::new(std::time::Instant::now() - Duration::from_secs(60)),
            instance: std::sync::Mutex::new(None),
        });
        (
            Arc::new(NativeModelLifecycle {
                models: BTreeMap::from([("test".into(), residency)]),
                targets: snapshot,
                intents,
            }),
            receiver,
            targets,
        )
    }

    fn finish_load(intent: ModelIntent, targets: &watch::Sender<ModelTargets>) {
        let ModelIntent::Load {
            completion: Some(completion),
            ..
        } = intent
        else {
            panic!("expected load")
        };
        targets.send_modify(|targets| {
            targets
                .targets
                .insert("loaded-test".into(), vec![InferenceTarget::Local(1234)]);
        });
        completion
            .send(Ok(crate::api::RuntimeLoadResponse {
                model_ref: "test".into(),
                model: "loaded-test".into(),
                instance_id: "runtime-1".into(),
                profile: String::new(),
                backend: None,
                context_length: None,
            }))
            .unwrap();
    }

    #[tokio::test]
    async fn simultaneous_requests_share_one_load_and_hold_stream_leases() {
        let (lifecycle, mut intents, targets) = fixture(1);
        let first = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test", true).await.unwrap().unwrap() })
        };
        let second = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test", true).await.unwrap().unwrap() })
        };
        finish_load(intents.recv().await.unwrap(), &targets);
        let (first, second) = (first.await.unwrap(), second.await.unwrap());
        assert!(intents.try_recv().is_err());
        assert_eq!(lifecycle.execution_model("test"), "loaded-test");
        assert!(lifecycle.manages("loaded-test"));
        let alias = lifecycle.acquire("loaded-test", true).await.unwrap();
        assert!(intents.try_recv().is_err());
        drop(alias);
        lifecycle.evict_idle().await;
        assert!(intents.try_recv().is_err());
        drop(first);
        lifecycle.evict_idle().await;
        assert!(intents.try_recv().is_err());
        drop(second);
        assert!(
            lifecycle.models["test"]
                .idle_since
                .lock()
                .unwrap()
                .elapsed()
                < Duration::from_secs(1)
        );
    }

    #[tokio::test]
    async fn idle_eviction_targets_exact_instance_and_next_request_reloads() {
        let (lifecycle, mut intents, targets) = fixture(1);
        *lifecycle.models["test"].instance.lock().unwrap() =
            Some(("runtime-1".into(), "loaded-test".into()));
        let eviction = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.evict_idle().await })
        };
        let ModelIntent::Unload {
            target,
            completion: Some(completion),
            options,
            ..
        } = intents.recv().await.unwrap()
        else {
            panic!("expected unload")
        };
        assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));
        assert!(!options.force);
        completion
            .send(Ok(crate::api::RuntimeUnloadResponse {
                model: "loaded-test".into(),
                instance_id: "runtime-1".into(),
                unloaded: true,
            }))
            .unwrap();
        eviction.await.unwrap();
        let request = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test", true).await.unwrap() })
        };
        finish_load(intents.recv().await.unwrap(), &targets);
        assert!(request.await.unwrap().is_some());
    }

    /// Rejections and lost completions wait another idle interval before retrying.
    #[tokio::test]
    async fn deferred_idle_unloads_back_off_before_retrying() {
        for error in [
            Some("maintained owner intent"),
            Some("instance no longer exists"),
            None,
        ] {
            let (lifecycle, mut intents, _targets) = fixture(30);
            let residency = &lifecycle.models["test"];
            *residency.instance.lock().unwrap() = Some(("runtime-1".into(), "loaded-test".into()));
            let eviction = {
                let lifecycle = lifecycle.clone();
                tokio::spawn(async move { lifecycle.evict_idle().await })
            };
            let ModelIntent::Unload {
                completion: Some(completion),
                ..
            } = intents.recv().await.unwrap()
            else {
                panic!("expected unload")
            };
            if let Some(error) = error {
                completion.send(Err(anyhow::anyhow!(error))).unwrap();
            } else {
                drop(completion);
            }
            eviction.await.unwrap();
            assert!(residency.instance.lock().unwrap().is_some());
            lifecycle.evict_idle().await;
            assert!(
                intents.try_recv().is_err(),
                "must not retry on the next maintenance tick"
            );
            *residency.idle_since.lock().unwrap() =
                std::time::Instant::now() - Duration::from_secs(15);
            lifecycle.evict_idle().await;
            assert!(
                intents.try_recv().is_err(),
                "must wait the entire idle interval"
            );
            *residency.idle_since.lock().unwrap() =
                std::time::Instant::now() - Duration::from_secs(31);
            let retry = {
                let lifecycle = lifecycle.clone();
                tokio::spawn(async move { lifecycle.evict_idle().await })
            };
            let ModelIntent::Unload {
                completion: Some(completion),
                ..
            } = intents.recv().await.unwrap()
            else {
                panic!("expected retry")
            };
            completion
                .send(Ok(crate::api::RuntimeUnloadResponse {
                    model: "loaded-test".into(),
                    instance_id: "runtime-1".into(),
                    unloaded: true,
                }))
                .unwrap();
            retry.await.unwrap();
            assert!(residency.instance.lock().unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn resident_policy_and_unowned_instances_are_never_evicted() {
        for timeout in [0, 1] {
            let (lifecycle, mut intents, _targets) = fixture(timeout);
            if timeout == 0 {
                *lifecycle.models["test"].instance.lock().unwrap() =
                    Some(("runtime-1".into(), "loaded-test".into()));
            }
            lifecycle.evict_idle().await;
            assert!(intents.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn failed_load_releases_waiters_and_allows_retry() {
        let (lifecycle, mut intents, targets) = fixture(1);
        let first = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test", true).await })
        };
        let ModelIntent::Load {
            completion: Some(completion),
            ..
        } = intents.recv().await.unwrap()
        else {
            panic!("expected load")
        };
        completion
            .send(Err(anyhow::anyhow!("capacity exhausted")))
            .unwrap();
        assert!(first.await.unwrap().is_err());
        let retry = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test", true).await })
        };
        finish_load(intents.recv().await.unwrap(), &targets);
        assert!(retry.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn cancelling_request_does_not_lose_load_ownership() {
        let (lifecycle, mut intents, targets) = fixture(1);
        let request = {
            let lifecycle = lifecycle.clone();
            tokio::spawn(async move { lifecycle.acquire("test", true).await })
        };
        let intent = intents.recv().await.unwrap();
        request.abort();
        finish_load(intent, &targets);
        // Acquiring again synchronizes with the detached load job.
        let _lease = lifecycle.acquire("test", true).await.unwrap();
        assert!(lifecycle.models["test"].instance.lock().unwrap().is_some());
        assert!(intents.try_recv().is_err());
    }

    #[test]
    fn assigned_peer_execution_never_uses_a_remote_ready_target() {
        let peer = iroh::SecretKey::from_bytes(&[7; 32]).public();
        let mut targets = ModelTargets::default();
        targets
            .targets
            .insert("test".into(), vec![InferenceTarget::Remote(peer)]);
        assert!(ready(&targets, "test", false));
        assert!(!ready(&targets, "test", true));
    }
}

#[cfg(test)]
mod intent_tests {
    use super::*;

    #[test]
    fn idle_eviction_respects_owner_ensure_and_refuses_split_controllers() {
        use super::super::model_reconciliation::ModelTargetReconciliationState;
        use super::super::run_auto::{RuntimeUnloadCandidate, RuntimeUnloadOwner};
        let mut state = ModelTargetReconciliationState::default();
        let mut candidate = RuntimeUnloadCandidate {
            owner: RuntimeUnloadOwner::Runtime,
            instance_id: "runtime-1".into(),
            model_name: "loaded-test".into(),
            profile: String::new(),
        };
        assert!(idle_unload_allowed(&state, Some(&candidate), Some("test")));
        state.add_desired("test", "", IntentSource::OwnerEnsure);
        assert!(!idle_unload_allowed(&state, Some(&candidate), Some("test")));
        candidate.owner = RuntimeUnloadOwner::Managed;
        assert!(!idle_unload_allowed(
            &ModelTargetReconciliationState::default(),
            Some(&candidate),
            Some("test")
        ));
        assert!(!idle_unload_allowed(&state, None, None));
    }
}

#[cfg(test)]
mod configuration_tests {
    use super::super::{RuntimeOptions, startup_models::build_startup_model_specs};
    use super::*;

    /// Each uniquely named entry has either a startup spec or a residency.
    #[test]
    fn unique_on_demand_entries_are_managed_and_startup_entries_remain_eager() {
        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "cold-model"
[models.lifecycle]
load = "on_demand"
[[models]]
model = "warm-model"
"#,
        )
        .unwrap();
        let residencies = configured_residencies(&config);
        let specs = build_startup_model_specs(&RuntimeOptions::default(), &config).unwrap();
        assert_eq!(
            residencies.keys().map(String::as_str).collect::<Vec<_>>(),
            ["cold-model"]
        );
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].config_model_id.as_deref(), Some("warm-model"));
    }

    /// Programmatic duplicate profiles must not fall between both launch paths.
    #[test]
    fn duplicate_model_profiles_keep_their_startup_specs() {
        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "shared-model"
ctx_size = 1024
[models.lifecycle]
load = "on_demand"
[[models]]
model = "shared-model"
ctx_size = 2048
[models.lifecycle]
load = "on_demand"
"#,
        )
        .unwrap();
        assert!(configured_residencies(&config).is_empty());
        let specs = build_startup_model_specs(&RuntimeOptions::default(), &config).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(
            specs.iter().map(|spec| spec.ctx_size).collect::<Vec<_>>(),
            [Some(1024), Some(2048)]
        );
        assert_ne!(specs[0].profile, specs[1].profile);
    }
}
