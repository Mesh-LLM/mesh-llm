//! Request leases for opt-in native residency. The reconciler owns execution.
use std::{collections::BTreeMap, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use mesh_llm_config::managed_models::{ModelLifecycle, ModelLoadPolicy};
use tokio::sync::{
    Mutex, OwnedMutexGuard, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, mpsc, watch,
};

pub(in crate::runtime) mod sources;
type LoadResult = Option<Result<(), String>>;

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
    #[cfg(test)]
    gate_wait_checkpoint: std::sync::Mutex<Option<mpsc::UnboundedSender<&'static str>>>,
}

struct Residency {
    policy: ModelLifecycle,
    spec: String,
    names: Vec<String>,
    loading: Mutex<Option<watch::Receiver<LoadResult>>>,
    gate: Arc<RwLock<()>>,
    operation: Arc<Mutex<()>>,
    idle_since: std::sync::Mutex<std::time::Instant>,
    instance: std::sync::Mutex<Option<(String, String, u16)>>,
}

tokio::task_local! {
    static CURRENT_REQUEST_LEASE: RequestLease;
}

#[derive(Clone)]
pub(crate) struct RequestLease {
    inner: Arc<RequestLeaseInner>,
}

struct RequestLeaseInner {
    residency: Arc<Residency>,
    _gate: OwnedRwLockReadGuard<()>,
}

impl RequestLease {
    fn new(residency: Arc<Residency>, gate: OwnedRwLockReadGuard<()>) -> Self {
        Self {
            inner: Arc::new(RequestLeaseInner {
                residency,
                _gate: gate,
            }),
        }
    }
}

/// Run `future` with `lease` visible to nested `lease_port` calls on this task
/// so selected-instance admission reuses the existing read lock instead of
/// acquiring a second read on the same write-preferring `RwLock`.
pub(crate) async fn with_request_lease<F: std::future::Future>(
    lease: Option<RequestLease>,
    future: F,
) -> F::Output {
    match lease {
        Some(lease) => CURRENT_REQUEST_LEASE.scope(lease, future).await,
        None => future.await,
    }
}

fn current_request_lease_for(residency: &Arc<Residency>) -> Option<RequestLease> {
    CURRENT_REQUEST_LEASE
        .try_with(|lease| Arc::ptr_eq(&lease.inner.residency, residency).then(|| lease.clone()))
        .ok()
        .flatten()
}

/// Exclusive local GPU lease held while forwarding to an `exclusive_gpu`
/// inference plugin. Unloads all resident on-demand native models before
/// returning and blocks native reloads until dropped.
#[derive(Debug)]
pub(crate) struct ExclusiveGpuHandoffLease {
    _gates: Vec<OwnedRwLockWriteGuard<()>>,
}

impl Drop for RequestLeaseInner {
    fn drop(&mut self) {
        *self
            .residency
            .idle_since
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = std::time::Instant::now();
    }
}

impl NativeModelLifecycle {
    #[cfg(test)]
    pub(crate) fn test_service(
        config: &crate::plugin::MeshConfig,
    ) -> (
        Arc<Self>,
        mpsc::Receiver<ModelIntent>,
        watch::Sender<ModelTargets>,
    ) {
        let (intents, receiver) = mpsc::channel(8);
        let (targets, snapshot) = watch::channel(ModelTargets::default());
        (
            Arc::new(Self {
                models: configured_residencies(config),
                targets: snapshot,
                intents,
                gate_wait_checkpoint: std::sync::Mutex::new(None),
            }),
            receiver,
            targets,
        )
    }

    #[cfg(test)]
    fn install_gate_wait_checkpoint(&self) -> mpsc::UnboundedReceiver<&'static str> {
        let (tx, rx) = mpsc::unbounded_channel();
        *self
            .gate_wait_checkpoint
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(tx);
        rx
    }

    #[cfg(test)]
    async fn await_gate_with_checkpoint<F>(&self, fut: F, label: &'static str) -> F::Output
    where
        F: std::future::Future,
    {
        let checkpoint = self
            .gate_wait_checkpoint
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut fut = std::pin::pin!(fut);
        let mut notified = false;
        std::future::poll_fn(|cx| {
            let result = fut.as_mut().poll(cx);
            if !notified {
                notified = true;
                if let Some(tx) = &checkpoint {
                    let _ = tx.send(label);
                }
            }
            result
        })
        .await
    }

    /// Resolve a configured identifier or loaded runtime alias to shared state.
    fn residency(&self, model: &str) -> Option<(&String, &Arc<Residency>)> {
        self.models.get_key_value(model).or_else(|| {
            let mut matches = self.models.iter().filter(|(_, residency)| {
                residency
                    .instance
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .is_some_and(|(_, loaded, _)| loaded == model)
                    || residency.names.iter().any(|name| name == model)
            });
            let first = matches.next()?;
            matches.next().is_none().then_some(first)
        })
    }

    /// Report whether requests for this name need a residency lease.
    pub(crate) fn manages(&self, model: &str) -> bool {
        self.residency(model).is_some()
    }

    /// List configured on-demand model identifiers, including cold models.
    pub(crate) fn model_names(&self) -> impl Iterator<Item = &String> {
        self.models
            .values()
            .flat_map(|residency| residency.names.iter())
            .filter(|name| self.manages(name))
    }

    /// Add configured-name aliases to the current runtime routing snapshot.
    pub(crate) fn targets(&self) -> ModelTargets {
        let mut targets = ModelTargets::default();
        for (model, residency) in &self.models {
            if let Some((_, loaded, port)) = residency
                .instance
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                && self
                    .targets
                    .borrow()
                    .candidates(loaded)
                    .contains(&InferenceTarget::Local(*port))
            {
                for name in residency.names.iter().chain([model, loaded]) {
                    targets
                        .targets
                        .insert(name.clone(), vec![InferenceTarget::Local(*port)]);
                }
            }
        }
        targets
    }

    /// Restrict dispatch to the exact instance owned by this configured identity.
    pub(crate) fn targets_for(&self, model: &str) -> ModelTargets {
        let mut targets = ModelTargets::default();
        if let Some((key, residency)) = self.residency(model)
            && let Some((_, loaded, port)) = residency
                .instance
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
            && self
                .targets
                .borrow()
                .candidates(loaded)
                .contains(&InferenceTarget::Local(*port))
        {
            for name in residency.names.iter().chain([key, loaded]) {
                targets
                    .targets
                    .insert(name.clone(), vec![InferenceTarget::Local(*port)]);
            }
        }
        targets
    }

    /// Bind ownership before load completion is delivered to waiting callers.
    pub(crate) fn record_loaded(
        &self,
        model: &str,
        instance_id: &str,
        loaded_name: &str,
        port: u16,
    ) {
        if let Some((_, residency)) = self.residency(model) {
            *residency.instance.lock().unwrap_or_else(|e| e.into_inner()) =
                Some((instance_id.to_owned(), loaded_name.to_owned(), port));
        }
    }

    async fn acquire_gate_read(
        &self,
        residency: &Residency,
        #[cfg_attr(not(test), allow(unused_variables))] label: &'static str,
    ) -> OwnedRwLockReadGuard<()> {
        let fut = residency.gate.clone().read_owned();
        #[cfg(test)]
        {
            self.await_gate_with_checkpoint(fut, label).await
        }
        #[cfg(not(test))]
        {
            fut.await
        }
    }

    async fn acquire_gate_write(
        &self,
        residency: &Residency,
        #[cfg_attr(not(test), allow(unused_variables))] label: &'static str,
    ) -> OwnedRwLockWriteGuard<()> {
        let fut = residency.gate.clone().write_owned();
        #[cfg(test)]
        {
            self.await_gate_with_checkpoint(fut, label).await
        }
        #[cfg(not(test))]
        {
            fut.await
        }
    }

    /// Lease the actual selected local port, including routes bypassing HTTP ingress.
    pub(crate) async fn lease_port(&self, port: u16) -> Option<RequestLease> {
        let residency = self
            .models
            .values()
            .find(|r| {
                r.instance
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .is_some_and(|(_, _, p)| *p == port)
            })?
            .clone();
        if let Some(existing) = current_request_lease_for(&residency) {
            let still_owned = residency
                .instance
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_some_and(|(_, _, p)| *p == port);
            return still_owned.then_some(existing);
        }
        let gate = self.acquire_gate_read(&residency, "lease_port_read").await;
        let still_owned = residency
            .instance
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|(_, _, p)| *p == port);
        still_owned.then_some(RequestLease::new(residency, gate))
    }

    /// Keep this lease until dispatch (including streaming) has finished.
    pub(crate) async fn acquire(self: &Arc<Self>, model: &str) -> Result<Option<RequestLease>> {
        let Some((model, residency)) = self.residency(model) else {
            return Ok(None);
        };
        let model = model.clone();
        let residency = residency.clone();
        tokio::time::timeout(LOAD_TIMEOUT, async {
            loop {
                let gate = self.acquire_gate_read(&residency, "acquire_read").await;
                if ready(&self.targets(), &model, true) {
                    return Ok(Some(RequestLease::new(residency, gate)));
                }
                // Release the read lock before loading so a waiting write lock
                // (`acquire_exclusive_gpu_handoff`) cannot deadlock with
                // `ensure_loaded`'s inner read acquisition under writer-preferring RwLock.
                drop(gate);
                self.ensure_loaded(model.clone(), residency.clone()).await?;
            }
        })
        .await
        .context("native model residency wait timed out")?
    }

    /// Wait for in-flight on-demand native requests to drain, unload any loaded
    /// on-demand native instances immediately, and hold exclusive write gates so
    /// native models stay unloaded until the plugin request completes.
    pub(crate) async fn acquire_exclusive_gpu_handoff(
        self: &Arc<Self>,
    ) -> Result<ExclusiveGpuHandoffLease> {
        self.acquire_exclusive_gpu_handoff_with_timeout(LOAD_TIMEOUT)
            .await
    }

    async fn acquire_exclusive_gpu_handoff_with_timeout(
        self: &Arc<Self>,
        timeout: Duration,
    ) -> Result<ExclusiveGpuHandoffLease> {
        tokio::time::timeout(timeout, async {
            let mut gates = Vec::with_capacity(self.models.len());
            for (model, residency) in &self.models {
                let operation = residency.operation.clone().lock_owned().await;
                let gate = self
                    .acquire_gate_write(residency, "exclusive_gpu_handoff_write")
                    .await;
                let instance = residency
                    .instance
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                let gate = if let Some((instance_id, _, _)) = instance {
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
                    self.intents
                        .send(intent)
                        .await
                        .context("native model lifecycle unavailable for plugin GPU handoff")?;
                    // Once the unload intent is accepted by the control loop,
                    // transfer ownership of the write gate and operation lock
                    // to a detached task so timeout or caller cancellation
                    // keeps the gate held until unload finishes and still
                    // clears the matching instance record.
                    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                    let residency_for_task = residency.clone();
                    let model_name = model.clone();
                    tokio::spawn(async move {
                        let _operation = operation;
                        let outcome = result
                            .await
                            .context("native model unload interrupted during plugin GPU handoff")
                            .and_then(|unload_result| {
                                unload_result.with_context(|| {
                                    format!(
                                        "failed to unload native model '{model_name}' for exclusive_gpu plugin handoff"
                                    )
                                })
                            });
                        if outcome.is_ok() {
                            let mut slot = residency_for_task
                                .instance
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            if slot
                                .as_ref()
                                .is_some_and(|(id, _, _)| id == &instance_id)
                            {
                                *slot = None;
                            }
                        }
                        let _ = done_tx.send(outcome.map(|_| gate));
                    });
                    done_rx
                        .await
                        .context("native model unload task stopped during plugin GPU handoff")??
                } else {
                    drop(operation);
                    gate
                };
                gates.push(gate);
            }
            Ok(ExclusiveGpuHandoffLease { _gates: gates })
        })
        .await
        .context("exclusive GPU handoff wait timed out")?
    }

    /// Spawn only the real load; cancelled callers leave no queued acquisition jobs.
    async fn ensure_loaded(
        self: &Arc<Self>,
        model: String,
        residency: Arc<Residency>,
    ) -> Result<()> {
        let mut slot = residency.loading.lock().await;
        let mut result = match slot
            .as_ref()
            .filter(|result| result.borrow().is_none() && result.has_changed().is_ok())
        {
            Some(result) => result.clone(),
            None => {
                let operation = residency.operation.clone().lock_owned().await;
                if ready(&self.targets(), &model, true) {
                    return Ok(());
                }
                let gate = residency.gate.clone().read_owned().await;
                let (sender, result) = watch::channel(None);
                *slot = Some(result.clone());
                let lifecycle = self.clone();
                let residency = residency.clone();
                tokio::spawn(async move {
                    let _gate = gate;
                    let loaded = lifecycle
                        .load(&model, &residency, operation)
                        .await
                        .map_err(|e| format!("{e:#}"));
                    sender.send_replace(Some(loaded));
                });
                result
            }
        };
        drop(slot);
        loop {
            if let Some(loaded) = result.borrow_and_update().clone() {
                return loaded.map_err(anyhow::Error::msg);
            }
            result
                .changed()
                .await
                .context("native model load task stopped")?;
        }
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
                    .map(|(_, name, _)| name.clone())
            })
            .filter(|name| ready(&self.targets_for(model), name, true))
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
                sizing: crate::api::RuntimeLoadSizing::default(),
                completion: Some(completion),
            })
            .await
            .context("native model lifecycle unavailable")?;
        let loaded = result.await.context("native model load interrupted")??;
        anyhow::ensure!(
            residency
                .instance
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_some_and(|(id, _, _)| *id == loaded.instance_id),
            "native instance did not register its serving port"
        );
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
            let Some((instance_id, _, _)) = instance.as_ref() else {
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
    model.provider.is_none()
        && model.lifecycle.load == ModelLoadPolicy::OnDemand
        && config
            .models
            .iter()
            .filter(|entry| entry.provider.is_none() && entry.model == model.model)
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
                    names: sources::model_names(config, model),
                    loading: Mutex::new(None),
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
) -> Result<()> {
    sources::validate_managed_sources(config)?;
    let Some(intents) = node.model_intent_tx.lock().await.clone() else {
        return Ok(());
    };
    let models = configured_residencies(config);
    let lifecycle = Arc::new(NativeModelLifecycle {
        models,
        targets,
        intents,
        #[cfg(test)]
        gate_wait_checkpoint: std::sync::Mutex::new(None),
    });
    if node.native_model_lifecycle.set(lifecycle.clone()).is_err() {
        return Ok(());
    }
    sources::spawn_prefetch(config, &lifecycle.models)?;
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
    Ok(())
}

#[cfg(test)]
mod tests;
