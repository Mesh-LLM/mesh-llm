//! Single lifetime owner of native weights and aggregate serving capacity.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use skippy_inference_api::InferenceBackend;
use tokio::sync::{Mutex, watch};

use crate::inference::skippy::{SkippyModelHandle, SkippyModelLoadOptions};
use crate::mesh::{
    ModelSourceKind, ServedModelDescriptor, ServedModelIdentity, ServedModelMetadata,
};

use super::admission::Admission;
use super::configuration::{
    ServingMembershipConfig, SharedServingConfig, private_directory, private_file,
};
use super::membership::{AttachmentInputs, Membership, MembershipPhase, MembershipStatus};

#[cfg(test)]
mod tests;

struct MemberRecord {
    config: ServingMembershipConfig,
    attachment: Mutex<Option<Membership>>,
    status: watch::Sender<MembershipStatus>,
}

/// A dropped management request must not leave a membership permanently
/// reporting "joining" after its partially built attachment has been dropped.
struct JoiningStatus<'a> {
    status: &'a watch::Sender<MembershipStatus>,
    owners: &'a std::sync::Mutex<BTreeMap<String, String>>,
    name: &'a str,
}

impl Drop for JoiningStatus<'_> {
    /// Mark an interrupted join as failed and release only its mesh ownership claim.
    fn drop(&mut self) {
        self.status.send_modify(|status| {
            if matches!(status.phase, MembershipPhase::Joining) {
                self.owners
                    .lock()
                    .expect("mesh ownership lock poisoned")
                    .retain(|_, owner| owner != self.name);
                status.phase = MembershipPhase::Failed;
                status.last_error = Some("Membership join interrupted; retry to reconnect.".into());
            }
        });
    }
}

#[derive(Serialize)]
pub struct SharedServingStatus {
    pub model_id: String,
    pub loaded_models: usize,
    pub shared_concurrency: usize,
    pub shared_queue_capacity: usize,
    pub active_requests: usize,
    pub queued_requests: usize,
    pub runtime_id: Option<String>,
    pub native_lane_count: Option<u32>,
    pub memberships: Vec<MembershipStatus>,
}

/// Dropping or pausing one attachment never drops the model. Only this host
/// owns its lifetime; mesh model-election and stage-control loops are absent.
pub struct SharedServingHost {
    config: SharedServingConfig,
    descriptor: ServedModelDescriptor,
    backend: Arc<dyn InferenceBackend>,
    admission: Arc<Admission>,
    members: BTreeMap<String, MemberRecord>,
    mesh_owners: std::sync::Mutex<BTreeMap<String, String>>,
    ingress_slots: Arc<tokio::sync::Semaphore>,
    stopping: AtomicBool,
    _native: Option<SkippyModelHandle>,
    _state_lock: std::fs::File,
}

impl SharedServingHost {
    /// Validate inputs, lock the private state directory, and load one local GGUF.
    ///
    /// Memberships remain detached until connected explicitly. Runtime discovery uses
    /// installed or bundled native code; this path never downloads a backend.
    pub async fn load(config: SharedServingConfig) -> Result<Arc<Self>> {
        config.validate()?;
        let lock = lock_state(&config)?;
        let mut options =
            SkippyModelLoadOptions::for_direct_gguf(&config.model_id, &config.model_path)
                .with_ctx_size(config.context_size)
                .with_generation_concurrency(config.concurrency);
        options.n_gpu_layers = config.gpu_layers;
        let native = tokio::task::spawn_blocking(move || {
            // This opt-in CLI/SDK path does not run ordinary mesh startup.
            // Resolve only an installed/bundled runtime; never inherit another
            // mesh's model configuration or silently download a new backend.
            #[cfg(feature = "dynamic-native-runtime")]
            crate::system::native_runtime::load_local_native_runtime_for_embedded_serving(
                &crate::system::native_runtime::NativeRuntimeStartupSelection::current(),
            )?;
            SkippyModelHandle::load(options)
        })
        .await
        .context("shared native model worker failed")??;
        let descriptor = ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: config.model_id.clone(),
                source_kind: ModelSourceKind::LocalGguf,
                is_primary: true,
                ..Default::default()
            },
            metadata: Some(ServedModelMetadata {
                workload_class: Some(native.workload_class()?),
                native_context_length: Some(config.context_size),
                ..Default::default()
            }),
            ..Default::default()
        };
        Self::assemble(config, descriptor, native.backend(), Some(native), lock)
    }

    /// Combine a backend, state lock and detached memberships under one capacity owner.
    fn assemble(
        config: SharedServingConfig,
        descriptor: ServedModelDescriptor,
        backend: Arc<dyn InferenceBackend>,
        native: Option<SkippyModelHandle>,
        lock: std::fs::File,
    ) -> Result<Arc<Self>> {
        let admission = Admission::new(config.concurrency, config.queue_capacity)?;
        let ingress_slots = Arc::new(tokio::sync::Semaphore::new(
            config.concurrency + config.queue_capacity + config.memberships.len(),
        ));
        let members = config
            .memberships
            .iter()
            .map(|config| {
                let (status, _) = watch::channel(MembershipStatus {
                    name: config.name.clone(),
                    phase: MembershipPhase::Detached,
                    endpoint_id: None,
                    mesh_id: None,
                    peers: 0,
                    admission: Default::default(),
                    last_error: None,
                });
                (
                    config.name.clone(),
                    MemberRecord {
                        config: config.clone(),
                        attachment: Mutex::new(None),
                        status,
                    },
                )
            })
            .collect();
        Ok(Arc::new(Self {
            config,
            descriptor,
            backend,
            admission,
            members,
            mesh_owners: std::sync::Mutex::default(),
            ingress_slots,
            stopping: AtomicBool::new(false),
            _native: native,
            _state_lock: lock,
        }))
    }

    /// Join independently, requiring at least one connected membership at completion.
    /// Partial failures remain visible and can be retried without stopping healthy meshes.
    pub async fn connect_all(self: &Arc<Self>) -> Result<()> {
        let joins = self.members.keys().map(|name| async move {
            if let Err(error) = self.reconnect(name).await {
                tracing::warn!(membership = name, %error, "shared serving membership unavailable");
            }
        });
        futures_util::future::join_all(joins).await;
        let connected = self
            .members
            .values()
            .filter(|member| matches!(member.status.borrow().phase, MembershipPhase::Connected))
            .count();
        let total = self.members.len();
        anyhow::ensure!(
            connected > 0,
            "shared serving startup failed: no memberships connected (0/{total}); inspect membership diagnostics"
        );
        let message = format!("Shared serving ready: {connected}/{total} memberships connected");
        let event = if connected == total {
            mesh_llm_events::OutputEvent::Info {
                message,
                context: None,
            }
        } else {
            mesh_llm_events::OutputEvent::Warning {
                message: format!("{message}; retry failed memberships through the control API"),
                context: None,
            }
        };
        let _ = mesh_llm_events::emit_event(event);
        Ok(())
    }

    /// Replace one attachment after rereading its token, trust and binary proof.
    ///
    /// Retain its persisted endpoint identity but allocate a fresh execution scope.
    /// Reject duplicate mesh attachments and joins racing with host shutdown;
    /// a failed or interrupted join leaves the membership available for retry.
    pub async fn reconnect(&self, name: &str) -> Result<()> {
        let record = self.record(name)?;
        let mut attachment = record.attachment.lock().await;
        anyhow::ensure!(
            !self.stopping.load(Ordering::Acquire),
            "shared host is stopping"
        );
        record.status.send_modify(|status| {
            status.phase = MembershipPhase::Joining;
            status.endpoint_id = None;
            status.mesh_id = None;
            status.last_error = None;
            status.peers = 0;
            status.admission = Default::default();
        });
        let _joining = JoiningStatus {
            status: &record.status,
            owners: &self.mesh_owners,
            name,
        };
        if let Some(previous) = attachment.take() {
            previous.close().await;
        }
        self.mesh_owners
            .lock()
            .expect("mesh ownership lock poisoned")
            .retain(|_, owner| owner != name);
        let result = Membership::attach(AttachmentInputs {
            release_attestation: crate::runtime::release_attestation::load_for_current_binary(),
            config: &record.config,
            state_dir: &self.config.state_dir,
            backend: self.backend.clone(),
            descriptor: self.descriptor.clone(),
            admission: self.admission.clone(),
            queue_capacity: self.config.membership_queue_capacity,
            lan_only: self.config.lan_only,
            bind_ip: self.config.bind_ip,
            ingress_slots: self.ingress_slots.clone(),
            ingress_capacity: self.config.concurrency
                + self
                    .config
                    .membership_queue_capacity
                    .min(self.config.queue_capacity)
                + 1,
        })
        .await;
        match result {
            Ok(joined) => {
                if self.stopping.load(Ordering::Acquire) {
                    joined.close().await;
                    anyhow::bail!("shared host is stopping");
                }
                let endpoint_id = joined.node.id().to_string();
                let mesh_id = joined.node.mesh_id().await;
                if let Some(mesh_id) = &mesh_id {
                    let duplicate = {
                        let mut owners = self
                            .mesh_owners
                            .lock()
                            .expect("mesh ownership lock poisoned");
                        if let std::collections::btree_map::Entry::Vacant(entry) =
                            owners.entry(mesh_id.clone())
                        {
                            entry.insert(name.to_owned());
                            false
                        } else {
                            true
                        }
                    };
                    if duplicate {
                        joined.close().await;
                        record.status.send_modify(|status| {
                            status.phase = MembershipPhase::Failed;
                            status.last_error =
                                Some("This mesh already has an attachment to this host.".into());
                        });
                        anyhow::bail!("mesh already attached to this host");
                    }
                }
                joined.advertise(true).await;
                let peers = joined.node.peers().await.len();
                record.status.send_modify(|status| {
                    status.phase = MembershipPhase::Connected;
                    status.endpoint_id = Some(endpoint_id);
                    status.mesh_id = mesh_id;
                    status.peers = peers;
                });
                *attachment = Some(joined);
                Ok(())
            }
            Err(error) => {
                record.status.send_modify(|status| {
                    status.phase = MembershipPhase::Failed;
                    status.endpoint_id = None;
                    status.mesh_id = None;
                    status.last_error =
                        Some("Membership join failed; inspect local diagnostics and retry.".into());
                });
                Err(error)
            }
        }
    }

    /// Stop new and queued work for one membership and withdraw its model offer.
    /// Already admitted execution may finish; the model stays loaded.
    pub async fn pause(&self, name: &str) -> Result<()> {
        let record = self.record(name)?;
        let attachment = record.attachment.lock().await;
        let attachment = attachment.as_ref().context("membership is not connected")?;
        record
            .status
            .send_modify(|status| status.phase = MembershipPhase::Paused);
        // Admission changes synchronously before advertisement can suspend.
        // Cancellation during gossip must not leave the old effective phase.
        attachment.pause().await;
        Ok(())
    }

    /// Reopen a connected or paused membership and advertise its model offer.
    /// Detached memberships must reconnect; a stopping host cannot resume.
    pub async fn resume(&self, name: &str) -> Result<()> {
        anyhow::ensure!(
            !self.stopping.load(Ordering::Acquire),
            "shared host is stopping"
        );
        let record = self.record(name)?;
        let attachment = record.attachment.lock().await;
        let attachment = attachment
            .as_ref()
            .context("membership is detached; reconnect first")?;
        anyhow::ensure!(
            !self.stopping.load(Ordering::Acquire),
            "shared host is stopping"
        );
        record
            .status
            .send_modify(|status| status.phase = MembershipPhase::Connected);
        attachment.resume().await;
        Ok(())
    }

    /// Detach one membership, cancel its requests and withdraw its model offer.
    /// Other memberships keep serving, and the host retains ownership of the model.
    pub async fn leave(&self, name: &str) -> Result<()> {
        let record = self.record(name)?;
        let mut attachment = record.attachment.lock().await;
        let previous = attachment.take();
        self.mesh_owners
            .lock()
            .expect("mesh ownership lock poisoned")
            .retain(|_, owner| owner != name);
        record.status.send_modify(|status| {
            status.phase = MembershipPhase::Detached;
            status.peers = 0;
            status.admission = Default::default();
        });
        if let Some(previous) = previous {
            previous.close().await;
        }
        Ok(())
    }

    /// Snapshot aggregate capacity and membership state without waiting for a join.
    /// Membership counters may reflect the last published state during a lifecycle operation.
    pub async fn status(&self) -> SharedServingStatus {
        let mut memberships = Vec::with_capacity(self.members.len());
        for record in self.members.values() {
            let mut status = record.status.borrow().clone();
            if let Ok(attachment) = record.attachment.try_lock()
                && let Some(attachment) = attachment.as_ref()
            {
                status.admission = self.admission.status(attachment.member);
                status.peers = attachment.node.peers().await.len();
            }
            memberships.push(status);
        }
        let (active_requests, queued_requests) = self.admission.totals();
        let native = self._native.as_ref().map(SkippyModelHandle::status);
        SharedServingStatus {
            model_id: self.config.model_id.clone(),
            // Shutdown disables execution, but the host still owns the weights
            // until its final Arc and backend references are dropped.
            loaded_models: 1,
            shared_concurrency: self.config.concurrency,
            shared_queue_capacity: self.config.queue_capacity,
            active_requests,
            queued_requests,
            runtime_id: native.as_ref().map(|native| native.run_id.clone()),
            native_lane_count: native.map(|native| native.lane_count),
            memberships,
        }
    }

    /// Prevent new joins, detach every membership and wait up to 30 seconds for workers.
    /// Only after leases drain is native execution shut down; a timeout returns an error.
    pub async fn shutdown(&self) -> Result<()> {
        self.stopping.store(true, Ordering::Release);
        futures_util::future::try_join_all(self.members.keys().map(|name| self.leave(name)))
            .await?;
        tokio::time::timeout(Duration::from_secs(30), async {
            while self.admission.active() > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("native requests are still cancelling; shared host has not drained")?;
        if let Some(native) = &self._native {
            native.shutdown();
        }
        Ok(())
    }

    /// Look up an exact configured membership name, rejecting unknown labels.
    fn record(&self, name: &str) -> Result<&MemberRecord> {
        self.members
            .get(name)
            .context("unknown shared-serving membership")
    }
}

/// Hold an exclusive process-lifetime lock in the private state directory.
fn lock_state(config: &SharedServingConfig) -> Result<std::fs::File> {
    private_directory(&config.state_dir)?;
    let lock = private_file(&config.state_dir.join("host.lock"))?;
    lock.try_lock()
        .context("shared-serving state directory is already in use")?;
    Ok(lock)
}
