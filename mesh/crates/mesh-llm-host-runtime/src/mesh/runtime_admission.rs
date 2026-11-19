//! Per-port lifecycle and cached provider checks for local runtime requests.
use super::Node;
use crate::runtime::{InstanceAdmissionError, InstanceLifecycleRecord, InstanceRequestGuard};
use iroh::EndpointId;
use std::sync::{
    Arc, Weak,
    atomic::{AtomicBool, Ordering},
};

pub(crate) struct RuntimeInstanceAdmission {
    lifecycle: Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    providers: Option<Weak<RuntimeProviderDependencies>>,
    accepting: AtomicBool,
}

pub(crate) struct RuntimeProviderDependencies {
    node: Node,
    providers: Vec<EndpointId>,
}

impl RuntimeProviderDependencies {
    pub(crate) fn new(node: Node, providers: Vec<EndpointId>) -> Self {
        Self { node, providers }
    }

    async fn require_current(&self) -> Result<(), InstanceAdmissionError> {
        self.node
            .require_current_peer_evidence_set(&self.providers)
            .await
            .map_err(|_| InstanceAdmissionError::ProviderIneligible)
    }
}

impl Node {
    #[cfg(test)]
    pub(crate) fn register_runtime_instance_lifecycle(
        &self,
        port: u16,
        lifecycle: Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    ) {
        self.register_runtime_instance_admission(port, lifecycle, None);
    }

    pub(crate) fn register_runtime_instance_admission(
        &self,
        port: u16,
        lifecycle: Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
        providers: Option<&Arc<RuntimeProviderDependencies>>,
    ) {
        let mut instances = self
            .runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(previous) = instances.insert(
            port,
            Arc::new(RuntimeInstanceAdmission {
                lifecycle,
                // The runtime handle owns the coordinator context, avoiding
                // a Node -> admission registry -> Node ownership cycle.
                providers: providers.map(Arc::downgrade),
                accepting: AtomicBool::new(true),
            }),
        ) {
            previous.accepting.store(false, Ordering::Release);
        }
    }

    pub(crate) fn retire_runtime_instance_lifecycle(
        &self,
        port: u16,
        lifecycle: &Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    ) -> Option<Arc<RuntimeInstanceAdmission>> {
        let instance = self
            .runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&port)
            .filter(|instance| Arc::ptr_eq(&instance.lifecycle, lifecycle))
            .cloned()?;
        instance.accepting.store(false, Ordering::Release);
        Some(instance)
    }

    pub(crate) fn unregister_retired_runtime_instance(
        &self,
        port: u16,
        retired: Option<&Arc<RuntimeInstanceAdmission>>,
    ) {
        let Some(retired) = retired else {
            return;
        };
        let mut instances = self
            .runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if instances
            .get(&port)
            .is_some_and(|current| Arc::ptr_eq(current, retired))
        {
            instances.remove(&port);
        }
    }

    #[cfg(test)]
    pub(crate) fn unregister_runtime_instance_lifecycle(&self, port: u16) {
        if let Some(instance) = self
            .runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&port)
        {
            instance.accepting.store(false, Ordering::Release);
        }
    }

    pub(crate) async fn begin_runtime_instance_request(
        &self,
        port: u16,
    ) -> Result<Option<InstanceRequestGuard>, InstanceAdmissionError> {
        let instance = self
            .runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&port)
            .cloned();
        let Some(instance) = instance else {
            return Ok(None);
        };
        let residency = match self.native_model_lifecycle.get() {
            Some(service) => service.lease_port(port).await,
            None => None,
        };
        let record = instance.lifecycle.lock().await;
        if !record.is_accepting_work() {
            return Err(InstanceAdmissionError::NotAccepting {
                state: record.state(),
            });
        }
        if let Some(providers) = &instance.providers {
            providers
                .upgrade()
                .ok_or(InstanceAdmissionError::ProviderIneligible)?
                .require_current()
                .await?;
        }
        if !instance.accepting.load(Ordering::Acquire) {
            return Err(InstanceAdmissionError::Retired);
        }
        Ok(Some(
            InstanceRequestGuard::new(record.in_flight_tracker()).with_residency(residency),
        ))
    }
}
