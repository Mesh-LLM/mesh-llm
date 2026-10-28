//! Operator-fixed, exclusive local provider connections. No domain policy.
use super::{PluginManager, config::ExternalPluginSpec};
use mesh_llm_config::{MeshConfig, PluginOperationBinding};
use mesh_llm_plugin::operation_binding::{
    BoundOperationRequest, BoundOperationResponse as Response,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

#[derive(Clone)]
struct Binding {
    grant: PluginOperationBinding,
    handle: Option<(String, u64, u64)>,
    capacity: Arc<Semaphore>,
}

#[derive(Default)]
pub(super) struct BindingRegistry(Mutex<BTreeMap<(String, String), Binding>>);

impl BindingRegistry {
    pub(super) fn from_specs(specs: &[ExternalPluginSpec]) -> anyhow::Result<Self> {
        let mut bindings = BTreeMap::new();
        let mut providers = std::collections::BTreeSet::new();
        for spec in specs {
            for grant in &spec.operation_bindings {
                grant.validate(&spec.name).map_err(anyhow::Error::msg)?;
                anyhow::ensure!(
                    providers.insert(grant.provider.clone()),
                    "provider requires one exclusive binding"
                );
                if !specs.iter().any(|provider| provider.name == grant.provider) {
                    tracing::warn!(provider = %grant.provider, "bound provider unavailable at startup");
                }
                anyhow::ensure!(
                    bindings
                        .insert(
                            (spec.name.clone(), grant.name.clone()),
                            Binding {
                                grant: grant.clone(),
                                handle: None,
                                capacity: Arc::new(Semaphore::new(8)),
                            }
                        )
                        .is_none(),
                    "duplicate binding"
                );
            }
        }
        Ok(Self(Mutex::new(bindings)))
    }
}

impl PluginManager {
    /// Changed grants are revoked immediately. New grants take effect on restart.
    pub(crate) fn revoke_changed_operation_bindings(&self, config: &MeshConfig) {
        self.inner
            .operation_bindings
            .0
            .lock()
            .unwrap()
            .retain(|(caller, _), binding| {
                config.plugins.iter().any(|entry| {
                    entry.name == *caller
                        && entry.enabled != Some(false)
                        && entry.operation_bindings.contains(&binding.grant)
                }) && !config.plugins.iter().any(|entry| {
                    entry.name == binding.grant.provider && entry.enabled == Some(false)
                })
            });
    }

    pub(crate) async fn bound_operation(
        &self,
        caller: &str,
        caller_generation: u64,
        request: BoundOperationRequest,
    ) -> Response {
        let key = (caller.to_owned(), request.binding.clone());
        let Some(binding) = self
            .inner
            .operation_bindings
            .0
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
        else {
            return denied("binding is not authorized");
        };
        let Ok(_capacity) = binding
            .capacity
            .clone()
            .try_acquire_many_owned(if request.handle.is_none() { 8 } else { 1 })
        else {
            return denied("binding capacity exhausted");
        };
        let Some(caller_plugin) = self.inner.plugins.get(caller) else {
            return denied("caller unavailable");
        };
        if caller_plugin.bound_generation().await != Some(caller_generation) {
            return denied("caller generation unavailable");
        }
        let Some(provider) = self.inner.plugins.get(&binding.grant.provider) else {
            return denied("provider unavailable");
        };
        let Some(generation) = provider.bound_generation().await else {
            return denied("provider not initialized");
        };
        let setup = request.handle.is_none();
        let (operation, input) = if setup {
            if !request.operation.is_empty() || !request.input_json.is_empty() {
                return denied("setup accepts no caller payload");
            }
            (
                &binding.grant.setup_operation,
                &binding.grant.setup_input_json,
            )
        } else {
            if !binding.grant.operations.contains(&request.operation) {
                return denied("operation not allowed");
            }
            if !binding
                .handle
                .as_ref()
                .is_some_and(|(handle, owner, target)| {
                    Some(handle) == request.handle.as_ref()
                        && *owner == caller_generation
                        && *target == generation
                })
            {
                return denied("handle requires setup and identity revalidation");
            }
            (&request.operation, &request.input_json)
        };
        if input.len() > 65_536 {
            return denied("operation input exceeds bound");
        }
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        let admission = match provider
            .admit_bound_operation(generation, operation, input, deadline)
            .await
        {
            Ok(admission) => admission,
            Err(response) => return response,
        };
        let dispatched = {
            let mut registry = self.inner.operation_bindings.0.lock().unwrap();
            let Some(current) = registry
                .get_mut(&key)
                .filter(|current| Arc::ptr_eq(&current.capacity, &binding.capacity))
            else {
                return denied("binding revoked");
            };
            if setup {
                current.handle = None;
            }
            admission.send()
        };
        let mut response = dispatched.response(deadline).await;
        if let Response::Responded {
            handle, is_error, ..
        } = &mut response
        {
            let mut registry = self.inner.operation_bindings.0.lock().unwrap();
            let Some(current) = registry
                .get_mut(&key)
                .filter(|current| Arc::ptr_eq(&current.capacity, &binding.capacity))
            else {
                return Response::Uncertain {
                    reason: "binding revoked during operation".into(),
                };
            };
            if setup && !*is_error {
                let value = uuid::Uuid::new_v4().to_string();
                current.handle = Some((value.clone(), caller_generation, generation));
                *handle = value;
            } else if !setup {
                *handle = request.handle.unwrap_or_default();
            }
        }
        response
    }
}

fn denied(reason: &str) -> Response {
    Response::NotDispatched {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
