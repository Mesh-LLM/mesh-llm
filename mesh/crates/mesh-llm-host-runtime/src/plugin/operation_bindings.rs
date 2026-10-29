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
    /// Granted by the installed package's manifest rather than a config entry.
    installed_grant: bool,
}

#[derive(Default)]
pub(super) struct BindingRegistry(Mutex<BTreeMap<(String, String), Binding>>);

impl BindingRegistry {
    pub(super) fn from_specs(specs: &[ExternalPluginSpec]) -> anyhow::Result<Self> {
        let mut bindings = BTreeMap::new();
        let mut providers = std::collections::BTreeSet::new();
        for spec in specs {
            let installed_grants = spec
                .installed_metadata
                .as_ref()
                .map(super::installed::installed_operation_bindings)
                .unwrap_or_default();
            for grant in &spec.operation_bindings {
                grant.validate(&spec.name).map_err(anyhow::Error::msg)?;
                if grant.capability.is_none() {
                    anyhow::ensure!(
                        providers.insert(grant.provider.clone()),
                        "provider requires one exclusive binding"
                    );
                    if !specs.iter().any(|provider| provider.name == grant.provider) {
                        tracing::warn!(provider = %grant.provider, "bound provider unavailable at startup");
                    }
                }
                anyhow::ensure!(
                    bindings
                        .insert(
                            (spec.name.clone(), grant.name.clone()),
                            Binding {
                                grant: grant.clone(),
                                handle: None,
                                capacity: Arc::new(Semaphore::new(8)),
                                installed_grant: installed_grants.contains(grant),
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
    /// A grant from an installed package stands until a config entry for the
    /// caller names its own bindings (which replace it) or disables the plugin.
    pub(crate) fn revoke_changed_operation_bindings(&self, config: &MeshConfig) {
        self.inner
            .operation_bindings
            .0
            .lock()
            .unwrap()
            .retain(|(caller, _), binding| {
                let caller_entry = config.plugins.iter().find(|entry| entry.name == *caller);
                let granted = match caller_entry {
                    Some(entry) if entry.enabled == Some(false) => false,
                    Some(entry) if !entry.operation_bindings.is_empty() => {
                        entry.operation_bindings.contains(&binding.grant)
                    }
                    Some(_) | None => binding.installed_grant,
                };
                granted
                    && !config.plugins.iter().any(|entry| {
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
        let provider_name = match self.resolve_binding_provider(&key, &binding).await {
            Ok(name) => name,
            Err(response) => return response,
        };
        let Some(provider) = self.inner.plugins.get(&provider_name) else {
            return denied("provider unavailable");
        };
        let Some(generation) = provider.bound_generation().await else {
            return denied("provider not initialized");
        };
        let setup = request.handle.is_none();
        let (operation, input) = if setup {
            if !request.operation.is_empty() {
                return denied("setup accepts no caller operation");
            }
            let input = if binding.grant.setup_directory {
                match setup_directory_payload(&request.input_json, &provider_name) {
                    Ok(payload) => payload,
                    Err(error) => return denied(&format!("setup directory unavailable: {error}")),
                }
            } else if !request.input_json.is_empty() {
                return denied("setup accepts no caller payload");
            } else {
                binding.grant.setup_input_json.clone()
            };
            (binding.grant.setup_operation.clone(), input)
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
            (request.operation.clone(), request.input_json.clone())
        };
        if input.len() > 65_536 {
            return denied("operation input exceeds bound");
        }
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        let admission = match provider
            .admit_bound_operation(generation, &operation, &input, deadline)
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

impl PluginManager {
    /// The provider a binding reaches. Named bindings are fixed by the
    /// operator. A capability binding (from an installed package) resolves to
    /// the single running plugin serving that capability; the choice is kept
    /// for the registry's lifetime so a later restart cannot hand the binding
    /// to another provider. Two candidates are refused: the operator names one.
    async fn resolve_binding_provider(
        &self,
        key: &(String, String),
        binding: &Binding,
    ) -> Result<String, Response> {
        if !binding.grant.provider.is_empty() {
            return Ok(binding.grant.provider.clone());
        }
        let Some(capability) = binding.grant.capability.as_deref() else {
            return Err(denied("binding names no provider"));
        };
        let mut candidates: Vec<String> = self
            .capability_providers()
            .await
            .map_err(|error| denied(&format!("capability lookup failed: {error}")))?
            .into_iter()
            .filter(|provider| provider.capability == capability && provider.available)
            .map(|provider| provider.plugin_name)
            .filter(|name| name != &key.0)
            .collect();
        candidates.sort();
        candidates.dedup();
        match candidates.as_slice() {
            [only] => {
                let mut registry = self.inner.operation_bindings.0.lock().unwrap();
                if let Some(current) = registry
                    .get_mut(key)
                    .filter(|current| Arc::ptr_eq(&current.capacity, &binding.capacity))
                {
                    current.grant.provider = only.clone();
                }
                tracing::info!(caller = %key.0, binding = %key.1, provider = %only, capability, "Capability binding resolved");
                Ok(only.clone())
            }
            [] => Err(denied(&format!("no running plugin serves {capability}"))),
            several => Err(denied(&format!(
                "several plugins serve {capability} ({}); name one with [[plugin.operation_bindings]]",
                several.join(", ")
            ))),
        }
    }
}

/// Setup payload for a `setup_directory` binding. The caller names a base
/// directory it owns (`{"directory": <absolute base>}`); the host scopes it to
/// the resolved provider (`<base>/<provider>`) and creates it owner-only, so
/// the caller never learns or chooses which provider it was bound to, and each
/// provider keeps its own state (e.g. `payments/wallets/lexe-wallet`).
fn setup_directory_payload(caller_payload: &str, provider: &str) -> anyhow::Result<String> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Payload {
        directory: std::path::PathBuf,
    }
    let payload: Payload = serde_json::from_str(caller_payload).map_err(|error| {
        anyhow::anyhow!("setup payload must be {{\"directory\": <base>}}: {error}")
    })?;
    anyhow::ensure!(
        payload.directory.is_absolute(),
        "setup directory base must be an absolute path"
    );
    anyhow::ensure!(
        !provider.is_empty()
            && provider
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            && provider != "."
            && provider != "..",
        "'{provider}' cannot be used as a directory name"
    );
    let directory = payload.directory.join(provider);
    std::fs::create_dir_all(&directory)?;
    mesh_llm_config::restrict_private_directory(&directory)?;
    Ok(serde_json::json!({ "directory": directory.display().to_string() }).to_string())
}

fn denied(reason: &str) -> Response {
    // Callers retry refused setups silently for a while; the reason must be
    // visible to an operator running with debug logging.
    tracing::debug!(reason, "Bound operation refused before dispatch");
    Response::NotDispatched {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
