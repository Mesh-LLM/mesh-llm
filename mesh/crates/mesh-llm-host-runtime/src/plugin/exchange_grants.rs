//! Operator grant revisions revoke in-flight observation and signing authority.

use mesh_llm_config::{MeshConfig, OpenAiExchangeGrant};
use std::collections::BTreeMap;
use std::sync::Mutex;
use tokio::sync::watch;

struct GrantState {
    effective: Option<OpenAiExchangeGrant>,
    revision: watch::Sender<u64>,
}

impl GrantState {
    fn new(effective: Option<OpenAiExchangeGrant>) -> Self {
        Self {
            effective,
            revision: watch::channel(0).0,
        }
    }
    fn replace(&mut self, effective: Option<OpenAiExchangeGrant>) -> bool {
        if self.effective == effective {
            return false;
        }
        self.effective = effective;
        self.revision
            .send_modify(|revision| *revision = revision.wrapping_add(1));
        true
    }
}

#[derive(Default)]
pub(super) struct ExchangeGrantRegistry(Mutex<BTreeMap<String, GrantState>>);
impl ExchangeGrantRegistry {
    pub(super) fn from_specs(specs: &[super::config::ExternalPluginSpec]) -> Self {
        Self(Mutex::new(
            specs
                .iter()
                .map(|spec| {
                    (
                        spec.name.clone(),
                        GrantState::new(spec.openai_exchange_grant.as_deref().cloned()),
                    )
                })
                .collect(),
        ))
    }
}

impl super::PluginManager {
    pub(super) fn exchange_grant_snapshot(&self) -> BTreeMap<String, OpenAiExchangeGrant> {
        let states = self.inner.exchange_grants.0.lock().unwrap();
        let mut grants: BTreeMap<_, _> = states
            .iter()
            .filter_map(|(name, state)| state.effective.clone().map(|grant| (name.clone(), grant)))
            .collect();
        for (name, plugin) in &self.inner.plugins {
            if !states.contains_key(name)
                && let Some(grant) = plugin.exchange_grant()
            {
                grants.insert(name.clone(), grant.clone());
            }
        }
        grants
    }
    /// Subscribe before obtaining a grant snapshot, so a concurrent revocation
    /// is observable even if it occurs between authorization and dispatch.
    pub(crate) fn exchange_grant_revision(&self, name: &str) -> watch::Receiver<u64> {
        let initial = self
            .inner
            .plugins
            .get(name)
            .and_then(|plugin| plugin.exchange_grant())
            .cloned();
        self.inner
            .exchange_grants
            .0
            .lock()
            .unwrap()
            .entry(name.to_owned())
            .or_insert_with(|| GrantState::new(initial))
            .revision
            .subscribe()
    }

    pub(crate) fn effective_exchange_grant(&self, name: &str) -> Option<OpenAiExchangeGrant> {
        let states = self.inner.exchange_grants.0.lock().unwrap();
        match states.get(name) {
            Some(state) => state.effective.clone(),
            None => self
                .inner
                .plugins
                .get(name)
                .and_then(|plugin| plugin.exchange_grant())
                .cloned(),
        }
    }

    /// Persisted owner config takes effect for security grants immediately,
    /// even when unrelated settings require a runtime restart.
    pub(crate) async fn apply_exchange_grants(&self, config: &MeshConfig) {
        let revoked = {
            let mut states = self.inner.exchange_grants.0.lock().unwrap();
            let mut changed = Vec::new();
            let names: std::collections::BTreeSet<_> = states
                .keys()
                .cloned()
                .chain(self.inner.plugins.keys().cloned())
                .chain(
                    config
                        .plugins
                        .iter()
                        .filter(|entry| entry.enabled != Some(false))
                        .map(|entry| entry.name.clone()),
                )
                .collect();
            for name in names {
                let effective = config
                    .plugins
                    .iter()
                    .find(|entry| entry.name == name)
                    .filter(|entry| entry.enabled != Some(false))
                    .and_then(|entry| entry.openai_exchange_grant.clone());
                let state = states.entry(name.clone()).or_insert_with(|| {
                    GrantState::new(
                        self.inner
                            .plugins
                            .get(&name)
                            .and_then(|plugin| plugin.exchange_grant())
                            .cloned(),
                    )
                });
                if state.replace(effective) {
                    changed.push(name.clone());
                }
            }
            changed
        };
        // Changed grants revoke previous delegation certificates, including
        // reduced scope/TTL and grants removed while callbacks are active.
        let mut registry = self.inner.identity_registry.lock().await;
        for name in revoked {
            registry.revoke_plugin(&name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_grant(required: bool) -> OpenAiExchangeGrant {
        OpenAiExchangeGrant {
            endpoints: vec!["chat_completions".into()],
            phases: vec!["request_received".into()],
            deadline_ms: 100,
            max_body_bytes: 1024,
            max_queue_bytes: 1024,
            max_in_flight: 1,
            failure_policy: if required {
                mesh_llm_config::OpenAiExchangeFailurePolicy::Required
            } else {
                mesh_llm_config::OpenAiExchangeFailurePolicy::BestEffort
            },
            ..Default::default()
        }
    }

    async fn unavailable_manager(
        grant: Option<OpenAiExchangeGrant>,
        lazy: bool,
    ) -> super::super::PluginManager {
        use super::super::{
            PluginManager, PluginStartupOptions,
            config::{ExternalPluginSpec, PluginHostMode, ResolvedPlugins},
        };
        let spec = ExternalPluginSpec {
            name: "unavailable-observer".into(),
            command: "/definitely-nonexistent-mesh-llm-plugin-1331".into(),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
            startup: PluginStartupOptions {
                optional: true,
                lazy_start: lazy,
                ..Default::default()
            },
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            installed_metadata: None,
            openai_exchange_grant: grant.map(Box::new),
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        PluginManager::start(
            &ResolvedPlugins {
                externals: vec![spec],
                inactive: Vec::new(),
            },
            PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            tx,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn required_grant_survives_failed_process_start_and_removal_disables_it() {
        let manager = unavailable_manager(Some(fixture_grant(true)), false).await;
        assert!(!manager.inner.plugins.contains_key("unavailable-observer"));
        assert!(manager.inner.inactive.contains_key("unavailable-observer"));
        assert!(manager.has_exchange_hooks().await);
        let result = manager
            .exchange_permissions_preflight("chat_completions")
            .await;
        assert_eq!(result.error_status(), Some(503));
        assert!(result.evidence_unavailable && result.incomplete);
        assert_eq!(
            manager.exchange_health_status("unavailable-observer"),
            "permissions_unavailable"
        );
        manager.apply_exchange_grants(&MeshConfig::default()).await;
        assert!(!manager.has_exchange_hooks().await);
        assert!(
            manager
                .exchange_permissions_preflight("chat_completions")
                .await
                .error_status()
                .is_none()
        );
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn missing_live_manifest_is_required_failure_but_best_effort_and_old_plugins_do_not_block()
     {
        let manager = unavailable_manager(Some(fixture_grant(true)), true).await;
        assert!(manager.inner.plugins.contains_key("unavailable-observer"));
        assert_eq!(
            manager
                .exchange_permissions_preflight("chat_completions")
                .await
                .error_status(),
            Some(503)
        );
        manager.shutdown().await;
        let manager = unavailable_manager(Some(fixture_grant(false)), false).await;
        let result = manager
            .exchange_permissions_preflight("chat_completions")
            .await;
        assert!(result.error_status().is_none());
        assert!(result.incomplete && result.evidence_unavailable);
        manager.shutdown().await;
        let manager = unavailable_manager(None, false).await;
        assert!(!manager.has_exchange_hooks().await);
        assert!(
            !manager
                .exchange_permissions_preflight("chat_completions")
                .await
                .incomplete
        );
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn owner_apply_enforces_required_new_plugin_until_available_or_removed() {
        let manager = unavailable_manager(None, true).await;
        let config = MeshConfig { plugins: vec![serde_json::from_value(serde_json::json!({"name":"new-unloaded-observer","openai_exchange_grant":fixture_grant(true)})).unwrap()], ..Default::default() };
        manager.apply_exchange_grants(&config).await;
        assert!(manager.has_exchange_hooks().await);
        assert_eq!(
            manager
                .exchange_permissions_preflight("chat_completions")
                .await
                .error_status(),
            Some(503)
        );
        manager.apply_exchange_grants(&MeshConfig::default()).await;
        assert!(!manager.has_exchange_hooks().await);
        manager.shutdown().await;
    }

    #[test]
    fn reduced_or_removed_grants_notify_existing_subscribers() {
        let mut state = GrantState::new(Some(OpenAiExchangeGrant {
            request_body: true,
            ..Default::default()
        }));
        let mut receiver = state.revision.subscribe();
        assert!(!receiver.has_changed().unwrap());
        assert!(state.replace(Some(OpenAiExchangeGrant::default())));
        assert!(receiver.has_changed().unwrap());
        receiver.borrow_and_update();
        assert!(!state.replace(Some(OpenAiExchangeGrant::default())));
        assert!(!receiver.has_changed().unwrap());
        assert!(state.replace(None));
        assert!(receiver.has_changed().unwrap());
        assert!(state.effective.is_none());
    }

    #[tokio::test]
    async fn persisted_grant_reduction_overrides_running_plugin_registration() {
        use super::super::{
            PluginManager, PluginStartupOptions,
            config::{ExternalPluginSpec, PluginHostMode, ResolvedPlugins},
        };
        let original = OpenAiExchangeGrant {
            endpoints: vec!["chat".into()],
            phases: vec!["request_received".into()],
            request_body: true,
            deadline_ms: 100,
            max_body_bytes: 1024,
            max_queue_bytes: 1024,
            max_in_flight: 1,
            ..Default::default()
        };
        let spec = ExternalPluginSpec {
            name: "observer".into(),
            command: "deferred-fixture".into(),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
            startup: PluginStartupOptions {
                lazy_start: true,
                ..Default::default()
            },
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            installed_metadata: None,
            openai_exchange_grant: Some(Box::new(original.clone())),
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        let manager = PluginManager::start(
            &ResolvedPlugins {
                externals: vec![spec],
                inactive: Vec::new(),
            },
            PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            tx,
        )
        .await
        .unwrap();
        let mut revision = manager.exchange_grant_revision("observer");
        assert!(
            manager
                .effective_exchange_grant("observer")
                .unwrap()
                .request_body
        );
        let mut reduced = original;
        reduced.request_body = false;
        let config = MeshConfig {
            plugins: vec![
                serde_json::from_value(
                    serde_json::json!({"name":"observer","openai_exchange_grant":reduced}),
                )
                .unwrap(),
            ],
            ..Default::default()
        };
        manager.apply_exchange_grants(&config).await;
        assert!(revision.has_changed().unwrap());
        revision.borrow_and_update();
        assert!(
            !manager
                .effective_exchange_grant("observer")
                .unwrap()
                .request_body
        );
        manager.apply_exchange_grants(&MeshConfig::default()).await;
        assert!(revision.has_changed().unwrap());
        assert!(manager.effective_exchange_grant("observer").is_none());
        manager.shutdown().await;
    }
}
