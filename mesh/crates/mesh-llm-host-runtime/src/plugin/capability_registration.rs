//! Previously declared capability presence is not a live health report.
//! Remember declarations for this manager's lifetime without retaining callable
//! operations, endpoints, or manifests after a plugin disconnects.

use std::collections::BTreeSet;
use std::sync::Mutex;

#[cfg(any(feature = "payments", test))]
use anyhow::{Result, anyhow};

use super::PluginCapabilityProvider;
#[cfg(any(feature = "payments", test))]
use super::PluginManager;

#[derive(Default)]
pub(super) struct CapabilityRegistration(Mutex<BTreeSet<String>>);

impl CapabilityRegistration {
    pub(super) fn record(&self, providers: &[PluginCapabilityProvider]) {
        // A poisoned registry stays poisoned: readers must fail closed rather
        // than interpret loss of registration evidence as an absent provider.
        if let Ok(mut known) = self.0.lock() {
            known.extend(providers.iter().map(|provider| provider.capability.clone()));
        }
    }

    #[cfg(any(feature = "payments", test))]
    fn contains(&self, capability: &str) -> Result<bool> {
        Ok(self
            .0
            .lock()
            .map_err(|_| anyhow!("capability registration lock poisoned"))?
            .contains(capability))
    }
}

#[cfg(any(feature = "payments", test))]
impl PluginManager {
    /// Whether a provider has declared this capability during this manager's
    /// lifetime. This does not authorize invocation; live provider selection
    /// must still succeed. Registration resets only when the manager is replaced
    /// (normally on restart), not when a provider restarts or loses its manifest.
    pub(crate) fn has_registered_capability(&self, capability: &str) -> Result<bool> {
        self.inner.capability_registration.contains(capability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_survives_health_report_removal_without_enabling_calls() -> Result<()> {
        let manager = PluginManager::for_test_summaries(Vec::new());
        assert!(!manager.has_registered_capability("payments.v1")?);
        manager.publish_plugin_providers(
            "external-payments",
            vec![PluginCapabilityProvider {
                capability: "payments.v1".into(),
                plugin_name: "external-payments".into(),
                plugin_status: "ready".into(),
                endpoint_id: None,
                available: true,
                detail: None,
            }],
        );
        manager
            .plugin_summary_producer("external-payments")
            .clear_plugin_reports("external-payments");
        manager.publish_plugin_providers("external-payments", Vec::new());
        assert!(manager.has_registered_capability("payments.v1")?);
        assert!(
            manager
                .inner
                .runtime_data
                .plugins_snapshot()
                .providers
                .is_empty()
        );
        let replacement = PluginManager::for_test_summaries(Vec::new());
        assert!(!replacement.has_registered_capability("payments.v1")?);
        Ok(())
    }
}
