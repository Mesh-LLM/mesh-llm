//! Configuration shared by host-managed inference providers.

use serde::{Deserialize, Serialize};

/// When an inference provider should construct its model sessions.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadPolicy {
    /// Load during normal model startup.
    #[default]
    Startup,
    /// Load when an explicitly named inference request needs this provider.
    OnDemand,
}

/// Model residency policy. Disk assets remain cached after idle unloading.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelLifecycle {
    /// Defaults to startup loading.
    pub load: ModelLoadPolicy,
    /// Idle interval after the last response; zero keeps weights resident.
    pub idle_timeout_secs: u64,
    /// Resolve disk assets in the background without constructing sessions.
    pub prefetch: bool,
}

impl Default for ModelLifecycle {
    fn default() -> Self {
        Self {
            load: ModelLoadPolicy::Startup,
            idle_timeout_secs: 0,
            prefetch: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residency_policy_round_trips_without_changing_legacy_defaults() {
        let config: crate::ModelConfigEntry = toml::from_str("model = 'test'").unwrap();
        assert_eq!(config.lifecycle, ModelLifecycle::default());
        assert!(!toml::to_string(&config).unwrap().contains("lifecycle"));
        let input = "model = 'test'\n[lifecycle]\nload = 'on_demand'\nidle_timeout_secs = 30\nprefetch = false";
        let config: crate::ModelConfigEntry = toml::from_str(input).unwrap();
        assert_eq!(config.lifecycle.load, ModelLoadPolicy::OnDemand);
        let saved = toml::to_string(&config).unwrap();
        let restored: crate::ModelConfigEntry = toml::from_str(&saved).unwrap();
        assert_eq!(restored.lifecycle, config.lifecycle);
    }

    #[test]
    fn rejects_misspelled_policy_and_negative_timeout() {
        for invalid in [
            "load = 'on-demand'",
            "idle_timeout_secs = -1",
            "preftech = true",
        ] {
            assert!(toml::from_str::<ModelLifecycle>(invalid).is_err());
        }
    }
}
