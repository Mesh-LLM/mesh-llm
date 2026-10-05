//! Runtime lifecycle and activity policy configuration.
//!
//! All enums use strict serde with lowercase variants for TOML round-trips.
//! Missing mode/activity fields preserve backward-compatible defaults
//! (`Serve` / no-throttle). Manual activity override is deliberately not
//! persisted; process restart is required for persistent settings.

use serde::{Deserialize, Serialize};

// ── Runtime Mode ────────────────────────────────────────────────────────────

/// Top-level runtime operating mode.
///
/// Absent in config resolves to `Serve` for backward compatibility.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    /// Read-only client: join mesh, route requests, never serve local models.
    Client,
    /// Full serve mode (default). Local models load and serve inference.
    #[default]
    Serve,
    /// On-demand: start without models; only load when explicitly requested.
    OnDemand,
}

impl RuntimeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::Serve => "serve",
            Self::OnDemand => "on_demand",
        }
    }

    /// Whether this mode allows loading local models at all.
    pub fn allows_local_models(self) -> bool {
        !matches!(self, Self::Client)
    }
}

// ── Startup Failure Policy ──────────────────────────────────────────────────

/// How the runtime reacts when a model fails to load during startup.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupFailurePolicy {
    /// Continue starting; log errors and move on (default).
    #[default]
    BestEffort,
    /// Abort startup if any configured model fails to load.
    FailFast,
}

impl StartupFailurePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BestEffort => "best_effort",
            Self::FailFast => "fail_fast",
        }
    }
}

// ── Drain Configuration ─────────────────────────────────────────────────────

/// Default drain timeout in seconds before a model instance is forcibly unloaded.
pub const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 30;
/// Default maximum drain timeout cap in seconds.
pub const DEFAULT_DRAIN_TIMEOUT_MAX_SECS: u64 = 300;

// ── Activity Policy Response ────────────────────────────────────────────────

/// What the runtime does when host activity is detected and policy is enabled.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityResponse {
    /// Pause remote model routing; keep local models serving.
    #[default]
    PauseRemote,
    /// Pause all inference (local and remote).
    PauseAll,
    /// Keep serving but reduce priority/scheduling weight.
    ReducePriority,
}

impl ActivityResponse {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PauseRemote => "pause_remote",
            Self::PauseAll => "pause_all",
            Self::ReducePriority => "reduce_priority",
        }
    }
}

// ── Activity Advertisement Mode ─────────────────────────────────────────────

/// How much admission state is shared with mesh peers.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityAdvertisement {
    /// Never advertise activity-related admission state.
    None,
    /// Only publish hosted/serving availability (known-empty when non-admitting).
    AvailabilityOnly,
    /// Emit coarse admission enum to all peers (default).
    #[default]
    CoarseState,
    /// Emit coarse admission on private meshes only; known-empty publicly.
    PrivateCoarseState,
}

impl ActivityAdvertisement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::AvailabilityOnly => "availability_only",
            Self::CoarseState => "coarse_state",
            Self::PrivateCoarseState => "private_coarse_state",
        }
    }

    /// Whether this mode emits the coarse admission enum at all.
    pub fn emits_admission_enum(self) -> bool {
        matches!(self, Self::CoarseState | Self::PrivateCoarseState)
    }

    /// Whether this mode publishes known-empty availability to public peers
    /// when non-admitting (for backward-compatible withdrawal signaling).
    pub fn withdraws_availability_when_paused(self) -> bool {
        matches!(self, Self::AvailabilityOnly | Self::CoarseState)
    }
}

// ── Activity Configuration ──────────────────────────────────────────────────

/// Default idle threshold in seconds before activity policy triggers.
pub const DEFAULT_ACTIVITY_IDLE_AFTER_SECS: u64 = 300;
/// Default poll interval for the activity detector in seconds.
pub const DEFAULT_ACTIVITY_POLL_INTERVAL_SECS: u64 = 5;
/// Default debounce before resuming inference after idle→active transition.
pub const DEFAULT_ACTIVITY_RESUME_DEBOUNCE_SECS: u64 = 30;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct RuntimeActivityConfig {
    /// Enable host activity detection and response (disabled by default).
    #[serde(default)]
    pub enabled: bool,
    /// Seconds of inactivity before the policy triggers (default 300, valid 30..=86400).
    #[serde(default = "default_idle_after_secs")]
    pub idle_after_secs: u64,
    /// How often to poll the activity detector in seconds (default 5, valid 1..=60).
    #[serde(default = "default_poll_interval_secs")]
    pub poll_interval_secs: u64,
    /// Debounce before resuming inference after idle→active transition (default 30, valid 0..=300).
    #[serde(default = "default_resume_debounce_secs")]
    pub resume_debounce_secs: u64,
    /// Response when activity is detected (default pause_remote).
    #[serde(default)]
    pub response: ActivityResponse,
    /// How to advertise admission state to mesh peers (default coarse_state).
    #[serde(default)]
    pub advertisement: ActivityAdvertisement,
}

impl Default for RuntimeActivityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_after_secs: DEFAULT_ACTIVITY_IDLE_AFTER_SECS,
            poll_interval_secs: DEFAULT_ACTIVITY_POLL_INTERVAL_SECS,
            resume_debounce_secs: DEFAULT_ACTIVITY_RESUME_DEBOUNCE_SECS,
            response: ActivityResponse::default(),
            advertisement: ActivityAdvertisement::default(),
        }
    }
}

fn default_idle_after_secs() -> u64 {
    DEFAULT_ACTIVITY_IDLE_AFTER_SECS
}

fn default_poll_interval_secs() -> u64 {
    DEFAULT_ACTIVITY_POLL_INTERVAL_SECS
}

fn default_resume_debounce_secs() -> u64 {
    DEFAULT_ACTIVITY_RESUME_DEBOUNCE_SECS
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // RuntimeMode round-trips
    #[test]
    fn runtime_mode_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&RuntimeMode::Client).unwrap(),
            "\"client\""
        );
        assert_eq!(
            serde_json::to_string(&RuntimeMode::Serve).unwrap(),
            "\"serve\""
        );
        assert_eq!(
            serde_json::to_string(&RuntimeMode::OnDemand).unwrap(),
            "\"on_demand\""
        );
    }

    #[test]
    fn runtime_mode_deserializes_lowercase() {
        let client: RuntimeMode = serde_json::from_str("\"client\"").unwrap();
        assert_eq!(client, RuntimeMode::Client);

        let serve: RuntimeMode = serde_json::from_str("\"serve\"").unwrap();
        assert_eq!(serve, RuntimeMode::Serve);

        let on_demand: RuntimeMode = serde_json::from_str("\"on_demand\"").unwrap();
        assert_eq!(on_demand, RuntimeMode::OnDemand);
    }

    #[test]
    fn runtime_mode_default_is_serve() {
        assert_eq!(RuntimeMode::default(), RuntimeMode::Serve);
    }

    #[test]
    fn runtime_mode_allows_local_models() {
        assert!(!RuntimeMode::Client.allows_local_models());
        assert!(RuntimeMode::Serve.allows_local_models());
        assert!(RuntimeMode::OnDemand.allows_local_models());
    }

    // StartupFailurePolicy round-trips
    #[test]
    fn startup_failure_policy_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&StartupFailurePolicy::BestEffort).unwrap(),
            "\"best_effort\""
        );
        assert_eq!(
            serde_json::to_string(&StartupFailurePolicy::FailFast).unwrap(),
            "\"fail_fast\""
        );
    }

    #[test]
    fn startup_failure_policy_deserializes_snake_case() {
        let best: StartupFailurePolicy = serde_json::from_str("\"best_effort\"").unwrap();
        assert_eq!(best, StartupFailurePolicy::BestEffort);

        let fail: StartupFailurePolicy = serde_json::from_str("\"fail_fast\"").unwrap();
        assert_eq!(fail, StartupFailurePolicy::FailFast);
    }

    #[test]
    fn startup_failure_policy_default_is_best_effort() {
        assert_eq!(
            StartupFailurePolicy::default(),
            StartupFailurePolicy::BestEffort
        );
    }

    // ActivityResponse round-trips
    #[test]
    fn activity_response_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&ActivityResponse::PauseRemote).unwrap(),
            "\"pause_remote\""
        );
        assert_eq!(
            serde_json::to_string(&ActivityResponse::PauseAll).unwrap(),
            "\"pause_all\""
        );
        assert_eq!(
            serde_json::to_string(&ActivityResponse::ReducePriority).unwrap(),
            "\"reduce_priority\""
        );
    }

    #[test]
    fn activity_response_deserializes_snake_case() {
        let pr: ActivityResponse = serde_json::from_str("\"pause_remote\"").unwrap();
        assert_eq!(pr, ActivityResponse::PauseRemote);

        let pa: ActivityResponse = serde_json::from_str("\"pause_all\"").unwrap();
        assert_eq!(pa, ActivityResponse::PauseAll);

        let rp: ActivityResponse = serde_json::from_str("\"reduce_priority\"").unwrap();
        assert_eq!(rp, ActivityResponse::ReducePriority);
    }

    #[test]
    fn activity_response_default_is_pause_remote() {
        assert_eq!(ActivityResponse::default(), ActivityResponse::PauseRemote);
    }

    // ActivityAdvertisement round-trips
    #[test]
    fn activity_advertisement_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&ActivityAdvertisement::None).unwrap(),
            "\"none\""
        );
        assert_eq!(
            serde_json::to_string(&ActivityAdvertisement::AvailabilityOnly).unwrap(),
            "\"availability_only\""
        );
        assert_eq!(
            serde_json::to_string(&ActivityAdvertisement::CoarseState).unwrap(),
            "\"coarse_state\""
        );
        assert_eq!(
            serde_json::to_string(&ActivityAdvertisement::PrivateCoarseState).unwrap(),
            "\"private_coarse_state\""
        );
    }

    #[test]
    fn activity_advertisement_deserializes_snake_case() {
        let none: ActivityAdvertisement = serde_json::from_str("\"none\"").unwrap();
        assert_eq!(none, ActivityAdvertisement::None);

        let avail: ActivityAdvertisement = serde_json::from_str("\"availability_only\"").unwrap();
        assert_eq!(avail, ActivityAdvertisement::AvailabilityOnly);

        let coarse: ActivityAdvertisement = serde_json::from_str("\"coarse_state\"").unwrap();
        assert_eq!(coarse, ActivityAdvertisement::CoarseState);

        let priv_coarse: ActivityAdvertisement =
            serde_json::from_str("\"private_coarse_state\"").unwrap();
        assert_eq!(priv_coarse, ActivityAdvertisement::PrivateCoarseState);
    }

    #[test]
    fn activity_advertisement_default_is_coarse_state() {
        assert_eq!(
            ActivityAdvertisement::default(),
            ActivityAdvertisement::CoarseState
        );
    }

    #[test]
    fn activity_advertisement_emits_enum_logic() {
        assert!(!ActivityAdvertisement::None.emits_admission_enum());
        assert!(!ActivityAdvertisement::AvailabilityOnly.emits_admission_enum());
        assert!(ActivityAdvertisement::CoarseState.emits_admission_enum());
        assert!(ActivityAdvertisement::PrivateCoarseState.emits_admission_enum());
    }

    #[test]
    fn activity_advertisement_withdraws_availability_logic() {
        assert!(!ActivityAdvertisement::None.withdraws_availability_when_paused());
        assert!(ActivityAdvertisement::AvailabilityOnly.withdraws_availability_when_paused());
        assert!(ActivityAdvertisement::CoarseState.withdraws_availability_when_paused());
        assert!(!ActivityAdvertisement::PrivateCoarseState.withdraws_availability_when_paused());
    }

    // RuntimeActivityConfig defaults
    #[test]
    fn activity_config_defaults() {
        let cfg = RuntimeActivityConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.idle_after_secs, 300);
        assert_eq!(cfg.poll_interval_secs, 5);
        assert_eq!(cfg.resume_debounce_secs, 30);
        assert_eq!(cfg.response, ActivityResponse::PauseRemote);
        assert_eq!(cfg.advertisement, ActivityAdvertisement::CoarseState);
    }

    #[test]
    fn activity_config_round_trip_with_toml() {
        let cfg = RuntimeActivityConfig {
            enabled: true,
            idle_after_secs: 600,
            poll_interval_secs: 10,
            resume_debounce_secs: 60,
            response: ActivityResponse::PauseAll,
            advertisement: ActivityAdvertisement::PrivateCoarseState,
        };

        let toml_str = toml::to_string(&cfg).unwrap();
        let parsed: RuntimeActivityConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn activity_config_missing_fields_defaults() {
        // TOML with only enabled=true; rest should default.
        let toml_str = r#"enabled = true"#;
        let parsed: RuntimeActivityConfig = toml::from_str(toml_str).unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.idle_after_secs, 300);
        assert_eq!(parsed.poll_interval_secs, 5);
        assert_eq!(parsed.resume_debounce_secs, 30);
        assert_eq!(parsed.response, ActivityResponse::PauseRemote);
        assert_eq!(parsed.advertisement, ActivityAdvertisement::CoarseState);
    }

    // Drain timeout constants
    #[test]
    fn drain_timeout_constants() {
        assert_eq!(DEFAULT_DRAIN_TIMEOUT_SECS, 30);
        assert_eq!(DEFAULT_DRAIN_TIMEOUT_MAX_SECS, 300);
    }

    // Activity constant defaults
    #[test]
    fn activity_constant_defaults() {
        assert_eq!(DEFAULT_ACTIVITY_IDLE_AFTER_SECS, 300);
        assert_eq!(DEFAULT_ACTIVITY_POLL_INTERVAL_SECS, 5);
        assert_eq!(DEFAULT_ACTIVITY_RESUME_DEBOUNCE_SECS, 30);
    }
}
