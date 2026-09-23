//! Resolving whether this machine reports analytics, and why.
//!
//! Every input is inspected in one place so `mesh-llm analytics status` can
//! explain the decision instead of leaving users to guess.

use std::env;

/// Environment override for the analytics opt-out.
pub const ENV_ANALYTICS: &str = "MESH_LLM_ANALYTICS";
/// The cross-vendor opt-out convention (<https://consoledonottrack.com>).
pub const ENV_DO_NOT_TRACK: &str = "DO_NOT_TRACK";
/// Runtime override for the ingestion project key.
pub const ENV_POSTHOG_KEY: &str = "MESH_LLM_POSTHOG_KEY";
/// Runtime override for the ingestion host, for self-hosted PostHog.
pub const ENV_POSTHOG_HOST: &str = "MESH_LLM_POSTHOG_HOST";

/// Default ingestion host. PostHog US cloud.
pub const DEFAULT_POSTHOG_HOST: &str = "https://us.i.posthog.com";

/// The project key baked in at release build time.
///
/// Absent in source and development builds, which is deliberate: a build that
/// was not produced by the release pipeline reports nothing at all.
const BUILT_IN_POSTHOG_KEY: Option<&str> = option_env!("MESH_LLM_POSTHOG_KEY");

/// Why analytics is on or off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Disposition {
    /// Reporting, with no override in play.
    EnabledByDefault,
    /// Reporting because an override turned it on explicitly.
    EnabledByEnv,
    /// `MESH_LLM_ANALYTICS` is set to a falsey value.
    DisabledByEnv,
    /// `DO_NOT_TRACK` is set.
    DisabledByDoNotTrack,
    /// `[analytics] enabled = false` in the config file.
    DisabledByConfig,
    /// The config file exists but could not be read or parsed.
    ///
    /// A privacy control has to fail closed. A config with one mistyped key
    /// anywhere fails to deserialize as a whole, and treating that as "no
    /// preference stated" would silently re-enable reporting for someone who
    /// had opted out.
    DisabledConfigUnreadable,
    /// No project key, so there is nowhere to report. Source builds land here.
    DisabledNoKey,
    /// A continuous-integration environment was detected.
    DisabledInCi,
}

impl Disposition {
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::EnabledByDefault | Self::EnabledByEnv)
    }

    /// A one-line explanation for `mesh-llm analytics status`.
    #[must_use]
    pub const fn explain(self) -> &'static str {
        match self {
            Self::EnabledByDefault => "enabled (default; run `mesh-llm analytics disable` to stop)",
            Self::EnabledByEnv => "enabled by MESH_LLM_ANALYTICS",
            Self::DisabledByEnv => "disabled by MESH_LLM_ANALYTICS",
            Self::DisabledByDoNotTrack => "disabled by DO_NOT_TRACK",
            Self::DisabledByConfig => "disabled by [analytics] enabled = false in config.toml",
            Self::DisabledConfigUnreadable => {
                "disabled: the config file could not be read, so the recorded preference is unknown"
            }
            Self::DisabledNoKey => {
                "disabled: no analytics key is available (none compiled in, and MESH_LLM_POSTHOG_KEY is unset or empty)"
            }
            Self::DisabledInCi => "disabled: continuous integration environment detected",
        }
    }
}

/// What the config file says about analytics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConfigPreference {
    /// The config loaded and did not mention analytics.
    #[default]
    Unstated,
    /// `[analytics] enabled` was stated explicitly.
    Stated(bool),
    /// The config exists but could not be read or parsed.
    ///
    /// Distinct from [`ConfigPreference::Unstated`] on purpose: an absent
    /// config is an ordinary first run, while an unreadable one may be hiding
    /// an opt-out and must not be read as consent.
    Unreadable,
}

/// Inputs to the consent decision, gathered from the environment.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConsentInputs {
    /// `MESH_LLM_ANALYTICS`, if set.
    pub env_override: Option<String>,
    /// Whether `DO_NOT_TRACK` is set to a truthy value.
    pub do_not_track: bool,
    /// What the config file says.
    pub config: ConfigPreference,
    /// Whether a project key is available.
    pub has_key: bool,
    /// Whether this looks like CI.
    pub in_ci: bool,
}

impl ConsentInputs {
    /// Read every input from the process environment.
    #[must_use]
    pub fn from_env(config: ConfigPreference) -> Self {
        Self {
            env_override: env::var(ENV_ANALYTICS).ok(),
            do_not_track: env::var(ENV_DO_NOT_TRACK)
                .ok()
                .is_some_and(|value| is_truthy(&value)),
            config,
            has_key: project_key().is_some(),
            in_ci: detect_ci(),
        }
    }

    /// Resolve the inputs into a single disposition.
    ///
    /// An explicit `MESH_LLM_ANALYTICS` wins over everything else, including
    /// the CI check, so a deliberate test run can still exercise the path.
    #[must_use]
    pub fn resolve(&self) -> Disposition {
        if !self.has_key {
            return Disposition::DisabledNoKey;
        }
        if let Some(raw) = self.env_override.as_deref() {
            return if is_truthy(raw) {
                Disposition::EnabledByEnv
            } else {
                Disposition::DisabledByEnv
            };
        }
        if self.do_not_track {
            return Disposition::DisabledByDoNotTrack;
        }
        match self.config {
            ConfigPreference::Stated(false) => return Disposition::DisabledByConfig,
            ConfigPreference::Unreadable => return Disposition::DisabledConfigUnreadable,
            ConfigPreference::Stated(true) | ConfigPreference::Unstated => {}
        }
        if self.in_ci {
            return Disposition::DisabledInCi;
        }
        Disposition::EnabledByDefault
    }
}

/// The ingestion project key, preferring the runtime override.
#[must_use]
pub fn project_key() -> Option<String> {
    if let Ok(key) = env::var(ENV_POSTHOG_KEY) {
        let key = key.trim().to_owned();
        if !key.is_empty() {
            return Some(key);
        }
    }
    BUILT_IN_POSTHOG_KEY
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_owned)
}

/// Normalize a configured ingestion host, or `None` when nothing is left of it.
///
/// Split out from [`ingestion_host`] so the normalization is testable without
/// mutating the process environment, which is global to the test binary.
fn normalize_host(raw: &str) -> Option<String> {
    let host = raw.trim().trim_end_matches('/');
    (!host.is_empty()).then(|| host.to_owned())
}

/// The ingestion host, preferring the runtime override.
#[must_use]
pub fn ingestion_host() -> String {
    env::var(ENV_POSTHOG_HOST)
        .ok()
        .and_then(|host| normalize_host(&host))
        .unwrap_or_else(|| DEFAULT_POSTHOG_HOST.to_owned())
}

fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Variables set by the common CI providers.
const CI_MARKERS: &[&str] = &[
    "CI",
    "CONTINUOUS_INTEGRATION",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
];

fn detect_ci() -> bool {
    CI_MARKERS.iter().any(|marker| {
        env::var(marker).is_ok_and(|value| {
            let value = value.trim();
            // `CI` is conventionally truthy-by-presence, but an explicit
            // `CI=false` should not count as continuous integration.
            !value.is_empty() && !matches!(value.to_ascii_lowercase().as_str(), "0" | "false")
        })
    })
}

#[cfg(test)]
#[path = "consent/tests.rs"]
mod tests;
