//! Typed rows of the checked-in runner role matrix, `ci/runner-roles.json`.
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Matrix {
    pub(super) schema_version: u32,
    pub(super) roles: Vec<Role>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Role {
    pub(super) id: String,
    pub(super) summary: String,
    pub(super) trust: BTreeSet<Trust>,
    pub(super) architecture: BTreeSet<Architecture>,
    pub(super) providers: BTreeSet<Provider>,
    pub(super) labels: BTreeSet<String>,
    pub(super) preparation_profiles: BTreeSet<Profile>,
    pub(super) cache_mode: CacheMode,
    pub(super) artifact_dependencies: Vec<String>,
    pub(super) hosted_fallback: HostedFallback,
    pub(super) main: Decision,
    pub(super) pull_request: Decision,
    pub(super) hosted_required_inputs: BTreeMap<String, bool>,
    pub(super) readiness: Readiness,
    pub(super) selectors: Vec<String>,
    pub(super) jobs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Trust {
    PullRequest,
    MainPush,
    ManualDispatch,
    ReleaseDispatch,
    Schedule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Architecture {
    Amd64,
    Arm64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Provider {
    GithubHosted,
    Depot,
    SelfHosted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Profile {
    Image,
    HostedBare,
    /// `hosted-bare` behind a `runner.environment == 'github-hosted'` guard.
    HostedBareGithubHostedOnly,
    PreinstalledMacos,
    PreinstalledLinux,
    RestoreAutomation,
    CiController,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum CacheMode {
    None,
    ProviderDerived,
    HostedExactKey,
    HostedTrustedPublisher,
    JobLocal,
    DiagnosticProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum HostedFallback {
    Selector,
    AlwaysHosted,
    Queue,
    OperatorInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Decision {
    DepotGated,
    NoDepot,
    SentinelRef,
    OperatorInputHosted,
    ManualDiagnostic,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Readiness {
    pub(super) state: ReadinessState,
    pub(super) evidence: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ReadinessState {
    Excluded,
    Supported,
    Prepared,
    Qualified,
    Active,
}
