#![recursion_limit = "256"]

mod api;
mod capture;
pub mod command_support;
pub mod config_schema;
pub mod crypto;
pub mod discovery;
pub mod inference;
mod logging;
mod mesh;
pub mod models;
mod network;
pub mod plugin;
mod plugins;
mod protocol;
mod runtime;
mod runtime_data;
mod system;

pub mod sdk;

pub mod proto {
    pub use mesh_llm_protocol::proto::*;
}

use anyhow::Result;
pub use crypto::{
    ReleaseAttestationClaims, ReleaseAttestationStatus, ReleaseAttestationSummary,
    ReleaseBuildAttestation, ReleaseSignerTrustStore, TrustedReleaseSigner,
    default_release_signer_trust_store_path, load_release_signer_trust_store,
    parse_release_signer_public_key, release_signer_key_id, save_release_signer_trust_store,
    verify_release_attestation,
};
pub use mesh::requirements::{
    BootstrapStatus, DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS, DirectNodeAdmissionProof,
    DirectPeerProofStatus, MeshGenesisPolicy, MeshRequirementDecision,
    MeshRequirementEvaluationInput, MeshRequirementRejectReason, MeshRequirements,
    NodeVersionBounds, PeerReleaseAttestationStatus, ProtocolGenerationBounds,
    ReleaseAttestationRequirement, SignedBootstrapToken, SignedMeshGenesisPolicy,
};
use std::path::Path;
use std::sync::{Arc, LazyLock, RwLock};

use logging::foundation::LoggingFoundation;
use logging::{LoggingDynamicLimits, LoggingRuntimeApplyError, LoggingRuntimeState};

/// The current process-local logging foundation.
///
/// Startup can be invoked more than once by embedded hosts and tests.  This
/// must therefore be replaceable instead of using a `OnceLock`: a one-shot
/// holder would silently keep a foundation resolved from an earlier config.
static LOGGING_FOUNDATION: LazyLock<RwLock<Option<Arc<LoggingFoundation>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Durable local logging resources derived from the current foundation.
///
/// This stays separate from the foundation so a store or artifact failure can
/// be represented without changing root-resolution semantics.
static LOGGING_RUNTIME_STATE: LazyLock<RwLock<Option<Arc<LoggingRuntimeState>>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn logging_foundation() -> Option<Arc<LoggingFoundation>> {
    LOGGING_FOUNDATION
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Return the process-local metadata/artifact capability state for internal
/// runtime consumers. No filesystem paths or storage errors cross this API.
pub fn logging_runtime_state() -> Option<Arc<LoggingRuntimeState>> {
    LOGGING_RUNTIME_STATE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Apply the logging values whose exported schema explicitly allows live
/// mutation. The result deliberately contains no filesystem details.
pub(crate) fn apply_live_logging_limits(
    config: &mesh_llm_config::LoggingConfig,
) -> Result<(), LoggingRuntimeApplyError> {
    let state = LOGGING_RUNTIME_STATE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .ok_or(LoggingRuntimeApplyError::Unavailable)?;
    state.apply_dynamic_limits(LoggingDynamicLimits::from_config(config))
}

pub fn logging_health_summary() -> String {
    match LOGGING_FOUNDATION
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_deref()
    {
        Some(foundation) => foundation.health_summary(),
        None => "logging not initialized".to_string(),
    }
}

pub const BUILD_VERSION: &str = mesh_llm_build_info::BUILD_VERSION;
pub const RELEASE_VERSION: &str = mesh_llm_build_info::RELEASE_VERSION;
pub const VERSION: &str = RELEASE_VERSION;

pub use runtime::{
    MeshGuardrailMode, RuntimeOptions, RuntimeSurface, console_session_mode_for_runtime_surface,
};

pub async fn run() -> Result<()> {
    initialize_host_runtime().await?;
    runtime::run().await
}

pub async fn run_runtime(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    initialize_host_runtime_for_options(&options).await?;
    run_runtime_initialized(options, explicit_surface, legacy_warning).await
}

pub async fn run_runtime_initialized(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    runtime::run_cli(options, explicit_surface, legacy_warning).await
}

pub async fn initialize_host_runtime() -> Result<()> {
    initialize_host_runtime_with_config(None).await
}

pub async fn initialize_host_runtime_for_options(options: &RuntimeOptions) -> Result<()> {
    if options.plugin.is_some() {
        return Ok(());
    }
    if !runtime_options_require_native_runtime(options) {
        let config = plugin::load_config(options.config.as_deref())?;
        initialize_logging_foundation(&config.logging);
        return Ok(());
    }
    initialize_host_runtime_with_config(options.config.as_deref()).await
}

fn runtime_options_require_native_runtime(options: &RuntimeOptions) -> bool {
    !options.client && options.plugin.is_none()
}

pub async fn initialize_host_runtime_with_config(config_path: Option<&Path>) -> Result<()> {
    let config = plugin::load_config(config_path)?;

    // Logging config is validated as part of config loading and must be resolved
    // before native runtime setup. Foundation failures remain fail-open so they
    // cannot prevent serving from starting.
    initialize_logging_foundation(&config.logging);

    #[cfg(feature = "dynamic-native-runtime")]
    {
        let native_runtime = config.runtime.native_runtime;
        let startup_selection = match native_runtime.mesh_version {
            Some(mesh_version) => {
                let runtime_selection = mesh_llm_native_runtime::RuntimeSelection::parse(
                    native_runtime.selection.as_deref(),
                )?;
                system::native_runtime::NativeRuntimeStartupSelection::explicit(
                    mesh_version,
                    native_runtime.skippy_abi,
                    runtime_selection,
                )
            }
            None => system::native_runtime::NativeRuntimeStartupSelection::current(),
        };
        if let Some(runtime) =
            system::native_runtime::try_load_installed_native_runtime(startup_selection).await?
        {
            tracing::info!(
                native_runtime_id = %runtime.native_runtime_id,
                libraries = ?runtime.libraries,
                "Loaded MeshLLM native runtime"
            );
        }
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        let _ = config;
    }

    Ok(())
}

/// Install the process-local logging resources for one validated runtime config.
///
/// This is shared by the CLI/native startup path and the embedded SDK entrypoint.
/// The holders are intentionally replaceable: an embedded host may start a new
/// runtime with a different config in the same process.
pub(crate) fn initialize_logging_foundation(config: &mesh_llm_config::LoggingConfig) {
    let foundation = logging_foundation_from_config(config);
    if !foundation.is_healthy() {
        tracing::warn!(
            summary = %foundation.health_summary(),
            "Logging foundation initialized unhealthy (fail-open)"
        );
    }
    let runtime_state = LoggingRuntimeState::initialize(&foundation, config);
    if !runtime_state.health().metadata_available {
        tracing::warn!("Logging metadata storage unavailable (fail-open)");
    }
    *LOGGING_RUNTIME_STATE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(runtime_state));
    *LOGGING_FOUNDATION
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(foundation));
}

fn logging_foundation_from_config(config: &mesh_llm_config::LoggingConfig) -> LoggingFoundation {
    LoggingFoundation::init(config.enabled, config.application_state_root.as_ref())
}

#[cfg(test)]
#[test]
fn logging_foundation_uses_the_configured_application_state_root() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let configured_root = temporary_directory.path().join("configured-root");
    let config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(configured_root.clone()),
        ..Default::default()
    };

    let foundation = logging_foundation_from_config(&config);

    assert!(foundation.is_healthy());
    assert_eq!(foundation.app_state_root(), configured_root);
}

#[cfg(test)]
#[test]
fn disabled_logging_config_creates_no_application_state_layout() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let configured_root = temporary_directory.path().join("disabled-root");
    let config = mesh_llm_config::LoggingConfig {
        enabled: false,
        application_state_root: Some(configured_root.clone()),
        ..Default::default()
    };

    let foundation = logging_foundation_from_config(&config);

    assert!(!foundation.is_healthy());
    assert!(
        !configured_root.exists(),
        "disabled logging must not create the configured application-state root"
    );
}

#[cfg(test)]
#[test]
#[serial_test::serial]
fn logging_foundation_install_replaces_a_previous_in_process_config() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let first_root = temporary_directory.path().join("first-root");
    let second_root = temporary_directory.path().join("second-root");
    let first_config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(first_root),
        ..Default::default()
    };
    let second_config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(second_root.clone()),
        ..Default::default()
    };

    initialize_logging_foundation(&first_config);
    initialize_logging_foundation(&second_config);

    let installed = logging_foundation().expect("logging foundation should be installed");
    assert_eq!(installed.app_state_root(), second_root);
}

#[cfg(test)]
#[test]
#[serial_test::serial]
fn live_logging_limit_application_updates_the_installed_service() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let mut config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("live-limits")),
        retention_ttl_secs: 3_600,
        replay_capacity: 4,
        ..Default::default()
    };
    initialize_logging_foundation(&config);

    config.retention_ttl_secs = 7_200;
    config.replay_capacity = 2;
    apply_live_logging_limits(&config).expect("healthy installed logging runtime");

    assert_eq!(
        logging_runtime_state()
            .expect("installed runtime state")
            .dynamic_limits(),
        Some(LoggingDynamicLimits {
            retention_ttl_secs: 7_200,
            replay_capacity: 2,
        })
    );

    let config_path = temporary_directory.path().join("owner-control-config.toml");
    let mut config_state = runtime::config_state::ConfigState::load(&config_path)
        .expect("load owner-control config state");
    let mut persisted = plugin::MeshConfig {
        logging: mesh_llm_config::LoggingConfig {
            application_state_root: config.application_state_root.clone(),
            retention_ttl_secs: 7_200,
            replay_capacity: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let revision = match config_state.apply(persisted.clone(), 0) {
        runtime::config_state::ApplyResult::Applied { revision, .. }
        | runtime::config_state::ApplyResult::AppliedWithRestartRequired { revision, .. } => {
            revision
        }
        result => panic!("baseline config application failed: {result:?}"),
    };

    persisted.logging.retention_ttl_secs = 10_800;
    persisted.logging.replay_capacity = 3;
    match config_state.apply_with_live_logging(persisted, revision) {
        runtime::config_state::ApplyResult::Applied {
            apply_mode: runtime::config_state::ConfigApplyMode::Live,
            ..
        } => {}
        result => panic!("expected live dynamic logging apply, got {result:?}"),
    }
}

#[cfg(test)]
include!("exact_test_wrappers.rs");
