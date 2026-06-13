#![recursion_limit = "256"]

mod api;
mod capture;
pub mod command_support;
pub mod crypto;
pub mod discovery;
pub mod inference;
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

use anyhow::Result;

pub const BUILD_VERSION: &str = mesh_llm_build_info::BUILD_VERSION;
pub const RELEASE_VERSION: &str = mesh_llm_build_info::RELEASE_VERSION;
pub const VERSION: &str = RELEASE_VERSION;

pub use runtime::{
    MeshGuardrailMode, RuntimeOptions, RuntimeSurface, console_session_mode_for_runtime_surface,
};

pub async fn run() -> Result<()> {
    initialize_host_runtime()?;
    runtime::run().await
}

pub async fn run_runtime(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    initialize_host_runtime()?;
    run_runtime_initialized(options, explicit_surface, legacy_warning).await
}

pub async fn run_runtime_initialized(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    runtime::run_cli(options, explicit_surface, legacy_warning).await
}

pub fn initialize_host_runtime() -> Result<()> {
    #[cfg(feature = "dynamic-native-runtime")]
    if let Some(runtime) = system::native_runtime::try_load_installed_native_runtime()? {
        tracing::info!(
            native_runtime_id = %runtime.native_runtime_id,
            libraries = ?runtime.libraries,
            "Loaded MeshLLM native runtime"
        );
    }
    Ok(())
}

#[cfg(test)]
include!("exact_test_wrappers.rs");
