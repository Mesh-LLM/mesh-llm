#![recursion_limit = "256"]

mod api;
mod capture;
mod cli;
pub mod crypto;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
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

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn run() -> Result<()> {
    initialize_host_runtime()?;
    runtime::run().await
}

pub async fn run_cli(
    cli: mesh_llm_cli::Cli,
    explicit_surface: Option<mesh_llm_cli::RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    initialize_host_runtime()?;
    run_cli_initialized(cli, explicit_surface, legacy_warning).await
}

pub async fn run_cli_initialized(
    cli: mesh_llm_cli::Cli,
    explicit_surface: Option<mesh_llm_cli::RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    runtime::run_cli(cli, explicit_surface, legacy_warning).await
}

pub fn resolved_plugin_list_rows(
    cli: &mesh_llm_cli::Cli,
) -> Result<mesh_llm_commands::plugin::PluginListRows> {
    let resolved = runtime::load_resolved_plugins(cli)?;
    Ok(mesh_llm_commands::plugin::PluginListRows {
        externals: resolved
            .externals
            .into_iter()
            .map(|spec| mesh_llm_commands::plugin::RuntimePluginRow {
                name: spec.name,
                command: spec.command,
                args: spec.args,
            })
            .collect(),
        inactive: resolved
            .inactive
            .into_iter()
            .map(|summary| mesh_llm_commands::plugin::InactivePluginRow {
                name: summary.name,
                kind: summary.kind,
                status: summary.status,
                error: summary.error,
            })
            .collect(),
    })
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
