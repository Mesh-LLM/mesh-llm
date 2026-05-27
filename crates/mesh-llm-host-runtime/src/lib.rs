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
use std::time::Duration;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn run() -> Result<()> {
    runtime::run().await
}

pub async fn run_main() -> i32 {
    match run().await {
        Ok(()) => 0,
        Err(err) => {
            let _ = cli::output::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}

#[cfg(test)]
include!("exact_test_wrappers.rs");
