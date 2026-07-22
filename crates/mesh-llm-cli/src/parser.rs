mod commands;
mod normalization;
mod runtime_surface_help;
mod validation;

pub use commands::{
    AuthCommand, BinaryFlavor, Cli, Command, ConfigCommand, DiscoveryScope, DoctorCommand,
    GpuCommand, MeshDiscoveryMode, MeshGuardrailCliMode, PluginCommand, SkillAgentArg,
    SkillCommand, TrustCommand, TrustPolicy,
};
pub use normalization::{
    NormalizedRuntimeArgs, RuntimeSurface, legacy_runtime_surface_warning,
    normalize_runtime_surface_args,
};
pub use runtime_surface_help::runtime_surface_help;
pub use validation::validate_discovery_mode_args;

#[cfg(test)]
mod setup_tests;

#[cfg(test)]
mod uninstall_tests;

#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
use clap::Parser;

#[cfg(test)]
use crate::runtime::RuntimeCommand;

#[cfg(test)]
pub fn assert_mesh_requirements_docs_examples_parse() {
    let unrestricted_args =
        normalize_runtime_surface_args(["mesh-llm", "serve", "--model", "Qwen3-8B-Q4_K_M"]);
    let unrestricted = Cli::parse_from(unrestricted_args.normalized.clone());
    assert!(unrestricted.command.is_none());
    assert_eq!(unrestricted.model, vec![PathBuf::from("Qwen3-8B-Q4_K_M")]);
    assert!(!unrestricted.publish);

    let signed_public_args = normalize_runtime_surface_args([
        "mesh-llm",
        "serve",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--publish",
        "--require-release-attestation",
        "--release-signer-key",
        "ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "--owner-key",
        "~/.mesh-llm/owner-keystore.json",
        "--owner-required",
        "--trust-policy",
        "require-owned",
        "--node-label",
        "lab-a",
    ]);
    let signed_public = Cli::parse_from(signed_public_args.normalized.clone());
    assert!(signed_public.command.is_none());
    assert_eq!(signed_public.model, vec![PathBuf::from("Qwen3-8B-Q4_K_M")]);
    assert!(signed_public.publish);
    assert!(signed_public.require_release_attestation);
    assert_eq!(
        signed_public.release_signer_key,
        vec![
            "ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
        ]
    );
    assert_eq!(
        signed_public.owner_key,
        Some(PathBuf::from("~/.mesh-llm/owner-keystore.json"))
    );
    assert!(signed_public.owner_required);
    assert_eq!(signed_public.trust_policy, Some(TrustPolicy::RequireOwned));
    assert_eq!(signed_public.node_label, Some("lab-a".to_string()));

    let signed_bootstrap_args =
        normalize_runtime_surface_args(["mesh-llm", "serve", "--join", "signed-bootstrap-token"]);
    let signed_bootstrap = Cli::parse_from(signed_bootstrap_args.normalized.clone());
    assert!(signed_bootstrap.command.is_none());
    assert_eq!(
        signed_bootstrap.join,
        vec!["signed-bootstrap-token".to_string()]
    );

    let runtime_bootstrap = Cli::parse_from(["mesh-llm", "runtime", "bootstrap", "--port", "3131"]);
    match runtime_bootstrap.command.expect("runtime command expected") {
        Command::Runtime {
            command: Some(RuntimeCommand::Bootstrap { port, json }),
        } => {
            assert_eq!(port, 3131);
            assert!(!json);
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn mesh_requirements_docs_examples_parse() {
        super::assert_mesh_requirements_docs_examples_parse();
    }

    #[test]
    fn split_topology_lock_requires_split_mode() {
        let error = Cli::try_parse_from([
            "mesh-llm",
            "--model",
            "model.gguf",
            "--split-topology-lock",
            "topology.json",
        ])
        .expect_err("topology lock without --split should fail");

        assert!(error.to_string().contains("--split"));
    }

    #[test]
    fn split_topology_lock_parses_with_split_mode() {
        let cli = Cli::try_parse_from([
            "mesh-llm",
            "--model",
            "model.gguf",
            "--split",
            "--split-topology-lock",
            "topology.json",
        ])
        .expect("locked split CLI should parse");

        assert!(cli.split);
        assert_eq!(
            cli.split_topology_lock,
            Some(std::path::PathBuf::from("topology.json"))
        );
    }
}
