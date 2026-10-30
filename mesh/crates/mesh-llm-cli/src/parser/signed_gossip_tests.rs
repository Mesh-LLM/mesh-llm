use super::{Cli, normalize_runtime_surface_args};
use clap::Parser;

#[test]
fn signed_gossip_is_opt_in_and_independent_of_release_certification() {
    let default_args = normalize_runtime_surface_args(["mesh-llm", "serve"]);
    let defaults = Cli::parse_from(default_args.normalized);
    assert!(!defaults.mesh_policy.require_signed_gossip);

    let strict_args = normalize_runtime_surface_args([
        "mesh-llm",
        "serve",
        "--require-signed-gossip",
        "--owner-key",
        "owner-keystore.json",
    ]);
    let strict = Cli::parse_from(strict_args.normalized);
    assert!(strict.mesh_policy.require_signed_gossip);
    assert!(!strict.mesh_policy.require_release_attestation);

    let certified_args =
        normalize_runtime_surface_args(["mesh-llm", "serve", "--require-release-attestation"]);
    let certified = Cli::parse_from(certified_args.normalized);
    assert!(!certified.mesh_policy.require_signed_gossip);
    assert!(certified.mesh_policy.require_release_attestation);
}
