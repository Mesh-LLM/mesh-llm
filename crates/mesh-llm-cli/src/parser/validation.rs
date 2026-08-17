use super::commands::{Cli, Command, MeshDiscoveryMode};

pub fn validate_discovery_mode_args(cli: &Cli) -> anyhow::Result<()> {
    if cli.mesh_discovery_mode != MeshDiscoveryMode::Mdns {
        return Ok(());
    }

    if !cli.nostr_relay.is_empty() {
        anyhow::bail!("--nostr-relay is only valid with --mesh-discovery-mode nostr");
    }
    if !cli.relay.is_empty() {
        anyhow::bail!("--relay is only valid with --mesh-discovery-mode nostr");
    }
    if !cli.relay_auth.is_empty() {
        anyhow::bail!("--relay-auth is only valid with --mesh-discovery-mode nostr");
    }
    if let Some(Command::Discover { relay, .. }) = cli.command.as_ref()
        && !relay.is_empty()
    {
        anyhow::bail!("discover --relay is only valid with --mesh-discovery-mode nostr");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::parser::{Cli, normalize_runtime_surface_args, validate_discovery_mode_args};
    use clap::Parser;

    #[test]
    fn cli_rejects_nostr_relays_in_mdns_mode() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--mesh-discovery-mode",
            "mdns",
            "--nostr-relay",
            "wss://relay.example",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        let err = validate_discovery_mode_args(&cli)
            .expect_err("mdns mode must reject Nostr relay overrides");
        assert!(err.to_string().contains("--nostr-relay"));
    }

    #[test]
    fn cli_rejects_iroh_relays_in_mdns_mode() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--mesh-discovery-mode",
            "mdns",
            "--relay",
            "https://relay.example/",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        let err = validate_discovery_mode_args(&cli)
            .expect_err("mdns mode must reject iroh relay overrides");
        assert!(err.to_string().contains("--relay"));
    }

    #[test]
    fn cli_rejects_relay_auth_in_mdns_mode() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--mesh-discovery-mode",
            "mdns",
            "--relay-auth",
            "https://relay.example/=secret-token",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        let err = validate_discovery_mode_args(&cli).expect_err("mdns mode must reject relay auth");
        assert!(err.to_string().contains("--relay-auth"));
    }
}
