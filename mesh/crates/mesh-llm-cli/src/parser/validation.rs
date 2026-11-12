use super::commands::{Cli, Command, MeshDiscoveryMode};
use super::subcommand_options::DiscoverArgs;

pub fn validate_discovery_mode_args(cli: &Cli) -> anyhow::Result<()> {
    if cli.effective_mesh_discovery_mode() != MeshDiscoveryMode::Mdns {
        return Ok(());
    }

    if !cli.network.nostr_relay.is_empty() {
        anyhow::bail!("--nostr-relay is only valid with --mesh-discovery-mode nostr");
    }
    if !cli.network.relay.is_empty() {
        anyhow::bail!("--relay is only valid with --mesh-discovery-mode nostr");
    }
    if !cli.network.relay_auth.is_empty() {
        anyhow::bail!("--relay-auth is only valid with --mesh-discovery-mode nostr");
    }
    if let Some(Command::Discover(DiscoverArgs { relay, .. })) = cli.command.as_ref()
        && !relay.is_empty()
    {
        anyhow::bail!("discover --relay is only valid with --mesh-discovery-mode nostr");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::parser::{
        Cli, MeshDiscoveryMode, normalize_runtime_surface_args, validate_discovery_mode_args,
    };
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

    #[test]
    fn discover_rejects_relays_when_mdns_is_set_after_the_subcommand() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "discover",
            "--mesh-discovery-mode",
            "mdns",
            "--relay",
            "wss://relay.example",
        ]);

        assert_eq!(cli.effective_mesh_discovery_mode(), MeshDiscoveryMode::Mdns);
        let err = validate_discovery_mode_args(&cli)
            .expect_err("subcommand-position mdns must still validate discover --relay");
        assert!(err.to_string().contains("discover --relay"));
    }

    #[test]
    fn discover_rejects_relays_when_mdns_is_set_before_the_subcommand() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "--mesh-discovery-mode",
            "mdns",
            "discover",
            "--relay",
            "wss://relay.example",
        ]);

        assert_eq!(cli.effective_mesh_discovery_mode(), MeshDiscoveryMode::Mdns);
        let err = validate_discovery_mode_args(&cli)
            .expect_err("top-level mdns position must also validate discover --relay");
        assert!(err.to_string().contains("discover --relay"));
    }

    #[test]
    fn discover_subcommand_mode_overrides_the_top_level_one() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "--mesh-discovery-mode",
            "nostr",
            "discover",
            "--mesh-discovery-mode",
            "mdns",
        ]);

        assert_eq!(cli.joining.mesh_discovery_mode, MeshDiscoveryMode::Nostr);
        assert_eq!(cli.effective_mesh_discovery_mode(), MeshDiscoveryMode::Mdns);
    }

    #[test]
    fn effective_mode_falls_back_to_the_top_level_default() {
        let cli = Cli::parse_from(["mesh-llm", "discover"]);

        assert_eq!(
            cli.effective_mesh_discovery_mode(),
            MeshDiscoveryMode::Nostr
        );
    }

    #[test]
    fn top_level_mesh_discovery_mode_before_discover_is_effective() {
        // The position the docs taught while the flag was global.
        let cli = Cli::parse_from(["mesh-llm", "--mesh-discovery-mode", "mdns", "discover"]);

        assert_eq!(cli.effective_mesh_discovery_mode(), MeshDiscoveryMode::Mdns);
    }

    #[test]
    fn discover_accepts_mesh_discovery_mode_after_the_subcommand_token() {
        let cli = Cli::parse_from(["mesh-llm", "discover", "--mesh-discovery-mode", "mdns"]);

        assert_eq!(cli.effective_mesh_discovery_mode(), MeshDiscoveryMode::Mdns);
    }

    #[test]
    fn non_discovery_subcommands_reject_mesh_discovery_mode() {
        // The flag used to be clap-global: every subcommand accepted it and
        // silently ignored it. Only `discover` (and the runtime surfaces)
        // consume it, so everywhere else clap now reports the unknown flag.
        for argv in [
            [
                "mesh-llm",
                "plugins",
                "list",
                "--mesh-discovery-mode",
                "mdns",
            ]
            .as_slice(),
            ["mesh-llm", "goose", "--mesh-discovery-mode", "mdns"].as_slice(),
            ["mesh-llm", "status", "--mesh-discovery-mode", "mdns"].as_slice(),
        ] {
            let err = Cli::try_parse_from(argv)
                .expect_err("non-discovery subcommands must reject --mesh-discovery-mode");
            assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
            assert!(
                err.to_string().contains("--mesh-discovery-mode"),
                "error should name the rejected flag: {err}"
            );
        }
    }

    #[test]
    fn top_level_mesh_discovery_mode_before_a_non_discovery_subcommand_is_accepted() {
        // Pinned residual, so it cannot change silently: top-level runtime
        // options placed before a one-shot subcommand are parsed by the
        // top-level surface and ignored by that subcommand, like --port or
        // --model. The scoping fix removes only the after-subcommand position,
        // where the subcommand's own help had advertised the flag.
        //
        // Known pre-existing gap this does NOT bless: the agent launchers
        // (goose/claude/pi/opencode) drop the mode entirely when they
        // auto-join, so `--mesh-discovery-mode mdns goose` still joins via
        // the default provider. Fixing that means plumbing joining options
        // into dispatch_agent_command — separate work.
        let cli = Cli::parse_from([
            "mesh-llm",
            "--mesh-discovery-mode",
            "mdns",
            "plugins",
            "list",
        ]);

        assert_eq!(cli.joining.mesh_discovery_mode, MeshDiscoveryMode::Mdns);
        assert!(cli.command.is_some());
    }
}
