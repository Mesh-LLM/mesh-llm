use super::commands::{Cli, Command, MeshDiscoveryMode};
use super::subcommand_options::DiscoverArgs;

impl Cli {
    /// The discovery provider this invocation runs with.
    ///
    /// `discover` accepts `--mesh-discovery-mode` after the subcommand token
    /// too; that copy wins over the top-level one. For the runtime surfaces
    /// (serve/client, no subcommand) this equals the top-level value by
    /// construction, so `runtime_options_from_cli` may keep reading the field
    /// directly.
    #[must_use]
    pub fn effective_mesh_discovery_mode(&self) -> MeshDiscoveryMode {
        if let Some(Command::Discover(DiscoverArgs {
            mesh_discovery_mode: Some(mode),
            ..
        })) = self.command.as_ref()
        {
            return *mode;
        }
        self.joining.mesh_discovery_mode
    }
}

/// Discovery-mode constraint for the one-shot `discover` path.
///
/// Only the flags `discover` actually reads are checked here: in mdns mode the
/// command's Nostr relay list (`discover --relay`) contradicts the selected
/// provider. Top-level relay flags (`--relay`, `--nostr-relay`, `--relay-auth`)
/// are inert on this path — discover sets up no iroh transport and uses its own
/// relay list — so they are accepted and ignored like other top-level runtime
/// options before a one-shot subcommand. The runtime surfaces enforce the
/// top-level relay constraint against themselves at startup
/// (`RuntimeOptions::validate_discovery_mode_args`).
pub fn validate_discovery_mode_args(cli: &Cli) -> anyhow::Result<()> {
    if cli.effective_mesh_discovery_mode() != MeshDiscoveryMode::Mdns {
        return Ok(());
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
    fn discover_ignores_inert_top_level_relay_flags() {
        // Top-level relay flags do nothing on the one-shot discover path, so
        // mdns mode does not reject them there — the runtime surfaces check
        // them against themselves at startup instead.
        let cli = Cli::parse_from([
            "mesh-llm",
            "--relay",
            "https://iroh.example/",
            "--nostr-relay",
            "wss://relay.example",
            "discover",
            "--mesh-discovery-mode",
            "mdns",
        ]);

        validate_discovery_mode_args(&cli)
            .expect("top-level relay flags are inert on discover and must not be rejected");
    }

    #[test]
    fn nostr_mode_accepts_discover_relays() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "discover",
            "--mesh-discovery-mode",
            "nostr",
            "--relay",
            "wss://relay.example",
        ]);

        validate_discovery_mode_args(&cli).expect("nostr mode must accept discover --relay");
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

    #[test]
    fn serve_surface_normalization_still_feeds_the_runtime_validator() {
        // The runtime surfaces keep enforcing their own top-level relay
        // constraint at startup; this only pins that the CLI parse of the
        // mdns + top-level-relay combination still succeeds so the runtime
        // path (RuntimeOptions::validate_discovery_mode_args) is the one that
        // reports it.
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--mesh-discovery-mode",
            "mdns",
            "--relay",
            "https://relay.example/",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert!(normalized.explicit_surface.is_some());
        assert_eq!(cli.joining.mesh_discovery_mode, MeshDiscoveryMode::Mdns);
        assert!(!cli.network.relay.is_empty());
    }
}
