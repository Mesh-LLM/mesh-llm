use super::client_args;
use clap::Parser;
use mesh_llm_cli::Cli;

/// The child's arguments for a launcher invoked as `argv`.
fn child_args(argv: &[&str], port: u16, console_port: Option<u16>) -> Vec<String> {
    let cli = Cli::parse_from(argv);
    client_args(port, console_port, &cli.joining)
        .into_iter()
        .map(|arg| arg.into_string().expect("UTF-8 argument"))
        .collect()
}

#[test]
fn child_without_joining_flags_auto_joins_over_nostr() {
    assert_eq!(
        child_args(&["mesh-llm", "goose"], 9337, Some(3131)),
        [
            "client",
            "--auto",
            "--port",
            "9337",
            "--console",
            "3131",
            "--mesh-discovery-mode",
            "nostr",
        ]
    );
}

#[test]
fn child_keeps_the_mdns_discovery_mode() {
    assert_eq!(
        child_args(
            &["mesh-llm", "--mesh-discovery-mode", "mdns", "claude"],
            9337,
            Some(3131),
        ),
        [
            "client",
            "--auto",
            "--port",
            "9337",
            "--console",
            "3131",
            "--mesh-discovery-mode",
            "mdns",
        ]
    );
}

#[test]
fn child_keeps_every_join_token_in_order() {
    assert_eq!(
        child_args(
            &["mesh-llm", "--join", "token-a", "-j", "token-b", "pi"],
            9447,
            None,
        ),
        [
            "client",
            "--auto",
            "--port",
            "9447",
            "--mesh-discovery-mode",
            "nostr",
            "--join",
            "token-a",
            "--join",
            "token-b",
        ]
    );
}

#[test]
fn child_keeps_join_files_and_the_mode_together() {
    assert_eq!(
        child_args(
            &[
                "mesh-llm",
                "--mesh-discovery-mode",
                "mdns",
                "--join-file",
                "invite.token",
                "opencode",
            ],
            9337,
            None,
        ),
        [
            "client",
            "--auto",
            "--port",
            "9337",
            "--mesh-discovery-mode",
            "mdns",
            "--join-file",
            "invite.token",
        ]
    );
}
