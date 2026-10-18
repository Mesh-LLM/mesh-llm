use super::{Cli, Command};
use clap::Parser;

fn console_of(args: &[&str]) -> u16 {
    match Cli::parse_from(args).command {
        Some(Command::Goose { console, .. } | Command::Claude { console, .. }) => console,
        _ => panic!("expected an agent launcher command"),
    }
}

#[test]
fn agent_launchers_default_to_the_default_console_port() {
    assert_eq!(console_of(&["mesh-llm", "goose"]), 3131);
    assert_eq!(console_of(&["mesh-llm", "claude"]), 3131);
}

#[test]
fn agent_launchers_accept_the_node_console_port() {
    assert_eq!(
        console_of(&["mesh-llm", "goose", "--console", "13131"]),
        13131
    );
    assert_eq!(
        console_of(&["mesh-llm", "claude", "--console", "13131"]),
        13131
    );
}
