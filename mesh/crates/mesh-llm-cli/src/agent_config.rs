//! Config-only harness connection options; never launch an agent or a node.
use clap::Args;
use std::path::PathBuf;

#[derive(Debug, Clone, Args)]
pub struct AgentConfigArgs {
    /// Write the Mesh provider configuration (required; no launcher is provided).
    #[arg(long, required = true)]
    pub write: bool,
    /// Running Mesh API host or URL.
    #[arg(long, default_value = "127.0.0.1:9337")]
    pub host: String,
    /// Wire model ID; auto lets Mesh route each request.
    #[arg(long, default_value = "auto")]
    pub model: String,
    /// Harness config file, including a custom profile's config.
    #[arg(long)]
    pub config_path: Option<PathBuf>,
    /// Explicit agent context budget; otherwise use served metadata or an 8192-token fallback.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1024..))]
    pub context_length: Option<u32>,
}

#[cfg(test)]
mod tests {
    use crate::{Cli, Command};
    use clap::Parser;

    #[test]
    fn config_commands_require_write_and_default_to_auto() {
        for name in ["hermes", "openclaw"] {
            assert!(Cli::try_parse_from(["mesh-llm", name]).is_err());
            let cli = Cli::try_parse_from(["mesh-llm", name, "--write"]).unwrap();
            let args = match cli.command.unwrap() {
                Command::Hermes(args) | Command::Openclaw(args) => args,
                _ => panic!("wrong command"),
            };
            assert_eq!(args.model, "auto");
            assert!(args.write);
        }
    }
}
