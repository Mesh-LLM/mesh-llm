use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
pub struct PromptArgs {
    #[arg(long, default_value = "http://127.0.0.1:9337/v1")]
    pub endpoint: String,
    #[arg(
        long,
        help = "Model ID; defaults to the first model returned by /models"
    )]
    pub model: Option<String>,
    /// Override the server completion budget; omitted by default.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_new_tokens: Option<u32>,
    #[arg(long, help = "Use /completions with each line as a raw prompt")]
    pub raw: bool,
    #[arg(long, help = "Disable model thinking through reasoning_effort=none")]
    pub no_think: bool,
    #[arg(long)]
    pub history_path: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};

    #[test]
    fn prompt_accepts_a_running_endpoint_without_model_files() {
        let cli = Cli::try_parse_from([
            "skippy",
            "prompt",
            "--endpoint",
            "http://127.0.0.1:9337/v1",
            "--no-think",
        ])
        .unwrap();
        let Command::Prompt(args) = cli.command else {
            panic!("expected prompt command");
        };
        assert_eq!(args.endpoint, "http://127.0.0.1:9337/v1");
        assert!(args.no_think);
        assert!(args.model.is_none());
    }

    #[test]
    fn prompt_inherits_server_defaults_and_only_sends_explicit_output_limits() {
        let cli = Cli::try_parse_from(["skippy", "prompt"]).unwrap();
        let Command::Prompt(args) = cli.command else {
            panic!("expected prompt");
        };
        assert_eq!(args.max_new_tokens, None);
        let cli = Cli::try_parse_from(["skippy", "prompt", "--max-new-tokens", "2048"]).unwrap();
        let Command::Prompt(args) = cli.command else {
            panic!("expected prompt");
        };
        assert_eq!(args.max_new_tokens, Some(2048));
        assert!(Cli::try_parse_from(["skippy", "prompt", "--max-new-tokens", "0"]).is_err());
    }
}
