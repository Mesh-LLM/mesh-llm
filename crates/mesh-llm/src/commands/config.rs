use anyhow::Result;
use mesh_llm_cli::{Cli, ConfigCommand};

pub(crate) fn dispatch_config_command(cli: &Cli, command: &ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Validate { config_path, json } => {
            mesh_llm_commands::config::run_config_validate(cli, config_path.as_deref(), *json)
        }
    }
}
