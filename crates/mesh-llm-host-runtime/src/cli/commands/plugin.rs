use anyhow::Result;

use crate::cli::{Cli, PluginCommand};

pub(crate) async fn run_plugin_command(command: &PluginCommand, cli: &Cli) -> Result<()> {
    let rows = if matches!(command, PluginCommand::List) {
        Some(crate::resolved_plugin_list_rows(cli)?)
    } else {
        None
    };
    mesh_llm_commands::plugin::run_plugin_command(command, rows.as_ref()).await?;
    Ok(())
}
