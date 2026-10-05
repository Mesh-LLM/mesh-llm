//! Dispatch wiring for `mesh-llm analytics`.
//!
//! The command behavior lives in `mesh-llm-commands`; this crate carries
//! dispatch only.

use anyhow::Result;
use mesh_llm_cli::AnalyticsCommand;
use std::path::Path;

pub(crate) fn dispatch_analytics_command(
    command: &AnalyticsCommand,
    config_override: Option<&Path>,
) -> Result<()> {
    mesh_llm_commands::analytics::dispatch_analytics_command(command, config_override)
}
