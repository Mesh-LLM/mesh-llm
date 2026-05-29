use anyhow::Result;

use crate::cli::Cli;
use crate::cli::Command;
use crate::system::autoupdate;

pub async fn run_update(cli: &Cli) -> Result<()> {
    let (requested_version, flavor, detect_flavor) = match &cli.command {
        Some(Command::Update {
            version,
            flavor,
            detect_flavor,
        }) => (version.as_deref(), *flavor, *detect_flavor),
        _ => (None, None, false),
    };
    autoupdate::run_update_command(autoupdate::UpdateCommandOptions {
        flavor,
        detect_flavor,
        requested_version,
        current_version: crate::VERSION,
    })
    .await
}
