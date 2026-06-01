use anyhow::Result;
use mesh_llm_cli::{Cli, Command};

pub async fn dispatch(cli: &Cli) -> Result<bool> {
    let Some(command) = cli.command.as_ref() else {
        return Ok(false);
    };

    match command {
        Command::Gpus { json, command } => {
            mesh_llm_commands::gpus::dispatch_gpu_command(*json, command.as_ref())?;
            Ok(true)
        }
        Command::Runtime { command } => dispatch_runtime_command(command.as_ref()).await,
        Command::Update { .. } => {
            mesh_llm_commands::update::run_update(cli).await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

async fn dispatch_runtime_command(
    command: Option<&mesh_llm_cli::runtime::RuntimeCommand>,
) -> Result<bool> {
    use mesh_llm_cli::runtime::RuntimeCommand;

    match command {
        Some(RuntimeCommand::List {
            available,
            manifest,
            bundle_dirs,
            cache_dir,
            json,
            ..
        }) => {
            mesh_llm_commands::runtime_native::run_native_runtime_list(
                *available,
                manifest.as_deref(),
                bundle_dirs,
                cache_dir.as_deref(),
                *json,
            )
            .await?;
            Ok(true)
        }
        Some(RuntimeCommand::Install {
            runtime,
            manifest,
            bundle_dirs,
            cache_dir,
            json,
        }) => {
            mesh_llm_commands::runtime_native::run_native_runtime_install(
                runtime.as_deref(),
                manifest.as_deref(),
                bundle_dirs,
                cache_dir.as_deref(),
                *json,
            )
            .await?;
            Ok(true)
        }
        Some(RuntimeCommand::Remove {
            native_runtime_id,
            mesh_version,
            cache_dir,
            json,
        }) => {
            mesh_llm_commands::runtime_native::run_native_runtime_remove(
                native_runtime_id,
                mesh_version.as_deref(),
                cache_dir.as_deref(),
                *json,
            )?;
            Ok(true)
        }
        Some(RuntimeCommand::Prune {
            active_only,
            mesh_version,
            cache_dir,
            json,
        }) => {
            mesh_llm_commands::runtime_native::run_native_runtime_prune(
                *active_only,
                mesh_version.as_deref(),
                cache_dir.as_deref(),
                *json,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
