mod analytics;
mod config;
mod discover;
mod doctor;
mod download;
#[cfg(test)]
mod load_tests;
mod models;
mod plugin_cli;
mod runtime;
mod setup;

use anyhow::Result;
use mesh_llm_cli::{Cli, Command, DiscoverArgs, ModelPrepareArgs, UninstallArgs};

use self::analytics::dispatch_analytics_command;
use self::config::dispatch_config_command;
use self::discover::{DiscoverOptions, run_discover, run_stop};
use self::doctor::dispatch_doctor_command;
use self::download::dispatch_download_command;
use self::models::dispatch_models_command;
use self::plugin_cli::run_external_plugin_command;
use self::runtime::{LoadSizing, dispatch_runtime_command, run_drop, run_load, run_status};
use self::setup::dispatch_setup_command;

pub async fn dispatch(cli: &Cli) -> Result<bool> {
    let Some(cmd) = cli.command.as_ref() else {
        return Ok(false);
    };
    if matches!(cmd, Command::Serve | Command::Client) {
        return Ok(false);
    }
    let family = match cmd {
        Command::ExternalPlugin(args) => plugin_cli::external_cli_command_family(cli, args),
        _ => mesh_llm_commands::operational_logging::command_family(cmd),
    };
    let command_boundary =
        mesh_llm_commands::operational_logging::CommandDispatchBoundary::start_with_cli(
            cli, family,
        );
    let result = dispatch_command(cli, cmd).await;
    command_boundary.finish(&result);
    result?;
    Ok(true)
}

async fn dispatch_command(cli: &Cli, cmd: &Command) -> Result<()> {
    match cmd {
        Command::Auth { command } => mesh_llm_commands::auth::run_auth_command(command),
        Command::KvCache { command } => {
            mesh_llm_commands::kv_cache::dispatch_kv_cache_command(command).await
        }
        Command::Runtime { command } => {
            dispatch_runtime_command(
                command.as_ref(),
                cli.config.as_deref(),
                cli.placement.llama_flavor,
            )
            .await
        }
        Command::ModelPrepare(_) => dispatch_model_prepare(cmd).await,
        Command::Hermes(args) => mesh_llm_commands::agent_cli::config_write::run(args, true).await,
        Command::Openclaw(args) => {
            mesh_llm_commands::agent_cli::config_write::run(args, false).await
        }
        _ => dispatch_general_command(cli, cmd).await,
    }
}

/// `gpus`, model and download commands read this host's devices and memory.
/// With a dynamic native runtime those come from the loaded runtime, so load
/// the one `serve` would select (bundled or installed, never downloaded)
/// before surveying. Without it the survey falls back to what the OS reports
/// and can disagree with what `serve` advertises. A runtime that fails to load
/// leaves that fallback in place rather than failing a read-only command.
///
/// The model subcommands that render the local fit budget (`search`, `show`)
/// or resolve a repo ref fit-aware (`download`) need the same survey; the
/// others never read device memory and skip the load.
async fn load_installed_native_runtime(cli: &Cli) {
    if let Err(error) = mesh_llm_host_runtime::load_installed_native_runtime_for_cli(
        cli.config.as_deref(),
        cli.placement.llama_flavor.map(crate::map_binary_flavor),
    )
    .await
    {
        use std::io::Write as _;
        if cli.logging.debug {
            // The full chain is debug-only: it names settings these commands
            // forced themselves (downloads off), which reads as advice about
            // config the user never touched.
            let _ = writeln!(
                mesh_llm_events::console_err(),
                "native runtime load failed: {error:#}"
            );
        }
        // stderr, so `--json` output on stdout stays machine-readable.
        let _ = writeln!(
            mesh_llm_events::console_err(),
            "⚠️ Native runtime not loaded; this host's device memory comes from what the OS reports and may differ from what `serve` advertises. Run `mesh-llm runtime install` to match `serve`."
        );
    }
}

/// Model and download commands size and select models for this host with the
/// same local fit budget as `serve`, so they honour `gpu.host_ram_offload`
/// from the config. An absent config leaves the default, off; an unreadable or
/// invalid one fails the command, as it fails `serve`, rather than selecting
/// against a budget the owner did not ask for.
fn apply_config_host_ram_offload(config_path: Option<&std::path::Path>) -> Result<()> {
    let config = mesh_llm_host_runtime::command_support::plugin::load_config(config_path)?;
    mesh_llm_system::capacity::set_process_host_ram_offload(
        config.gpu.host_ram_offload.unwrap_or(false),
    );
    Ok(())
}

/// The `models` subcommands that render the local fit budget in their output
/// (`search`, `show`) or resolve a repo ref fit-aware (`download`), so they
/// read this host's device memory and need the runtime-backed survey.
fn models_command_reads_fit_budget(command: &mesh_llm_cli::models::ModelsCommand) -> bool {
    matches!(
        command,
        mesh_llm_cli::models::ModelsCommand::Search { .. }
            | mesh_llm_cli::models::ModelsCommand::Show { .. }
            | mesh_llm_cli::models::ModelsCommand::Download { .. }
    )
}

/// Whether a `gpus` subcommand reads the hardware survey. `run-benchmark`
/// resolves and runs its own benchmark tool and never reads the survey, so it
/// skips the runtime load.
fn gpu_command_reads_survey(command: Option<&mesh_llm_cli::GpuCommand>) -> bool {
    !matches!(command, Some(mesh_llm_cli::GpuCommand::RunBenchmark { .. }))
}

/// Dispatch general CLI commands and propagate the selected command's result.
async fn dispatch_general_command(cli: &Cli, cmd: &Command) -> Result<()> {
    match cmd {
        Command::Serve | Command::Client => Ok(()),
        Command::Models { command } => {
            apply_config_host_ram_offload(cli.config.as_deref())?;
            if models_command_reads_fit_budget(command) {
                load_installed_native_runtime(cli).await;
            }
            dispatch_models_command(command).await?;
            Ok(())
        }
        Command::Download { name, draft } => {
            apply_config_host_ram_offload(cli.config.as_deref())?;
            load_installed_native_runtime(cli).await;
            dispatch_download_command(name.as_deref(), *draft).await
        }
        Command::Update { .. } => mesh_llm_commands::update::run_update(cli).await,
        Command::Gpus { json, command } => {
            if gpu_command_reads_survey(command.as_ref()) {
                load_installed_native_runtime(cli).await;
            }
            mesh_llm_commands::gpus::dispatch_gpu_command(
                *json,
                command.as_ref(),
                cli.config.as_deref(),
            )?;
            Ok(())
        }
        Command::Runtime { .. } | Command::KvCache { .. } => {
            unreachable!("runtime and kv-cache commands are dispatched before general commands")
        }
        Command::Setup(_) => {
            dispatch_setup_command(cmd, cli.config.as_deref(), cli.placement.llama_flavor).await
        }
        Command::Analytics { command } => {
            dispatch_analytics_command(command, cli.config.as_deref())
        }
        Command::Uninstall(UninstallArgs {
            dry_run,
            yes,
            keep_cache,
            keep_service_files,
            purge_config,
            keep_config,
            binary_path,
            json,
            verbose,
        }) => mesh_llm_commands::uninstall::run_uninstall_command(
            mesh_llm_commands::uninstall::UninstallOptions {
                dry_run: *dry_run,
                yes: *yes,
                keep_cache: *keep_cache,
                keep_service_files: *keep_service_files,
                purge_config: *purge_config,
                keep_config: *keep_config,
                binary_path: binary_path.clone(),
                json: *json,
                verbose: *verbose,
            },
            run_stop,
        ),
        Command::Config { command } => dispatch_config_command(cli, command),
        Command::Doctor { command, json } => {
            dispatch_doctor_command(
                command.as_ref(),
                cli.config.as_deref(),
                cli.placement.llama_flavor,
                *json,
            )
            .await
        }
        Command::Load {
            name,
            port,
            ctx_size,
            parallel,
        } => run_load(name, *port, LoadSizing::from_flags(*ctx_size, *parallel)).await,
        Command::Unload { name, port } => run_drop(name, *port).await,
        Command::Status { port } => run_status(*port).await,
        Command::Stop => run_stop(),
        Command::ServiceHost => mesh_llm_commands::setup::run_service_host(),
        Command::Discover(DiscoverArgs {
            name,
            model,
            min_vram,
            region,
            auto,
            relay,
            // resolved through cli.effective_mesh_discovery_mode()
            mesh_discovery_mode: _,
        }) => {
            // The runtime surfaces enforce their own relay/mode constraint at
            // startup; the one-shot discover path has no runtime options, so
            // the CLI-level check is enforced here.
            mesh_llm_cli::validate_discovery_mode_args(cli)?;
            run_discover(DiscoverOptions {
                name: name.clone(),
                model: model.clone(),
                min_vram_gb: *min_vram,
                region: region.clone(),
                auto_join: *auto,
                relays: relay.clone(),
                discovery_mode: cli.effective_mesh_discovery_mode(),
                supplied_join_tokens: cli.joining.join.clone(),
            })
            .await
        }
        Command::RotateKey => {
            mesh_llm_host_runtime::command_support::discovery::nostr::rotate_keys()
        }
        Command::Goose { .. }
        | Command::Claude { .. }
        | Command::Pi { .. }
        | Command::Opencode { .. } => dispatch_agent_command(cmd).await,
        Command::Skills { command } => mesh_llm_commands::skills::run_skills_command(command),
        Command::Plugin { command } => run_plugin_command(command, cli).await,
        Command::Benchmark { command } => {
            mesh_llm_commands::benchmark::dispatch_benchmark_command(cli.config.as_deref(), command)
                .await
        }
        Command::ModelPrepare(_)
        | Command::Auth { .. }
        | Command::Hermes(_)
        | Command::Openclaw(_) => unreachable!("handled by dispatch_command"),
        Command::ExternalPlugin(args) => run_external_plugin_command(cli, args).await,
    }
}

async fn dispatch_agent_command(cmd: &Command) -> Result<()> {
    match cmd {
        Command::Goose {
            model,
            port,
            console,
        } => mesh_llm_commands::agent_cli::run_goose(model.clone(), *port, *console).await,
        Command::Claude {
            model,
            port,
            console,
        } => mesh_llm_commands::agent_cli::run_claude(model.clone(), *port, *console).await,
        Command::Pi { model, host, write } => {
            mesh_llm_commands::agent_cli::run_pi(model.clone(), host, *write).await
        }
        Command::Opencode { model, host, write } => {
            mesh_llm_commands::agent_cli::run_opencode(model.clone(), host, *write).await
        }
        _ => unreachable!("dispatch_agent_command called for non-agent command"),
    }
}

async fn dispatch_model_prepare(cmd: &Command) -> Result<()> {
    let Command::ModelPrepare(ModelPrepareArgs {
        source_repo,
        quant,
        target,
        model_id,
        flavor,
        timeout,
        mesh_llm_ref,
        dry_run,
        confirm,
        follow,
        json,
        status,
        logs,
        cancel,
        list,
        update_script,
    }) = cmd
    else {
        unreachable!("dispatch_model_prepare called for non-model-prepare command");
    };

    mesh_llm_commands::model_package::dispatch_model_package(
        mesh_llm_commands::model_package::ModelPrepareArgs {
            source_repo: source_repo.as_deref(),
            quant: quant.as_deref(),
            target: target.as_deref(),
            model_id: model_id.as_deref(),
            generation_defaults: None,
            flavor,
            timeout,
            mesh_llm_ref,
            experimental: false,
            dry_run: *dry_run,
            confirm: *confirm,
            follow: *follow,
            json: *json,
            status: status.as_deref(),
            logs: logs.as_deref(),
            cancel: cancel.as_deref(),
            list: *list,
            update_script: *update_script,
        },
    )
    .await
}

async fn run_plugin_command(command: &mesh_llm_cli::PluginCommand, cli: &Cli) -> Result<()> {
    match command {
        mesh_llm_cli::PluginCommand::List => {
            let mut rows = resolved_plugin_list_rows(cli)?;
            rows.providers =
                mesh_llm_commands::plugin::fetch_capability_providers(cli.node.console).await;
            mesh_llm_commands::plugin::run_plugin_command(command, Some(&rows)).await?;
        }
        mesh_llm_cli::PluginCommand::Info { .. } => {
            if !mesh_llm_commands::plugin::run_plugin_command(command, None).await? {
                let rows = resolved_plugin_list_rows(cli)?;
                mesh_llm_commands::plugin::run_plugin_command(command, Some(&rows)).await?;
            }
        }
        _ => {
            mesh_llm_commands::plugin::run_plugin_command(command, None).await?;
        }
    }
    Ok(())
}

fn resolved_plugin_list_rows(cli: &Cli) -> Result<mesh_llm_commands::plugin::PluginListRows> {
    let options = plugin_runtime_options_from_cli(cli);
    let resolved = mesh_llm_host_runtime::command_support::plugin::load_resolved_plugins(&options)?;
    Ok(mesh_llm_commands::plugin::PluginListRows {
        externals: resolved
            .externals
            .into_iter()
            .map(|spec| mesh_llm_commands::plugin::RuntimePluginRow {
                name: spec.name,
                command: spec.command,
                args: spec.args,
            })
            .collect(),
        inactive: resolved
            .inactive
            .into_iter()
            .map(|summary| mesh_llm_commands::plugin::InactivePluginRow {
                name: summary.name,
                kind: summary.kind,
                status: summary.status,
                error: summary.error,
            })
            .collect(),
        providers: None,
    })
}

fn plugin_runtime_options_from_cli(cli: &Cli) -> mesh_llm_host_runtime::RuntimeOptions {
    mesh_llm_host_runtime::RuntimeOptions {
        config: cli.config.clone(),
        publish: cli.mesh_policy.publish,
        nostr_discovery: cli.nostr_discovery,
        ..mesh_llm_host_runtime::RuntimeOptions::default()
    }
}

#[cfg(test)]
mod discovery_mode_dispatch_tests {
    use super::dispatch_general_command;
    use clap::Parser;
    use mesh_llm_cli::{Cli, Command};

    #[tokio::test]
    async fn discover_dispatch_rejects_relays_in_mdns_mode_before_any_discovery() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "discover",
            "--mesh-discovery-mode",
            "mdns",
            "--relay",
            "wss://relay.example",
        ]);
        let command = cli.command.as_ref().expect("discover subcommand");
        assert!(matches!(command, Command::Discover { .. }));

        let err = dispatch_general_command(&cli, command)
            .await
            .expect_err("dispatch must enforce the relay/mode constraint before discovering");
        assert!(
            err.to_string().contains("discover --relay"),
            "error should name the discover relay constraint: {err}"
        );
    }
}
