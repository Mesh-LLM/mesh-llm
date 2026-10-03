mod cli;
mod conversion;
mod disk_cache;
mod local_model;
mod local_resource_planning;
mod native_logging;
mod runtime;
mod serve;

#[cfg(unix)]
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command, OutputFormat};
use std::io::IsTerminal;
use std::sync::Arc;

fn main() -> std::process::ExitCode {
    let startup_warnings = prepare_model_download_directories();
    let cli = match parse_cli() {
        Ok(Some(cli)) => cli,
        Ok(None) => return std::process::ExitCode::SUCCESS,
        Err(error) => {
            let _ = skippy_commands::console::failure(&error);
            return std::process::ExitCode::FAILURE;
        }
    };
    let native_logs = Arc::new(native_logging::NativeDiagnostics::new(cli.debug));
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = skippy_commands::console::failure(&error.into());
            return std::process::ExitCode::FAILURE;
        }
    };
    let result = runtime.block_on(run_main(cli, startup_warnings, native_logs.clone()));
    runtime.shutdown_timeout(std::time::Duration::from_secs(1));
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            native_logs.flush_on_error();
            let _ = skippy_commands::console::failure(&error);
            std::process::ExitCode::FAILURE
        }
    }
}

fn prepare_model_download_directories() -> Vec<String> {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    match skippy_model_hf::prepare_cli_download_directories() {
        Ok(prepared) => {
            let warnings = prepared
                .fallbacks
                .iter()
                .map(|fallback| format!("⚠ {fallback}"))
                .collect();
            // SAFETY: This runs in synchronous main before Tokio creates worker threads.
            unsafe { prepared.apply_to_process_environment() };
            warnings
        }
        Err(error) => vec![format!(
            "⚠ Unable to prepare model download directories: {error:#}. Model downloads may fail; set MESH_LLM_DATA_DIR to a writable directory."
        )],
    }
}

async fn run_main(
    cli: Cli,
    startup_warnings: Vec<String>,
    native_logs: Arc<native_logging::NativeDiagnostics>,
) -> Result<()> {
    let output = match (&cli.command, cli.output) {
        (Command::Models { command }, _) if command.json() => OutputFormat::Json,
        (Command::Serve(_), OutputFormat::Auto) if !std::io::stdout().is_terminal() => {
            OutputFormat::Jsonl
        }
        (Command::Models { .. } | Command::Prompt(_), OutputFormat::Auto) => OutputFormat::Human,
        _ => cli.output,
    };
    skippy_commands::console::install(output.into());
    for warning in startup_warnings {
        skippy_commands::console::status(&warning)?;
    }
    if matches!(&cli.command, Command::Prompt(_))
        && skippy_commands::console::mode() != skippy_commands::console::OutputMode::Human
    {
        anyhow::bail!("interactive prompt requires human output; agents should call the API");
    }
    if let Command::Serve(args) = &cli.command {
        serve::validate(args)?;
    }
    #[cfg(feature = "dynamic-native-runtime")]
    let automatic_runtime = cli.native_runtime.bundle_dirs.is_empty()
        && cli.native_runtime.release.is_none()
        && cli.native_runtime.selection.is_none();
    let native_options = runtime::resolve_options(cli.native_runtime)?;
    #[cfg(feature = "dynamic-native-runtime")]
    if matches!(&cli.command, Command::Serve(_) | Command::PlanSplit(_)) {
        runtime::prepare_native_runtime(&native_options, automatic_runtime).await?;
    }
    skippy_runtime::logging::set_native_log_sink(native_logs);
    match cli.command {
        Command::Doctor => runtime::doctor(&native_options),
        Command::Prompt(args) => {
            tokio::task::spawn_blocking(move || {
                skippy_commands::prompt::run(skippy_commands::prompt::PromptCommand {
                    endpoint: args.endpoint,
                    model: args.model,
                    max_new_tokens: args.max_new_tokens,
                    raw: args.raw,
                    no_think: args.no_think,
                    history_path: args.history_path,
                })
            })
            .await?
        }
        Command::Serve(args) => serve::run(*args).await,
        Command::Models { command } => skippy_commands::models::run_standalone(&command).await,
        Command::PlanSplit(args) => skippy_commands::split::run(args.into()),
        Command::Runtime { command } => {
            skippy_commands::runtime::run(
                command.into(),
                &runtime::command_options(&native_options),
            )
            .await
        }
        Command::ExampleConfig => {
            skippy_commands::console::write_json(&skippy_config::example_config())
        }
    }
}

fn parse_cli() -> Result<Option<Cli>> {
    match Cli::try_parse() {
        Ok(cli) => Ok(Some(cli)),
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error.print()?;
            Ok(None)
        }
        Err(error) => {
            if requested_jsonl_output() {
                skippy_commands::console::install(skippy_commands::console::OutputMode::Jsonl);
            }
            anyhow::bail!(error.to_string())
        }
    }
}

fn requested_jsonl_output() -> bool {
    let args = std::env::args_os().collect::<Vec<_>>();
    args.windows(2)
        .any(|pair| pair[0] == "--output" && pair[1] == "jsonl")
        || args.iter().any(|arg| arg == "--output=jsonl")
}

fn shutdown_signal() -> Result<impl std::future::Future<Output = ()> + Send + 'static> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .context("install SIGTERM handler")?;
        let mut interrupt =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                .context("install SIGINT handler")?;
        Ok(async move {
            tokio::select! {
                _ = interrupt.recv() => {}
                _ = terminate.recv() => {}
            }
        })
    }
    #[cfg(not(unix))]
    {
        Ok(async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                let _ = skippy_events::diagnostics::emit(
                    skippy_events::diagnostics::ServingDiagnostic::Warning {
                        message: format!("interrupt handler failed: {error}"),
                        context: None,
                    },
                );
            }
        })
    }
}
