mod cli;
mod conversion;
mod local_model;
mod runtime;

#[cfg(unix)]
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command};

#[tokio::main]
async fn main() -> Result<()> {
    skippy_commands::console::install();
    let cli = Cli::parse();
    let native_options = runtime::resolve_options(cli.native_runtime)?;
    #[cfg(feature = "dynamic-native-runtime")]
    if matches!(
        &cli.command,
        Command::Serve(_)
            | Command::ServeBinary(_)
            | Command::ServeOpenAi(_)
            | Command::PlanSplit(_)
    ) {
        skippy_api::native_runtime::load_local_native_runtime(&native_options)?;
    }
    match cli.command {
        Command::Serve(args) => {
            skippy_server::http::serve_stage_http_with_shutdown(
                conversion::stage_http_options(args)?,
                shutdown_signal()?,
            )
            .await
        }
        Command::ServeBinary(args) => {
            skippy_server::binary_transport::serve_binary_stage_with_shutdown(
                conversion::binary_stage_options(args)?,
                shutdown_signal()?,
            )
            .await
        }
        Command::ServeOpenAi(args) => {
            skippy_api::serving::serve_local_openai_with_shutdown(
                conversion::local_openai_options(args)?,
                shutdown_signal()?,
            )
            .await
        }
        Command::Models { cache_dir, command } => {
            skippy_commands::models::run(cache_dir, command.into()).await
        }
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
