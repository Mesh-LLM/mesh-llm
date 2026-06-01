#![recursion_limit = "256"]

use std::time::Duration;

use clap::Parser;

pub use mesh_llm_host_runtime::*;

pub async fn run_main() -> i32 {
    match run_cli_entrypoint().await {
        Ok(()) => 0,
        Err(err) => {
            let _ = mesh_llm_tui::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}

async fn run_cli_entrypoint() -> anyhow::Result<()> {
    let normalized_args = mesh_llm_cli::normalize_runtime_surface_args(std::env::args_os());
    let cli = mesh_llm_cli::Cli::parse_from(normalized_args.normalized.clone());
    let warning = mesh_llm_cli::legacy_runtime_surface_warning(
        &cli,
        &normalized_args.original,
        normalized_args.explicit_surface,
    );

    mesh_llm_host_runtime::run_cli(cli, normalized_args.explicit_surface, warning).await
}
