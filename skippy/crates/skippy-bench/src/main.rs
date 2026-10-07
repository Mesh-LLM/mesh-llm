mod chat_corpus;
mod cli;
mod direct_return_listener;
mod distributed;
mod evals;
mod l2_tier;
mod local_single;
mod local_split;
mod model_identity;
mod support;
mod telemetry_report;
mod token_lengths;
mod verify_window_local;

use anyhow::Result;
use clap::Parser;

use crate::{
    chat_corpus::chat_corpus,
    cli::{Cli, CommandKind},
    distributed::{focused_runtime, run_distributed},
    evals::eval_command,
    local_single::local_single,
    local_split::{
        local_split_binary, local_split_chain_binary, local_split_compare, local_split_inprocess,
    },
    token_lengths::token_lengths,
    verify_window_local::verify_window_local,
};

fn prepare_model_download_directories() {
    let prepared = match skippy_model_hf::prepare_download_directories() {
        Ok(prepared) => prepared,
        Err(error) => {
            eprintln!(
                "⚠ Unable to prepare model download directories: {error:#}. \
                 Model downloads may fail; set MESH_LLM_DATA_DIR to a writable directory."
            );
            return;
        }
    };
    for fallback in &prepared.fallbacks {
        eprintln!("⚠ {fallback}");
    }
    // SAFETY: runs before any Tokio runtime, process is single-threaded.
    unsafe { prepared.apply_to_process_environment() };
}

fn needs_model_download_preparation(command: &CommandKind) -> bool {
    let CommandKind::Eval(args) = command else {
        return true;
    };
    !matches!(
        &args.command,
        cli::EvalCommandKind::PortReady(_)
            | cli::EvalCommandKind::PatchSwerexIndex(_)
            | cli::EvalCommandKind::PatchSwerexModal(_)
            | cli::EvalCommandKind::PrepareMcp(_)
            | cli::EvalCommandKind::PrepareSwe(_)
    ) && !matches!(&args.command, cli::EvalCommandKind::Run(args) if matches!(args.eval, cli::EvalId::McpAtlas | cli::EvalId::SweBenchPro))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if needs_model_download_preparation(&cli.command) {
        prepare_model_download_directories();
    }
    match cli.command {
        CommandKind::LocalSingle(args) => local_single(args),
        CommandKind::LocalSplitInprocess(args) => local_split_inprocess(args),
        CommandKind::LocalSplitBinary(args) => local_split_binary(args),
        CommandKind::LocalSplitCompare(args) => local_split_compare(args),
        CommandKind::LocalSplitChainBinary(args) => local_split_chain_binary(args),
        CommandKind::VerifyWindowLocal(args) => verify_window_local(args),
        CommandKind::L2Tier(args) => l2_tier::l2_tier(args),
        CommandKind::ChatCorpus(args) => chat_corpus(args),
        CommandKind::TokenLengths(args) => token_lengths(args),
        CommandKind::FocusedRuntime(args) => focused_runtime(args),
        CommandKind::Eval(args) => eval_command(args),
        CommandKind::Run(args) => run_distributed(args),
    }
}

#[cfg(test)]
mod startup_tests {
    use super::*;
    #[test]
    fn mcp_optional_preparation_and_run_defer_model_cache_but_other_evals_preserve_startup() {
        let prepare = Cli::try_parse_from([
            "bench",
            "eval",
            "prepare-mcp",
            "--uv",
            "/uv",
            "--python",
            "/python",
        ])
        .unwrap();
        assert!(!needs_model_download_preparation(&prepare.command));
        let swe = Cli::try_parse_from([
            "bench",
            "eval",
            "prepare-swe",
            "--uv",
            "/uv",
            "--python",
            "/python3.11",
            "--deployment",
            "modal",
        ])
        .unwrap();
        assert!(!needs_model_download_preparation(&swe.command));
        for eval in [
            "mcp-atlas",
            "speed-bench",
            "terminal-bench",
            "swe-gym",
            "swe-bench-pro",
        ] {
            let cli = Cli::try_parse_from(["bench", "eval", "run", eval]).unwrap();
            assert_eq!(
                needs_model_download_preparation(&cli.command),
                !matches!(eval, "mcp-atlas" | "swe-bench-pro")
            );
        }
        let info = Cli::try_parse_from(["bench", "eval", "info", "mcp-atlas"]).unwrap();
        assert!(needs_model_download_preparation(&info.command));
        let probe = Cli::try_parse_from(["bench", "eval", "port-ready", "1984"]).unwrap();
        assert!(!needs_model_download_preparation(&probe.command));
    }
}
