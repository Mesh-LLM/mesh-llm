mod chat_corpus;
mod cli;
mod direct_return_listener;
mod distributed;
mod evals;
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
    let prepared = match model_hf::prepare_download_directories() {
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

fn main() -> Result<()> {
    prepare_model_download_directories();
    match Cli::parse().command {
        CommandKind::LocalSingle(args) => local_single(args),
        CommandKind::LocalSplitInprocess(args) => local_split_inprocess(args),
        CommandKind::LocalSplitBinary(args) => local_split_binary(args),
        CommandKind::LocalSplitCompare(args) => local_split_compare(args),
        CommandKind::LocalSplitChainBinary(args) => local_split_chain_binary(args),
        CommandKind::VerifyWindowLocal(args) => verify_window_local(args),
        CommandKind::ChatCorpus(args) => chat_corpus(args),
        CommandKind::TokenLengths(args) => token_lengths(args),
        CommandKind::FocusedRuntime(args) => focused_runtime(args),
        CommandKind::Eval(args) => eval_command(args),
        CommandKind::Run(args) => run_distributed(args),
    }
}
