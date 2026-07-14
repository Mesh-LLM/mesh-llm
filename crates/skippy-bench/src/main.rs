mod chat_corpus;
mod cli;
mod distributed;
mod evals;
mod local_single;
mod local_split;
mod model_identity;
mod support;
mod telemetry_report;
mod token_lengths;
mod verify_span_local;

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
    verify_span_local::verify_span_local,
};

fn main() -> Result<()> {
    match Cli::parse().command {
        CommandKind::LocalSingle(args) => local_single(args),
        CommandKind::LocalSplitInprocess(args) => local_split_inprocess(args),
        CommandKind::LocalSplitBinary(args) => local_split_binary(args),
        CommandKind::LocalSplitCompare(args) => local_split_compare(args),
        CommandKind::LocalSplitChainBinary(args) => local_split_chain_binary(args),
        CommandKind::VerifySpanLocal(args) => verify_span_local(args),
        CommandKind::ChatCorpus(args) => chat_corpus(args),
        CommandKind::TokenLengths(args) => token_lengths(args),
        CommandKind::FocusedRuntime(args) => focused_runtime(args),
        CommandKind::Eval(args) => eval_command(args),
        CommandKind::Run(args) => run_distributed(args),
    }
}
