//! Convert a paired DFlash GGUF into a compiled Core ML draft.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use skippy_ane::dflash::conversion::{ConversionOptions, convert};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    draft: PathBuf,
    #[arg(long)]
    target: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    embeddings_out: PathBuf,
    #[arg(long, default_value_t = 512)]
    context: usize,
    #[arg(long, default_value_t = 7)]
    rows: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    convert(&ConversionOptions {
        draft: args.draft,
        target: args.target,
        out: args.out,
        embeddings_out: args.embeddings_out,
        context: args.context,
        rows: args.rows,
    })
}
