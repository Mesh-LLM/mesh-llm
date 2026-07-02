use clap::{Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub enum BenchmarkCommand {
    /// Tune model-serving settings by running isolated throughput trials.
    Tune {
        /// Tune exactly one local/configured model target.
        #[arg(long, conflicts_with = "models")]
        model: Option<String>,
        /// Tune multiple local/configured model targets from a comma-separated list.
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// Print machine-readable JSON output.
        #[arg(long)]
        json: bool,
        /// Context sizes to benchmark, as a comma-separated token list.
        #[arg(long, value_delimiter = ',')]
        ctx_sizes: Vec<u32>,
        /// Batch sizes to benchmark, as a comma-separated list.
        #[arg(long, value_delimiter = ',')]
        batch_sizes: Vec<u32>,
        /// Micro-batch sizes to benchmark, as a comma-separated list.
        #[arg(long, value_delimiter = ',')]
        ubatch_sizes: Vec<u32>,
        /// mmap values to benchmark independently: auto, enabled, disabled.
        #[arg(long = "mmap-values", value_delimiter = ',')]
        mmap_values: Vec<BenchmarkBoolOrAuto>,
        /// mlock values to benchmark independently: enabled, disabled.
        #[arg(long = "mlock-values", value_delimiter = ',')]
        mlock_values: Vec<BenchmarkBool>,
        /// Treat candidates within this percent of the raw best tok/s as throughput-equivalent.
        #[arg(long, default_value_t = 10.0)]
        throughput_tolerance_pct: f64,
        /// Maximum generated tokens per benchmark request.
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        /// Startup wait limit for each benchmark trial.
        #[arg(long, default_value_t = 600)]
        startup_timeout_secs: u64,
        /// HTTP request timeout for each benchmark request.
        #[arg(long, default_value_t = 600)]
        request_timeout_secs: u64,
        /// Prompt sent during benchmark trials.
        #[arg(
            long,
            default_value = "Write a concise paragraph about distributed GPU inference."
        )]
        prompt: String,
    },
    /// Import a prompt corpus from a supported online source into local JSONL.
    #[command(name = "import-prompts")]
    ImportPrompts {
        /// Online source to import.
        #[arg(long, value_enum)]
        source: PromptImportSource,
        /// Maximum number of prompts to import.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Optional per-prompt decode budget hint written into the corpus.
        #[arg(long)]
        max_tokens: Option<u32>,
        /// Output JSONL path.
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BenchmarkBoolOrAuto {
    Auto,
    #[value(alias = "true")]
    Enabled,
    #[value(alias = "false")]
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BenchmarkBool {
    #[value(alias = "true")]
    Enabled,
    #[value(alias = "false")]
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum GpuBenchmarkBackend {
    Metal,
    Cuda,
    Hip,
    Intel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PromptImportSource {
    MtBench,
    Gsm8k,
    Humaneval,
}
