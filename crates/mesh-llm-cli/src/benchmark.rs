use clap::{Args, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Subcommand, Debug, Clone)]
pub enum BenchmarkCommand {
    /// Tune model-serving settings by running isolated throughput trials.
    Tune(Box<BenchmarkTuneCommand>),
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

#[derive(Args, Debug, Clone)]
pub struct BenchmarkTuneCommand {
    /// Tune exactly one local/configured model target.
    #[arg(long, conflicts_with = "models")]
    pub model: Option<String>,
    /// Tune multiple local/configured model targets from a comma-separated list.
    #[arg(long, value_delimiter = ',')]
    pub models: Vec<String>,
    /// Print machine-readable JSON output.
    #[arg(long)]
    pub json: bool,
    /// Context sizes to benchmark, as a comma-separated token list.
    #[arg(long, value_delimiter = ',')]
    pub ctx_sizes: Vec<u32>,
    /// Batch sizes to benchmark, as a comma-separated list.
    #[arg(long, value_delimiter = ',')]
    pub batch_sizes: Vec<u32>,
    /// Micro-batch sizes to benchmark, as a comma-separated list.
    #[arg(long, value_delimiter = ',')]
    pub ubatch_sizes: Vec<u32>,
    /// mmap values to benchmark independently: auto, enabled, disabled.
    #[arg(long = "mmap-values", value_delimiter = ',')]
    pub mmap_values: Vec<BenchmarkBoolOrAuto>,
    /// mlock values to benchmark independently: enabled, disabled.
    #[arg(long = "mlock-values", value_delimiter = ',')]
    pub mlock_values: Vec<BenchmarkBool>,
    /// Flash attention values to benchmark independently: on, off.
    #[arg(long = "flash-attention", value_delimiter = ',')]
    pub flash_attention: Vec<BenchmarkFlashAttention>,
    /// Speculative decoding types to benchmark: auto, disabled, mtp, draft, mtp-ngram.
    #[arg(
        long = "speculative-types",
        value_delimiter = ',',
        conflicts_with = "no_speculative_tune"
    )]
    pub speculative_types: Vec<BenchmarkSpeculativeType>,
    /// Disable speculative decoding sweeps and only benchmark the disabled baseline.
    #[arg(
        long = "no-speculative-tune",
        conflicts_with_all = [
            "speculative_types",
            "spec_draft_models",
            "spec_draft_max_tokens",
            "spec_draft_min_tokens",
            "spec_draft_acceptance_threshold",
            "spec_draft_split_probability",
            "spec_ngram_min",
            "spec_ngram_max"
        ]
    )]
    pub no_speculative_tune: bool,
    /// Candidate draft GGUF paths to benchmark for speculative draft mode.
    #[arg(long = "spec-draft-models", value_delimiter = ',')]
    pub spec_draft_models: Vec<PathBuf>,
    /// Candidate maximum draft-token windows for MTP and draft speculation.
    #[arg(long = "spec-draft-max-tokens", value_delimiter = ',')]
    pub spec_draft_max_tokens: Vec<u32>,
    /// Candidate minimum draft-token windows for MTP and draft speculation.
    #[arg(long = "spec-draft-min-tokens", value_delimiter = ',')]
    pub spec_draft_min_tokens: Vec<u32>,
    /// Candidate minimum match lengths for MTP + N-gram speculation.
    #[arg(long = "spec-ngram-min", value_delimiter = ',')]
    pub spec_ngram_min: Vec<u32>,
    /// Candidate maximum match lengths for MTP + N-gram speculation.
    #[arg(long = "spec-ngram-max", value_delimiter = ',')]
    pub spec_ngram_max: Vec<u32>,
    /// Candidate draft-acceptance-threshold values for speculative draft sweeps.
    #[arg(long = "spec-draft-acceptance-threshold", value_delimiter = ',')]
    pub spec_draft_acceptance_threshold: Vec<f64>,
    /// Candidate draft-split-probability values for speculative draft sweeps.
    #[arg(long = "spec-draft-split-probability", value_delimiter = ',')]
    pub spec_draft_split_probability: Vec<f64>,
    /// Persist the recommended settings to the local config file.
    #[arg(long)]
    pub apply: bool,
    /// Replace existing writable config fields instead of preserving existing values.
    #[arg(long, requires = "apply")]
    pub replace_existing: bool,
    /// Print launch-argument output instead of applying or reporting recommended fields.
    #[arg(long)]
    pub launch_args: bool,
    /// Treat candidates within this percent of the raw best tok/s as throughput-equivalent.
    #[arg(long, default_value_t = 10.0)]
    pub throughput_tolerance_pct: f64,
    /// Maximum generated tokens per benchmark request.
    #[arg(long, default_value_t = 128)]
    pub max_tokens: u32,
    /// Startup wait limit for each benchmark trial.
    #[arg(long, default_value_t = 600)]
    pub startup_timeout_secs: u64,
    /// HTTP request timeout for each benchmark request.
    #[arg(long, default_value_t = 600)]
    pub request_timeout_secs: u64,
    /// Capture Skippy debug telemetry in each trial log.
    #[arg(long)]
    pub debug_telemetry: bool,
    /// Prompt sent during benchmark trials.
    #[arg(
        long,
        default_value = "Write a concise paragraph about distributed GPU inference."
    )]
    pub prompt: String,
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
pub enum BenchmarkFlashAttention {
    #[value(alias = "enabled", alias = "true", alias = "1")]
    On,
    #[value(alias = "disabled", alias = "false", alias = "0")]
    Off,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BenchmarkSpeculativeType {
    Auto,
    Disabled,
    Mtp,
    Draft,
    MtpNgram,
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
