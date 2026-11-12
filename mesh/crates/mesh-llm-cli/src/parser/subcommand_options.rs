//! Options of the subcommands with the most arguments, one `Args` struct each.
//!
//! clap's derive code defines every argument of a subcommand enum in that
//! enum's `augment_subcommands`, and in a debug build that frame grows with each
//! argument (clap-rs/clap#5134). That frame stays on the stack while the nested
//! subcommand enums are defined; each struct here defines its arguments in its
//! own function, which returns before that. Field order and attributes are
//! unchanged, so `--help` doesn't change.

use clap::Args;

#[derive(Args, Debug)]
pub struct ModelPrepareArgs {
    /// Source HuggingFace model ref (e.g. unsloth/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL).
    pub source_repo: Option<String>,

    /// Quantization variant (deprecated; prefer source refs like repo:Q4_K_M).
    #[arg(long)]
    pub quant: Option<String>,

    /// Target repo for the layer package (auto-derived if omitted).
    #[arg(long)]
    pub target: Option<String>,

    /// Override model ID in the manifest.
    #[arg(long)]
    pub model_id: Option<String>,

    /// HF Job hardware flavor. Use auto for the default CPU splitter baseline.
    #[arg(long, default_value = "auto")]
    pub flavor: String,

    /// Requested job timeout; raised automatically by model-size minimums.
    #[arg(long, default_value = "1h")]
    pub timeout: String,

    /// Branch or tag of mesh-llm to build in the job [default: main].
    #[arg(long, default_value = "main")]
    pub mesh_llm_ref: String,

    /// Explicitly keep this as a dry run. This is the default unless --confirm is set.
    #[arg(long)]
    pub dry_run: bool,

    /// Actually submit the HF Job. Without this, the command only prints plan, spec, and max cost.
    #[arg(long)]
    pub confirm: bool,

    /// Stream job logs after submission until completion.
    #[arg(long)]
    pub follow: bool,

    /// Emit JSON output.
    #[arg(long)]
    pub json: bool,

    /// Check status of a previously submitted job.
    #[arg(long)]
    pub status: Option<String>,

    /// Fetch logs for a previously submitted job.
    #[arg(long)]
    pub logs: Option<String>,

    /// Cancel a running job.
    #[arg(long)]
    pub cancel: Option<String>,

    /// List recent skippy-model-package jobs.
    #[arg(long)]
    pub list: bool,

    /// Upload the latest job script to the meshllm bucket (requires org access).
    #[arg(long)]
    pub update_script: bool,
}

#[derive(Args, Debug)]
pub struct UninstallArgs {
    /// Print what would be removed without changing the machine.
    #[arg(long)]
    pub dry_run: bool,
    /// Do not prompt before removing files and services.
    #[arg(long)]
    pub yes: bool,
    /// Preserve native runtime caches.
    #[arg(long)]
    pub keep_cache: bool,
    /// Preserve setup-owned service helper files.
    #[arg(long)]
    pub keep_service_files: bool,
    /// Also remove ~/.mesh-llm configuration and identity data.
    #[arg(long, conflicts_with = "keep_config")]
    pub purge_config: bool,
    /// Explicitly preserve ~/.mesh-llm configuration and identity data.
    #[arg(long, conflicts_with = "purge_config")]
    pub keep_config: bool,
    /// Override the installed binary path to remove.
    #[arg(long)]
    pub binary_path: Option<std::path::PathBuf>,
    /// Print machine-readable JSON.
    #[arg(long)]
    pub json: bool,
    /// Print detailed cleanup steps and removed paths.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
pub struct SetupArgs {
    /// Automatically answer yes to prompts.
    #[arg(long)]
    pub yes: bool,
    /// Run without prompting for interactive input.
    #[arg(long = "no-interactive")]
    pub no_interactive: bool,
    /// Explicitly install this catalog plugin set instead of offering the
    /// recommended one. Suppresses the plugin-set prompt.
    #[arg(
        long = "plugin-set",
        value_name = "NAME",
        conflicts_with_all = ["skip_plugin_install", "no_default_plugins"]
    )]
    pub plugin_set: Option<String>,
    /// Skip optional plugin installation and its prompt.
    #[arg(long = "skip-plugin-install")]
    pub skip_plugin_install: bool,
    /// Deprecated alias for --skip-plugin-install. Bundled default
    /// plugins were replaced by catalog plugin sets.
    #[arg(long, hide = true, conflicts_with = "plugin_set")]
    pub no_default_plugins: bool,
    /// Install and enable the mesh-llm service.
    #[arg(long, conflicts_with = "no_service")]
    pub service: bool,
    /// Skip installing and enabling the mesh-llm service.
    #[arg(long = "no-service", conflicts_with = "service")]
    pub no_service: bool,
    /// Skip downloading or configuring the native runtime.
    #[arg(long = "skip-runtime")]
    pub skip_runtime: bool,
    /// Print detailed setup paths, commands, and follow-up guidance.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
pub struct DiscoverArgs {
    /// Filter by mesh name (case-insensitive exact match)
    #[arg(long)]
    pub name: Option<String>,
    /// Filter by model name (substring match)
    #[arg(long)]
    pub model: Option<String>,
    /// Filter by minimum VRAM (GB)
    #[arg(long)]
    pub min_vram: Option<f64>,
    /// Filter by region
    #[arg(long)]
    pub region: Option<String>,
    /// Print the invite token of the best match (for piping to --join)
    #[arg(long)]
    pub auto: bool,
    /// Nostr relay URLs (default: see DEFAULT_RELAYS)
    #[arg(long)]
    pub relay: Vec<String>,
}
