use clap::Subcommand;
use std::path::PathBuf;

use crate::MeshGuardrailCliMode;

#[derive(Subcommand, Debug)]
pub enum RuntimeCommand {
    /// List available or installed native runtimes.
    List {
        /// List release-manifest or bundled runtimes instead of installed runtimes.
        #[arg(long, conflicts_with = "installed")]
        available: bool,
        /// List installed native runtimes. This is the default when no list mode is supplied.
        #[arg(long, conflicts_with = "available")]
        installed: bool,
        /// Release manifest JSON to inspect.
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Packaged native runtime directory to inspect. Repeatable.
        #[arg(long = "bundle-dir")]
        bundle_dirs: Vec<PathBuf>,
        /// Override the native runtime cache root.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Install the recommended native runtime, or an explicit flavor/runtime ID.
    Install {
        /// Optional runtime flavor or native runtime ID. Omit to install the recommended runtime.
        runtime: Option<String>,
        /// Release manifest JSON to resolve against.
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Packaged native runtime directory to install from. Repeatable.
        #[arg(long = "bundle-dir")]
        bundle_dirs: Vec<PathBuf>,
        /// Override the native runtime cache root.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove an installed native runtime.
    Remove {
        /// Native runtime ID to remove.
        native_runtime_id: String,
        /// MeshLLM version. Defaults to the running MeshLLM version.
        #[arg(long)]
        mesh_version: Option<String>,
        /// Override the native runtime cache root.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Prune old native runtimes from the cache.
    Prune {
        /// Remove every runtime not matching the active MeshLLM version.
        #[arg(long)]
        active_only: bool,
        /// Override the active MeshLLM version. Defaults to the running version.
        #[arg(long)]
        mesh_version: Option<String>,
        /// Override the native runtime cache root.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show local model status on a running mesh-llm instance.
    #[command(hide = true)]
    Status {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Show the local-only owner-control bootstrap policy for a running mesh-llm instance.
    #[command(hide = true)]
    Bootstrap {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
    /// Fetch config from a remote owner-control endpoint through the local management API.
    #[command(hide = true)]
    GetConfig {
        /// Explicit owner-control endpoint token for the target node.
        #[arg(long)]
        endpoint: String,
        /// Console/API port of the local mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
    /// Refresh local inventory on a remote owner-control endpoint through the local management API.
    #[command(hide = true)]
    RefreshInventory {
        /// Explicit owner-control endpoint token for the target node.
        #[arg(long)]
        endpoint: String,
        /// Console/API port of the local mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
    /// Apply config to a remote owner-control endpoint through the local management API.
    #[command(hide = true)]
    ApplyConfig {
        /// Explicit owner-control endpoint token for the target node.
        #[arg(long)]
        endpoint: String,
        /// Expected remote config revision for CAS.
        #[arg(long)]
        expected_revision: u64,
        /// TOML config file to apply remotely.
        #[arg(long)]
        config: PathBuf,
        /// Console/API port of the local mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
    /// Load a local model into a running mesh-llm instance.
    #[command(hide = true)]
    Load {
        /// Model name/path/url to load
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Unload a local model from a running mesh-llm instance.
    #[command(alias = "drop", hide = true)]
    Unload {
        /// Model name to unload
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Set mesh guardrail mode on running Skippy-backed models without restart.
    #[command(hide = true)]
    Guardrails {
        /// Guardrail mode to apply to active Skippy-backed OpenAI surfaces.
        #[arg(long, value_enum)]
        mode: MeshGuardrailCliMode,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print the raw JSON payload.
        #[arg(long)]
        json: bool,
    },
}
