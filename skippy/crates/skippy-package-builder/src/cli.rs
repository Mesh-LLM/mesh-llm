use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "skippy-package-builder")]
#[command(about = "Inspect, write, and verify Skippy model packages")]
pub(crate) struct Args {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Admit complete local GGUF shard custody against caller-supplied byte pins.
    AdmitSource {
        model: PathBuf,
        #[arg(long = "pin", required = true)]
        pins: Vec<String>,
        #[arg(long)]
        minimum_context: u64,
    },
    /// Verify local selected package/tokenizer parts without an independent full source.
    AdmitPackage {
        package: PathBuf,
        #[arg(long)]
        manifest_sha256: String,
        #[arg(long)]
        model_id: String,
        #[arg(long)]
        layer_start: u32,
        #[arg(long)]
        layer_end: u32,
        #[arg(long)]
        minimum_context: u64,
    },
    Inspect {
        model: PathBuf,
    },
    /// Emit the source-complete v2 package used by graph-admitted serving.
    WritePackage {
        model: String,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long = "projector")]
        projectors: Vec<PathBuf>,
        /// Immutable publisher configuration files to copy into metadata/.
        /// Supported basenames: config.json, generation_config.json,
        /// tokenizer_config.json, chat_template.jinja, hf_quant_config.json.
        #[arg(long = "publisher-metadata")]
        publisher_metadata: Vec<PathBuf>,
        #[arg(long)]
        after_artifact_command: Option<PathBuf>,
        #[arg(long)]
        transform_artifact_command: Option<PathBuf>,
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long)]
        source_repo: Option<String>,
        #[arg(long)]
        source_revision: Option<String>,
        #[arg(long)]
        source_file: Option<String>,
        /// JSON file containing typed package generation request defaults.
        #[arg(long)]
        generation_defaults: Option<PathBuf>,
        #[arg(long)]
        resume_existing_artifacts: bool,
        /// Maximum payload bytes per artifact; oversized common and layer
        /// groups are split into deterministic part artifacts. Defaults to 8 GiB.
        #[arg(long)]
        max_artifact_bytes: Option<u64>,
    },
    /// Verify byte-preserving v2 packages against independent local source files.
    VerifyPackageV2 {
        package: PathBuf,
        #[arg(long)]
        source: PathBuf,
        /// Logical source primary filename, if different from the local filename.
        #[arg(long)]
        source_file: Option<String>,
        /// Independent originals for all declared projector sidecars.
        #[arg(long = "source-projector")]
        source_projectors: Vec<PathBuf>,
    },
    ValidateGlmDsaContract {
        package: PathBuf,
        #[arg(long)]
        require_generation_policy: bool,
    },
    RepairGlmDsaGenerationPolicy {
        package: PathBuf,
        #[arg(long)]
        in_place: bool,
    },
}

impl Command {
    pub(crate) fn requires_download_preparation(&self) -> bool {
        !matches!(
            self,
            Self::Inspect { .. }
                | Self::VerifyPackageV2 { .. }
                | Self::AdmitSource { .. }
                | Self::AdmitPackage { .. }
        )
    }
}
