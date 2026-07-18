use anyhow::Result;
use clap::Parser;

mod cli;
mod gguf_header;
mod hash;
mod inspect;
mod package;
mod plan;
mod preflight;
mod progress;
#[cfg(test)]
mod tests;
mod validate;
mod write;

use cli::{Args, Command};
use package::{ArtifactHook, ExplicitSourceIdentity};

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Inspect { model } => inspect::inspect(model),
        Command::Plan { model, stages } => plan::build_plan(&model, stages).and_then(|output| {
            println!("{}", serde_json::to_string_pretty(&output)?);
            Ok(())
        }),
        Command::Write {
            model,
            layers,
            out,
            stage_index,
            include_embeddings,
            include_output,
            manifest,
        } => write::write_one(
            model,
            layers,
            out,
            stage_index,
            include_embeddings,
            include_output,
            manifest,
        ),
        Command::WriteStages {
            model,
            stages,
            out_dir,
        } => write::write_stages(model, stages, out_dir),
        Command::WritePackage {
            model,
            out_dir,
            projectors,
            after_artifact_command,
            transform_artifact_command,
            model_id,
            source_repo,
            source_revision,
            source_file,
            resume_existing_artifacts,
        } => package::write_package(
            model,
            out_dir,
            projectors,
            ArtifactHook {
                command: after_artifact_command,
            },
            ArtifactHook {
                command: transform_artifact_command,
            },
            ExplicitSourceIdentity {
                model_id,
                source_repo,
                source_revision,
                source_file,
            },
            resume_existing_artifacts,
        ),
        Command::Validate { full, slices } => validate::validate(full, slices),
        Command::ValidatePackage { full, package } => validate::validate_package(full, package),
        Command::Preflight {
            package,
            stages,
            verify_sha256,
        } => validate::run_preflight(package, stages, verify_sha256),
    }
}
