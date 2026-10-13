use anyhow::{Context, Result};
use clap::Parser;

mod artifact_admission;
mod cli;
mod generation_manifest;
mod glm_dsa_contract;
mod glm_dsa_generation_policy;
mod hash;
mod inspect;
mod layer_package_fetch;
mod layer_package_inspection;
mod layer_package_planning;
mod package;
mod package_reference;
mod package_v2;
mod part_writer;
mod progress;
mod source_inventory;
mod tensor_payload;
#[cfg(test)]
mod test_gguf;
mod verify_v2;
mod write;

use cli::{Args, Command};
use package::{ArtifactHook, ExplicitSourceIdentity};

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

// ponytail: main runs on a child thread because the Windows main thread has a
// 1 MB stack. sha256 over a multi-GB GGUF plus FFI slice writing blows that
// stack in debug builds. 8 MB matches the mesh-llm runtime default. If a real
// recursion sink appears, raise this or fix the recursion — don't go lower.
const MAIN_STACK_SIZE: usize = 8 * 1024 * 1024;

fn main() -> Result<()> {
    let args = Args::parse();
    // Local inspection and verification must not touch download caches.
    if args.command.requires_download_preparation() {
        prepare_model_download_directories();
    }

    let handle = std::thread::Builder::new()
        .stack_size(MAIN_STACK_SIZE)
        .spawn(move || run(args))
        .context("spawn skippy-model-package worker thread")?;
    handle.join().unwrap_or_else(|panic| {
        std::panic::resume_unwind(panic);
    })
}

fn run(args: Args) -> Result<()> {
    run_with_output(args, &mut std::io::stdout().lock())
}

fn run_with_output(args: Args, output: &mut dyn std::io::Write) -> Result<()> {
    match args.command {
        Command::FetchLayerPackageWorker {
            reference,
            cache_root,
            stage_index,
            stage_count,
            timeout_millis,
            expected_layer_count,
            expected_activation_width,
        } => {
            let report = layer_package_fetch::fetch_worker(layer_package_fetch::Input {
                reference: &reference,
                cache_root: &cache_root,
                stage: stage_index.zip(stage_count),
                timeout: std::time::Duration::from_millis(timeout_millis),
                expected_layers: expected_layer_count,
                expected_width: expected_activation_width,
            })?;
            write_admission_receipt(output, &report)
        }
        Command::FetchLayerPackage {
            reference,
            cache_root,
            stage_index,
            stage_count,
            timeout_secs,
            expected_layer_count,
            expected_activation_width,
        } => {
            let report = layer_package_fetch::fetch(layer_package_fetch::Input {
                reference: &reference,
                cache_root: &cache_root,
                stage: stage_index.zip(stage_count),
                timeout: std::time::Duration::from_secs(timeout_secs),
                expected_layers: expected_layer_count,
                expected_width: expected_activation_width,
            })?;
            write_admission_receipt(output, &report)
        }
        Command::ResolveLayerPackageCache {
            reference,
            cache_root,
        } => {
            let reference =
                skippy_model_ref::package_reference::PackageReference::parse(&reference)?;
            let snapshot = skippy_model_hf::package_cache::resolve(&reference, &cache_root)?;
            write_admission_receipt(output, &snapshot)
        }
        Command::PlanLayerPackageArtifacts {
            manifest,
            stage_index,
            stage_count,
            layer_start,
            layer_end,
        } => layer_package_planning::write_artifacts(
            &manifest,
            stage_index,
            stage_count,
            layer_start,
            layer_end,
            output,
        ),
        Command::EvenLayerStageRange {
            stage_index,
            stage_count,
            layer_count,
        } => layer_package_planning::write_range(stage_index, stage_count, layer_count, output),
        Command::ParsePackageReference { reference } => {
            package_reference::write(&reference, output)
        }
        Command::AdmitSource {
            model,
            pins,
            minimum_context,
        } => {
            let receipt = artifact_admission::source(&model, &pins, minimum_context)?;
            write_admission_receipt(output, &receipt)?;
            Ok(())
        }
        Command::AdmitPackage {
            package,
            manifest_sha256,
            model_id,
            layer_start,
            layer_end,
            minimum_context,
        } => {
            let receipt = artifact_admission::package(
                &package,
                &manifest_sha256,
                &model_id,
                layer_start,
                layer_end,
                minimum_context,
            )?;
            write_admission_receipt(output, &receipt)?;
            Ok(())
        }
        Command::Inspect { model } => inspect::inspect(model),
        Command::InspectLayerPackage {
            package,
            expected_layer_count,
            expected_activation_width,
        } => {
            let report = layer_package_inspection::inspect(
                &package,
                expected_layer_count,
                expected_activation_width,
            )?;
            write_admission_receipt(output, &report)
        }
        Command::WritePackage {
            model,
            out_dir,
            projectors,
            publisher_metadata,
            after_artifact_command,
            transform_artifact_command,
            model_id,
            source_repo,
            source_revision,
            source_file,
            generation_defaults,
            resume_existing_artifacts,
            max_artifact_bytes,
        } => package_v2::write_package(
            model,
            out_dir,
            package_v2::PackageSidecars {
                projectors,
                publisher_metadata,
            },
            ArtifactHook {
                command: after_artifact_command,
            },
            ArtifactHook {
                command: transform_artifact_command,
            },
            package_v2::PackageWriteOptions {
                explicit: ExplicitSourceIdentity {
                    model_id,
                    source_repo,
                    source_revision,
                    source_file,
                },
                generation_defaults,
                resume_existing_artifacts,
                max_artifact_bytes,
            },
        ),
        Command::VerifyPackageV2 {
            package,
            source,
            source_file,
            source_projectors,
        } => {
            let report = verify_v2::verify_package(
                &package,
                &source,
                source_file.as_deref(),
                &source_projectors,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::ValidateGlmDsaContract {
            package,
            require_generation_policy,
        } => {
            let report = glm_dsa_contract::validate_path_with_options(
                &package,
                glm_dsa_contract::GlmDsaContractOptions {
                    require_generation_policy,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            anyhow::ensure!(
                report.valid,
                "GLM-DSA contract validation failed for {}",
                package.display()
            );
            Ok(())
        }
        Command::RepairGlmDsaGenerationPolicy { package, in_place } => {
            glm_dsa_generation_policy::repair_package(&package, in_place)
        }
    }
}

fn write_admission_receipt(
    output: &mut dyn std::io::Write,
    receipt: &impl serde::Serialize,
) -> Result<()> {
    serde_json::to_writer_pretty(&mut *output, receipt)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}
#[cfg(test)]
mod admission_output_tests {
    use super::*;
    struct Refusal {
        bytes: Vec<u8>,
        reject_write: bool,
    }
    impl std::io::Write for Refusal {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.reject_write {
                return Err(std::io::ErrorKind::BrokenPipe.into());
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
    }
    #[test]
    fn admission_machine_output_is_one_json_document_with_terminal_newline() {
        let mut output = Vec::new();
        write_admission_receipt(
            &mut output,
            &serde_json::json!({"admitted":true,"path":"a b"}),
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&output).unwrap(),
            serde_json::json!({"admitted":true,"path":"a b"})
        );
        assert!(output.ends_with(b"}\n"));
        assert!(!output.ends_with(b"\n\n"));
    }
    #[test]
    fn admission_machine_output_refuses_write_and_flush_failures() {
        for reject_write in [true, false] {
            let mut output = Refusal {
                bytes: Vec::new(),
                reject_write,
            };
            assert!(
                write_admission_receipt(&mut output, &serde_json::json!({"admitted":true}))
                    .is_err()
            );
            assert_eq!(output.bytes.is_empty(), reject_write);
        }
    }
}
