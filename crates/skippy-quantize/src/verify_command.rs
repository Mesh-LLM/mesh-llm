use std::path::Path;

use anyhow::Result;

use crate::llama_load::{LlamaLoadOptions, validate_llama_load};
use crate::manifest::read_manifest;
use crate::verify::{first_artifact_path, verify_manifest};

pub(crate) fn verify_job(
    manifest_path: &Path,
    llama_load: bool,
    llama_cli: Option<&Path>,
    check_tensors: bool,
    json: bool,
) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    let report = verify_manifest(&manifest)?;
    let llama_load = if llama_load || llama_cli.is_some() {
        Some(validate_llama_load(
            &first_artifact_path(&manifest),
            llama_cli,
            LlamaLoadOptions { check_tensors },
        )?)
    } else {
        None
    };
    if json {
        if let Some(llama_load) = llama_load {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "artifact": report,
                    "llama_load": llama_load,
                }))?
            );
        } else {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    } else {
        println!(
            "verified artifact: {}/{} shards prefix={} basename={}",
            report.completed_count, report.expected_splits, report.prefix, report.basename
        );
        if let Some(llama_load) = llama_load {
            println!(
                "llama_load_valid=true model={} llama_cli={}",
                llama_load.model.display(),
                llama_load.llama_cli.display()
            );
        }
    }
    Ok(())
}
