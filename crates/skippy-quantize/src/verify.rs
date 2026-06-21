use std::path::Path;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::llama_load::{LlamaLoadOptions, validate_llama_load};
use crate::manifest::{Manifest, manifest_progress, read_manifest};
use crate::output::{print_info, print_success};
use crate::splits::{Progress, split_status_for_basename};

#[derive(Debug, Serialize)]
pub struct VerificationReport {
    pub root: String,
    pub prefix: String,
    pub basename: String,
    pub expected_splits: u32,
    pub completed_count: usize,
    pub first_missing: Option<u32>,
    pub last_present: Option<u32>,
    pub first_shard: String,
    pub last_shard: String,
    pub complete: bool,
}

pub fn verify_manifest(manifest: &Manifest) -> Result<VerificationReport> {
    let progress = split_status_for_basename(
        &manifest.target,
        &manifest.target_prefix,
        &manifest.output_basename,
        manifest.expected_splits,
    )?;
    let report = report_from_progress(manifest, progress);
    ensure!(
        report.complete,
        "manifest artifact is incomplete: {}/{} shards first_missing={:?}",
        report.completed_count,
        report.expected_splits,
        report.first_missing
    );
    Ok(report)
}

pub fn verify_manifest_path_if_complete(path: &Path) -> Result<Option<VerificationReport>> {
    let manifest = read_manifest(path)?;
    let progress = manifest_progress(&manifest)?;
    if !progress.complete {
        return Ok(None);
    }
    verify_manifest(&manifest).map(Some)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VerifyOnCompleteOptions<'a> {
    pub(crate) enabled: bool,
    pub(crate) llama_load: bool,
    pub(crate) llama_cli: Option<&'a Path>,
    pub(crate) check_tensors: bool,
}

pub fn print_verify_on_complete(
    manifest_path: &Path,
    options: VerifyOnCompleteOptions<'_>,
) -> Result<()> {
    if !options.enabled {
        return Ok(());
    }
    let manifest = read_manifest(manifest_path)?;
    let Some(report) = verify_manifest_path_if_complete(manifest_path)? else {
        return Ok(());
    };
    print_success(format!(
        "Verified complete artifact: {}/{} shards prefix={} basename={}",
        report.completed_count, report.expected_splits, report.prefix, report.basename
    ));
    if options.llama_load || options.llama_cli.is_some() {
        let llama_load = validate_llama_load(
            &first_artifact_path(&manifest),
            options.llama_cli,
            LlamaLoadOptions {
                check_tensors: options.check_tensors,
            },
        )?;
        print_info(format!(
            "llama load validation passed: model={} llama_cli={}",
            llama_load.model.display(),
            llama_load.llama_cli.display()
        ));
    }
    Ok(())
}

pub(crate) fn first_artifact_path(manifest: &Manifest) -> std::path::PathBuf {
    let root = manifest.target.join(&manifest.target_prefix);
    let unsplit = root.join(format!("{}.gguf", manifest.output_basename));
    if manifest.expected_splits == 1 && unsplit.is_file() {
        return unsplit;
    }
    root.join(format!(
        "{}-00001-of-{:05}.gguf",
        manifest.output_basename, manifest.expected_splits
    ))
}

fn report_from_progress(manifest: &Manifest, progress: Progress) -> VerificationReport {
    VerificationReport {
        root: manifest.target.display().to_string(),
        prefix: manifest.target_prefix.clone(),
        basename: manifest.output_basename.clone(),
        expected_splits: manifest.expected_splits,
        completed_count: progress.completed_count,
        first_missing: progress.first_missing,
        last_present: progress.last_present,
        first_shard: shard_name(&manifest.output_basename, 1, manifest.expected_splits),
        last_shard: shard_name(
            &manifest.output_basename,
            manifest.expected_splits,
            manifest.expected_splits,
        ),
        complete: progress.complete,
    }
}

fn shard_name(basename: &str, index: u32, total: u32) -> String {
    format!("{basename}-{index:05}-of-{total:05}.gguf")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::manifest::MANIFEST_VERSION;
    use crate::records::unix_timestamp_ms;
    use crate::types::JobKind;

    use super::*;

    #[test]
    fn verifies_exact_manifest_artifact() {
        let root = std::env::temp_dir().join(format!(
            "skippy-quantize-verify-test-{}",
            unix_timestamp_ms()
        ));
        let prefix_root = root.join("Q2_K");
        fs::create_dir_all(&prefix_root).unwrap();
        fs::write(prefix_root.join("out-00001-of-00002.gguf"), b"1").unwrap();
        fs::write(prefix_root.join("out-00002-of-00002.gguf"), b"2").unwrap();

        let manifest = Manifest {
            schema_version: MANIFEST_VERSION,
            kind: JobKind::QuantizeGguf,
            source: PathBuf::from("/source"),
            source_prefix: Some("BF16".to_string()),
            target: root.clone(),
            target_prefix: "Q2_K".to_string(),
            output_basename: "out".to_string(),
            expected_splits: 2,
            window_size: 1,
            quant: Some("Q2_K".to_string()),
            output_type: None,
            tensor_type_file: None,
        };

        let report = verify_manifest(&manifest).unwrap();
        assert!(report.complete);
        assert_eq!(report.first_shard, "out-00001-of-00002.gguf");
        assert_eq!(report.last_shard, "out-00002-of-00002.gguf");
        fs::remove_dir_all(root).unwrap();
    }
}
