use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::command_reports::{SplitValidation, TensorTypeValidation};
use crate::manifest::{manifest_progress, read_manifest};
use crate::output::{
    format_shard_ranges, format_window, print_info, print_json_pretty, print_progress_line,
    print_success, print_warn,
};
use crate::quantize::ensure_tensor_type_entry;
use crate::splits::{Progress, split_status, split_status_for_basename};

pub(crate) fn run_status(manifest_path: &Path, json: bool) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    let progress = manifest_progress(&manifest)?;
    if json {
        print_json_pretty(&progress)?;
    } else {
        print_progress(&progress);
    }
    Ok(())
}

pub(crate) fn run_next_window(manifest_path: &Path, json: bool) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    let progress = manifest_progress(&manifest)?;
    if json {
        print_json_pretty(&progress.next_window)?;
    } else if let Some(window) = progress.next_window {
        print_info(format!("Next window: {}", format_window(window)));
    } else {
        print_success("No next window; job is complete");
    }
    Ok(())
}

pub(crate) fn validate_tensor_types_command(path: &Path, json: bool) -> Result<()> {
    let validation = validate_tensor_types(path)?;
    if json {
        print_json_pretty(&validation)?;
    } else {
        print_success(format!(
            "Valid tensor type file: {} entries",
            validation.entry_count
        ));
    }
    Ok(())
}

pub(crate) fn validate_splits_command(
    root: &Path,
    prefix: &str,
    expected_splits: Option<u32>,
    basename: Option<&str>,
    json: bool,
) -> Result<()> {
    let progress = if let Some(basename) = basename {
        split_status_for_basename(
            root,
            prefix,
            basename,
            expected_splits.context("--expected-splits is required with --basename")?,
        )?
    } else {
        split_status(root, prefix, expected_splits)?
    };
    let validation = SplitValidation {
        root: root.to_path_buf(),
        prefix: prefix.to_string(),
        expected_splits: progress.expected_splits,
        completed_count: progress.completed_count,
        first_missing: progress.first_missing,
        last_present: progress.last_present,
        complete: progress.complete,
    };
    if json {
        print_json_pretty(&validation)?;
    } else if validation.complete {
        print_progress_line(
            "split artifact",
            validation.completed_count,
            validation.expected_splits,
        );
        print_success("Split artifact is complete");
    } else {
        print_progress_line(
            "split artifact",
            validation.completed_count,
            validation.expected_splits,
        );
        print_warn(format!(
            "Split artifact is incomplete; first missing shard: {:?}",
            validation.first_missing
        ));
    }
    ensure!(validation.complete, "split artifact is incomplete");
    Ok(())
}

pub(crate) fn validate_tensor_types(path: &Path) -> Result<TensorTypeValidation> {
    let data = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut entry_count = 0;
    for token in data.split_whitespace() {
        ensure_tensor_type_entry(token)?;
        entry_count += 1;
    }
    Ok(TensorTypeValidation {
        valid: true,
        entry_count,
    })
}

fn print_progress(progress: &Progress) {
    print_progress_line(
        "job status",
        progress.completed_count,
        progress.expected_splits,
    );
    if progress.complete {
        print_success("All shards complete");
    } else {
        print_warn(format!("Missing shards: {}", progress.missing_count));
    }
    if !progress.missing_ranges.is_empty() {
        print_info(format!(
            "Missing ranges: {}",
            format_shard_ranges(&progress.missing_ranges)
        ));
    }
    match progress.next_window {
        Some(window) => print_info(format!("Next window: {}", format_window(window))),
        None => print_success("Next window: complete"),
    }
}
