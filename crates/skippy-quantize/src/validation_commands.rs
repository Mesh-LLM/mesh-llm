use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::command_reports::{SplitValidation, TensorTypeValidation};
use crate::manifest::{manifest_progress, read_manifest};
use crate::quantize::ensure_tensor_type_entry;
use crate::splits::{Progress, split_status, split_status_for_basename};

pub(crate) fn run_status(manifest_path: &Path, json: bool) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    let progress = manifest_progress(&manifest)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&progress)?);
    } else {
        print_progress(&progress);
    }
    Ok(())
}

pub(crate) fn run_next_window(manifest_path: &Path, json: bool) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    let progress = manifest_progress(&manifest)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&progress.next_window)?);
    } else if let Some(window) = progress.next_window {
        println!("{}..{}", window.first_split, window.last_split);
    } else {
        println!("complete");
    }
    Ok(())
}

pub(crate) fn validate_tensor_types_command(path: &Path, json: bool) -> Result<()> {
    let validation = validate_tensor_types(path)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&validation)?);
    } else {
        println!("valid tensor type file: {} entries", validation.entry_count);
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
        println!("{}", serde_json::to_string_pretty(&validation)?);
    } else if validation.complete {
        println!(
            "complete split artifact: {}/{} shards",
            validation.completed_count, validation.expected_splits
        );
    } else {
        println!(
            "incomplete split artifact: {}/{} shards first_missing={:?}",
            validation.completed_count, validation.expected_splits, validation.first_missing
        );
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
    println!(
        "{}/{} shards complete ({:.2}%)",
        progress.completed_count, progress.expected_splits, progress.completed_percent
    );
    println!("missing shards: {}", progress.missing_count);
    if !progress.missing_ranges.is_empty() {
        let ranges = progress
            .missing_ranges
            .iter()
            .map(|range| {
                if range.first_split == range.last_split {
                    range.first_split.to_string()
                } else {
                    format!("{}..{}", range.first_split, range.last_split)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("missing ranges: {ranges}");
    }
    match progress.next_window {
        Some(window) => println!("next window: {}..{}", window.first_split, window.last_split),
        None => println!("next window: complete"),
    }
}
