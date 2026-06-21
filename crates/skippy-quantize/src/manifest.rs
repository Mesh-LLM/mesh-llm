use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::output::print_path_event;
use crate::splits::{Progress, next_missing_window, split_status_for_basename};
use crate::types::{ConvertOutputType, JobKind};

pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub kind: JobKind,
    pub source: PathBuf,
    pub source_prefix: Option<String>,
    pub target: PathBuf,
    pub target_prefix: String,
    pub output_basename: String,
    pub expected_splits: u32,
    pub window_size: u32,
    pub quant: Option<String>,
    pub output_type: Option<ConvertOutputType>,
    pub tensor_type_file: Option<PathBuf>,
}

pub fn ensure_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    if path.exists() {
        let existing = read_manifest(path)?;
        ensure!(
            existing == *manifest,
            "existing manifest does not match requested job: {}",
            path.display()
        );
        print_path_event("📄", "Resuming manifest", path);
        return Ok(());
    }
    write_manifest(path, manifest)?;
    print_path_event("📄", "Created manifest", path);
    Ok(())
}

pub fn read_manifest(path: &Path) -> Result<Manifest> {
    let data = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let manifest: Manifest =
        serde_json::from_str(&data).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        manifest.schema_version == MANIFEST_VERSION,
        "unsupported manifest schema version {}",
        manifest.schema_version
    );
    Ok(manifest)
}

pub fn write_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_vec_pretty(manifest)?)
        .with_context(|| format!("write {}", path.display()))
}

pub fn manifest_progress(manifest: &Manifest) -> Result<Progress> {
    split_status_for_basename(
        &manifest.target,
        &manifest.target_prefix,
        &manifest.output_basename,
        manifest.expected_splits,
    )
    .map(|mut progress| {
        progress.next_window = next_missing_window(&progress.missing_ranges, manifest.window_size);
        progress
    })
}
