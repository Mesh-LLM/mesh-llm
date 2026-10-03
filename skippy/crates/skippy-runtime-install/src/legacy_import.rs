//! The explicit boundary for unversioned Mesh-era native runtime manifests.

use crate::{NativeRuntimeImportOutcome, import_runtime_copy};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use skippy_native_runtime::{
    NativeRuntimeArtifact, NativeRuntimeBackend, NativeRuntimeCache, NativeRuntimeManifest,
    NativeRuntimePlatform,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LegacyRuntimeImportEntry {
    Skipped {
        source: PathBuf,
        reason: String,
    },
    Failed {
        source: PathBuf,
        error: String,
    },
    Runtime {
        outcome: NativeRuntimeImportOutcome,
        warnings: Vec<String>,
    },
}

#[derive(Debug, Default, Serialize)]
pub struct LegacyRuntimeImportReport {
    pub entries: Vec<LegacyRuntimeImportEntry>,
}

impl LegacyRuntimeImportReport {
    pub fn has_failures(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| matches!(entry, LegacyRuntimeImportEntry::Failed { .. }))
    }
}

#[derive(Deserialize)]
struct LegacyManifest {
    runtime: LegacyArtifact,
}

// Kept separate from the normal wire reader: the old spelling is interpreted
// only after the user explicitly selects a source cache for import.
#[derive(Deserialize)]
struct LegacyArtifact {
    id: String,
    mesh_version: String,
    skippy_abi: String,
    platform: NativeRuntimePlatform,
    backend: NativeRuntimeBackend,
    #[serde(default)]
    rank: i64,
    libraries: Vec<String>,
    #[serde(default)]
    files: BTreeMap<String, String>,
    #[serde(default)]
    tools: BTreeMap<String, String>,
    url: Option<String>,
    sha256: Option<String>,
    signature: Option<String>,
}

fn decode_legacy(path: &Path, release: &str, id: &str) -> Result<NativeRuntimeManifest> {
    if !fs::symlink_metadata(path)?.is_file() {
        bail!("legacy manifest must be a regular file: {}", path.display());
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    if value.get("schema_version").is_some() {
        bail!(
            "legacy import accepts only unversioned manifests: {}",
            path.display()
        );
    }
    let legacy: LegacyManifest = serde_json::from_value(value)?;
    let old = legacy.runtime;
    if old.mesh_version != release || old.id != id {
        bail!(
            "legacy runtime identity does not match its cache directory: {}",
            path.display()
        );
    }
    Ok(NativeRuntimeManifest {
        runtime: NativeRuntimeArtifact {
            id: old.id,
            release_version: Some(old.mesh_version),
            skippy_abi: old.skippy_abi,
            platform: old.platform,
            backend: old.backend,
            rank: old.rank,
            libraries: old.libraries,
            files: old.files,
            tools: old.tools,
            url: old.url,
            sha256: old.sha256,
            signature: old.signature,
        },
    })
}

/// Import an explicitly selected legacy cache, preserving the source tree.
///
/// Missing manifests and unknown version directories are reported as skips.
/// Invalid metadata/checksums and destination collisions fail only that entry;
/// callers must inspect `has_failures()` when choosing an exit status. ABI
/// mismatches are preserved with a warning, not made eligible for execution.
pub fn import_legacy_runtime_cache(
    source_cache: &Path,
    destination_cache: &NativeRuntimeCache,
    required_abi: &str,
    dry_run: bool,
) -> Result<LegacyRuntimeImportReport> {
    let mut report = LegacyRuntimeImportReport::default();
    for version_dir in sorted_entries(source_cache)? {
        let release = version_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if !fs::symlink_metadata(&version_dir)?.is_dir() || semver::Version::parse(release).is_err()
        {
            report.entries.push(LegacyRuntimeImportEntry::Skipped {
                source: version_dir,
                reason: "not a legacy release directory".into(),
            });
            continue;
        }
        let entries = match sorted_entries(&version_dir) {
            Ok(entries) => entries,
            Err(error) => {
                report.entries.push(LegacyRuntimeImportEntry::Failed {
                    source: version_dir,
                    error: format!("{error:#}"),
                });
                continue;
            }
        };
        for source in entries {
            let source_metadata = match fs::symlink_metadata(&source) {
                Ok(metadata) => metadata,
                Err(error) => {
                    report.entries.push(LegacyRuntimeImportEntry::Failed {
                        source,
                        error: error.to_string(),
                    });
                    continue;
                }
            };
            if !source_metadata.is_dir() {
                report.entries.push(LegacyRuntimeImportEntry::Skipped {
                    source,
                    reason: "not a runtime directory (symlinks are not traversed)".into(),
                });
                continue;
            }
            let manifest_path = source.join("manifest.json");
            let result = (|| -> Result<_> {
                if !manifest_path.try_exists()? {
                    return Ok(LegacyRuntimeImportEntry::Skipped {
                        source: source.clone(),
                        reason: "missing manifest.json".into(),
                    });
                }
                let id = source
                    .file_name()
                    .and_then(|name| name.to_str())
                    .context("legacy runtime directory name is not UTF-8")?;
                let manifest = decode_legacy(&manifest_path, release, id)?;
                let warnings = if manifest.runtime.skippy_abi == required_abi {
                    Vec::new()
                } else {
                    vec![format!(
                        "preserved runtime ABI {} differs from required ABI {}; resolver eligibility is unchanged",
                        manifest.runtime.skippy_abi, required_abi
                    )]
                };
                let outcome = import_runtime_copy(&source, &manifest, destination_cache, dry_run)?;
                Ok(LegacyRuntimeImportEntry::Runtime { outcome, warnings })
            })();
            report.entries.push(match result {
                Ok(entry) => entry,
                Err(error) => LegacyRuntimeImportEntry::Failed {
                    source,
                    error: format!("{error:#}"),
                },
            });
        }
    }
    Ok(report)
}

fn sorted_entries(root: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = fs::read_dir(root)
        .with_context(|| format!("read legacy runtime cache {}", root.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NativeRuntimeImportStatus;
    use sha2::{Digest, Sha256};

    fn legacy_fixture(root: &Path, release: &str, id: &str) -> PathBuf {
        let dir = root.join(release).join(id);
        fs::create_dir_all(dir.join("lib")).unwrap();
        fs::write(dir.join("lib/runtime.bin"), b"legacy payload").unwrap();
        let hash = hex::encode(Sha256::digest(b"legacy payload"));
        fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "runtime": {"id": id, "mesh_version": release, "skippy_abi": "0.1.0",
                    "platform": {"os": "macos", "arch": "aarch64"},
                    "backend": {"kind": "cpu"}, "libraries": ["lib/runtime.bin"],
                    "files": {"lib/runtime.bin": hash}},
                "build": {"provenance": "legacy fixture"}
            }))
            .unwrap(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn legacy_scan_reports_each_entry_and_preserves_source_during_apply() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("legacy");
        let valid = legacy_fixture(&source, "1.2.3", "good");
        let original = fs::read(valid.join("manifest.json")).unwrap();
        let corrupt = legacy_fixture(&source, "1.2.3", "bad");
        fs::write(corrupt.join("lib/runtime.bin"), b"corrupt").unwrap();
        fs::create_dir_all(source.join("1.2.3/missing")).unwrap();
        fs::create_dir_all(source.join("unknown-release")).unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("new-cache"));
        let preview = import_legacy_runtime_cache(&source, &cache, "0.1.57", true).unwrap();
        assert!(preview.has_failures());
        assert_eq!(preview.entries.len(), 4);
        assert!(!cache.root().exists());
        let report = import_legacy_runtime_cache(&source, &cache, "0.1.57", false).unwrap();
        assert!(report.has_failures());
        assert_eq!(
            report
                .entries
                .iter()
                .filter(|e| matches!(e, LegacyRuntimeImportEntry::Failed { .. }))
                .count(),
            1
        );
        assert_eq!(
            report
                .entries
                .iter()
                .filter(|e| matches!(e, LegacyRuntimeImportEntry::Skipped { .. }))
                .count(),
            2
        );
        let imported = report
            .entries
            .iter()
            .find_map(|entry| match entry {
                LegacyRuntimeImportEntry::Runtime { outcome, warnings } => {
                    Some((outcome, warnings))
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(imported.0.status, NativeRuntimeImportStatus::Imported);
        assert_eq!(imported.1.len(), 1);
        assert!(imported.1[0].contains("ABI 0.1.0"));
        assert_eq!(fs::read(valid.join("manifest.json")).unwrap(), original);
        assert_eq!(
            fs::read(valid.join("lib/runtime.bin")).unwrap(),
            b"legacy payload"
        );
        assert_eq!(
            fs::read(imported.0.destination.join("lib/runtime.bin")).unwrap(),
            b"legacy payload"
        );
        let repeat = import_legacy_runtime_cache(&source, &cache, "0.1.57", false).unwrap();
        assert!(repeat.entries.iter().any(|entry| matches!(entry,
            LegacyRuntimeImportEntry::Runtime { outcome, .. } if outcome.status == NativeRuntimeImportStatus::AlreadyPresent)));
        assert!(!cache.runtime_dir("1.2.3", "bad").exists());
    }

    #[test]
    fn legacy_reader_rejects_generation_markers_and_directory_identity_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("legacy");
        let dir = legacy_fixture(&source, "1.2.3", "runtime");
        let path = dir.join("manifest.json");
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        for (pointer, value) in [
            ("schema_version", serde_json::json!(2)),
            ("schema_version", serde_json::Value::Null),
        ] {
            let mut changed = original.clone();
            changed[pointer] = value;
            fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            assert!(
                decode_legacy(&path, "1.2.3", "runtime")
                    .unwrap_err()
                    .to_string()
                    .contains("unversioned")
            );
        }
        fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
        assert!(decode_legacy(&path, "2.0.0", "runtime").is_err());
        assert!(decode_legacy(&path, "1.2.3", "other").is_err());
    }
}
