//! Explicit, non-destructive payload copying for runtime cache migrations.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_native_runtime::{NativeRuntimeCache, NativeRuntimeManifest};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeImportStatus {
    Planned,
    Imported,
    AlreadyPresent,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeRuntimeImportOutcome {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub status: NativeRuntimeImportStatus,
}

/// Copy a decoded runtime into a different cache without changing source bytes.
///
/// The caller owns legacy metadata decoding. This boundary validates its typed
/// manifest and all payload checksums, copies bytes (never hardlinks), and emits
/// current metadata. Existing destinations must match exactly; they are never
/// merged or replaced. Dry runs validate both sides but create nothing.
pub fn import_runtime_copy(
    source: &Path,
    manifest: &NativeRuntimeManifest,
    cache: &NativeRuntimeCache,
    dry_run: bool,
) -> Result<NativeRuntimeImportOutcome> {
    let release = manifest
        .runtime
        .mesh_version
        .as_deref()
        .context("runtime import requires an explicit release")?;
    single_component(release)?;
    single_component(&manifest.runtime.id)?;
    let source = source
        .canonicalize()
        .context("resolve runtime import source")?;
    // The shared verifier checks relative paths, canonical containment and hashes.
    manifest.verify_payload(&source)?;
    let files: BTreeSet<_> = manifest
        .runtime
        .files
        .keys()
        .chain(manifest.runtime.tools.keys())
        .collect();
    let destination = cache.runtime_dir(release, &manifest.runtime.id);
    let outcome = |status| NativeRuntimeImportOutcome {
        source: source.clone(),
        destination: destination.clone(),
        status,
    };
    if destination.try_exists()? {
        let existing = NativeRuntimeManifest::read_from_dir(&destination).with_context(|| {
            format!(
                "runtime import collision: {} -> {}",
                source.display(),
                destination.display()
            )
        })?;
        if existing != *manifest {
            bail!(
                "runtime import collision: {} -> {} (different manifest or payload)",
                source.display(),
                destination.display()
            );
        }
        return Ok(outcome(NativeRuntimeImportStatus::AlreadyPresent));
    }
    if dry_run {
        return Ok(outcome(NativeRuntimeImportStatus::Planned));
    }
    fs::create_dir_all(
        destination
            .parent()
            .context("runtime destination has no parent")?,
    )?;
    // Atomic ownership claim: never replace even an empty pre-existing entry.
    fs::create_dir(&destination)
        .with_context(|| format!("claim runtime import destination {}", destination.display()))?;
    let result = (|| -> Result<()> {
        for relative in files {
            let target = destination.join(relative);
            fs::create_dir_all(target.parent().context("runtime payload has no parent")?)?;
            fs::copy(source.join(relative), &target)
                .with_context(|| format!("copy runtime payload {}", target.display()))?;
        }
        manifest.verify_payload(&destination)?;
        // Discovery ignores the entry until its manifest exists. Publish it only
        // after all copied payloads verify, then exercise the normal reader.
        manifest.write_to_dir(&destination)?;
        NativeRuntimeManifest::read_from_dir(&destination)?;
        Ok(())
    })();
    if let Err(error) = result {
        // Only this invocation's newly claimed entry is eligible for cleanup.
        fs::remove_dir_all(&destination).with_context(|| {
            format!(
                "import failed ({error:#}); could not remove incomplete destination {}",
                destination.display()
            )
        })?;
        return Err(error);
    }
    Ok(outcome(NativeRuntimeImportStatus::Imported))
}

fn single_component(value: &str) -> Result<()> {
    let mut parts = Path::new(value).components();
    if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
        bail!("runtime import identity must be a single path component: {value:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn fixture(source: &Path) -> NativeRuntimeManifest {
        fs::create_dir_all(source.join("lib")).unwrap();
        fs::write(source.join("lib/runtime.bin"), b"verified runtime bytes").unwrap();
        let checksum = hex::encode(Sha256::digest(b"verified runtime bytes"));
        let manifest: NativeRuntimeManifest = serde_json::from_value(serde_json::json!({
            "runtime": {"id": "test-runtime", "mesh_version": "1.2.3",
                "skippy_abi": "0.1.57", "platform": {"os": "macos", "arch": "aarch64"},
                "backend": {"kind": "cpu"}, "libraries": ["lib/runtime.bin"],
                "files": {"lib/runtime.bin": checksum}}
        }))
        .unwrap();
        manifest.write_to_dir(source).unwrap();
        manifest
    }

    #[test]
    fn copy_import_is_dry_run_safe_idempotent_and_preserves_source() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("legacy");
        let manifest = fixture(&source);
        let original_metadata = fs::read(source.join("manifest.json")).unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("new-cache"));
        let planned = import_runtime_copy(&source, &manifest, &cache, true).unwrap();
        assert_eq!(planned.status, NativeRuntimeImportStatus::Planned);
        assert!(!cache.root().exists());
        let applied = import_runtime_copy(&source, &manifest, &cache, false).unwrap();
        assert_eq!(applied.status, NativeRuntimeImportStatus::Imported);
        assert_eq!(
            NativeRuntimeManifest::read_from_dir(&applied.destination).unwrap(),
            manifest
        );
        assert_eq!(
            fs::read(source.join("manifest.json")).unwrap(),
            original_metadata
        );
        assert_eq!(
            import_runtime_copy(&source, &manifest, &cache, false)
                .unwrap()
                .status,
            NativeRuntimeImportStatus::AlreadyPresent
        );
        // A destination write must not reach the legacy payload through a hardlink.
        fs::write(
            applied.destination.join("lib/runtime.bin"),
            b"changed destination",
        )
        .unwrap();
        assert_eq!(
            fs::read(source.join("lib/runtime.bin")).unwrap(),
            b"verified runtime bytes"
        );
        assert!(
            import_runtime_copy(&source, &manifest, &cache, false)
                .unwrap_err()
                .to_string()
                .contains("collision")
        );
        assert_eq!(
            fs::read(applied.destination.join("lib/runtime.bin")).unwrap(),
            b"changed destination"
        );
    }

    #[test]
    fn copy_import_refuses_invalid_source_and_existing_empty_destination() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("legacy");
        let manifest = fixture(&source);
        let cache = NativeRuntimeCache::new(temp.path().join("new-cache"));
        fs::write(source.join("lib/runtime.bin"), b"corrupt").unwrap();
        assert!(import_runtime_copy(&source, &manifest, &cache, true).is_err());
        assert!(import_runtime_copy(&source, &manifest, &cache, false).is_err());
        assert!(!cache.root().exists());
        fs::write(source.join("lib/runtime.bin"), b"verified runtime bytes").unwrap();
        let destination = cache.runtime_dir("1.2.3", "test-runtime");
        fs::create_dir_all(&destination).unwrap();
        let error = import_runtime_copy(&source, &manifest, &cache, false).unwrap_err();
        assert!(format!("{error:#}").contains("collision"));
        assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
        let mut unsafe_manifest = manifest;
        unsafe_manifest.runtime.mesh_version = Some("../escape".into());
        assert!(import_runtime_copy(&source, &unsafe_manifest, &cache, false).is_err());
        assert!(!temp.path().join("escape").exists());
    }

    #[cfg(unix)]
    #[test]
    fn copy_import_rejects_payload_symlinks_outside_source() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("legacy");
        let manifest = fixture(&source);
        let external = temp.path().join("outside");
        fs::rename(source.join("lib/runtime.bin"), &external).unwrap();
        std::os::unix::fs::symlink(&external, source.join("lib/runtime.bin")).unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("new-cache"));
        assert!(
            import_runtime_copy(&source, &manifest, &cache, false)
                .unwrap_err()
                .to_string()
                .contains("escapes")
        );
        assert!(!cache.root().exists());
        assert_eq!(fs::read(external).unwrap(), b"verified runtime bytes");
    }
}
