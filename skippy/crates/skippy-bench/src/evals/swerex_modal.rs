//! Finite Modal SDK patch admission. No recursive directory traversal or deployment execution.
use anyhow::{Result, bail};
use std::path::Path;

#[cfg(unix)]
mod contract;
#[cfg(unix)]
mod transform;
#[cfg(unix)]
use super::swerex_source::AdmittedSource;
#[cfg(unix)]
use sha2::{Digest, Sha256};

#[cfg(unix)]
fn digest(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    let hex = b"0123456789abcdef";
    let mut result = String::with_capacity(64);
    for byte in hash {
        result.push(char::from(hex[usize::from(byte >> 4)]));
        result.push(char::from(hex[usize::from(byte & 15)]));
    }
    result
}

#[cfg(unix)]
struct Prepared {
    destination: AdmittedSource,
    path: std::path::PathBuf,
    replacement: Vec<u8>,
    original_sha256: String,
    already_patched: bool,
    changes_source: bool,
}

#[cfg(unix)]
fn prepare(
    package: &Path,
    environment_root: &Path,
    patch_root: &Path,
    mapping: &contract::Mapping<'_>,
    rewrite: fn(&str, &[u8]) -> Result<Vec<u8>>,
) -> Result<Prepared> {
    let original = mapping.original_sha256.ok_or_else(|| {
        anyhow::anyhow!("official SWE-ReX 1.4.0 destination compatibility is not admitted")
    })?;
    let replacement = AdmittedSource::open(&patch_root.join(mapping.source), patch_root)?;
    if digest(&replacement.bytes) != mapping.source_sha256 {
        bail!("SWE-agent Modal replacement does not match its pinned source");
    }
    let destination = AdmittedSource::open(&package.join(mapping.destination), environment_root)?;
    let current = digest(&destination.bytes);
    let changes_source = original != mapping.replacement_sha256;
    let already_patched = changes_source && current == mapping.replacement_sha256;
    if current != original && !already_patched {
        bail!("SWE-ReX Modal destination is neither original nor the admitted replacement");
    }
    let backup = destination.backup_bytes()?;
    if already_patched && backup.is_none() {
        bail!("patched SWE-ReX Modal destination has no original backup");
    }
    if backup
        .as_deref()
        .is_some_and(|bytes| digest(bytes) != original)
    {
        bail!("SWE-ReX Modal original backup has unknown bytes");
    }
    let original_bytes = if already_patched {
        backup
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("missing admitted original backup"))?
    } else {
        destination.bytes.as_slice()
    };
    let rewritten = rewrite(mapping.destination, original_bytes)?;
    if digest(&rewritten) != mapping.replacement_sha256 {
        bail!("rewritten SWE-ReX source differs from the reviewed finite transform");
    }
    Ok(Prepared {
        destination,
        path: package.join(mapping.destination),
        replacement: rewritten,
        original_sha256: original.to_owned(),
        already_patched,
        changes_source,
    })
}

#[cfg(unix)]
fn apply_with(
    module_source: &Path,
    environment_root: &Path,
    patch_root: &Path,
    metadata_sha256: Option<&str>,
    mappings: &[contract::Mapping<'_>; 3],
    rewrite: fn(&str, &[u8]) -> Result<Vec<u8>>,
) -> Result<()> {
    let metadata_sha256 = metadata_sha256
        .ok_or_else(|| anyhow::anyhow!("official SWE-ReX 1.4.0 SDK metadata is not admitted"))?;
    if mappings
        .iter()
        .any(|mapping| mapping.original_sha256.is_none())
    {
        bail!("official SWE-ReX 1.4.0 destination compatibility is not admitted");
    }
    if module_source.file_name() != Some(std::ffi::OsStr::new("__init__.py")) {
        bail!("SWE-ReX Modal locator must name the installed package initializer");
    }
    let package = module_source
        .parent()
        .ok_or_else(|| anyhow::anyhow!("missing SDK package"))?;
    if package.file_name() != Some(std::ffi::OsStr::new("swerex")) {
        bail!("SWE-ReX Modal locator must name the swerex package");
    }
    let _locator = AdmittedSource::open(module_source, environment_root)?;
    let site_packages = package
        .parent()
        .ok_or_else(|| anyhow::anyhow!("missing SDK site packages"))?;
    let metadata = AdmittedSource::open(
        &site_packages.join("swe_rex-1.4.0.dist-info/METADATA"),
        environment_root,
    )?;
    if digest(&metadata.bytes) != metadata_sha256 {
        bail!("installed SWE-ReX SDK metadata does not match the admitted 1.4.0 distribution");
    }
    // Admit all source/destination/backup bytes before the first mutation.
    let prepared = mappings
        .iter()
        .map(|mapping| prepare(package, environment_root, patch_root, mapping, rewrite))
        .collect::<Result<Vec<_>>>()?;
    for file in prepared {
        if file.changes_source && !file.already_patched {
            file.destination.preserve_backup()?;
            file.destination.replace(&file.replacement)?;
        }
        let current = AdmittedSource::open(&file.path, environment_root)?;
        if current.bytes != file.replacement
            || file.changes_source
                && current
                    .backup_bytes()?
                    .as_deref()
                    .is_none_or(|bytes| digest(bytes) != file.original_sha256)
        {
            bail!("SWE-ReX Modal patch custody failed after replacement");
        }
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn patch(
    module_source: &Path,
    environment_root: &Path,
    patch_root: &Path,
) -> Result<()> {
    apply_with(
        module_source,
        environment_root,
        patch_root,
        contract::SDK_METADATA_SHA256,
        &contract::MAPPINGS,
        transform::rewrite,
    )
}

#[cfg(not(unix))]
pub(super) fn patch(
    _module_source: &Path,
    _environment_root: &Path,
    _patch_root: &Path,
) -> Result<()> {
    bail!("the SWE-ReX Modal source patch requires Unix");
}

#[cfg(all(test, unix))]
mod tests;
/// Admit the current compiled Modal profile without rewriting any installed SDK files.
#[cfg(unix)]
pub(super) fn admit_patched(environment_root: &Path) -> Result<()> {
    let package = environment_root.join("lib/python3.11/site-packages/swerex");
    for mapping in &contract::MAPPINGS {
        let admitted = AdmittedSource::open(&package.join(mapping.destination), environment_root)?;
        if digest(&admitted.bytes) != mapping.replacement_sha256 {
            bail!("prepared Modal SDK differs from current compiled profile");
        }
    }
    Ok(())
}
#[cfg(not(unix))]
pub(super) fn admit_patched(_environment_root: &Path) -> Result<()> {
    bail!("prepared Modal SDK admission requires Unix");
}
