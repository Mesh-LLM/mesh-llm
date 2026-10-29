//! Where a release's bundled default plugins are.
//!
//! A release archive may carry each default plugin's release archive,
//! unchanged, in `plugins/` beside the `mesh-llm` binary. A package or a
//! formula may keep that directory under its own prefix instead. The node
//! installs a default only from here, after checking the archive against this
//! build's pin; it never downloads one.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::Digest;

use crate::catalog::{PinnedRelease, is_sha256_hex};
use crate::install::{
    InstallOutcome, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter,
    extract_for_install,
};
use crate::store::{InstalledPluginMetadata, PluginStore};
use crate::target::ArchiveExt;

/// Names a bundled-plugins directory to use instead of the one beside the
/// binary (a development build, or a test).
pub const BUNDLED_PLUGINS_DIR_ENV: &str = "MESH_LLM_BUNDLED_PLUGINS_DIR";

/// The release these plugins were bundled with; versioned package installs
/// keep them under `lib/mesh-llm/<release>/plugins`.
pub const BUNDLED_RELEASE: &str = env!("CARGO_PKG_VERSION");

/// The file name a plugin's release archive has in the bundle, the name its
/// own release published it under.
pub fn bundled_archive_name(name: &str, version: &str, target: &str) -> String {
    format!("{name}-{version}-{target}.tar.gz")
}

/// The directories a bundled plugin may be in, in order, for the binary at
/// `executable` of release `release`: beside the binary (the release archive
/// as installed by `install.sh`), then the layouts packages and formulas use
/// for the native runtime (`lib/mesh-llm/<release>`, `lib/mesh-llm`,
/// `libexec`).
pub fn bundled_plugin_dir_candidates(executable: &Path, release: &str) -> Vec<PathBuf> {
    let executable = executable
        .canonicalize()
        .unwrap_or_else(|_| executable.to_path_buf());
    let Some(executable_dir) = executable.parent() else {
        return Vec::new();
    };
    let mut candidates = vec![executable_dir.join("plugins")];
    if let Some(prefix) = executable_dir.parent() {
        candidates.push(
            prefix
                .join("lib")
                .join("mesh-llm")
                .join(release)
                .join("plugins"),
        );
        candidates.push(prefix.join("lib").join("mesh-llm").join("plugins"));
        candidates.push(prefix.join("libexec").join("plugins"));
    }
    candidates
}

/// This install's bundled-plugins directory: `MESH_LLM_BUNDLED_PLUGINS_DIR`
/// when set, else the first candidate that exists. `None` when there is none
/// (a development build, or a platform the release bundles nothing for).
pub fn bundled_plugins_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(BUNDLED_PLUGINS_DIR_ENV).filter(|dir| !dir.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    let executable = std::env::current_exe().ok()?;
    bundled_plugin_dir_candidates(&executable, BUNDLED_RELEASE)
        .into_iter()
        .find(|dir| dir.is_dir())
}

/// Install default plugin `name` at `pin` from this release's bundled copy
/// (`options.bundled_plugins_dir`), the release archive the release pipeline
/// verified. The archive must match `pin`, a pin held in reviewed code,
/// before anything is extracted. Nothing is ever downloaded: `Ok(None)` when
/// the release bundles no copy for this platform.
pub fn install_bundled_default(
    name: &str,
    pin: PinnedRelease<'_>,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<Option<InstallOutcome>> {
    if !is_sha256_hex(pin.sha256) {
        bail!(
            "default plugin '{name}' {} pins no SHA-256 for {} ({:?}); not installed",
            pin.version,
            options.target.triple(),
            pin.sha256
        );
    }
    let asset = bundled_archive_name(name, pin.version, options.target.triple());
    let Some(archive_path) = options
        .bundled_plugins_dir
        .as_ref()
        .map(|dir| dir.join(&asset))
        .filter(|path| path.is_file())
    else {
        return Ok(None);
    };
    let (verified_archive, digest) = copy_and_hash_bundled_archive(&archive_path)?;
    if digest != pin.sha256 {
        bail!(
            "the bundled {asset} is {digest}, not the reviewed {} {}; not installed",
            pin.version,
            pin.sha256
        );
    }
    progress.report(PluginProgressEvent::Extracting {
        asset: asset.clone(),
    });
    let store = PluginStore::new(&options.store_root);
    let current = store.load_optional(name)?;
    let extracted = extract_for_install(verified_archive.path(), ArchiveExt::TarGz, name, options)?;
    let metadata = InstalledPluginMetadata {
        name: name.to_string(),
        source_repository: format!("bundled:{}", archive_path.display()),
        installed_version: pin.version.to_string(),
        target_triple: options.target.triple().to_string(),
        downloaded_asset_name: asset,
        install_path: extracted.install_path,
        enabled: current.as_ref().map(|item| item.enabled).unwrap_or(true),
        default_managed: true,
        requires: extracted.requires,
        manifest: extracted.manifest,
        last_protocol_version: current.as_ref().and_then(|item| item.last_protocol_version),
        last_status: current.as_ref().and_then(|item| item.last_status.clone()),
        last_error: None,
    };
    store.save(&metadata)?;
    progress.report(PluginProgressEvent::Installed {
        installed: Box::new(metadata.clone()),
    });
    Ok(Some(InstallOutcome {
        metadata,
        changed: true,
    }))
}

/// For tests elsewhere in the crate: the digest a bundled archive must match.
#[cfg(test)]
pub(crate) fn tests_sha256_file(path: &std::path::Path) -> String {
    sha256_file(path).unwrap()
}

#[cfg(test)]
fn sha256_file(path: &std::path::Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("open bundled plugin {}", path.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)
            .with_context(|| format!("read bundled plugin {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn copy_and_hash_bundled_archive(
    path: &std::path::Path,
) -> Result<(tempfile::NamedTempFile, String)> {
    let mut source =
        fs::File::open(path).with_context(|| format!("open bundled plugin {}", path.display()))?;
    let mut verified_archive = tempfile::NamedTempFile::new()
        .context("create private temporary copy of bundled plugin")?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = std::io::Read::read(&mut source, &mut buffer)
            .with_context(|| format!("read bundled plugin {}", path.display()))?;
        if read == 0 {
            break;
        }
        verified_archive
            .write_all(&buffer[..read])
            .context("copy bundled plugin to private temporary file")?;
        hasher.update(&buffer[..read]);
    }
    verified_archive
        .flush()
        .context("flush private temporary copy of bundled plugin")?;
    Ok((verified_archive, hex::encode(hasher.finalize())))
}

#[cfg(test)]
mod tests {
    use flate2::{Compression, write::GzEncoder};
    use tempfile::TempDir;

    use super::*;
    use crate::target::PluginTarget;

    fn write_tar_gz(archive_path: &Path, plugin_name: &str, files: &[(&str, &[u8])]) -> Result<()> {
        let archive_file = fs::File::create(archive_path)?;
        let encoder = GzEncoder::new(archive_file, Compression::default());
        let mut archive = tar::Builder::new(encoder);
        for (relative_path, contents) in files {
            let path = format!("{plugin_name}/{relative_path}");
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            archive.append_data(&mut header, path, *contents)?;
        }
        archive.finish()?;
        archive.into_inner()?.finish()?;
        Ok(())
    }

    /// Options for installing for `os`/`arch`, which need not be the host.
    fn options_for(root: &Path, os: &str, arch: &str) -> PluginInstallOptions {
        PluginInstallOptions {
            store_root: root.join("store"),
            install_root: root.join("installed"),
            catalog_url: "unused".to_string(),
            target: PluginTarget::from_os_arch(os, arch).unwrap(),
            bundled_plugins_dir: Some(root.join("bundle")),
        }
    }

    /// Each target with the executable name its build carries, whatever the host.
    const TARGETS: [(&str, &str, &str); 2] = [
        ("windows", "x86_64", "demo.exe"),
        ("linux", "x86_64", "demo"),
    ];

    #[test]
    fn candidates_follow_the_native_runtime_layouts() {
        let temp = tempfile::tempdir().unwrap();
        let prefix = temp.path().canonicalize().unwrap();
        let binary = prefix.join("bin").join("mesh-llm");
        assert_eq!(
            bundled_plugin_dir_candidates(&binary, "0.79.0"),
            vec![
                prefix.join("bin").join("plugins"),
                prefix.join("lib/mesh-llm/0.79.0/plugins"),
                prefix.join("lib/mesh-llm/plugins"),
                prefix.join("libexec/plugins"),
            ]
        );
    }

    #[test]
    fn a_bundled_archive_copy_keeps_the_bytes_that_were_hashed() {
        let temp = TempDir::new().unwrap();
        let archive_path = temp.path().join("demo.tar.gz");
        let pinned_bytes = b"verified plugin archive";
        fs::write(&archive_path, pinned_bytes).unwrap();

        let (verified_archive, digest) = copy_and_hash_bundled_archive(&archive_path).unwrap();
        fs::write(&archive_path, b"replacement plugin archive").unwrap();

        assert_ne!(verified_archive.path(), archive_path);
        assert_eq!(digest, hex::encode(sha2::Sha256::digest(pinned_bytes)));
        assert_eq!(fs::read(verified_archive.path()).unwrap(), pinned_bytes);
    }

    #[test]
    fn a_bundled_default_for_another_target_installs_and_its_executable_is_where_the_host_looks() {
        for (os, arch, executable) in TARGETS {
            let temp = TempDir::new().unwrap();
            let options = options_for(temp.path(), os, arch);
            let bundle = temp.path().join("bundle");
            fs::create_dir_all(&bundle).unwrap();
            let archive_path = bundle.join(crate::bundled::bundled_archive_name(
                "demo",
                "0.1.0",
                options.target.triple(),
            ));
            write_tar_gz(
                &archive_path,
                "demo",
                &[("plugin.toml", b"name = \"demo\""), (executable, b"")],
            )
            .unwrap();
            let digest = sha256_file(&archive_path).unwrap();
            let pin = PinnedRelease {
                version: "0.1.0",
                sha256: &digest,
            };

            let outcome = install_bundled_default("demo", pin, &options, &mut |_| {})
                .unwrap_or_else(|error| panic!("{os}/{arch}: {error:#}"))
                .unwrap_or_else(|| panic!("{os}/{arch}: the bundled copy was not found"));

            assert_eq!(
                outcome.metadata.executable_path(),
                outcome.metadata.install_path.join(executable),
                "{os}/{arch}"
            );
            assert!(outcome.metadata.executable_path().is_file(), "{os}/{arch}");
        }
    }

    #[test]
    fn the_archive_name_is_the_plugin_releases_own() {
        assert_eq!(
            bundled_archive_name("capsules", "0.1.4", "x86_64-unknown-linux-gnu"),
            "capsules-0.1.4-x86_64-unknown-linux-gnu.tar.gz"
        );
    }
}
