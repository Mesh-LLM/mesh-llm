//! Standalone native-runtime command execution.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use skippy_runtime_install::{NativeRuntimeCache, NativeRuntimeManifest};

/// Resolved native runtime inputs for command execution, decoupled from clap.
///
/// The CLI assembles this from parsed arguments plus the `skippy-config`
/// path policy and hands it over with the parsed [`RuntimeAction`].
#[derive(Debug, Clone, Default)]
pub struct RuntimeRunOptions {
    /// Resolved runtime cache root; `None` fails the command at execution.
    pub cache_dir: Option<PathBuf>,
    pub release: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub selection: Option<String>,
}

/// Parsed `skippy runtime` action, decoupled from clap.
#[derive(Debug, Clone)]
pub enum RuntimeAction {
    List,
    /// Install a checksum-verified runtime from an explicit release catalog.
    Install {
        manifest: Option<PathBuf>,
        manifest_url: Option<String>,
    },
    /// Copy a verified bundle into the Skippy cache; leave the source unchanged.
    Import {
        source: PathBuf,
        dry_run: bool,
    },
    /// Explicitly migrate a Mesh-era runtime cache without modifying its contents.
    ImportLegacy {
        source: PathBuf,
        dry_run: bool,
    },
}

pub async fn run(command: RuntimeAction, options: &RuntimeRunOptions) -> Result<()> {
    let cache = NativeRuntimeCache::new(
        options
            .cache_dir
            .as_ref()
            .context("runtime cache not resolved")?,
    );
    match command {
        RuntimeAction::List => crate::console::write_json(&cache.installed()?),
        RuntimeAction::Install {
            manifest,
            manifest_url,
        } => {
            use skippy_runtime_install::{
                NativeRuntimeBundleInstallPolicy, NativeRuntimeCatalog,
                NativeRuntimeInstallOptions, RuntimeSelection,
            };
            anyhow::ensure!(
                manifest.is_some() != manifest_url.is_some(),
                "supply exactly one runtime catalog file or URL"
            );
            let release = options
                .release
                .as_deref()
                .unwrap_or(skippy_runtime_install::runtime_release_version());
            // The CLI requires an explicit catalog; this default URL is never used.
            let catalog = NativeRuntimeCatalog {
                releases_url: String::new(),
                release_tags: Default::default(),
                rolling_release: None,
            };
            let mut install = NativeRuntimeInstallOptions::new(release, catalog);
            install.manifest_path = manifest;
            install.manifest_url = manifest_url;
            install.cache_dir = options.cache_dir.clone();
            install.bundle_dirs = options.bundle_dirs.clone();
            install.selection = RuntimeSelection::parse(options.selection.as_deref())?;
            install.skippy_abi_version = Some(skippy_runtime_install::current_skippy_abi_version());
            install.bundle_install_policy =
                NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache;
            let outcome = skippy_runtime_install::install_native_runtime_explicit(install).await?;
            crate::console::write_json(&outcome)
        }
        RuntimeAction::Import { source, dry_run } => {
            let manifest = NativeRuntimeManifest::read_from_dir(&source)?;
            let outcome =
                skippy_runtime_install::import_runtime_copy(&source, &manifest, &cache, dry_run)?;
            crate::console::write_json(&outcome)
        }
        RuntimeAction::ImportLegacy { source, dry_run } => {
            let report = skippy_runtime_install::import_legacy_runtime_cache(
                &source,
                &cache,
                &skippy_runtime_install::current_skippy_abi_version(),
                dry_run,
            )?;
            crate::console::write_json(&report)?;
            if report.has_failures() {
                bail!("one or more legacy runtime imports failed; see the JSON report");
            }
            Ok(())
        }
    }
}
