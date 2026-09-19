use crate::cli::{NativeRuntimeArgs, RuntimeCommand};
use anyhow::{Context, Result, bail};
use skippy_api::native_runtime::NativeRuntimeOptions;
use skippy_runtime_install::{NativeRuntimeCache, NativeRuntimeManifest};

pub fn resolve_options(args: NativeRuntimeArgs) -> Result<NativeRuntimeOptions> {
    let mut options: NativeRuntimeOptions = args.into();
    if options.cache_dir.is_none() {
        options.cache_dir = Some(
            match std::env::var_os("SKIPPY_NATIVE_RUNTIME_CACHE_DIR").filter(|v| !v.is_empty()) {
                Some(path) => path.into(),
                None => dirs::cache_dir()
                    .or_else(|| dirs::home_dir().map(|p| p.join(".cache")))
                    .context(
                        "cannot determine Skippy runtime cache directory; supply --runtime-cache",
                    )?
                    .join("skippy")
                    .join("native-runtimes"),
            },
        );
    }
    if let Some(paths) = std::env::var_os("SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR") {
        options
            .bundle_dirs
            .extend(std::env::split_paths(&paths).filter(|p| !p.as_os_str().is_empty()));
    }
    Ok(options)
}

pub async fn run(command: RuntimeCommand, options: &NativeRuntimeOptions) -> Result<()> {
    let cache = NativeRuntimeCache::new(
        options
            .cache_dir
            .as_ref()
            .context("runtime cache not resolved")?,
    );
    match command {
        RuntimeCommand::List => crate::console::write_json(&cache.installed()?),
        RuntimeCommand::Install {
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
        RuntimeCommand::Import { source, dry_run } => {
            let manifest = NativeRuntimeManifest::read_from_dir(&source)?;
            let outcome =
                skippy_runtime_install::import_runtime_copy(&source, &manifest, &cache, dry_run)?;
            crate::console::write_json(&outcome)
        }
        RuntimeCommand::ImportLegacy { source, dry_run } => {
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
