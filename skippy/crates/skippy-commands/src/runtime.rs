//! Native-runtime commands using the same cache, discovery, and installer as Mesh.

use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use skippy_runtime_install::{
    NativeRuntimeBundleInstallPolicy, NativeRuntimeCache, NativeRuntimeCatalog,
    NativeRuntimeInstallOptions, NativeRuntimeManifestOptions, NativeRuntimePruneMode,
    NativeRuntimeResolver, RuntimeSelection,
};

#[derive(Debug, Clone, Default)]
pub struct RuntimeRunOptions {
    pub cache_dir: Option<PathBuf>,
    pub release: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub selection: Option<String>,
}

#[derive(Debug, Clone)]
pub enum RuntimeAction {
    List {
        available: bool,
        manifest: Option<PathBuf>,
    },
    Install {
        runtime: Option<String>,
        manifest: Option<PathBuf>,
    },
    Remove {
        native_runtime_id: String,
        release: Option<String>,
    },
    Prune {
        active_only: bool,
        release: Option<String>,
    },
}

pub async fn run(command: RuntimeAction, options: &RuntimeRunOptions) -> Result<()> {
    match command {
        RuntimeAction::List {
            available,
            manifest,
        } => list(available, manifest, options).await,
        RuntimeAction::Install { runtime, manifest } => {
            install(runtime.as_deref(), manifest, options).await
        }
        RuntimeAction::Remove {
            native_runtime_id,
            release,
        } => remove(&native_runtime_id, release.as_deref(), options),
        RuntimeAction::Prune {
            active_only,
            release,
        } => prune(active_only, release.as_deref(), options),
    }
}

fn release<'a>(options: &'a RuntimeRunOptions, override_release: Option<&'a str>) -> &'a str {
    override_release
        .or(options.release.as_deref())
        .unwrap_or(skippy_runtime_install::runtime_release_version())
}

fn catalog(runtime_release: &str) -> NativeRuntimeCatalog {
    let build = option_env!("MESH_LLM_BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"));
    skippy_runtime_install::publication_catalog(build, env!("CARGO_PKG_VERSION"), runtime_release)
}

fn cache(options: &RuntimeRunOptions) -> Result<NativeRuntimeCache> {
    let root = options
        .cache_dir
        .as_ref()
        .context("runtime cache not resolved")?;
    Ok(NativeRuntimeCache::new(root))
}

async fn list(
    available: bool,
    manifest: Option<PathBuf>,
    options: &RuntimeRunOptions,
) -> Result<()> {
    let release = release(options, None);
    let cache = cache(options)?;
    if !available {
        let installed = skippy_runtime_install::discover_local_native_runtimes_in(
            &options.bundle_dirs,
            &cache,
            release,
            |_| true,
        )?;
        return crate::console::present(&installed, |output| {
            if installed.is_empty() {
                writeln!(output, "No native runtimes installed.")?;
            }
            for runtime in &installed {
                writeln!(
                    output,
                    "⚙️  {} ({})",
                    runtime.native_runtime_id, runtime.flavor
                )?;
                writeln!(output, "   {}", runtime.path.display())?;
            }
            Ok(())
        });
    }

    let mut manifest_options = NativeRuntimeManifestOptions::new(release, catalog(release));
    manifest_options.manifest_path = manifest;
    manifest_options.bundle_dirs = options.bundle_dirs.clone();
    let (manifest, sources) =
        skippy_runtime_install::load_release_manifest_with_sources(manifest_options).await?;
    let selection = RuntimeSelection::parse(options.selection.as_deref())?;
    let evaluated = NativeRuntimeResolver::new(
        release,
        skippy_runtime_install::host_runtime_profile(),
        manifest,
        cache,
    )
    .with_bundle_dirs(sources.bundle_dirs.clone())
    .with_skippy_abi_version(skippy_runtime_install::current_skippy_abi_version())
    .evaluate(&selection)?;
    let report = serde_json::json!({"sources": sources, "candidates": evaluated});
    crate::console::present(&report, |output| {
        if evaluated.is_empty() {
            writeln!(output, "No native runtimes available.")?;
        }
        for candidate in &evaluated {
            let status = if candidate.compatible { "✅" } else { "⛔" };
            writeln!(output, "{status} {}", candidate.artifact.id)?;
            for reason in &candidate.rejection_reasons {
                writeln!(output, "   {reason}")?;
            }
        }
        Ok(())
    })
}

async fn install(
    runtime: Option<&str>,
    manifest: Option<PathBuf>,
    options: &RuntimeRunOptions,
) -> Result<()> {
    let release = release(options, None);
    let mut install = NativeRuntimeInstallOptions::new(release, catalog(release));
    install.manifest_path = manifest;
    install.cache_dir = options.cache_dir.clone();
    install.bundle_dirs = options.bundle_dirs.clone();
    install.selection = RuntimeSelection::parse(runtime.or(options.selection.as_deref()))?;
    install.skippy_abi_version = Some(skippy_runtime_install::current_skippy_abi_version());
    install.bundle_install_policy =
        NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache;
    install.progress = Some(Arc::new(|progress| {
        if let Some(total) = progress.total_bytes {
            let _ = crate::console::progress("Native runtime", progress.downloaded_bytes, total);
        }
    }));
    let outcome = skippy_runtime_install::install_native_runtime(install).await?;
    crate::console::present(&outcome, |output| {
        writeln!(
            output,
            "✅ Native runtime ready: {}",
            outcome.runtime.native_runtime_id
        )?;
        writeln!(output, "   {}", outcome.runtime.path.display())
    })
}

fn remove(id: &str, override_release: Option<&str>, options: &RuntimeRunOptions) -> Result<()> {
    let release = release(options, override_release);
    let removed = cache(options)?.remove(release, id)?;
    let report =
        serde_json::json!({"native_runtime_id": id, "release": release, "removed": removed});
    crate::console::present(&report, |output| {
        if removed {
            writeln!(output, "🗑️ Removed native runtime {id}")
        } else {
            writeln!(
                output,
                "No installed native runtime {id} for release {release}"
            )
        }
    })
}

fn prune(
    active_only: bool,
    override_release: Option<&str>,
    options: &RuntimeRunOptions,
) -> Result<()> {
    let release = release(options, override_release);
    let mode = if active_only {
        NativeRuntimePruneMode::ActiveOnly
    } else {
        NativeRuntimePruneMode::KeepActiveAndPrevious
    };
    let plan = cache(options)?.prune(release, mode)?;
    crate::console::present(&plan, |output| {
        writeln!(
            output,
            "🧹 Pruned {} native runtime release(s)",
            plan.remove_dirs.len()
        )?;
        for path in &plan.remove_dirs {
            writeln!(output, "   {}", path.display())?;
        }
        Ok(())
    })
}
