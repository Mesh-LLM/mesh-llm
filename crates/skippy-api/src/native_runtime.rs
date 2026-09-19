//! Local native-library startup for standalone server entry points.

use anyhow::{Context, Result};
use skippy_native_runtime::{
    NativeRuntimeLoadPlan, RuntimeSelection, has_startup_compatibility_metadata,
};
use skippy_runtime_install::{
    NativeRuntimeCache, current_skippy_abi_version, discover_local_native_runtimes_in,
    host_runtime_profile,
};

/// Explicit native runtime selection supplied by the embedding application.
#[derive(Clone, Debug, Default)]
pub struct NativeRuntimeOptions {
    pub bundle_dirs: Vec<std::path::PathBuf>,
    pub cache_dir: Option<std::path::PathBuf>,
    pub release: Option<String>,
    pub selection: Option<String>,
}

/// Resolve without entering native code, preserving the shared Mesh selection rules.
pub fn local_native_runtime_plan(args: &NativeRuntimeOptions) -> Result<NativeRuntimeLoadPlan> {
    let release = args
        .release
        .as_deref()
        .unwrap_or(skippy_native_runtime::runtime_release_version());
    let abi = current_skippy_abi_version();
    let profile = host_runtime_profile();
    let cache = NativeRuntimeCache::new(
        args.cache_dir
            .as_ref()
            .context("native runtime cache directory must be supplied by the caller")?,
    );
    let selection = RuntimeSelection::parse(args.selection.as_deref())?;
    let runtimes =
        discover_local_native_runtimes_in(&args.bundle_dirs, &cache, release, |runtime| {
            has_startup_compatibility_metadata(&runtime.manifest.runtime, &profile)
        })?;
    skippy_runtime_install::startup::select_local_native_runtime_plan(
        &runtimes, &profile, release, Some(&abi), &selection,
    )?.with_context(|| format!(
        "no compatible local Skippy runtime for release {release}, ABI {abi}; supply --runtime-bundle or --runtime-cache containing a verified runtime"
    ))
}

/// Load the selected native libraries before any stage or tokenizer native call.
/// Runtime acquisition is explicit; this entry point never downloads automatically.
#[cfg(feature = "dynamic-native-runtime")]
pub fn load_local_native_runtime(args: &NativeRuntimeOptions) -> Result<()> {
    if skippy_runtime::native_runtime_loaded() {
        return Ok(());
    }
    let plan = local_native_runtime_plan(args)?;
    // The normal reader verified the selected bundle's ABI metadata and payload hashes.
    // The dynamic loader additionally checks the ABI exported by the actual library.
    unsafe { skippy_runtime::load_native_runtime_libraries(&plan.libraries) }.with_context(|| {
        format!(
            "load Skippy runtime {} from {}",
            plan.native_runtime_id,
            plan.root.display()
        )
    })
}
