use std::{path::PathBuf, sync::Arc};

use crate::errors::{FfiError, map_native_runtime_error};
use crate::native_runtime_types::{
    InstalledNativeRuntimeNative, NativeRuntimeInstallOptionsNative,
    NativeRuntimeInstallOutcomeNative, NativeRuntimeProgressListener, NativeRuntimePruneModeNative,
    NativeRuntimePruneResultNative,
};
use crate::runtime_blocking::block_on;

#[uniffi::export]
pub fn current_mesh_version() -> String {
    mesh_llm_sdk::native_runtime::CURRENT_MESH_VERSION.to_string()
}

#[uniffi::export]
pub fn current_skippy_abi_version() -> String {
    mesh_llm_sdk::native_runtime::current_skippy_abi_version()
}

#[uniffi::export]
pub fn install_native_runtime(
    options: NativeRuntimeInstallOptionsNative,
    progress: Option<Box<dyn NativeRuntimeProgressListener>>,
) -> Result<NativeRuntimeInstallOutcomeNative, FfiError> {
    let options = runtime_install_options(options, progress)?;
    block_on(mesh_llm_sdk::native_runtime::install_native_runtime(
        options,
    ))
    .map(NativeRuntimeInstallOutcomeNative::from)
    .map_err(map_native_runtime_error)
}

#[uniffi::export]
pub fn installed_native_runtimes(
    cache_dir: Option<String>,
) -> Result<Vec<InstalledNativeRuntimeNative>, FfiError> {
    native_runtime_cache(cache_dir)?
        .installed()
        .map(|runtimes| {
            runtimes
                .into_iter()
                .map(InstalledNativeRuntimeNative::from)
                .collect()
        })
        .map_err(map_native_runtime_error)
}

#[uniffi::export]
pub fn remove_native_runtime(
    cache_dir: Option<String>,
    mesh_version: String,
    native_runtime_id: String,
) -> Result<bool, FfiError> {
    native_runtime_cache(cache_dir)?
        .remove(&mesh_version, &native_runtime_id)
        .map_err(map_native_runtime_error)
}

#[uniffi::export]
pub fn prune_native_runtimes(
    cache_dir: Option<String>,
    active_mesh_version: Option<String>,
    mode: NativeRuntimePruneModeNative,
) -> Result<NativeRuntimePruneResultNative, FfiError> {
    let active_mesh_version = active_mesh_version
        .unwrap_or_else(|| mesh_llm_sdk::native_runtime::CURRENT_MESH_VERSION.to_string());
    native_runtime_cache(cache_dir)?
        .prune(&active_mesh_version, mode.into())
        .map(NativeRuntimePruneResultNative::from)
        .map_err(map_native_runtime_error)
}

fn runtime_install_options(
    options: NativeRuntimeInstallOptionsNative,
    progress: Option<Box<dyn NativeRuntimeProgressListener>>,
) -> Result<mesh_llm_sdk::native_runtime::NativeRuntimeInstallOptions, FfiError> {
    let progress = progress.map(runtime_progress_callback);
    Ok(mesh_llm_sdk::native_runtime::NativeRuntimeInstallOptions {
        mesh_version: options
            .mesh_version
            .unwrap_or_else(|| mesh_llm_sdk::native_runtime::CURRENT_MESH_VERSION.to_string()),
        skippy_abi_version: options.skippy_abi_version,
        selection: mesh_llm_sdk::native_runtime::RuntimeSelection::parse(Some(
            options.selection.as_str(),
        ))
        .map_err(map_native_runtime_error)?,
        manifest_path: options.manifest_path.map(PathBuf::from),
        manifest_url: options.manifest_url,
        bundle_dirs: options.bundle_dirs.into_iter().map(PathBuf::from).collect(),
        cache_dir: options.cache_dir.map(PathBuf::from),
        verification_policy: options.verification_policy.into(),
        bundle_install_policy: Default::default(),
        progress,
        allow_download: options.allow_download,
    })
}

fn runtime_progress_callback(
    listener: Box<dyn NativeRuntimeProgressListener>,
) -> mesh_llm_sdk::native_runtime::NativeRuntimeDownloadProgressCallback {
    let listener: Arc<dyn NativeRuntimeProgressListener> = Arc::from(listener);
    Arc::new(move |event| listener.on_progress(event.into()))
}

fn native_runtime_cache(
    cache_dir: Option<String>,
) -> Result<mesh_llm_sdk::native_runtime::NativeRuntimeCache, FfiError> {
    let cache_dir = cache_dir.map(PathBuf::from);
    mesh_llm_sdk::native_runtime::native_runtime_cache(cache_dir.as_deref())
        .map_err(map_native_runtime_error)
}
