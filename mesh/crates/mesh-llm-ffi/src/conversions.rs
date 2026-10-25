use crate::native_runtime_types::{
    InstalledNativeRuntimeNative, NativeRuntimeDownloadProgressNative,
    NativeRuntimeInstallOutcomeNative, NativeRuntimePruneModeNative,
    NativeRuntimePruneResultNative, NativeRuntimeVerificationPolicyNative,
};
use std::path::PathBuf;

fn path_to_string(path: PathBuf) -> String {
    path.to_string_lossy().into_owned()
}

impl From<NativeRuntimeVerificationPolicyNative>
    for mesh_llm_sdk::native_runtime::NativeRuntimeVerificationPolicy
{
    fn from(value: NativeRuntimeVerificationPolicyNative) -> Self {
        match value {
            NativeRuntimeVerificationPolicyNative::RequireChecksum => Self::RequireChecksum,
            NativeRuntimeVerificationPolicyNative::RequireChecksumAndSignature => {
                Self::RequireChecksumAndSignature
            }
        }
    }
}

impl From<NativeRuntimePruneModeNative> for mesh_llm_sdk::native_runtime::NativeRuntimePruneMode {
    fn from(value: NativeRuntimePruneModeNative) -> Self {
        match value {
            NativeRuntimePruneModeNative::KeepActiveAndPrevious => Self::KeepActiveAndPrevious,
            NativeRuntimePruneModeNative::ActiveOnly => Self::ActiveOnly,
        }
    }
}

impl From<mesh_llm_sdk::native_runtime::NativeRuntimeDownloadProgress>
    for NativeRuntimeDownloadProgressNative
{
    fn from(value: mesh_llm_sdk::native_runtime::NativeRuntimeDownloadProgress) -> Self {
        Self {
            native_runtime_id: value.native_runtime_id,
            url: value.url,
            downloaded_bytes: value.downloaded_bytes,
            total_bytes: value.total_bytes,
            finished: value.finished,
        }
    }
}

impl From<mesh_llm_sdk::native_runtime::InstalledNativeRuntime> for InstalledNativeRuntimeNative {
    fn from(value: mesh_llm_sdk::native_runtime::InstalledNativeRuntime) -> Self {
        Self {
            mesh_version: value.release_version,
            native_runtime_id: value.native_runtime_id,
            flavor: value.flavor,
            path: path_to_string(value.path),
            skippy_abi_version: Some(value.manifest.runtime.skippy_abi),
        }
    }
}

impl From<mesh_llm_sdk::native_runtime::NativeRuntimeInstallOutcome>
    for NativeRuntimeInstallOutcomeNative
{
    fn from(value: mesh_llm_sdk::native_runtime::NativeRuntimeInstallOutcome) -> Self {
        Self {
            status: match value.status {
                mesh_llm_sdk::native_runtime::NativeRuntimeInstallStatus::AlreadyInstalled => {
                    "already_installed".to_string()
                }
                mesh_llm_sdk::native_runtime::NativeRuntimeInstallStatus::Installed => {
                    "installed".to_string()
                }
            },
            runtime: value.runtime.into(),
            selected_native_runtime_id: value.resolution.selected.id,
            selected_source: native_runtime_source_name(&value.resolution.source),
        }
    }
}

impl From<mesh_llm_sdk::native_runtime::CachePrunePlan> for NativeRuntimePruneResultNative {
    fn from(value: mesh_llm_sdk::native_runtime::CachePrunePlan) -> Self {
        Self {
            removed_dirs: value.remove_dirs.into_iter().map(path_to_string).collect(),
        }
    }
}

fn native_runtime_source_name(
    source: &mesh_llm_sdk::native_runtime::NativeRuntimeSource,
) -> String {
    match source {
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Installed { .. } => "installed",
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Bundle { .. } => "bundle",
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Download { .. } => "download",
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Missing => "missing",
    }
    .to_string()
}
