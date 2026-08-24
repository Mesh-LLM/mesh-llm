use crate::events::ClientEvent;

#[derive(uniffi::Enum)]
pub enum NativeRuntimeVerificationPolicyNative {
    RequireChecksum,
    RequireChecksumAndSignature,
}

#[derive(uniffi::Enum)]
pub enum NativeRuntimePruneModeNative {
    KeepActiveAndPrevious,
    ActiveOnly,
}

#[derive(uniffi::Record)]
pub struct NativeRuntimeInstallOptionsNative {
    pub mesh_version: Option<String>,
    pub skippy_abi_version: Option<String>,
    pub selection: String,
    pub manifest_path: Option<String>,
    pub manifest_url: Option<String>,
    pub bundle_dirs: Vec<String>,
    pub cache_dir: Option<String>,
    pub verification_policy: NativeRuntimeVerificationPolicyNative,
    pub allow_download: bool,
}

#[derive(uniffi::Record)]
pub struct NativeRuntimeDownloadProgressNative {
    pub native_runtime_id: String,
    pub url: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub finished: bool,
}

#[derive(uniffi::Record)]
pub struct InstalledNativeRuntimeNative {
    pub mesh_version: String,
    pub native_runtime_id: String,
    pub flavor: String,
    pub path: String,
    pub skippy_abi_version: Option<String>,
}

#[derive(uniffi::Record)]
pub struct NativeRuntimeInstallOutcomeNative {
    pub status: String,
    pub runtime: InstalledNativeRuntimeNative,
    pub selected_native_runtime_id: String,
    pub selected_source: String,
}

#[derive(uniffi::Record)]
pub struct NativeRuntimePruneResultNative {
    pub removed_dirs: Vec<String>,
}

#[uniffi::export(callback_interface)]
pub trait EventListener: Send + Sync {
    fn on_event(&self, event: ClientEvent);
}

#[uniffi::export(callback_interface)]
pub trait NativeRuntimeProgressListener: Send + Sync {
    fn on_progress(&self, event: NativeRuntimeDownloadProgressNative);
}
