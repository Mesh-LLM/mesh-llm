#[derive(uniffi::Enum)]
pub enum CapabilityLevel {
    None,
    Likely,
    Supported,
}

#[derive(uniffi::Record)]
pub struct ModelCapabilities {
    pub multimodal: bool,
    pub vision: CapabilityLevel,
    pub audio: CapabilityLevel,
    pub reasoning: CapabilityLevel,
    pub tool_use: CapabilityLevel,
    pub moe: bool,
}

#[derive(uniffi::Record)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub capabilities: ModelCapabilities,
}

#[derive(uniffi::Record)]
pub struct ModelSearchQuery {
    pub query: String,
    pub limit: Option<u64>,
}

#[derive(uniffi::Enum)]
pub enum ModelSource {
    Catalog,
    HuggingFace,
    Local,
}

#[derive(uniffi::Enum)]
pub enum ModelKind {
    Gguf,
    Safetensors,
    LayerPackage,
    Unknown,
}

#[derive(uniffi::Record)]
pub struct ModelDetails {
    pub id: String,
    pub name: String,
    pub source: ModelSource,
    pub kind: ModelKind,
    pub model_ref: String,
    pub download_ref: String,
    pub path: Option<String>,
    pub size_bytes: Option<u64>,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub installed: bool,
    pub capabilities: ModelCapabilities,
}

#[derive(uniffi::Record)]
pub struct InstalledModel {
    pub model_ref: String,
    pub path: String,
    pub size_bytes: Option<u64>,
    pub capabilities: ModelCapabilities,
}

#[derive(uniffi::Record)]
pub struct ModelCacheStatus {
    pub cache_dir: Option<String>,
}

#[derive(uniffi::Record)]
pub struct DownloadedModel {
    pub model_ref: String,
    pub paths: Vec<String>,
    pub primary_path: Option<String>,
    pub details: Option<ModelDetails>,
}

#[derive(uniffi::Record)]
pub struct DeleteModelOptions {
    pub force: bool,
}

#[derive(uniffi::Record)]
pub struct DeleteModelResult {
    pub deleted_paths: Vec<String>,
    pub reclaimed_bytes: u64,
}

#[derive(uniffi::Record)]
pub struct CleanupPolicy {
    pub remove_all: bool,
}

#[derive(uniffi::Record)]
pub struct CleanupResult {
    pub deleted_paths: Vec<String>,
    pub reclaimed_bytes: u64,
    pub skipped_paths: Vec<String>,
}

#[derive(uniffi::Record)]
pub struct PrunePolicy {
    pub remove_all: bool,
}

#[derive(uniffi::Record)]
pub struct PruneResult {
    pub deleted_paths: Vec<String>,
    pub reclaimed_bytes: u64,
}

#[derive(uniffi::Enum)]
pub enum DevicePolicy {
    Auto,
    Cpu,
    Gpu { device_ids: Vec<String> },
}

#[derive(uniffi::Record)]
pub struct LoadModelOptions {
    pub device_policy: DevicePolicy,
    pub profile: String,
}

#[derive(uniffi::Enum)]
pub enum ServingModelState {
    Loading,
    Ready,
    Failed,
    Unloading,
    Stopped,
    Unknown { value: String },
}

#[derive(uniffi::Record)]
pub struct ServedModel {
    pub model_ref: String,
    pub profile: String,
    pub model_id: String,
    pub instance_id: Option<String>,
    pub state: ServingModelState,
    pub backend: Option<String>,
    pub capabilities: ModelCapabilities,
    pub context_length: Option<u32>,
    pub error: Option<String>,
}

#[derive(uniffi::Record)]
pub struct ServingStatus {
    pub enabled: bool,
    pub models: Vec<ServedModel>,
}

#[derive(uniffi::Enum)]
pub enum UnloadTarget {
    Model { model_id: String },
    Instance { instance_id: String },
}

#[derive(uniffi::Record)]
pub struct UnloadModelOptions {
    pub drain_timeout_ms: u64,
    pub force: bool,
}
