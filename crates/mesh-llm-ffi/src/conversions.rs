use std::time::Duration;

use mesh_llm_sdk::node as sdk_node;
use mesh_llm_sdk::node::{
    DevicePolicy as ApiDevicePolicy, ModelKind as ApiModelKind, ModelSource as ApiModelSource,
    ServingModelState as ApiServingModelState, UnloadModelOptions as ApiUnloadModelOptions,
    UnloadTarget as ApiUnloadTarget,
};
use mesh_llm_sdk::{
    ChatMessage, ChatRequest, PublicMeshQuery as ApiPublicMeshQuery, ResponsesRequest,
};

use crate::model_types::{
    CapabilityLevel, CleanupResult, DeleteModelResult, DevicePolicy, DownloadedModel,
    InstalledModel, ModelCacheStatus, ModelCapabilities, ModelDetails, ModelKind, ModelSource,
    ModelSummary, PruneResult, ServedModel, ServingModelState, ServingStatus, UnloadModelOptions,
    UnloadTarget,
};
use crate::native_runtime_types::{
    InstalledNativeRuntimeNative, NativeRuntimeDownloadProgressNative,
    NativeRuntimeInstallOutcomeNative, NativeRuntimePruneModeNative,
    NativeRuntimePruneResultNative, NativeRuntimeVerificationPolicyNative,
};
use crate::request_types::{
    ChatMessageNative, ChatRequestNative, PublicMesh, PublicMeshQuery, ResponsesRequestNative,
};

fn path_to_string(path: std::path::PathBuf) -> String {
    path.display().to_string()
}

impl From<ChatRequestNative> for ChatRequest {
    fn from(value: ChatRequestNative) -> Self {
        Self {
            model: value.model,
            messages: value.messages.into_iter().map(ChatMessage::from).collect(),
        }
    }
}

impl From<ChatMessageNative> for ChatMessage {
    fn from(value: ChatMessageNative) -> Self {
        Self {
            role: value.role,
            content: value.content,
        }
    }
}

impl From<ResponsesRequestNative> for ResponsesRequest {
    fn from(value: ResponsesRequestNative) -> Self {
        Self {
            model: value.model,
            input: value.input,
        }
    }
}

impl From<PublicMeshQuery> for ApiPublicMeshQuery {
    fn from(value: PublicMeshQuery) -> Self {
        Self {
            model: value.model,
            min_vram_gb: value.min_vram_gb,
            region: value.region,
            target_name: value.target_name,
            relays: value.relays,
        }
    }
}

impl From<sdk_node::PublicMesh> for PublicMesh {
    fn from(value: sdk_node::PublicMesh) -> Self {
        Self {
            invite_token: value.invite_token,
            serving: value.serving,
            wanted: value.wanted,
            on_disk: value.on_disk,
            total_vram_bytes: value.total_vram_bytes,
            node_count: value.node_count as u64,
            client_count: value.client_count as u64,
            max_clients: value.max_clients as u64,
            name: value.name,
            region: value.region,
            mesh_id: value.mesh_id,
            publisher_npub: value.publisher_npub,
            published_at: value.published_at,
            expires_at: value.expires_at,
        }
    }
}

impl From<sdk_node::CapabilityLevel> for CapabilityLevel {
    fn from(value: sdk_node::CapabilityLevel) -> Self {
        match value {
            sdk_node::CapabilityLevel::None => Self::None,
            sdk_node::CapabilityLevel::Likely => Self::Likely,
            sdk_node::CapabilityLevel::Supported => Self::Supported,
        }
    }
}

impl From<sdk_node::ModelCapabilities> for ModelCapabilities {
    fn from(value: sdk_node::ModelCapabilities) -> Self {
        Self {
            multimodal: value.multimodal,
            vision: value.vision.into(),
            audio: value.audio.into(),
            reasoning: value.reasoning.into(),
            tool_use: value.tool_use.into(),
            moe: value.moe,
        }
    }
}

impl From<sdk_node::ModelSummary> for ModelSummary {
    fn from(value: sdk_node::ModelSummary) -> Self {
        Self {
            id: value.id,
            name: value.name,
            size_label: value.size_label,
            description: value.description,
            capabilities: value.capabilities.into(),
        }
    }
}

impl From<ApiModelSource> for ModelSource {
    fn from(value: ApiModelSource) -> Self {
        match value {
            ApiModelSource::Catalog => Self::Catalog,
            ApiModelSource::HuggingFace => Self::HuggingFace,
            ApiModelSource::Local => Self::Local,
        }
    }
}

impl From<ApiModelKind> for ModelKind {
    fn from(value: ApiModelKind) -> Self {
        match value {
            ApiModelKind::Gguf => Self::Gguf,
            ApiModelKind::Safetensors => Self::Safetensors,
            ApiModelKind::LayerPackage => Self::LayerPackage,
            ApiModelKind::Unknown => Self::Unknown,
        }
    }
}

impl From<sdk_node::ModelDetails> for ModelDetails {
    fn from(value: sdk_node::ModelDetails) -> Self {
        Self {
            id: value.id,
            name: value.name,
            source: value.source.into(),
            kind: value.kind.into(),
            model_ref: value.model_ref,
            download_ref: value.download_ref,
            path: value.path.map(path_to_string),
            size_bytes: value.size_bytes,
            size_label: value.size_label,
            description: value.description,
            draft: value.draft,
            installed: value.installed,
            capabilities: value.capabilities.into(),
        }
    }
}

impl From<sdk_node::InstalledModel> for InstalledModel {
    fn from(value: sdk_node::InstalledModel) -> Self {
        Self {
            model_ref: value.model_ref,
            path: path_to_string(value.path),
            size_bytes: value.size_bytes,
            capabilities: value.capabilities.into(),
        }
    }
}

impl From<sdk_node::ModelCacheStatus> for ModelCacheStatus {
    fn from(value: sdk_node::ModelCacheStatus) -> Self {
        Self {
            cache_dir: value.cache_dir.map(path_to_string),
        }
    }
}

impl From<sdk_node::DownloadedModel> for DownloadedModel {
    fn from(value: sdk_node::DownloadedModel) -> Self {
        Self {
            model_ref: value.model_ref,
            paths: value.paths.into_iter().map(path_to_string).collect(),
            primary_path: value.primary_path.map(path_to_string),
            details: value.details.map(ModelDetails::from),
        }
    }
}

impl From<sdk_node::DeleteModelResult> for DeleteModelResult {
    fn from(value: sdk_node::DeleteModelResult) -> Self {
        Self {
            deleted_paths: value
                .deleted_paths
                .into_iter()
                .map(path_to_string)
                .collect(),
            reclaimed_bytes: value.reclaimed_bytes,
        }
    }
}

impl From<sdk_node::CleanupResult> for CleanupResult {
    fn from(value: sdk_node::CleanupResult) -> Self {
        Self {
            deleted_paths: value
                .deleted_paths
                .into_iter()
                .map(path_to_string)
                .collect(),
            reclaimed_bytes: value.reclaimed_bytes,
            skipped_paths: value
                .skipped_paths
                .into_iter()
                .map(path_to_string)
                .collect(),
        }
    }
}

impl From<sdk_node::PruneResult> for PruneResult {
    fn from(value: sdk_node::PruneResult) -> Self {
        Self {
            deleted_paths: value
                .deleted_paths
                .into_iter()
                .map(path_to_string)
                .collect(),
            reclaimed_bytes: value.reclaimed_bytes,
        }
    }
}

impl From<DevicePolicy> for ApiDevicePolicy {
    fn from(value: DevicePolicy) -> Self {
        match value {
            DevicePolicy::Auto => Self::Auto,
            DevicePolicy::Cpu => Self::Cpu,
            DevicePolicy::Gpu { device_ids } => Self::Gpu { device_ids },
        }
    }
}

impl From<ApiServingModelState> for ServingModelState {
    fn from(value: ApiServingModelState) -> Self {
        match value {
            ApiServingModelState::Loading => Self::Loading,
            ApiServingModelState::Ready => Self::Ready,
            ApiServingModelState::Failed => Self::Failed,
            ApiServingModelState::Unloading => Self::Unloading,
            ApiServingModelState::Stopped => Self::Stopped,
            ApiServingModelState::Unknown(value) => Self::Unknown { value },
        }
    }
}

impl From<sdk_node::ServedModel> for ServedModel {
    fn from(value: sdk_node::ServedModel) -> Self {
        Self {
            model_ref: value.model_ref,
            profile: value.profile,
            model_id: value.model_id,
            instance_id: value.instance_id,
            state: value.state.into(),
            backend: value.backend,
            capabilities: value.capabilities.into(),
            context_length: value.context_length,
            error: value.error,
        }
    }
}

impl From<sdk_node::ServingStatus> for ServingStatus {
    fn from(value: sdk_node::ServingStatus) -> Self {
        Self {
            enabled: value.enabled,
            models: value.models.into_iter().map(ServedModel::from).collect(),
        }
    }
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
            mesh_version: value.mesh_version,
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

impl From<UnloadTarget> for ApiUnloadTarget {
    fn from(value: UnloadTarget) -> Self {
        match value {
            UnloadTarget::Model { model_id } => Self::Model(model_id),
            UnloadTarget::Instance { instance_id } => Self::Instance(instance_id),
        }
    }
}

impl From<UnloadModelOptions> for ApiUnloadModelOptions {
    fn from(value: UnloadModelOptions) -> Self {
        Self {
            drain_timeout: Duration::from_millis(value.drain_timeout_ms),
            force: value.force,
        }
    }
}
