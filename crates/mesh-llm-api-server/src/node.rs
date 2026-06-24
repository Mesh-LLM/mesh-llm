use crate::events::EventListener;
use crate::{
    ChatRequest, ClientBuilder, InviteToken, MeshApiError, MeshClient, Model, OwnerKeypair,
    RequestId, ResponsesRequest, Status,
};
pub use mesh_llm_node::models::{CapabilityLevel, ModelCapabilities, ModelKind, ModelSource};
use mesh_llm_node::serving::ServingController;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[derive(Clone, Debug, Default)]
pub enum DevicePolicy {
    #[default]
    Auto,
    Cpu,
    Gpu {
        device_ids: Vec<String>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct DownloadOptions;

#[derive(Clone, Debug, Default)]
pub struct DeleteModelOptions {
    pub force: bool,
}

#[derive(Clone, Debug, Default)]
pub struct LoadModelOptions {
    pub device_policy: DevicePolicy,
    pub profile: Option<String>,
}

#[derive(Clone, Debug)]
pub struct UnloadModelOptions {
    pub drain_timeout: Duration,
    pub force: bool,
}

impl Default for UnloadModelOptions {
    fn default() -> Self {
        Self {
            drain_timeout: Duration::from_secs(30),
            force: false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum UnloadTarget {
    Model(String),
    Instance(String),
}

#[derive(Clone, Debug, Default)]
pub struct CleanupPolicy {
    pub remove_all: bool,
}

#[derive(Clone, Debug, Default)]
pub struct PrunePolicy {
    pub remove_all: bool,
}

#[derive(Clone, Debug)]
pub struct ModelSearchQuery {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug)]
pub struct ModelDetails {
    pub id: String,
    pub name: String,
    pub source: ModelSource,
    pub kind: ModelKind,
    pub model_ref: String,
    pub download_ref: String,
    pub path: Option<PathBuf>,
    pub size_bytes: Option<u64>,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub installed: bool,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug)]
pub struct InstalledModel {
    pub model_ref: String,
    pub path: PathBuf,
    pub size_bytes: Option<u64>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug, Default)]
pub struct ModelCacheStatus {
    pub cache_dir: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct DownloadId(pub String);

#[derive(Clone, Debug)]
pub struct DownloadedModel {
    pub model_ref: String,
    pub paths: Vec<PathBuf>,
    pub primary_path: Option<PathBuf>,
    pub details: Option<ModelDetails>,
}

#[derive(Clone, Debug, Default)]
pub struct DeleteModelResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CleanupResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
    pub skipped_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Default)]
pub struct PruneResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct ServedModel {
    pub model_ref: String,
    pub profile: Option<String>,
    pub model_id: String,
    pub instance_id: Option<String>,
    pub state: ServingModelState,
    pub backend: Option<String>,
    pub capabilities: ModelCapabilities,
    pub context_length: Option<u32>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub enum ServingModelState {
    Loading,
    #[default]
    Ready,
    Failed,
    Unloading,
    Stopped,
    Unknown(String),
}

#[derive(Clone, Debug, Default)]
pub struct ServingStatus {
    pub enabled: bool,
    pub models: Vec<ServedModel>,
}

#[derive(Clone, Debug)]
pub struct MeshNodeConfig {
    pub owner_keypair: OwnerKeypair,
    pub invite_token: InviteToken,
    pub user_agent: String,
    pub connect_timeout: Duration,
    pub cache_dir: Option<PathBuf>,
    pub runtime_dir: Option<PathBuf>,
    pub serving_enabled: bool,
    pub device_policy: DevicePolicy,
}

pub struct MeshNodeBuilder {
    owner_keypair: Option<OwnerKeypair>,
    invite_token: Option<InviteToken>,
    user_agent: String,
    connect_timeout: Duration,
    cache_dir: Option<PathBuf>,
    runtime_dir: Option<PathBuf>,
    serving_enabled: bool,
    device_policy: DevicePolicy,
    serving_controller: Option<Arc<dyn ServingController>>,
}

impl MeshNodeBuilder {
    pub fn identity(mut self, identity: OwnerKeypair) -> Self {
        self.owner_keypair = Some(identity);
        self
    }

    pub fn join(mut self, token: InviteToken) -> Self {
        self.invite_token = Some(token);
        self
    }

    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    pub fn runtime_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.runtime_dir = Some(path.into());
        self
    }

    pub fn serving_enabled(mut self, enabled: bool) -> Self {
        self.serving_enabled = enabled;
        self
    }

    pub fn device_policy(mut self, policy: DevicePolicy) -> Self {
        self.device_policy = policy;
        self
    }

    pub fn serving_controller(mut self, controller: Arc<dyn ServingController>) -> Self {
        self.serving_enabled = true;
        self.serving_controller = Some(controller);
        self
    }

    pub fn build(self) -> Result<MeshNode, MeshApiError> {
        let owner_keypair = self.owner_keypair.ok_or(MeshApiError::InvalidConfig {
            message: "MeshNode identity is required",
        })?;
        let invite_token = self.invite_token.ok_or(MeshApiError::InvalidConfig {
            message: "MeshNode join token is required",
        })?;
        let client = ClientBuilder::new(owner_keypair.clone(), invite_token.clone())
            .with_user_agent(self.user_agent.clone())
            .with_connect_timeout(self.connect_timeout)
            .build()?;
        let config = MeshNodeConfig {
            owner_keypair,
            invite_token,
            user_agent: self.user_agent,
            connect_timeout: self.connect_timeout,
            cache_dir: self.cache_dir,
            runtime_dir: self.runtime_dir,
            serving_enabled: self.serving_enabled,
            device_policy: self.device_policy,
        };

        Ok(MeshNode {
            inner: Arc::new(MeshNodeInner {
                client: Mutex::new(client),
                config,
                serving_controller: self.serving_controller,
            }),
        })
    }
}

impl Default for MeshNodeBuilder {
    fn default() -> Self {
        Self {
            owner_keypair: None,
            invite_token: None,
            user_agent: format!("mesh-llm-api-server/{}", env!("CARGO_PKG_VERSION")),
            connect_timeout: Duration::from_secs(30),
            cache_dir: None,
            runtime_dir: None,
            serving_enabled: false,
            device_policy: DevicePolicy::Auto,
            serving_controller: None,
        }
    }
}

struct MeshNodeInner {
    client: Mutex<MeshClient>,
    config: MeshNodeConfig,
    serving_controller: Option<Arc<dyn ServingController>>,
}

#[derive(Clone)]
pub struct MeshNode {
    inner: Arc<MeshNodeInner>,
}

impl MeshNode {
    pub fn builder() -> MeshNodeBuilder {
        MeshNodeBuilder::default()
    }

    pub async fn start(&self) -> Result<(), MeshApiError> {
        self.inner.client.lock().await.join().await
    }

    pub async fn stop(&self) -> Result<(), MeshApiError> {
        self.inner.client.lock().await.disconnect().await;
        Ok(())
    }

    pub async fn reconnect(&self) -> Result<(), MeshApiError> {
        self.inner.client.lock().await.reconnect().await
    }

    pub fn inference(&self) -> MeshInference {
        MeshInference {
            inner: self.inner.clone(),
        }
    }

    pub fn models(&self) -> MeshModels {
        MeshModels {
            inner: self.inner.clone(),
        }
    }

    pub fn serving(&self) -> MeshServing {
        MeshServing {
            inner: self.inner.clone(),
        }
    }

    pub fn status(&self) -> MeshStatusApi {
        MeshStatusApi {
            inner: self.inner.clone(),
        }
    }

    pub fn events(&self) -> MeshEvents {
        MeshEvents {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MeshInference {
    inner: Arc<MeshNodeInner>,
}

impl MeshInference {
    pub async fn list_models(&self) -> Result<Vec<Model>, MeshApiError> {
        self.inner.client.lock().await.list_models().await
    }

    pub async fn chat(
        &self,
        request: ChatRequest,
        listener: Arc<dyn EventListener>,
    ) -> Result<RequestId, MeshApiError> {
        Ok(self.inner.client.lock().await.chat(request, listener))
    }

    pub async fn responses(
        &self,
        request: ResponsesRequest,
        listener: Arc<dyn EventListener>,
    ) -> Result<RequestId, MeshApiError> {
        Ok(self.inner.client.lock().await.responses(request, listener))
    }

    pub async fn cancel(&self, request_id: RequestId) -> Result<(), MeshApiError> {
        self.inner.client.lock().await.cancel(request_id);
        Ok(())
    }
}

#[derive(Clone)]
pub struct MeshModels {
    inner: Arc<MeshNodeInner>,
}

impl MeshModels {
    pub async fn recommended(&self) -> Result<Vec<ModelSummary>, MeshApiError> {
        Ok(mesh_llm_node::models::recommended_models()
            .into_iter()
            .map(ModelSummary::from)
            .collect())
    }

    pub async fn search(&self, query: ModelSearchQuery) -> Result<Vec<ModelSummary>, MeshApiError> {
        Ok(mesh_llm_node::models::search_models(
            mesh_llm_node::models::ModelSearchQuery {
                query: query.query,
                limit: query.limit.unwrap_or(20),
            },
            self.model_cache_dir(),
        )
        .into_iter()
        .map(ModelSummary::from)
        .collect())
    }

    pub async fn show(&self, model_ref: impl AsRef<str>) -> Result<ModelDetails, MeshApiError> {
        let cache_dir = self.model_cache_dir();
        mesh_llm_node::models::show_model(model_ref, cache_dir)
            .await
            .map(ModelDetails::from)
            .map_err(model_management_error)
    }

    pub async fn installed(&self) -> Result<Vec<InstalledModel>, MeshApiError> {
        let cache_dir = self.model_cache_dir();
        Ok(mesh_llm_node::models::scan_installed_models(cache_dir)
            .into_iter()
            .map(InstalledModel::from)
            .collect())
    }

    pub async fn cache_status(&self) -> Result<ModelCacheStatus, MeshApiError> {
        Ok(ModelCacheStatus {
            cache_dir: self.inner.config.cache_dir.clone(),
        })
    }

    pub async fn download(
        &self,
        model_ref: impl AsRef<str>,
        _options: DownloadOptions,
    ) -> Result<DownloadedModel, MeshApiError> {
        let cache_dir = self.model_cache_dir();
        mesh_llm_node::models::download_model(model_ref, cache_dir)
            .await
            .map(DownloadedModel::from)
            .map_err(model_management_error)
    }

    pub async fn cancel_download(&self, _download_id: DownloadId) -> Result<(), MeshApiError> {
        Err(MeshApiError::Unsupported {
            feature: "download cancellation",
        })
    }

    pub async fn delete(
        &self,
        model_ref: impl AsRef<str>,
        options: DeleteModelOptions,
    ) -> Result<DeleteModelResult, MeshApiError> {
        mesh_llm_node::models::delete_model(
            model_ref,
            self.model_cache_dir(),
            mesh_llm_node::models::DeleteModelOptions {
                force: options.force,
            },
        )
        .await
        .map(DeleteModelResult::from)
        .map_err(model_management_error)
    }

    pub async fn cleanup(&self, policy: CleanupPolicy) -> Result<CleanupResult, MeshApiError> {
        mesh_llm_node::models::cleanup_models(
            self.model_cache_dir(),
            mesh_llm_node::models::CleanupPolicy {
                remove_all: policy.remove_all,
            },
        )
        .map(CleanupResult::from)
        .map_err(model_management_error)
    }

    pub async fn prune_derived_cache(
        &self,
        policy: PrunePolicy,
    ) -> Result<PruneResult, MeshApiError> {
        let Some(runtime_dir) = self.inner.config.runtime_dir.clone() else {
            return Ok(PruneResult::default());
        };
        mesh_llm_node::models::prune_derived_cache(
            runtime_dir,
            mesh_llm_node::models::PrunePolicy {
                remove_all: policy.remove_all,
            },
        )
        .map(PruneResult::from)
        .map_err(model_management_error)
    }
}

impl MeshModels {
    fn model_cache_dir(&self) -> PathBuf {
        self.inner
            .config
            .cache_dir
            .clone()
            .unwrap_or_else(mesh_llm_node::models::default_huggingface_cache_dir)
    }
}

fn model_management_error(error: anyhow::Error) -> MeshApiError {
    MeshApiError::ModelManagement {
        message: error.to_string(),
    }
}

impl From<mesh_llm_node::models::ModelSummary> for ModelSummary {
    fn from(value: mesh_llm_node::models::ModelSummary) -> Self {
        Self {
            id: value.id,
            name: value.name,
            size_label: value.size_label,
            description: value.description,
            capabilities: value.capabilities,
        }
    }
}

impl From<mesh_llm_node::models::ModelDetails> for ModelDetails {
    fn from(value: mesh_llm_node::models::ModelDetails) -> Self {
        Self {
            id: value.id,
            name: value.name,
            source: value.source,
            kind: value.kind,
            model_ref: value.model_ref,
            download_ref: value.download_ref,
            path: value.path,
            size_bytes: value.size_bytes,
            size_label: value.size_label,
            description: value.description,
            draft: value.draft,
            installed: value.installed,
            capabilities: value.capabilities,
        }
    }
}

impl From<mesh_llm_node::models::InstalledModel> for InstalledModel {
    fn from(value: mesh_llm_node::models::InstalledModel) -> Self {
        Self {
            model_ref: value.model_ref,
            path: value.path,
            size_bytes: value.size_bytes,
            capabilities: value.capabilities,
        }
    }
}

impl From<mesh_llm_node::models::DownloadedModel> for DownloadedModel {
    fn from(value: mesh_llm_node::models::DownloadedModel) -> Self {
        Self {
            model_ref: value.model_ref,
            paths: value.paths,
            primary_path: value.primary_path,
            details: value.details.map(ModelDetails::from),
        }
    }
}

impl From<mesh_llm_node::models::DeleteModelResult> for DeleteModelResult {
    fn from(value: mesh_llm_node::models::DeleteModelResult) -> Self {
        Self {
            deleted_paths: value.deleted_paths,
            reclaimed_bytes: value.reclaimed_bytes,
        }
    }
}

impl From<mesh_llm_node::models::CleanupResult> for CleanupResult {
    fn from(value: mesh_llm_node::models::CleanupResult) -> Self {
        Self {
            deleted_paths: value.deleted_paths,
            reclaimed_bytes: value.reclaimed_bytes,
            skipped_paths: value.skipped_paths,
        }
    }
}

impl From<mesh_llm_node::models::PruneResult> for PruneResult {
    fn from(value: mesh_llm_node::models::PruneResult) -> Self {
        Self {
            deleted_paths: value.deleted_paths,
            reclaimed_bytes: value.reclaimed_bytes,
        }
    }
}

impl From<DevicePolicy> for mesh_llm_node::serving::DevicePolicy {
    fn from(value: DevicePolicy) -> Self {
        match value {
            DevicePolicy::Auto => Self::Auto,
            DevicePolicy::Cpu => Self::Cpu,
            DevicePolicy::Gpu { device_ids } => Self::Gpu { device_ids },
        }
    }
}

impl From<UnloadModelOptions> for mesh_llm_node::serving::UnloadOptions {
    fn from(value: UnloadModelOptions) -> Self {
        Self {
            drain_timeout: value.drain_timeout,
            force: value.force,
        }
    }
}

impl From<UnloadTarget> for mesh_llm_node::serving::UnloadTarget {
    fn from(value: UnloadTarget) -> Self {
        match value {
            UnloadTarget::Model(model_id) => Self::Model(model_id),
            UnloadTarget::Instance(instance_id) => Self::Instance(instance_id),
        }
    }
}

impl From<mesh_llm_node::serving::ServingModelState> for ServingModelState {
    fn from(value: mesh_llm_node::serving::ServingModelState) -> Self {
        match value {
            mesh_llm_node::serving::ServingModelState::Loading => Self::Loading,
            mesh_llm_node::serving::ServingModelState::Ready => Self::Ready,
            mesh_llm_node::serving::ServingModelState::Failed => Self::Failed,
            mesh_llm_node::serving::ServingModelState::Unloading => Self::Unloading,
            mesh_llm_node::serving::ServingModelState::Stopped => Self::Stopped,
            mesh_llm_node::serving::ServingModelState::Unknown(value) => Self::Unknown(value),
        }
    }
}

impl From<mesh_llm_node::serving::ServedModel> for ServedModel {
    fn from(value: mesh_llm_node::serving::ServedModel) -> Self {
        Self {
            model_ref: value.model_ref,
            profile: value.profile,
            model_id: value.model_id,
            instance_id: value.instance_id,
            state: value.state.into(),
            backend: value.backend,
            capabilities: value.capabilities,
            context_length: value.context_length,
            error: value.error,
        }
    }
}

impl From<mesh_llm_node::serving::ServingStatus> for ServingStatus {
    fn from(value: mesh_llm_node::serving::ServingStatus) -> Self {
        Self {
            enabled: value.enabled,
            models: value.models.into_iter().map(ServedModel::from).collect(),
        }
    }
}

fn serving_error(error: anyhow::Error) -> MeshApiError {
    if let Some(error) = error.downcast_ref::<mesh_llm_node::serving::ServingError>() {
        return MeshApiError::Serving {
            message: error.to_string(),
        };
    }
    MeshApiError::Serving {
        message: error.to_string(),
    }
}

#[derive(Clone)]
pub struct MeshServing {
    inner: Arc<MeshNodeInner>,
}

impl MeshServing {
    pub async fn load(
        &self,
        model_ref: impl AsRef<str>,
        options: LoadModelOptions,
    ) -> Result<ServedModel, MeshApiError> {
        let controller = self.serving_controller()?;
        controller
            .load(mesh_llm_node::serving::LoadModelRequest {
                model_ref: model_ref.as_ref().to_string(),
                device_policy: options.device_policy.into(),
                profile: options.profile.clone(),
            })
            .await
            .map(ServedModel::from)
            .map_err(serving_error)
    }

    pub async fn unload(
        &self,
        target: UnloadTarget,
        options: UnloadModelOptions,
    ) -> Result<(), MeshApiError> {
        let controller = self.serving_controller()?;
        controller
            .unload(mesh_llm_node::serving::UnloadModelRequest {
                target: target.into(),
                options: options.into(),
            })
            .await
            .map_err(serving_error)
    }

    pub async fn unload_model(
        &self,
        model_id: impl AsRef<str>,
        options: UnloadModelOptions,
    ) -> Result<(), MeshApiError> {
        self.unload(UnloadTarget::Model(model_id.as_ref().to_string()), options)
            .await
    }

    pub async fn unload_instance(
        &self,
        instance_id: impl AsRef<str>,
        options: UnloadModelOptions,
    ) -> Result<(), MeshApiError> {
        self.unload(
            UnloadTarget::Instance(instance_id.as_ref().to_string()),
            options,
        )
        .await
    }

    pub async fn served_models(&self) -> Result<Vec<ServedModel>, MeshApiError> {
        let Some(controller) = self.inner.serving_controller.clone() else {
            return Ok(Vec::new());
        };
        controller
            .served_models()
            .await
            .map(|models| models.into_iter().map(ServedModel::from).collect())
            .map_err(serving_error)
    }

    pub async fn status(&self) -> Result<ServingStatus, MeshApiError> {
        let Some(controller) = self.inner.serving_controller.clone() else {
            return Ok(ServingStatus {
                enabled: self.inner.config.serving_enabled,
                models: Vec::new(),
            });
        };
        controller
            .status()
            .await
            .map(ServingStatus::from)
            .map_err(serving_error)
    }

    pub async fn set_device_policy(&self, policy: DevicePolicy) -> Result<(), MeshApiError> {
        let controller = self.serving_controller()?;
        controller
            .set_device_policy(policy.into())
            .await
            .map_err(serving_error)
    }

    fn serving_controller(&self) -> Result<Arc<dyn ServingController>, MeshApiError> {
        self.inner
            .serving_controller
            .clone()
            .ok_or(MeshApiError::Unsupported {
                feature: "in-process serving controller",
            })
    }
}

#[derive(Clone)]
pub struct MeshStatusApi {
    inner: Arc<MeshNodeInner>,
}

impl MeshStatusApi {
    pub async fn node(&self) -> Result<Status, MeshApiError> {
        Ok(self.inner.client.lock().await.status().await)
    }

    pub async fn models(&self) -> Result<Vec<Model>, MeshApiError> {
        self.inner.client.lock().await.list_models().await
    }

    pub async fn serving(&self) -> Result<ServingStatus, MeshApiError> {
        let Some(controller) = self.inner.serving_controller.clone() else {
            return Ok(ServingStatus {
                enabled: self.inner.config.serving_enabled,
                models: Vec::new(),
            });
        };
        controller
            .status()
            .await
            .map(ServingStatus::from)
            .map_err(serving_error)
    }
}

#[derive(Clone)]
pub struct MeshEvents {
    inner: Arc<MeshNodeInner>,
}

impl MeshEvents {
    pub fn is_supported(&self) -> bool {
        let _ = &self.inner;
        false
    }
}
