use mesh_llm_api_server::OwnerKeypair;
use mesh_llm_api_server::events::{Event, EventListener as CoreEventListener};
use mesh_llm_api_server::{
    ChatMessage, ChatRequest, ClientBuilder, DevicePolicy as ApiDevicePolicy, InviteToken,
    MeshApiError, MeshClient, MeshNode, ModelKind as ApiModelKind, ModelSource as ApiModelSource,
    PublicMeshQuery as ApiPublicMeshQuery, RequestId, ResponsesRequest,
    ServingModelState as ApiServingModelState, UnloadModelOptions as ApiUnloadModelOptions,
    UnloadTarget as ApiUnloadTarget, create_auto_client as sdk_create_auto_client,
    create_auto_node as sdk_create_auto_node, discover_public_meshes as sdk_discover_public_meshes,
};
#[cfg(feature = "embedded-runtime")]
use mesh_llm_host_runtime::sdk::{EmbeddedChatMessage, EmbeddedServingController};
use std::future::Future;
use std::sync::LazyLock;
use std::sync::{Arc, Mutex};
use std::time::Duration;

uniffi::setup_scaffolding!("mesh_ffi");

static SDK_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("mesh-llm-sdk")
        .build()
        .expect("create mesh-llm SDK runtime")
});

fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
        Err(_) => SDK_RUNTIME.block_on(future),
    }
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("invalid invite token: {0}")]
    InvalidInviteToken(String),
    #[error("invalid owner keypair: {0}")]
    InvalidOwnerKeypair(String),
    #[error("client build failed: {0}")]
    BuildFailed(String),
    #[error("join failed: {0}")]
    JoinFailed(String),
    #[error("discovery failed: {0}")]
    DiscoveryFailed(String),
    #[error("stream failed: {0}")]
    StreamFailed(String),
    #[error("cancelled: {0}")]
    Cancelled(String),
    #[error("reconnect failed: {0}")]
    ReconnectFailed(String),
    #[error("host unavailable: {0}")]
    HostUnavailable(String),
    #[error("model management failed: {0}")]
    ModelManagementFailed(String),
    #[error("serving failed: {0}")]
    ServingFailed(String),
    #[error("serving is unsupported by this node: {0}")]
    ServingUnsupported(String),
    #[error("console failed: {0}")]
    ConsoleFailed(String),
}

#[derive(uniffi::Record)]
pub struct ModelNative {
    pub id: String,
    pub name: String,
}

#[derive(uniffi::Record)]
pub struct ClientStatus {
    pub connected: bool,
    pub peer_count: u64,
}

#[derive(uniffi::Record)]
pub struct ConsoleOptionsNative {
    pub asset_dir: String,
    pub port: Option<u16>,
    pub listen_all: bool,
}

#[derive(uniffi::Record)]
pub struct PublicMeshQuery {
    pub model: Option<String>,
    pub min_vram_gb: Option<f64>,
    pub region: Option<String>,
    pub target_name: Option<String>,
    pub relays: Vec<String>,
}

#[derive(uniffi::Record)]
pub struct PublicMesh {
    pub invite_token: String,
    pub serving: Vec<String>,
    pub wanted: Vec<String>,
    pub on_disk: Vec<String>,
    pub total_vram_bytes: u64,
    pub node_count: u64,
    pub client_count: u64,
    pub max_clients: u64,
    pub name: Option<String>,
    pub region: Option<String>,
    pub mesh_id: Option<String>,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

#[derive(uniffi::Record)]
pub struct ChatRequestNative {
    pub model: String,
    pub messages: Vec<ChatMessageNative>,
}

#[derive(uniffi::Record)]
pub struct ChatMessageNative {
    pub role: String,
    pub content: String,
}

#[derive(uniffi::Record)]
pub struct ResponsesRequestNative {
    pub model: String,
    pub input: String,
}

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

#[derive(uniffi::Enum)]
pub enum ClientEvent {
    Connecting,
    Joined { node_id: String },
    ModelsUpdated { models: Vec<ModelNative> },
    TokenDelta { request_id: String, delta: String },
    Completed { request_id: String },
    Failed { request_id: String, error: String },
    Disconnected { reason: String },
}

#[uniffi::export(callback_interface)]
pub trait EventListener: Send + Sync {
    fn on_event(&self, event: ClientEvent);
}

struct EventListenerBridge {
    inner: Box<dyn EventListener>,
}

impl CoreEventListener for EventListenerBridge {
    fn on_event(&self, event: Event) {
        let native = match event {
            Event::Connecting => ClientEvent::Connecting,
            Event::Joined { node_id } => ClientEvent::Joined { node_id },
            Event::ModelsUpdated { models } => ClientEvent::ModelsUpdated {
                models: models
                    .into_iter()
                    .map(|m| ModelNative {
                        id: m.id,
                        name: m.name,
                    })
                    .collect(),
            },
            Event::TokenDelta { request_id, delta } => {
                ClientEvent::TokenDelta { request_id, delta }
            }
            Event::Completed { request_id } => ClientEvent::Completed { request_id },
            Event::Failed { request_id, error } => ClientEvent::Failed { request_id, error },
            Event::Disconnected { reason } => ClientEvent::Disconnected { reason },
        };
        self.inner.on_event(native);
    }
}

#[derive(uniffi::Object)]
pub struct MeshClientHandle {
    client: tokio::sync::Mutex<MeshClient>,
}

#[derive(uniffi::Object)]
pub struct MeshNodeHandle {
    node: MeshNode,
    #[cfg(feature = "embedded-runtime")]
    local_serving: Option<Arc<EmbeddedServingController>>,
}

#[derive(uniffi::Object)]
pub struct ConsoleHandle {
    inner: Mutex<Option<mesh_llm_console_server::ConsoleServerHandle>>,
    url: String,
}

/// Generate a fresh owner keypair, returning its hex-encoded form.
///
/// Callers should persist this value on first run and pass it back to
/// `create_node` on subsequent launches so the embedded node keeps a stable
/// identity. Generating a new keypair on every launch will make the app look
/// like a different owner to the mesh each time.
#[uniffi::export]
pub fn generate_owner_keypair_hex() -> String {
    OwnerKeypair::generate().to_hex()
}

#[uniffi::export]
pub fn discover_public_meshes(query: PublicMeshQuery) -> Result<Vec<PublicMesh>, FfiError> {
    block_on(sdk_discover_public_meshes(query.into()))
        .map(|meshes| meshes.into_iter().map(PublicMesh::from).collect())
        .map_err(map_mesh_api_error)
}

#[uniffi::export]
pub fn create_auto_client(
    owner_keypair_bytes_hex: String,
    query: PublicMeshQuery,
) -> Result<Arc<MeshClientHandle>, FfiError> {
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    block_on(sdk_create_auto_client(kp, query.into()))
        .map(|result| {
            Arc::new(MeshClientHandle {
                client: tokio::sync::Mutex::new(result.client),
            })
        })
        .map_err(map_mesh_api_error)
}

#[uniffi::export]
pub fn create_auto_node(
    owner_keypair_bytes_hex: String,
    query: PublicMeshQuery,
) -> Result<Arc<MeshNodeHandle>, FfiError> {
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    block_on(sdk_create_auto_node(kp, query.into()))
        .map(|result| {
            Arc::new(MeshNodeHandle {
                node: result.node,
                #[cfg(feature = "embedded-runtime")]
                local_serving: None,
            })
        })
        .map_err(map_mesh_api_error)
}

#[uniffi::export]
pub fn create_client(
    owner_keypair_bytes_hex: String,
    invite_token: String,
) -> Result<Arc<MeshClientHandle>, FfiError> {
    let token = invite_token
        .parse::<InviteToken>()
        .map_err(FfiError::InvalidInviteToken)?;
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    let client = ClientBuilder::new(kp, token)
        .build()
        .map_err(|error| FfiError::BuildFailed(error.to_string()))?;
    Ok(Arc::new(MeshClientHandle {
        client: tokio::sync::Mutex::new(client),
    }))
}

#[uniffi::export]
pub fn create_node(
    owner_keypair_bytes_hex: String,
    invite_token: String,
    cache_dir: Option<String>,
    runtime_dir: Option<String>,
    serving_enabled: bool,
) -> Result<Arc<MeshNodeHandle>, FfiError> {
    let token = invite_token
        .parse::<InviteToken>()
        .map_err(FfiError::InvalidInviteToken)?;
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    #[cfg(not(feature = "embedded-runtime"))]
    if serving_enabled {
        return Err(FfiError::ServingUnsupported(
            "this native library was built without embedded-runtime support".to_string(),
        ));
    }
    let mut builder = MeshNode::builder().identity(kp).join(token);
    #[cfg(feature = "embedded-runtime")]
    let local_serving = if serving_enabled {
        let controller = Arc::new(EmbeddedServingController::new());
        builder = builder.serving_controller(controller.clone());
        Some(controller)
    } else {
        builder = builder.serving_enabled(false);
        None
    };
    #[cfg(not(feature = "embedded-runtime"))]
    {
        builder = builder.serving_enabled(serving_enabled);
    }
    if let Some(path) = non_empty_path(cache_dir) {
        builder = builder.cache_dir(path);
    }
    if let Some(path) = non_empty_path(runtime_dir) {
        builder = builder.runtime_dir(path);
    }
    let node = builder
        .build()
        .map_err(|error| FfiError::BuildFailed(error.to_string()))?;
    Ok(Arc::new(MeshNodeHandle {
        node,
        #[cfg(feature = "embedded-runtime")]
        local_serving,
    }))
}

#[uniffi::export]
impl MeshClientHandle {
    pub fn start(&self) -> Result<(), FfiError> {
        block_on(async {
            let mut client = self.client.lock().await;
            client.join().await
        })
        .map_err(|error| FfiError::JoinFailed(error.to_string()))
    }

    pub fn stop(&self) {
        block_on(async {
            self.client.lock().await.disconnect().await;
        });
    }

    pub fn reconnect(&self) -> Result<(), FfiError> {
        block_on(async {
            let mut client = self.client.lock().await;
            client.reconnect().await
        })
        .map_err(|error| FfiError::ReconnectFailed(error.to_string()))
    }

    pub fn status(&self) -> ClientStatus {
        let status = block_on(async { self.client.lock().await.status().await });
        ClientStatus {
            connected: status.connected,
            peer_count: status.peer_count as u64,
        }
    }

    pub fn inference_list_models(&self) -> Result<Vec<ModelNative>, FfiError> {
        block_on(async { self.client.lock().await.list_models().await })
            .map(|models| {
                models
                    .into_iter()
                    .map(|m| ModelNative {
                        id: m.id,
                        name: m.name,
                    })
                    .collect()
            })
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))
    }

    pub fn chat(
        &self,
        request: ChatRequestNative,
        listener: Box<dyn EventListener>,
    ) -> Result<String, FfiError> {
        let bridge = Arc::new(EventListenerBridge { inner: listener });
        let request_id = block_on(async { self.client.lock().await.chat(request.into(), bridge) });
        Ok(request_id.0)
    }

    pub fn responses(
        &self,
        request: ResponsesRequestNative,
        listener: Box<dyn EventListener>,
    ) -> Result<String, FfiError> {
        let bridge = Arc::new(EventListenerBridge { inner: listener });
        let request_id =
            block_on(async { self.client.lock().await.responses(request.into(), bridge) });
        Ok(request_id.0)
    }

    pub fn cancel(&self, request_id: String) {
        block_on(async {
            self.client.lock().await.cancel(RequestId(request_id));
        });
    }
}

#[uniffi::export]
impl ConsoleHandle {
    pub fn url(&self) -> String {
        self.url.clone()
    }

    pub fn stop(&self) -> Result<(), FfiError> {
        let handle = self
            .inner
            .lock()
            .map_err(|error| FfiError::ConsoleFailed(error.to_string()))?
            .take();
        if let Some(handle) = handle {
            block_on(handle.stop());
        }
        Ok(())
    }
}

#[uniffi::export]
impl MeshNodeHandle {
    pub fn start(&self) -> Result<(), FfiError> {
        block_on(self.node.start()).map_err(|error| FfiError::JoinFailed(error.to_string()))
    }

    pub fn stop(&self) -> Result<(), FfiError> {
        block_on(self.node.stop()).map_err(|error| FfiError::HostUnavailable(error.to_string()))
    }

    pub fn reconnect(&self) -> Result<(), FfiError> {
        block_on(self.node.reconnect())
            .map_err(|error| FfiError::ReconnectFailed(error.to_string()))
    }

    pub fn status(&self) -> ClientStatus {
        let status = block_on(self.node.status().node()).unwrap_or(mesh_llm_api_server::Status {
            connected: false,
            peer_count: 0,
        });
        ClientStatus {
            connected: status.connected,
            peer_count: status.peer_count as u64,
        }
    }

    pub fn inference_list_models(&self) -> Result<Vec<ModelNative>, FfiError> {
        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = &self.local_serving {
            let models = block_on(controller.model_list());
            if !models.is_empty() {
                return Ok(models
                    .into_iter()
                    .map(|(id, name)| ModelNative { id, name })
                    .collect());
            }
        }
        block_on(self.node.inference().list_models())
            .map(|models| {
                models
                    .into_iter()
                    .map(|m| ModelNative {
                        id: m.id,
                        name: m.name,
                    })
                    .collect()
            })
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))
    }

    pub fn chat(
        &self,
        request: ChatRequestNative,
        listener: Box<dyn EventListener>,
    ) -> Result<String, FfiError> {
        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = self.local_controller_for_model(&request.model) {
            let request_id = new_request_id();
            let model = request.model.clone();
            let messages = request
                .messages
                .into_iter()
                .map(|message| EmbeddedChatMessage {
                    role: message.role,
                    content: message.content,
                })
                .collect();
            let content = block_on(controller.chat_completion_text(&model, messages))
                .map_err(|error| FfiError::StreamFailed(error.to_string()))?;
            listener.on_event(ClientEvent::TokenDelta {
                request_id: request_id.clone(),
                delta: content,
            });
            listener.on_event(ClientEvent::Completed {
                request_id: request_id.clone(),
            });
            return Ok(request_id);
        }
        let bridge = Arc::new(EventListenerBridge { inner: listener });
        block_on(self.node.inference().chat(request.into(), bridge))
            .map(|request_id| request_id.0)
            .map_err(map_stream_error)
    }

    pub fn responses(
        &self,
        request: ResponsesRequestNative,
        listener: Box<dyn EventListener>,
    ) -> Result<String, FfiError> {
        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = self.local_controller_for_model(&request.model) {
            let request_id = new_request_id();
            let content = block_on(controller.chat_completion_text(
                &request.model,
                vec![EmbeddedChatMessage {
                    role: "user".to_string(),
                    content: request.input,
                }],
            ))
            .map_err(|error| FfiError::StreamFailed(error.to_string()))?;
            listener.on_event(ClientEvent::TokenDelta {
                request_id: request_id.clone(),
                delta: content,
            });
            listener.on_event(ClientEvent::Completed {
                request_id: request_id.clone(),
            });
            return Ok(request_id);
        }
        let bridge = Arc::new(EventListenerBridge { inner: listener });
        block_on(self.node.inference().responses(request.into(), bridge))
            .map(|request_id| request_id.0)
            .map_err(map_stream_error)
    }

    pub fn cancel(&self, request_id: String) -> Result<(), FfiError> {
        block_on(self.node.inference().cancel(RequestId(request_id))).map_err(map_stream_error)
    }

    pub fn recommended_models(&self) -> Result<Vec<ModelSummary>, FfiError> {
        block_on(self.node.models().recommended())
            .map(|models| models.into_iter().map(ModelSummary::from).collect())
            .map_err(map_model_error)
    }

    pub fn search_models(&self, query: ModelSearchQuery) -> Result<Vec<ModelSummary>, FfiError> {
        block_on(
            self.node
                .models()
                .search(mesh_llm_api_server::ModelSearchQuery {
                    query: query.query,
                    limit: query.limit.map(|limit| limit as usize),
                }),
        )
        .map(|models| models.into_iter().map(ModelSummary::from).collect())
        .map_err(map_model_error)
    }

    pub fn show_model(&self, model_ref: String) -> Result<ModelDetails, FfiError> {
        block_on(self.node.models().show(model_ref))
            .map(ModelDetails::from)
            .map_err(map_model_error)
    }

    pub fn installed_models(&self) -> Result<Vec<InstalledModel>, FfiError> {
        block_on(self.node.models().installed())
            .map(|models| models.into_iter().map(InstalledModel::from).collect())
            .map_err(map_model_error)
    }

    pub fn model_cache_status(&self) -> Result<ModelCacheStatus, FfiError> {
        block_on(self.node.models().cache_status())
            .map(ModelCacheStatus::from)
            .map_err(map_model_error)
    }

    pub fn download_model(&self, model_ref: String) -> Result<DownloadedModel, FfiError> {
        block_on(
            self.node
                .models()
                .download(model_ref, mesh_llm_api_server::DownloadOptions),
        )
        .map(DownloadedModel::from)
        .map_err(map_model_error)
    }

    pub fn delete_model(
        &self,
        model_ref: String,
        options: DeleteModelOptions,
    ) -> Result<DeleteModelResult, FfiError> {
        block_on(self.node.models().delete(
            model_ref,
            mesh_llm_api_server::DeleteModelOptions {
                force: options.force,
            },
        ))
        .map(DeleteModelResult::from)
        .map_err(map_model_error)
    }

    pub fn cleanup_models(&self, policy: CleanupPolicy) -> Result<CleanupResult, FfiError> {
        block_on(
            self.node
                .models()
                .cleanup(mesh_llm_api_server::CleanupPolicy {
                    remove_all: policy.remove_all,
                }),
        )
        .map(CleanupResult::from)
        .map_err(map_model_error)
    }

    pub fn prune_derived_cache(&self, policy: PrunePolicy) -> Result<PruneResult, FfiError> {
        block_on(
            self.node
                .models()
                .prune_derived_cache(mesh_llm_api_server::PrunePolicy {
                    remove_all: policy.remove_all,
                }),
        )
        .map(PruneResult::from)
        .map_err(map_model_error)
    }

    pub fn load_serving_model(
        &self,
        model_ref: String,
        options: LoadModelOptions,
    ) -> Result<ServedModel, FfiError> {
        block_on(self.node.serving().load(
            model_ref,
            mesh_llm_api_server::LoadModelOptions {
                device_policy: options.device_policy.into(),
            },
        ))
        .map(ServedModel::from)
        .map_err(map_serving_error)
    }

    pub fn unload_serving_model(
        &self,
        target: UnloadTarget,
        options: UnloadModelOptions,
    ) -> Result<(), FfiError> {
        block_on(self.node.serving().unload(target.into(), options.into()))
            .map_err(map_serving_error)
    }

    pub fn unload_serving_model_by_id(
        &self,
        model_id: String,
        options: UnloadModelOptions,
    ) -> Result<(), FfiError> {
        block_on(self.node.serving().unload_model(model_id, options.into()))
            .map_err(map_serving_error)
    }

    pub fn unload_serving_instance(
        &self,
        instance_id: String,
        options: UnloadModelOptions,
    ) -> Result<(), FfiError> {
        block_on(
            self.node
                .serving()
                .unload_instance(instance_id, options.into()),
        )
        .map_err(map_serving_error)
    }

    pub fn served_models(&self) -> Result<Vec<ServedModel>, FfiError> {
        block_on(self.node.serving().served_models())
            .map(|models| models.into_iter().map(ServedModel::from).collect())
            .map_err(map_serving_error)
    }

    pub fn serving_status(&self) -> Result<ServingStatus, FfiError> {
        block_on(self.node.serving().status())
            .map(ServingStatus::from)
            .map_err(map_serving_error)
    }

    pub fn set_device_policy(&self, policy: DevicePolicy) -> Result<(), FfiError> {
        block_on(self.node.serving().set_device_policy(policy.into())).map_err(map_serving_error)
    }

    pub fn start_console(
        &self,
        options: ConsoleOptionsNative,
    ) -> Result<Arc<ConsoleHandle>, FfiError> {
        let handle = block_on(mesh_llm_console_server::start_file_console(
            mesh_llm_console_server::ConsoleServerOptions {
                asset_dir: options.asset_dir.into(),
                port: options.port.unwrap_or(0),
                listen_all: options.listen_all,
            },
        ))
        .map_err(|error| FfiError::ConsoleFailed(error.to_string()))?;
        let url = handle.url().to_string();
        Ok(Arc::new(ConsoleHandle {
            inner: Mutex::new(Some(handle)),
            url,
        }))
    }
}

#[cfg(feature = "embedded-runtime")]
impl MeshNodeHandle {
    fn local_controller_for_model(&self, model: &str) -> Option<&Arc<EmbeddedServingController>> {
        let controller = self.local_serving.as_ref()?;
        let is_loaded = block_on(controller.model_list())
            .into_iter()
            .any(|(model_id, model_ref)| model_id == model || model_ref == model);
        is_loaded.then_some(controller)
    }
}

#[cfg(feature = "embedded-runtime")]
fn new_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
    format!("local-{}", NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed))
}

fn non_empty_path(value: Option<String>) -> Option<String> {
    value.and_then(|path| {
        let trimmed = path.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn parse_owner_keypair(owner_keypair_bytes_hex: &str) -> Result<OwnerKeypair, FfiError> {
    // An empty keypair is rejected rather than silently generating a fresh identity:
    // a caller that forgets to pass their persisted owner keypair would otherwise
    // get a brand-new identity every launch with no error. Callers that genuinely
    // want a new keypair should create one explicitly before calling create_node.
    let trimmed = owner_keypair_bytes_hex.trim();
    if trimmed.is_empty() {
        return Err(FfiError::InvalidOwnerKeypair(
            "owner keypair must not be empty".to_string(),
        ));
    }
    OwnerKeypair::from_hex(trimmed)
        .map_err(|error| FfiError::InvalidOwnerKeypair(error.to_string()))
}

fn path_to_string(path: std::path::PathBuf) -> String {
    path.display().to_string()
}

fn map_mesh_api_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::Client(error) => FfiError::BuildFailed(error.to_string()),
        MeshApiError::Discovery { message } => FfiError::DiscoveryFailed(message),
        MeshApiError::NoPublicMeshFound => {
            FfiError::HostUnavailable("no public mesh matched the requested criteria".to_string())
        }
        MeshApiError::InvalidInviteToken { message } => FfiError::InvalidInviteToken(message),
        MeshApiError::InvalidConfig { message } => FfiError::BuildFailed(message.to_string()),
        MeshApiError::ModelManagement { message } => FfiError::ModelManagementFailed(message),
        MeshApiError::Serving { message } => FfiError::ServingFailed(message),
        MeshApiError::Unsupported { feature } => FfiError::HostUnavailable(feature.to_string()),
    }
}

fn map_model_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::ModelManagement { message } => FfiError::ModelManagementFailed(message),
        other => FfiError::ModelManagementFailed(other.to_string()),
    }
}

fn map_serving_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::Unsupported { feature } => FfiError::ServingUnsupported(feature.to_string()),
        MeshApiError::Serving { message } => FfiError::ServingFailed(message),
        other => FfiError::ServingFailed(other.to_string()),
    }
}

fn map_stream_error(error: MeshApiError) -> FfiError {
    match error {
        MeshApiError::Client(error) => FfiError::StreamFailed(error.to_string()),
        other => FfiError::StreamFailed(other.to_string()),
    }
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

impl From<mesh_llm_api_server::PublicMesh> for PublicMesh {
    fn from(value: mesh_llm_api_server::PublicMesh) -> Self {
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

impl From<mesh_llm_api_server::CapabilityLevel> for CapabilityLevel {
    fn from(value: mesh_llm_api_server::CapabilityLevel) -> Self {
        match value {
            mesh_llm_api_server::CapabilityLevel::None => Self::None,
            mesh_llm_api_server::CapabilityLevel::Likely => Self::Likely,
            mesh_llm_api_server::CapabilityLevel::Supported => Self::Supported,
        }
    }
}

impl From<mesh_llm_api_server::ModelCapabilities> for ModelCapabilities {
    fn from(value: mesh_llm_api_server::ModelCapabilities) -> Self {
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

impl From<mesh_llm_api_server::ModelSummary> for ModelSummary {
    fn from(value: mesh_llm_api_server::ModelSummary) -> Self {
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

impl From<mesh_llm_api_server::ModelDetails> for ModelDetails {
    fn from(value: mesh_llm_api_server::ModelDetails) -> Self {
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

impl From<mesh_llm_api_server::InstalledModel> for InstalledModel {
    fn from(value: mesh_llm_api_server::InstalledModel) -> Self {
        Self {
            model_ref: value.model_ref,
            path: path_to_string(value.path),
            size_bytes: value.size_bytes,
            capabilities: value.capabilities.into(),
        }
    }
}

impl From<mesh_llm_api_server::ModelCacheStatus> for ModelCacheStatus {
    fn from(value: mesh_llm_api_server::ModelCacheStatus) -> Self {
        Self {
            cache_dir: value.cache_dir.map(path_to_string),
        }
    }
}

impl From<mesh_llm_api_server::DownloadedModel> for DownloadedModel {
    fn from(value: mesh_llm_api_server::DownloadedModel) -> Self {
        Self {
            model_ref: value.model_ref,
            paths: value.paths.into_iter().map(path_to_string).collect(),
            primary_path: value.primary_path.map(path_to_string),
            details: value.details.map(ModelDetails::from),
        }
    }
}

impl From<mesh_llm_api_server::DeleteModelResult> for DeleteModelResult {
    fn from(value: mesh_llm_api_server::DeleteModelResult) -> Self {
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

impl From<mesh_llm_api_server::CleanupResult> for CleanupResult {
    fn from(value: mesh_llm_api_server::CleanupResult) -> Self {
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

impl From<mesh_llm_api_server::PruneResult> for PruneResult {
    fn from(value: mesh_llm_api_server::PruneResult) -> Self {
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

impl From<mesh_llm_api_server::ServedModel> for ServedModel {
    fn from(value: mesh_llm_api_server::ServedModel) -> Self {
        Self {
            model_ref: value.model_ref,
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

impl From<mesh_llm_api_server::ServingStatus> for ServingStatus {
    fn from(value: mesh_llm_api_server::ServingStatus) -> Self {
        Self {
            enabled: value.enabled,
            models: value.models.into_iter().map(ServedModel::from).collect(),
        }
    }
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
