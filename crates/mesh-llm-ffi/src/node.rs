#[cfg(feature = "embedded-runtime")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use mesh_llm_sdk::node as sdk_node;
use mesh_llm_sdk::node::{MeshNode, create_auto_node as sdk_create_auto_node};
use mesh_llm_sdk::{InviteToken, RequestId};

#[cfg(feature = "embedded-runtime")]
use mesh_llm_sdk::embedded_runtime::{EmbeddedChatMessage, EmbeddedServingController};

use crate::errors::{
    FfiError, map_mesh_api_error, map_model_error, map_openai_error, map_serving_error,
    map_stream_error,
};
#[cfg(feature = "embedded-runtime")]
use crate::events::{ClientEvent, OpenAiStreamEventNative};
use crate::events::{EventListenerBridge, OpenAiStreamListenerBridge};
#[cfg(feature = "embedded-runtime")]
use crate::handles::LocalOpenAiStreamCancellation;
use crate::handles::{ConsoleHandle, MeshNodeHandle};
use crate::identity::parse_owner_keypair;
use crate::model_types::{
    CleanupPolicy, CleanupResult, DeleteModelOptions, DeleteModelResult, DevicePolicy,
    DownloadedModel, InstalledModel, LoadModelOptions, ModelCacheStatus, ModelDetails,
    ModelSearchQuery, ModelSummary, PrunePolicy, PruneResult, ServedModel, ServingStatus,
    UnloadModelOptions, UnloadTarget,
};
use crate::native_runtime_types::{EventListener, OpenAiStreamListener};
use crate::request_types::{
    ChatRequestNative, ClientStatus, ConsoleOptionsNative, ModelNative, OpenAiResponseNative,
    PublicMeshQuery, ResponsesRequestNative,
};
use crate::runtime_blocking::block_on;

fn non_empty_path(value: Option<String>) -> Option<String> {
    value.and_then(|path| {
        let trimmed = path.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
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
                #[cfg(feature = "embedded-runtime")]
                local_openai_streams: Arc::new(Mutex::new(std::collections::HashMap::new())),
            })
        })
        .map_err(map_mesh_api_error)
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
        #[cfg(feature = "embedded-runtime")]
        local_openai_streams: Arc::new(Mutex::new(std::collections::HashMap::new())),
    }))
}

#[uniffi::export]
impl MeshNodeHandle {
    pub fn start(&self) -> Result<(), FfiError> {
        block_on(self.node.start()).map_err(|error| FfiError::JoinFailed(error.to_string()))
    }

    pub fn stop(&self) -> Result<(), FfiError> {
        #[cfg(feature = "embedded-runtime")]
        self.cancel_all_local_openai_streams();
        block_on(self.node.stop()).map_err(|error| FfiError::HostUnavailable(error.to_string()))
    }

    pub fn reconnect(&self) -> Result<(), FfiError> {
        block_on(self.node.reconnect())
            .map_err(|error| FfiError::ReconnectFailed(error.to_string()))
    }

    pub fn status(&self) -> ClientStatus {
        let status = block_on(self.node.status().node()).unwrap_or(sdk_node::Status {
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
                    .map(|(id, name)| ModelNative {
                        id,
                        name,
                        context_length: None,
                    })
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
                        context_length: m.context_length,
                    })
                    .collect()
            })
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))
    }

    pub fn openai_request(
        &self,
        path: String,
        body_json: String,
    ) -> Result<OpenAiResponseNative, FfiError> {
        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = self.local_controller_for_openai_body(&body_json) {
            let response = block_on(controller.openai_request(&path, body_json))
                .map_err(|error| FfiError::OpenAiRequestFailed(error.to_string()))?;
            return Ok(OpenAiResponseNative {
                status_code: response.status_code,
                content_type: response.content_type,
                body: response.body,
            });
        }
        block_on(self.node.inference().openai_request(&path, body_json))
            .map(OpenAiResponseNative::from)
            .map_err(map_openai_error)
    }

    pub fn openai_stream(
        &self,
        path: String,
        body_json: String,
        listener: Box<dyn OpenAiStreamListener>,
    ) -> Result<String, FfiError> {
        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = self.local_controller_for_openai_body(&body_json) {
            let mut stream = block_on(controller.openai_stream(&path, body_json))
                .map_err(|error| FfiError::OpenAiRequestFailed(error.to_string()))?;
            let request_id = new_request_id();
            let cancelled = Arc::new(AtomicBool::new(false));
            let cancel_notify = Arc::new(tokio::sync::Notify::new());
            self.local_openai_streams.lock().unwrap().insert(
                request_id.clone(),
                LocalOpenAiStreamCancellation {
                    cancelled: cancelled.clone(),
                    cancel_notify: cancel_notify.clone(),
                },
            );
            let streams = self.local_openai_streams.clone();
            let id = request_id.clone();
            crate::SDK_RUNTIME.spawn(async move {
                let status_code = stream.status_code();
                let content_type = stream.content_type().map(ToString::to_string);
                let result = if !stream.is_success() {
                    let body = stream.body_text().await.unwrap_or_default();
                    Err((
                        Some(status_code),
                        format!("OpenAI stream returned HTTP {status_code}"),
                        Some(body),
                    ))
                } else if !stream.is_event_stream() {
                    listener.on_event(OpenAiStreamEventNative::Started {
                        request_id: id.clone(),
                        status_code,
                        content_type,
                    });
                    let body = stream.body_text().await.unwrap_or_default();
                    Err((
                        Some(status_code),
                        "streaming response did not use text/event-stream".to_string(),
                        Some(body),
                    ))
                } else {
                    listener.on_event(OpenAiStreamEventNative::Started {
                        request_id: id.clone(),
                        status_code,
                        content_type,
                    });
                    loop {
                        if cancelled.load(Ordering::Acquire) {
                            break Err((None, "cancelled".to_string(), None));
                        }
                        tokio::select! {
                            _ = cancel_notify.notified() => {
                                break Err((None, "cancelled".to_string(), None));
                            }
                            event = stream.next_event() => match event {
                                Ok(Some(event)) => listener.on_event(OpenAiStreamEventNative::Sse {
                                    request_id: id.clone(),
                                    event_type: event.event_type,
                                    data: event.data,
                                    raw: event.raw,
                                }),
                                Ok(None) => break Ok(()),
                                Err(error) => break Err((None, error.to_string(), None)),
                            }
                        }
                    }
                };
                streams.lock().unwrap().remove(&id);
                match result {
                    Ok(()) => {
                        listener.on_event(OpenAiStreamEventNative::Completed { request_id: id })
                    }
                    Err((status_code, error, body)) => {
                        listener.on_event(OpenAiStreamEventNative::Failed {
                            request_id: id,
                            status_code,
                            error,
                            body,
                        })
                    }
                }
            });
            return Ok(request_id);
        }
        let bridge = Arc::new(OpenAiStreamListenerBridge { inner: listener });
        block_on(
            self.node
                .inference()
                .openai_stream(&path, body_json, bridge),
        )
        .map(|request_id| request_id.0)
        .map_err(map_openai_error)
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
        #[cfg(feature = "embedded-runtime")]
        if self.cancel_local_openai_stream(&request_id) {
            return Ok(());
        }
        block_on(self.node.inference().cancel(RequestId(request_id))).map_err(map_stream_error)
    }

    pub fn recommended_models(&self) -> Result<Vec<ModelSummary>, FfiError> {
        block_on(self.node.models().recommended())
            .map(|models| models.into_iter().map(ModelSummary::from).collect())
            .map_err(map_model_error)
    }

    pub fn search_models(&self, query: ModelSearchQuery) -> Result<Vec<ModelSummary>, FfiError> {
        block_on(self.node.models().search(sdk_node::ModelSearchQuery {
            query: query.query,
            limit: query.limit.map(|limit| limit as usize),
        }))
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
                .download(model_ref, sdk_node::DownloadOptions),
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
            sdk_node::DeleteModelOptions {
                force: options.force,
            },
        ))
        .map(DeleteModelResult::from)
        .map_err(map_model_error)
    }

    pub fn cleanup_models(&self, policy: CleanupPolicy) -> Result<CleanupResult, FfiError> {
        block_on(self.node.models().cleanup(sdk_node::CleanupPolicy {
            remove_all: policy.remove_all,
        }))
        .map(CleanupResult::from)
        .map_err(map_model_error)
    }

    pub fn prune_derived_cache(&self, policy: PrunePolicy) -> Result<PruneResult, FfiError> {
        block_on(
            self.node
                .models()
                .prune_derived_cache(sdk_node::PrunePolicy {
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
            sdk_node::LoadModelOptions {
                device_policy: options.device_policy.into(),
                profile: options.profile,
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
        let handle = block_on(mesh_llm_sdk::console::start_file_console(
            mesh_llm_sdk::console::ConsoleServerOptions {
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

    fn local_controller_for_openai_body(
        &self,
        body_json: &str,
    ) -> Option<Arc<EmbeddedServingController>> {
        let controller = self.local_serving.as_ref()?;
        block_on(controller.handles_openai_request(body_json)).then(|| controller.clone())
    }

    fn cancel_local_openai_stream(&self, request_id: &str) -> bool {
        let active = self.local_openai_streams.lock().unwrap().remove(request_id);
        if let Some(active) = active {
            active.cancelled.store(true, Ordering::Release);
            active.cancel_notify.notify_one();
            true
        } else {
            false
        }
    }

    fn cancel_all_local_openai_streams(&self) {
        let streams = std::mem::take(&mut *self.local_openai_streams.lock().unwrap());
        for active in streams.into_values() {
            active.cancelled.store(true, Ordering::Release);
            active.cancel_notify.notify_one();
        }
    }
}

#[cfg(feature = "embedded-runtime")]
fn new_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
    format!("local-{}", NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed))
}
