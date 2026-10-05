use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mesh_llm_sdk::embedded_node::{SseDecoder, SseFrame};
use mesh_llm_sdk::{MeshNode, MeshNodeBuilder, OpenAiClient};

use crate::errors::FfiError;
use crate::events::OpenAiStreamEventNative;
use crate::handles::MeshNodeHandle;
use crate::native_runtime_types::OpenAiStreamListener;
use crate::request_types::{ModelNative, NodeStatusNative, OpenAiResponseNative};
use crate::runtime_blocking::block_on;

#[uniffi::export]
pub fn create_node(
    mode: String,
    join_tokens: Vec<String>,
    models: Vec<String>,
    auto_join: bool,
    owner_key_path: Option<String>,
    api_port: u16,
    console_port: u16,
) -> Result<Arc<MeshNodeHandle>, FfiError> {
    let mut builder = MeshNode::builder()
        .join_tokens(join_tokens)
        .models(models)
        .auto_join(auto_join)
        .api_port(api_port)
        .console_port(console_port);
    builder = match mode.as_str() {
        "client" => builder.client(),
        "serve" => builder.serve_only(),
        "combined" => builder.serve(),
        _ => return Err(FfiError::BuildFailed(format!("unknown node mode: {mode}"))),
    };
    if let Some(path) = owner_key_path {
        builder = builder.owner_key(path);
    }
    Ok(Arc::new(MeshNodeHandle {
        builder,
        node: Mutex::new(None),
        streams: Arc::new(Mutex::new(HashMap::new())),
    }))
}

#[uniffi::export]
impl MeshNodeHandle {
    pub fn start(&self) -> Result<(), FfiError> {
        let mut node = self
            .node
            .lock()
            .map_err(|e| FfiError::HostUnavailable(e.to_string()))?;
        if node.is_none() {
            *node = Some(
                block_on(self.builder.clone().start())
                    .map_err(|e| FfiError::JoinFailed(e.to_string()))?,
            );
        }
        Ok(())
    }

    pub fn stop(&self) -> Result<(), FfiError> {
        let mut streams = self
            .streams
            .lock()
            .map_err(|e| FfiError::HostUnavailable(e.to_string()))?;
        for (_, handle) in streams.drain() {
            handle.abort();
        }
        drop(streams);
        let node = self
            .node
            .lock()
            .map_err(|e| FfiError::HostUnavailable(e.to_string()))?
            .take();
        if let Some(node) = node {
            block_on(node.stop()).map_err(|e| FfiError::HostUnavailable(e.to_string()))?;
        }
        Ok(())
    }

    pub fn status(&self) -> Result<NodeStatusNative, FfiError> {
        let node = self
            .node
            .lock()
            .map_err(|e| FfiError::HostUnavailable(e.to_string()))?;
        let Some(node) = node.as_ref() else {
            return Ok(NodeStatusNative {
                running: false,
                mode: mode_name(&self.builder),
                api_base_url: String::new(),
                console_url: String::new(),
                payload_json: "null".to_string(),
            });
        };
        let status =
            block_on(node.status()).map_err(|e| FfiError::HostUnavailable(e.to_string()))?;
        Ok(NodeStatusNative {
            running: true,
            mode: mode_name(&self.builder),
            api_base_url: status.api_base_url,
            console_url: status.console_url,
            payload_json: status.payload.to_string(),
        })
    }

    pub fn join_token(&self, token: String) -> Result<(), FfiError> {
        let node = self
            .node
            .lock()
            .map_err(|e| FfiError::HostUnavailable(e.to_string()))?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".to_string()))?;
        block_on(node.join_token(token)).map_err(|e| FfiError::JoinFailed(e.to_string()))
    }

    pub fn inference_list_models(&self) -> Result<Vec<ModelNative>, FfiError> {
        let client = openai_client(self)?;
        let value =
            block_on(client.models()).map_err(|e| FfiError::DiscoveryFailed(e.to_string()))?;
        let models = value
            .get("data")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| FfiError::DiscoveryFailed("invalid /v1/models response".to_string()))?;
        Ok(models
            .iter()
            .filter_map(|value| {
                value
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(|id| ModelNative {
                        id: id.to_string(),
                        name: id.to_string(),
                        context_length: value
                            .get("context_length")
                            .and_then(serde_json::Value::as_u64)
                            .and_then(|v| u32::try_from(v).ok()),
                    })
            })
            .collect())
    }

    pub fn openai_request(
        &self,
        path: String,
        body_json: String,
    ) -> Result<OpenAiResponseNative, FfiError> {
        let client = openai_client(self)?;
        let response = block_on(client.request(&path, body_json))
            .map_err(|e| FfiError::OpenAiRequestFailed(e.to_string()))?;
        Ok(OpenAiResponseNative {
            status_code: response.status_code,
            content_type: response.content_type,
            body: response.body,
        })
    }

    pub fn openai_stream(
        &self,
        path: String,
        body_json: String,
        listener: Box<dyn OpenAiStreamListener>,
    ) -> Result<String, FfiError> {
        let client = openai_client(self)?;
        let response = block_on(client.stream(&path, body_json))
            .map_err(|e| FfiError::StreamFailed(e.to_string()))?;
        let request_id = format!("stream-{}", uuid::Uuid::new_v4());
        let id = request_id.clone();
        let streams = self.streams.clone();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let task = crate::SDK_RUNTIME.spawn(async move {
            if ready_rx.await.is_err() {
                return;
            }
            send_stream_events(response, &id, listener).await;
            if let Ok(mut guard) = streams.lock() {
                guard.remove(&id);
            }
        });
        self.streams
            .lock()
            .map_err(|e| FfiError::StreamFailed(e.to_string()))?
            .insert(request_id.clone(), task.abort_handle());
        let _ = ready_tx.send(());
        Ok(request_id)
    }

    pub fn cancel(&self, request_id: String) {
        if let Ok(mut streams) = self.streams.lock()
            && let Some(task) = streams.remove(&request_id)
        {
            task.abort();
        }
    }
}

fn openai_client(handle: &MeshNodeHandle) -> Result<OpenAiClient, FfiError> {
    let node = handle
        .node
        .lock()
        .map_err(|e| FfiError::HostUnavailable(e.to_string()))?;
    node.as_ref()
        .ok_or_else(|| FfiError::HostUnavailable("node is not running".to_string()))?
        .openai_client()
        .map_err(|e| FfiError::ServingUnsupported(e.to_string()))
}

fn mode_name(builder: &MeshNodeBuilder) -> String {
    match builder.clone().build().mode {
        mesh_llm_sdk::embedded_node::EmbeddedMeshNodeMode::Client => "client",
        mesh_llm_sdk::embedded_node::EmbeddedMeshNodeMode::ServeOnly => "serve",
        mesh_llm_sdk::embedded_node::EmbeddedMeshNodeMode::Serve => "combined",
    }
    .to_string()
}

async fn send_stream_events(
    mut response: reqwest::Response,
    request_id: &str,
    listener: Box<dyn OpenAiStreamListener>,
) {
    let status_code = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string);
    listener.on_event(OpenAiStreamEventNative::Started {
        request_id: request_id.to_string(),
        status_code,
        content_type: content_type.clone(),
    });
    if !(200..300).contains(&status_code)
        || !content_type
            .as_deref()
            .is_some_and(|value| value.starts_with("text/event-stream"))
    {
        let body = response.text().await.ok();
        listener.on_event(OpenAiStreamEventNative::Failed {
            request_id: request_id.to_string(),
            status_code: Some(status_code),
            error: format!("expected streaming response; HTTP {status_code}"),
            body,
        });
        return;
    }
    let mut decoder = SseDecoder::default();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => match decoder.push(&chunk) {
                Ok(frames) => {
                    for frame in frames {
                        emit_frame(&*listener, request_id, frame);
                    }
                }
                Err(error) => {
                    stream_failed(&*listener, request_id, status_code, error.to_string());
                    return;
                }
            },
            Ok(None) => {
                match decoder.finish() {
                    Ok(Some(frame)) => emit_frame(&*listener, request_id, frame),
                    Ok(None) => {}
                    Err(error) => {
                        stream_failed(&*listener, request_id, status_code, error.to_string());
                        return;
                    }
                }
                listener.on_event(OpenAiStreamEventNative::Completed {
                    request_id: request_id.to_string(),
                });
                return;
            }
            Err(error) => {
                stream_failed(&*listener, request_id, status_code, error.to_string());
                return;
            }
        }
    }
}

fn emit_frame(listener: &dyn OpenAiStreamListener, request_id: &str, frame: SseFrame) {
    listener.on_event(OpenAiStreamEventNative::Sse {
        request_id: request_id.to_string(),
        event_type: frame.event_type,
        data: frame.data,
        raw: frame.raw,
    });
}

fn stream_failed(
    listener: &dyn OpenAiStreamListener,
    request_id: &str,
    status_code: u16,
    error: String,
) {
    listener.on_event(OpenAiStreamEventNative::Failed {
        request_id: request_id.to_string(),
        status_code: Some(status_code),
        error,
        body: None,
    });
}
