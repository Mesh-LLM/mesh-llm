use mesh_llm_sdk::events::{
    Event, EventListener as CoreEventListener, OpenAiStreamEvent as CoreOpenAiStreamEvent,
    OpenAiStreamListener as CoreOpenAiStreamListener,
};

use crate::native_runtime_types::{EventListener, OpenAiStreamListener};
use crate::request_types::ModelNative;

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

#[derive(uniffi::Enum)]
pub enum OpenAiStreamEventNative {
    Started {
        request_id: String,
        status_code: u16,
        content_type: Option<String>,
    },
    Sse {
        request_id: String,
        event_type: Option<String>,
        data: String,
        raw: String,
    },
    Completed {
        request_id: String,
    },
    Failed {
        request_id: String,
        status_code: Option<u16>,
        error: String,
        body: Option<String>,
    },
}

pub(super) struct EventListenerBridge {
    pub(super) inner: Box<dyn EventListener>,
}

pub(super) struct OpenAiStreamListenerBridge {
    pub(super) inner: Box<dyn OpenAiStreamListener>,
}

impl CoreOpenAiStreamListener for OpenAiStreamListenerBridge {
    fn on_event(&self, event: CoreOpenAiStreamEvent) {
        self.inner.on_event(match event {
            CoreOpenAiStreamEvent::Started {
                request_id,
                status_code,
                content_type,
            } => OpenAiStreamEventNative::Started {
                request_id,
                status_code,
                content_type,
            },
            CoreOpenAiStreamEvent::Sse {
                request_id,
                event_type,
                data,
                raw,
            } => OpenAiStreamEventNative::Sse {
                request_id,
                event_type,
                data,
                raw,
            },
            CoreOpenAiStreamEvent::Completed { request_id } => {
                OpenAiStreamEventNative::Completed { request_id }
            }
            CoreOpenAiStreamEvent::Failed {
                request_id,
                status_code,
                error,
                body,
            } => OpenAiStreamEventNative::Failed {
                request_id,
                status_code,
                error,
                body,
            },
        });
    }
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
