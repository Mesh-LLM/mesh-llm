use mesh_llm_sdk::events::{Event, EventListener as CoreEventListener};

use crate::native_runtime_types::EventListener;
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

pub(super) struct EventListenerBridge {
    pub(super) inner: Box<dyn EventListener>,
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
