use crate::Model;

#[derive(Debug, Clone)]
pub enum Event {
    Connecting,
    Joined { node_id: String },
    ModelsUpdated { models: Vec<Model> },
    TokenDelta { request_id: String, delta: String },
    Completed { request_id: String },
    Failed { request_id: String, error: String },
    Disconnected { reason: String },
}

pub trait EventListener: Send + Sync + 'static {
    fn on_event(&self, event: Event);
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum OpenAiStreamEvent {
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

pub trait OpenAiStreamListener: Send + Sync + 'static {
    fn on_event(&self, event: OpenAiStreamEvent);
}
