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
