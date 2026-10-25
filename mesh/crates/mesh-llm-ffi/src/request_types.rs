#[derive(uniffi::Record)]
pub struct ModelNative {
    pub id: String,
    pub name: String,
    pub context_length: Option<u32>,
}

#[derive(uniffi::Record)]
pub struct NodeStatusNative {
    pub running: bool,
    pub mode: String,
    pub api_base_url: String,
    pub console_url: String,
    pub payload_json: String,
}

#[derive(uniffi::Record)]
pub struct OpenAiResponseNative {
    pub status_code: u16,
    pub content_type: Option<String>,
    pub body: String,
}
