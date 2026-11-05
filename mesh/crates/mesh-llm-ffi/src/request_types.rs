#[derive(uniffi::Record)]
pub struct ModelNative {
    pub id: String,
    pub name: String,
    pub context_length: Option<u32>,
    /// Effort strings the serving provider observed in the loaded chat
    /// template, in its reported order. `None` is unknown — an older provider
    /// or an unprobed model — and an empty list is a known "no effort levels".
    pub thinking_efforts: Option<Vec<String>>,
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
