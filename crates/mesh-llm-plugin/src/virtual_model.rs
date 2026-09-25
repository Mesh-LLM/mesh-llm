use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VirtualModelCandidate {
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_count_b: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub supports_tools: bool,
    #[serde(default)]
    pub supports_vision: bool,
    #[serde(default)]
    pub supports_audio: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VirtualModelInvocation {
    pub request: serde_json::Value,
    #[serde(default)]
    pub candidates: Vec<VirtualModelCandidate>,
    #[serde(default)]
    pub response_adapter: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VirtualModelResponse {
    pub status_code: u16,
    pub body: serde_json::Value,
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    #[serde(default)]
    pub event_stream: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HostInferenceRequest {
    pub model_id: String,
    pub request: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HostInferenceResponse {
    pub status_code: u16,
    pub body: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_by: Option<String>,
}
