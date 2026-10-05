use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VirtualModelCandidate {
    pub model_id: String,
    /// Hex-encoded mesh endpoint that serves this concrete instance.
    ///
    /// `None` keeps compatibility with plugin-backed candidates that are not
    /// tied to a mesh host. Host-backed virtual models should preserve this
    /// value when they call back into inference so replicas of the same model
    /// remain distinct workers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_count_b: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// The host's admission state for this physical placement. Prefer ready
    /// replicas when a virtual model caps its committee.
    #[serde(default)]
    pub deprioritized: bool,
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
    /// Force this nested request to the physical candidate selected by the
    /// virtual model. This is the same endpoint identity accepted by
    /// `x-mesh-target` at normal ingress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_node_id: Option<String>,
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
