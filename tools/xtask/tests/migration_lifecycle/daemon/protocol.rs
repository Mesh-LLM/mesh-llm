use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Audit {
    pub pid: u32,
    pub cwd: std::path::PathBuf,
    pub arguments: Vec<String>,
    pub environment: std::collections::BTreeMap<String, String>,
}

#[derive(Default, Clone, Copy, Serialize, Deserialize)]
pub enum Behavior {
    #[default]
    Clean,
    Nonzero,
    Stubborn,
    EarlyZero,
    EarlyNonzero,
    ExitModels,
    SlowHeaders,
    SlowBody,
    CleanupRecord,
    HoldCleanup,
    DeleteFailure,
    Flood,
    HoldBind,
    ExternalModels,
    EndlessBody,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Plan {
    pub status: String,
    pub status_failures: usize,
    pub models_code: u16,
    pub models_body: Vec<u8>,
    #[serde(default)]
    pub status_wire: Option<Vec<u8>>,
    pub wire: Option<Vec<u8>>,
    pub record: String,
    pub before_body: bool,
    pub behavior: Behavior,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            status: r#"{"api_port":API,"token":"status-secret-never-print","local_instances":[{"pid":PID,"is_self":true}]}"#.into(),
            status_failures: 0,
            models_code: 200,
            models_body: br#"{"object":"list","data":[]}"#.to_vec(),
            status_wire: None,
            wire: None,
            record: "{\"request_id\":\"ID\",\"source\":\"direct_http\",\"route\":\"models\",\"method\":\"GET\",\"request_kind\":\"model_listing\",\"status_code\":CODE,\"event\":\"EVENT\",\"outcome\":\"OUTCOME\"}\n".into(),
            before_body: true,
            behavior: Behavior::Clean,
        }
    }
}
