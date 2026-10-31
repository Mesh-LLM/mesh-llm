//! Generic operator-authorized local provider RPC contract.
use serde::{Deserialize, Serialize};

pub const CAPABILITY: &str = "operation_bindings.v1";
pub const METHOD: &str = "plugin/bound_operation";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundOperationRequest {
    pub binding: String,
    /// Absent runs only the operator-fixed setup call.
    pub handle: Option<String>,
    /// Setup must leave these empty; it cannot supply replacement arguments.
    #[serde(default)]
    pub operation: String,
    #[serde(default)]
    pub input_json: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum BoundOperationResponse {
    NotDispatched {
        reason: String,
    },
    Responded {
        handle: String,
        output_json: String,
        is_error: bool,
    },
    Uncertain {
        reason: String,
    },
}
