//! Local image safety operation contract. Only an explicit `Allow` permits progress.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const REQUEST_GUARD_CAPABILITY: &str = "image-request-guard.v1";
pub const GENERATION_GUARD_CAPABILITY: &str = "image-generation-guard.v1";
pub const STATUS_OPERATION: &str = "status";
pub const TEXT_OPERATION: &str = "screen_text";
pub const IMAGE_OPERATION: &str = "screen_image";
pub const PROTOCOL_VERSION: u32 = 1;
/// Keep one image and JSON framing comfortably below the plugin control frame limit.
pub const MAX_IMAGE_DATA_URL_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_TEXT_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextScreenRequest {
    /// The host's complete effective conversation, serialized as text.
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageScreenRequest {
    /// One bounded inline image; remote URLs and filesystem paths are not accepted.
    pub data_url: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GuardDecision {
    Allow,
    Block,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GuardVerdict {
    pub decision: GuardDecision,
    pub model: String,
    pub policy_revision: String,
    /// Bounded machine-readable code; never classifier rationale or input content.
    pub reason_code: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GuardStatus {
    pub protocol_version: u32,
    pub ready: bool,
    pub screen_received_images: bool,
    pub model: String,
    pub policy_revision: String,
}
