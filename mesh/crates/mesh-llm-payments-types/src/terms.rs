use serde::{Deserialize, Serialize};

use crate::pricing::Pricing;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestTerms {
    /// Host evidence correlation, separate from the private recovery capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exchange_id: Option<String>,
    pub id: String,
    pub peer: String,
    #[serde(default)]
    pub payee: Option<String>,
    pub model: String,
    pub pricing: Pricing,
    pub input_tokens: u64,
    pub max_output_tokens: u64,
    pub max_total_msat: u64,
    pub expires_at_ms: u64,
}
