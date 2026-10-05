//! Shared single-machine serving defaults used by Skippy and Mesh.

pub const CTX_SIZE: u32 = 4096;
pub const BATCH: u32 = 512;
pub const UBATCH: u32 = 512;
// Automatic local serving uses the shared KV planner; this remains the
// fallback for callers that construct a stage without model metadata.
pub const PARALLEL: usize = 4;
pub const PREFILL_CHUNK_SIZE: usize = 64;
pub const PREFILL_ADAPTIVE_START: usize = 64;
pub const PREFILL_ADAPTIVE_STEP: usize = 64;
pub const PREFILL_ADAPTIVE_MAX: usize = 512;
pub const PREFILL_ADAPTIVE_TARGET_MS: f64 = 100.0;
pub const PREFILL_CHUNK_POLICY: &str = "fixed";

/// Completion ceiling when the client leaves its output budget unspecified.
/// Serving clamps it to the remaining context window.
pub const MAX_OUTPUT_TOKENS: u32 = 8192;
/// Default native-MTP proposal window when the operator sets no bound.
///
/// On Qwen3.8-27B-UD-Q4_K_XL (RTX 5090, CUDA 13, 8192 context), depth one
/// reached 42.751 tok/s versus 37.503 without speculation (+14.0%). Depth
/// three reached 37.938 tok/s, 11.3% slower than depth one. Increase only
/// after measuring a win with a reachable deeper verification window.
pub const NATIVE_MTP_DRAFT_TOKENS: usize = 1;
pub const DRAFT_MODEL_TOKENS: usize = 3;
pub const DOWNSTREAM_CONNECT_TIMEOUT_SECS: u64 = 30;
pub const CONTINUOUS_BATCHING: bool = true;

pub const THROUGHPUT_PROFILE: &str = "balanced";
pub const CONTINUOUS_BATCHING_POLICY: &str = "auto";

pub struct ThroughputProfileDefaults {
    pub batch: Option<u32>,
    pub ubatch: Option<u32>,
    pub parallel: Option<usize>,
    pub continuous_batching: Option<String>,
}

/// Resolve named throughput profiles before explicit settings are applied.
pub fn throughput_profile_defaults(policy: &str) -> ThroughputProfileDefaults {
    match policy {
        "throughput" => ThroughputProfileDefaults {
            batch: Some(BATCH * 2),
            ubatch: Some(UBATCH * 2),
            parallel: Some(2),
            continuous_batching: Some("true".to_string()),
        },
        "saver" => ThroughputProfileDefaults {
            batch: Some(BATCH / 2),
            ubatch: Some(UBATCH / 2),
            parallel: Some(1),
            continuous_batching: Some("false".to_string()),
        },
        _ => ThroughputProfileDefaults {
            batch: Some(BATCH),
            ubatch: Some(UBATCH),
            parallel: Some(PARALLEL),
            continuous_batching: Some(CONTINUOUS_BATCHING_POLICY.to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_profiles_preserve_mesh_tuning_behavior() {
        for (name, batch, parallel, continuous) in [
            ("balanced", 512, 4, "auto"),
            ("throughput", 1024, 2, "true"),
            ("saver", 256, 1, "false"),
        ] {
            let profile = throughput_profile_defaults(name);
            assert_eq!(profile.batch, Some(batch));
            assert_eq!(profile.ubatch, Some(batch));
            assert_eq!(profile.parallel, Some(parallel));
            assert_eq!(profile.continuous_batching.as_deref(), Some(continuous));
        }
    }
}
