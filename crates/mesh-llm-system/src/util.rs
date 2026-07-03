use std::path::Path;

/// Check if a path or string contains "mtp" as a marker for MTP-capable models.
/// Used to identify models that may have native MTP (Multi-Token Prediction) support.
pub fn contains_mtp_marker<T: AsRef<Path>>(path: T) -> bool {
    path.as_ref()
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().contains("mtp"))
}

/// Check if a string value contains "mtp" as a marker for MTP-capable models.
/// Used for checking model IDs, refs, and other string identifiers.
pub fn contains_mtp_marker_str(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("-mtp")
        || normalized.contains("_mtp")
        || normalized.contains("/mtp")
        || normalized.contains("mtp-gguf")
        || normalized.contains("mtp_gguf")
}

/// Validate that draft_min_tokens <= draft_max_tokens for speculative decoding.
pub fn validate_draft_min_max(draft_min_tokens: u32, draft_max_tokens: u32) -> Result<(), String> {
    if draft_min_tokens > draft_max_tokens {
        Err(
            "skippy speculative draft_min_tokens must be less than or equal to draft_max_tokens"
                .to_string(),
        )
    } else {
        Ok(())
    }
}
