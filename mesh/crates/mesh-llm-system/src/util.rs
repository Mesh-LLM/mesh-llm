use std::path::Path;

/// Check if a path or string contains "mtp" as a marker for MTP-capable models.
/// Used to identify models that may have native MTP (Multi-Token Prediction) support.
pub fn contains_mtp_marker<T: AsRef<Path>>(path: T) -> bool {
    contains_mtp_marker_str(&path.as_ref().to_string_lossy())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_mtp_marker_detects_marker_in_path_text() {
        assert!(contains_mtp_marker("/models/native-mtp/model.gguf"));
        assert!(contains_mtp_marker("/models/base/MTP-GGUF-model.gguf"));
        assert!(contains_mtp_marker("/models/base/model_mtp.gguf"));
        assert!(!contains_mtp_marker("/models/base/model.gguf"));
    }

    #[test]
    fn contains_mtp_marker_str_detects_supported_marker_patterns() {
        assert!(contains_mtp_marker_str("vendor/model-mtp"));
        assert!(contains_mtp_marker_str("vendor/model_mtp"));
        assert!(contains_mtp_marker_str("vendor/mtp/model"));
        assert!(contains_mtp_marker_str("vendor/model-mtp-gguf"));
        assert!(contains_mtp_marker_str("vendor/model_mtp_gguf"));
        assert!(!contains_mtp_marker_str("vendor/model"));
        assert!(!contains_mtp_marker_str("vendor/attempt-model"));
    }
}
