use crate::diagnostic::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSource,
    DiagnosticResult, invalid_value_diagnostic, rejected_field_diagnostic,
    unsupported_field_diagnostic,
};
use crate::model::{
    BoolOrAuto, ConfigPath, ConfigSupportState, canonicalize_built_in_config_identifier,
    resolve_built_in_config_identifier,
};
use semver::{BuildMetadata, Version};
use url::Url;

pub(crate) fn parsed_config_path(raw_path: &str) -> Option<ConfigPath> {
    ConfigPath::parse_rendered(raw_path).ok()
}

pub(crate) fn validation_diagnostic(
    raw_path: &str,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    let message = message.into();
    if let Some(diagnostic) = built_in_support_diagnostic(raw_path, message.clone()) {
        return diagnostic;
    }

    let mut diagnostic = ConfigDiagnostic::error(
        ConfigDiagnosticCode::InvalidValue,
        ConfigDiagnosticSource::Validation,
        message,
    );
    diagnostic.path = parsed_config_path(raw_path);
    diagnostic
}

pub fn canonical_builtin_diagnostic_path(raw_path: &str) -> Option<ConfigPath> {
    canonicalize_built_in_config_identifier(raw_path)
        .and_then(|path| ConfigPath::parse_rendered(&path).ok())
}

pub fn built_in_support_diagnostic(
    raw_path: &str,
    message: impl Into<String>,
) -> Option<ConfigDiagnostic> {
    let resolution = resolve_built_in_config_identifier(raw_path)?;
    let message = message.into();
    let mut diagnostic = match resolution.support {
        ConfigSupportState::Rejected => {
            rejected_field_diagnostic(resolution.canonical_path.clone(), message)
        }
        ConfigSupportState::Unsupported | ConfigSupportState::Unwired => {
            unsupported_field_diagnostic(resolution.canonical_path.clone(), message)
        }
        _ => invalid_value_diagnostic(resolution.canonical_path.clone(), message),
    };
    diagnostic.path = Some(resolution.requested_path);
    diagnostic.canonical_path = Some(resolution.canonical_path);
    diagnostic.schema_source = Some(ConfigDiagnosticSchemaSource::BuiltIn);
    Some(diagnostic)
}

pub(crate) fn validate_optional_u32_range(
    value: Option<u32>,
    path: &str,
    min: u32,
    max: u32,
) -> DiagnosticResult {
    if let Some(value) = value
        && (value < min || value > max)
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be between {min} and {max}, got {value}"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_optional_positive_u64(value: Option<u64>, path: &str) -> DiagnosticResult {
    if value == Some(0) {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be at least 1 when set"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_optional_positive_usize(
    value: Option<usize>,
    path: &str,
) -> DiagnosticResult {
    if value == Some(0) {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be at least 1 when set"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_non_empty(value: &str, path: &str) -> DiagnosticResult {
    if value.trim().is_empty() {
        return Err(validation_diagnostic(
            path,
            format!("{path} must not be empty when set"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_optional_enum(
    value: Option<&str>,
    allowed: &[&str],
    path: &str,
) -> DiagnosticResult {
    if let Some(value) = value {
        validate_allowed(value, allowed, path)?;
    }
    Ok(())
}

pub(crate) fn validate_optional_kv_cache_type(value: Option<&str>, path: &str) -> DiagnosticResult {
    validate_optional_enum(
        value,
        &[
            "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
        ],
        path,
    )
}

pub(crate) fn validate_allowed(value: &str, allowed: &[&str], path: &str) -> DiagnosticResult {
    validate_non_empty(value, path)?;
    if !allowed
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be one of: {}", allowed.join(", ")),
        ));
    }
    Ok(())
}

pub(crate) fn validate_bool_or_auto(value: Option<&BoolOrAuto>, path: &str) -> DiagnosticResult {
    if let Some(BoolOrAuto::String(value)) = value {
        validate_allowed(value, &["auto"], path)?;
    }
    Ok(())
}

pub(crate) fn validate_optional_http_url(value: Option<&str>, path: &str) -> DiagnosticResult {
    if let Some(value) = value {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            let url = Url::parse(trimmed).map_err(|_| {
                validation_diagnostic(
                    path,
                    format!("{path} must be a valid URL (http:// or https://)"),
                )
            })?;
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err(validation_diagnostic(
                    path,
                    format!("{path} must use http:// or https:// scheme"),
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_probability(value: Option<f64>, path: &str) -> DiagnosticResult {
    if let Some(value) = value
        && !(0.0..=1.0).contains(&value)
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be between 0.0 and 1.0"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_non_negative_f64(value: Option<f64>, path: &str) -> DiagnosticResult {
    if let Some(value) = value
        && value < 0.0
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be greater than or equal to 0.0"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_positive_f64(value: Option<f64>, path: &str) -> DiagnosticResult {
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be greater than 0.0"),
        ));
    }
    Ok(())
}

pub(crate) fn validate_hf_pair(
    repo: Option<&str>,
    file: Option<&str>,
    repo_path: &str,
    file_path: &str,
) -> DiagnosticResult {
    let repo_present = repo.is_some_and(|v| !v.trim().is_empty());
    let file_present = file.is_some_and(|v| !v.trim().is_empty());
    match (repo_present, file_present) {
        (true, false) => Err(validation_diagnostic(
            file_path,
            format!("{file_path} must be set when {repo_path} is set"),
        )),
        (false, true) => Err(validation_diagnostic(
            repo_path,
            format!("{repo_path} must be set when {file_path} is set"),
        )),
        _ => Ok(()),
    }
}

pub(crate) fn validate_string_list(values: &[String], path: &str) -> DiagnosticResult {
    for value in values {
        validate_non_empty(value, path)?;
    }
    Ok(())
}

pub(crate) fn validate_optional_path(value: Option<&str>, path: &str) -> DiagnosticResult {
    if let Some(value) = value {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            validate_path_chars(trimmed, path)?;
        }
    }
    Ok(())
}

/// Heuristic for distinguishing a model identifier (`Org/Name:Q4_K_M`) from a
/// bare filesystem path (`/models/draft.gguf`, `C:/models/draft.gguf`). The
/// strict identifier validator requires a `':'` separator, but a Windows-style
/// absolute path also contains a `':'` immediately after the drive letter.
/// Identifiers place the quantization marker *after* the last `/`, so this
/// returns true only when the value contains a `:` that follows a `/`.
pub(crate) fn looks_like_model_identifier(value: &str) -> bool {
    let Some(colon) = value.rfind(':') else {
        return false;
    };
    match value.rfind('/') {
        Some(slash) => colon > slash,
        None => false,
    }
}

/// Validate that a `draft_model` value is a model identifier (e.g. `Qwen/Qwen3-0.6B:Q4_K_M`),
/// not a bare file path. Identifiers must contain a `:` quantization marker that follows
/// the last `/`, so that Windows-style absolute paths like `C:/models/draft.gguf` are not
/// mistaken for identifiers. When `legacy_path_used` is true, the value is treated as a
/// filesystem path and the identifier-shape check is skipped; `validate_path_chars` is still
/// applied so NUL bytes and control characters are rejected on legacy paths too.
pub(crate) fn validate_model_identifier(
    value: Option<&str>,
    path: &str,
    legacy_path_used: bool,
) -> DiagnosticResult {
    if let Some(value) = value {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            // Reject NUL bytes and control characters regardless of which key
            // supplied the value; legacy paths should not bypass path-char
            // validation.
            validate_path_chars(trimmed, path)?;
            if !legacy_path_used && !looks_like_model_identifier(trimmed) {
                return Err(validation_diagnostic(
                    path,
                    format!(
                        "{path} must be a model identifier (e.g. \"Qwen/Qwen3-0.6B:Q4_K_M\"), \
                         not a bare file path; use the legacy `draft_model_path` key for local paths"
                    ),
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_path_chars(value: &str, path: &str) -> DiagnosticResult {
    if value.contains('\0') {
        return Err(validation_diagnostic(
            path,
            format!("{path} must not contain NUL bytes"),
        ));
    }
    for ch in value.chars() {
        if ch.is_control() {
            return Err(validation_diagnostic(
                path,
                format!("{path} must not contain control characters"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn parse_node_version(
    raw: &str,
    path: &str,
) -> std::result::Result<Version, ConfigDiagnostic> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements node version bounds must be valid semver strings (an optional leading 'v' is allowed)",
        ));
    }
    let normalized = normalized
        .strip_prefix('v')
        .or_else(|| normalized.strip_prefix('V'))
        .unwrap_or(normalized);
    Version::parse(normalized).map_err(|_| {
        validation_diagnostic(
            path,
            "mesh_requirements node version bounds must be valid semver strings (an optional leading 'v' is allowed)",
        )
    })
}

pub(crate) fn validate_release_signer_key_shape(raw: &str, path: &str) -> DiagnosticResult {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements.release_signer_keys entries must not be empty",
        ));
    }
    let Some(encoded) = normalized.strip_prefix("ed25519:") else {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements.release_signer_keys entries must be of the form 'ed25519:<64-character-hex-public-key>'",
        ));
    };
    if encoded.len() != 64 || !encoded.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements.release_signer_keys entries must be of the form 'ed25519:<64-character-hex-public-key>'",
        ));
    }
    Ok(())
}

pub(crate) fn version_precedence_cmp(left: &Version, right: &Version) -> std::cmp::Ordering {
    let mut left = left.clone();
    let mut right = right.clone();
    left.build = BuildMetadata::EMPTY;
    right.build = BuildMetadata::EMPTY;
    left.cmp(&right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::{
        ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
        alias_diagnostic,
    };
    use crate::model::ConfigPath;

    #[test]
    fn schema_diagnostic_constructors_preserve_paths_and_legacy_message() {
        let used_path = ConfigPath::from_fields(["models", "gpu_id"]);
        let canonical_path = ConfigPath::from_fields(["models", "hardware", "device"]);
        let diagnostic = alias_diagnostic(
            used_path.clone(),
            canonical_path.clone(),
            "legacy gpu_id alias resolved to models.hardware.device",
        )
        .with_help("Use models.hardware.device for new config writes.");

        assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Warning);
        assert_eq!(diagnostic.code, ConfigDiagnosticCode::AliasApplied);
        assert_eq!(
            diagnostic.schema_source,
            Some(ConfigDiagnosticSchemaSource::BuiltIn)
        );
        assert_eq!(diagnostic.path, Some(used_path));
        assert_eq!(diagnostic.canonical_path, Some(canonical_path));
        assert_eq!(
            diagnostic.legacy_message(),
            "legacy gpu_id alias resolved to models.hardware.device"
        );
        assert_eq!(
            diagnostic.help.as_deref(),
            Some("Use models.hardware.device for new config writes.")
        );
    }

    #[test]
    fn schema_diagnostics_round_trip_via_toml() {
        let diagnostic = rejected_field_diagnostic(
            ConfigPath::from_fields(["defaults", "request_defaults", "json_schema"]),
            "defaults.request_defaults.json_schema is documented-rejected and must not be set",
        );

        let encoded = toml::to_string(&diagnostic).expect("diagnostic should serialize");
        let decoded: ConfigDiagnostic =
            toml::from_str(&encoded).expect("diagnostic should deserialize");

        assert_eq!(decoded, diagnostic);
    }

    #[test]
    fn schema_diagnostic_helpers_cover_validation_and_support_cases() {
        let invalid = invalid_value_diagnostic(
            ConfigPath::from_fields(["gpu", "parallel"]),
            "gpu.parallel must be at least 1, got 0",
        );
        let unsupported = unsupported_field_diagnostic(
            ConfigPath::from_fields(["runtime", "sleep_idle_seconds"]),
            "runtime.sleep_idle_seconds is not supported",
        );

        assert_eq!(invalid.code, ConfigDiagnosticCode::InvalidValue);
        assert_eq!(invalid.severity, ConfigDiagnosticSeverity::Error);
        assert_eq!(
            invalid.schema_source,
            Some(ConfigDiagnosticSchemaSource::BuiltIn)
        );
        assert_eq!(unsupported.code, ConfigDiagnosticCode::UnsupportedField);
        assert_eq!(
            unsupported.canonical_path.as_ref().map(ConfigPath::render),
            Some("runtime.sleep_idle_seconds".to_string())
        );
    }

    #[test]
    fn canonical_path_aliases_use_stable_built_in_identifier() {
        assert_eq!(
            canonical_builtin_diagnostic_path("models[0].gpu_id")
                .as_ref()
                .map(ConfigPath::render),
            Some("models.<model-ref>.hardware.device".to_string())
        );

        let diagnostic = built_in_support_diagnostic(
            "models[0].gpu_id",
            "legacy gpu_id should report the canonical device path",
        )
        .expect("legacy built-in alias should resolve");

        assert_eq!(
            diagnostic.path.as_ref().map(ConfigPath::render),
            Some("models[0].gpu_id".to_string())
        );
        assert_eq!(
            diagnostic.canonical_path.as_ref().map(ConfigPath::render),
            Some("models.<model-ref>.hardware.device".to_string())
        );
    }
}
