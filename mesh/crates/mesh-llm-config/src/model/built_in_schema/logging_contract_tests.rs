//! Stable built-in schema contracts for operator logging settings.

use super::*;

#[test]
fn logging_settings_classify_only_retention_and_replay_as_dynamic() {
    for path in ["logging.retention_ttl_secs", "logging.replay_capacity"] {
        let setting = schema_setting(path);

        assert_eq!(
            setting.control_surfaces,
            vec![ConfigControlSurface::ConfigFile],
            "{path}"
        );
        assert_eq!(setting.apply_mode, ConfigApplyMode::DynamicApply, "{path}");
        assert_eq!(setting.restart_scope, ConfigRestartScope::None, "{path}");
    }

    for path in [
        "logging.enabled",
        "logging.application_state_root",
        "logging.summary_line_limit",
        "logging.event_buffer_size",
        "logging.retention_max_rows",
        "logging.queue_capacity",
        "logging.cleanup_cadence_secs",
        "logging.artifact.capture_mode",
        "logging.artifact.byte_limit_bytes",
        "logging.artifact.aggregate_limit_bytes",
        "logging.export_limit_bytes",
        "logging.webhook.enabled",
        "logging.webhook.url",
        "logging.webhook.max_attempts",
        "logging.webhook.timeout_secs",
        "logging.webhook.dead_letter_retention_secs",
    ] {
        let setting = schema_setting(path);

        assert_eq!(
            setting.control_surfaces,
            vec![ConfigControlSurface::ConfigFile],
            "{path}"
        );
        assert_eq!(setting.apply_mode, ConfigApplyMode::StaticOnLoad, "{path}");
        assert_eq!(
            setting.restart_scope,
            ConfigRestartScope::ProcessRestart,
            "{path}"
        );
    }
}

#[test]
fn logging_capacity_descriptions_are_stable_and_truthful() {
    for (path, expected) in [
        (
            "logging.summary_line_limit",
            "Maximum number of Unicode characters in each generated request summary.",
        ),
        (
            "logging.event_buffer_size",
            "Maximum number of event entries held in memory for replay.",
        ),
        (
            "logging.replay_capacity",
            "Number of recent events available to reconnecting console clients.",
        ),
        (
            "logging.queue_capacity",
            "Maximum number of pending log entries waiting for persistence and webhook dispatch.",
        ),
    ] {
        assert_eq!(schema_setting(path).description.as_deref(), Some(expected));
    }
}

#[test]
fn logging_presentation_exposes_friendly_choices_and_byte_units() {
    let capture = schema_setting("logging.artifact.capture_mode");
    let capture_presentation = capture.presentation.expect("capture presentation");
    assert_eq!(
        capture_presentation.label.as_deref(),
        Some("Captured content")
    );
    assert_eq!(capture_presentation.choices[0].label, "Metadata only");
    assert_eq!(capture_presentation.choices[1].label, "Redacted payloads");

    let byte_limit = schema_setting("logging.artifact.byte_limit_bytes");
    let byte_presentation = byte_limit.presentation.expect("byte presentation");
    assert_eq!(byte_presentation.renderer_id.as_deref(), Some("byte-size"));
    assert_eq!(
        byte_presentation
            .display_units
            .iter()
            .map(|unit| unit.label.as_str())
            .collect::<Vec<_>>(),
        vec!["B", "KB", "MB", "GB"]
    );

    assert_eq!(
        schema_setting("logging.audit.enabled").visibility,
        ConfigVisibility::Advanced
    );
    assert_eq!(
        schema_setting("logging.enabled").visibility,
        ConfigVisibility::User
    );
}

#[test]
fn audit_restart_contract_and_logging_bounds_are_exported() {
    for path in [
        "logging.audit.enabled",
        "logging.audit.log_path",
        "logging.audit.log_format",
        "logging.audit.log_level",
        "logging.audit.max_file_size_mb",
        "logging.audit.max_files",
    ] {
        let setting = schema_setting(path);
        assert_eq!(setting.apply_mode, ConfigApplyMode::StaticOnLoad, "{path}");
        assert_eq!(
            setting.restart_scope,
            ConfigRestartScope::ProcessRestart,
            "{path}"
        );
    }

    assert_has_range_constraint(
        &schema_setting("logging.queue_capacity"),
        Some("64"),
        Some("131072"),
    );
    assert_has_range_constraint(
        &schema_setting("logging.artifact.byte_limit_bytes"),
        Some("1024"),
        Some("16777216"),
    );
    assert_has_range_constraint(
        &schema_setting("logging.audit.max_file_size_mb"),
        Some("1"),
        None,
    );
    assert_has_range_constraint(&schema_setting("logging.audit.max_files"), Some("1"), None);
}

fn schema_setting(path: &str) -> ConfigSettingSchema {
    built_in_config_schema_descriptor(&schema_path(path)).expect("schema setting should exist")
}

fn assert_has_range_constraint(
    setting: &ConfigSettingSchema,
    expected_min: Option<&str>,
    expected_max: Option<&str>,
) {
    assert!(
        setting.constraints.iter().any(|constraint| {
            matches!(
                constraint,
                ConfigConstraint::Range { min, max }
                    if min.as_deref() == expected_min && max.as_deref() == expected_max
            )
        }),
        "expected range constraint min={expected_min:?} max={expected_max:?} on {}",
        setting.path.render()
    );
}
