use super::*;
#[test]
fn runtime_config_diagnostics_transport() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    let invalid: MeshConfig = toml::from_str(
        r#"version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
    )
    .expect("invalid fixture should still deserialize");

    match state.apply(invalid, 0) {
        ApplyResult::ValidationError { error, diagnostics } => {
            assert!(
                error.contains(
                    "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
                ),
                "unexpected legacy error: {error}"
            );
            assert!(diagnostics.iter().any(|diagnostic| {
                diagnostic.severity == ConfigDiagnosticSeverity::Error
                    && diagnostic
                        .message
                        .contains("reasoning_format must be one of")
                    && diagnostic
                        .path
                        .as_ref()
                        .map(|path| path.render())
                        .as_deref()
                        == Some("models[0].request_defaults.reasoning_format")
                    && diagnostic.help.is_none()
            }));
        }
        other => panic!("expected ValidationError, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial_test::serial]
fn runtime_config_success_preserves_warning_diagnostics() {
    with_plugin_store(
        &[installed_plugin_metadata(
            "blackboard",
            Some(legacy_unvalidated_schema("blackboard")),
        )],
        || {
            let dir = test_dir();
            let config_path = dir.join("config.toml");
            let mut state = ConfigState::load(&config_path).expect("load");
            let config: MeshConfig = toml::from_str(
                r#"
version = 1

[[plugin]]
name = "blackboard"

[plugin.settings]
arbitrary = "kept"
"#,
            )
            .expect("legacy plugin config should deserialize");

            match state.apply(config, 0) {
                ApplyResult::Applied {
                    revision,
                    apply_mode,
                    diagnostics,
                    ..
                } => {
                    assert_eq!(revision, 1);
                    assert_eq!(apply_mode, ConfigApplyMode::Staged);
                    assert!(diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == ConfigDiagnosticCode::LegacyUnvalidatedConfig
                            && diagnostic.severity == ConfigDiagnosticSeverity::Warning
                            && diagnostic
                                .canonical_path
                                .as_ref()
                                .map(|path| path.render())
                                .as_deref()
                                == Some("plugin.blackboard.settings")
                    }));
                }
                other => panic!("expected Applied with warning diagnostics, got {other:?}"),
            }

            std::fs::remove_dir_all(&dir).ok();
        },
    );
}

#[test]
#[serial_test::serial]
fn runtime_config_apply_legacy_plugin_schema_keeps_unknown_settings_but_rejects_bad_known_values() {
    with_plugin_store(
        &[installed_plugin_metadata(
            "blackboard",
            Some(strict_blackboard_schema("blackboard", true)),
        )],
        || {
            let dir = test_dir();
            let config_path = dir.join("config.toml");
            let mut state = ConfigState::load(&config_path).expect("load");
            let config: MeshConfig = toml::from_str(
                r#"
version = 1

[[plugin]]
name = "blackboard"

[plugin.settings]
retention_days = 0
mode = "mystery"
unknown = true
"#,
            )
            .expect("legacy plugin config should deserialize");

            match state.apply(config, 0) {
                ApplyResult::ValidationError { error, diagnostics } => {
                    assert!(
                        !error.is_empty(),
                        "legacy error summary should not be empty"
                    );
                    assert!(diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == ConfigDiagnosticCode::LegacyUnvalidatedConfig
                            && diagnostic.severity == ConfigDiagnosticSeverity::Warning
                            && diagnostic
                                .canonical_path
                                .as_ref()
                                .map(|path| path.render())
                                .as_deref()
                                == Some("plugin.blackboard.settings")
                    }));
                    assert!(diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == ConfigDiagnosticCode::InvalidValue
                            && diagnostic
                                .canonical_path
                                .as_ref()
                                .map(|path| path.render())
                                .as_deref()
                                == Some("plugin.blackboard.settings.retention_days")
                    }));
                    assert!(diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == ConfigDiagnosticCode::InvalidValue
                            && diagnostic
                                .canonical_path
                                .as_ref()
                                .map(|path| path.render())
                                .as_deref()
                                == Some("plugin.blackboard.settings.mode")
                    }));
                    assert!(!diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == ConfigDiagnosticCode::UnknownField
                            && diagnostic
                                .canonical_path
                                .as_ref()
                                .map(|path| path.render())
                                .as_deref()
                                == Some("plugin.blackboard.settings.unknown")
                    }));
                }
                other => panic!("expected ValidationError, got {other:?}"),
            }

            std::fs::remove_dir_all(&dir).ok();
        },
    );
}

#[test]
fn runtime_config_apply_accepts_schema_driven_valid_fixture() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    let valid: MeshConfig =
        toml::from_str(CONTROL_FIXTURE_VALID).expect("valid fixture should deserialize");

    match state.apply(valid, 0) {
        ApplyResult::Applied { diagnostics, .. } => assert!(diagnostics.is_empty()),
        other => panic!("expected Applied, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn runtime_config_apply_matches_validator_signatures_for_schema_driven_invalid_fixture() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    let invalid: MeshConfig =
        toml::from_str(CONTROL_FIXTURE_INVALID).expect("invalid fixture should deserialize");
    let expected = diagnostic_signatures(&validate_config_diagnostics(&invalid));

    match state.apply(invalid, 0) {
        ApplyResult::ValidationError { diagnostics, .. } => {
            assert_eq!(diagnostic_signatures(&diagnostics), expected);
        }
        other => panic!("expected ValidationError, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}
