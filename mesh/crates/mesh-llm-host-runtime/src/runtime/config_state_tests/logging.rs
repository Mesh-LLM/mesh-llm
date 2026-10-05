use super::*;
#[test]
fn retention_and_replay_logging_changes_remain_staged_without_restart() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");

    // Apply baseline config first.
    let base_config = minimal_valid_config();
    match state.apply(base_config.clone(), 0) {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(revision, 1);
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
        }
        other => panic!("expected Applied for baseline, got {other:?}"),
    }

    // When: change only DynamicApply fields. This config state does not
    // apply them to a running service; it stages the persisted revision.
    let mut changed = base_config;
    changed.logging.retention_ttl_secs = 72 * 3600;
    changed.logging.replay_capacity = 256;

    match state.apply(changed, 1) {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(revision, 2);
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
        }
        other => panic!("expected staged apply for dynamic change, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn logging_change_classifier_requires_restart_for_every_static_setting() {
    let base = minimal_valid_config().logging;
    let changes: [LoggingConfigChange; 16] = [
        ("enabled", |config| config.enabled = !config.enabled),
        ("application_state_root", |config| {
            config.application_state_root = Some(PathBuf::from("logging-state"));
        }),
        ("summary_line_limit", |config| {
            config.summary_line_limit += 1
        }),
        ("event_buffer_size", |config| config.event_buffer_size += 1),
        ("retention_max_rows", |config| {
            config.retention_max_rows += 1
        }),
        ("queue_capacity", |config| config.queue_capacity += 1),
        ("artifact.capture_mode", |config| {
            config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
        }),
        ("artifact.byte_limit_bytes", |config| {
            config.artifact.byte_limit_bytes += 1;
        }),
        ("artifact.aggregate_limit_bytes", |config| {
            config.artifact.aggregate_limit_bytes += 1;
        }),
        ("export_limit_bytes", |config| {
            config.export_limit_bytes += 1
        }),
        ("cleanup_cadence_secs", |config| {
            config.cleanup_cadence_secs += 1
        }),
        ("webhook.enabled", |config| {
            config.webhook.enabled = !config.webhook.enabled
        }),
        ("webhook.url", |config| {
            config.webhook.url = Some("https://example.test/logs".into());
        }),
        ("webhook.max_attempts", |config| {
            config.webhook.max_attempts += 1
        }),
        ("webhook.timeout_secs", |config| {
            config.webhook.timeout_secs += 1
        }),
        ("webhook.dead_letter_retention_secs", |config| {
            config.webhook.dead_letter_retention_secs += 1;
        }),
    ];

    for (name, change) in changes {
        let mut changed = base.clone();
        change(&mut changed);
        assert!(
            logging_changes_require_restart(&base, &changed),
            "logging.{name} must be restart-required"
        );
    }

    let dynamic_changes: [LoggingConfigChange; 2] = [
        (
            "retention_ttl_secs",
            (|config: &mut LoggingConfig| config.retention_ttl_secs += 1) as fn(&mut LoggingConfig),
        ),
        (
            "replay_capacity",
            (|config: &mut LoggingConfig| config.replay_capacity += 1) as fn(&mut LoggingConfig),
        ),
    ];
    for (name, change) in dynamic_changes {
        let mut changed = base.clone();
        change(&mut changed);
        assert!(
            !logging_changes_require_restart(&base, &changed),
            "logging.{name} must remain dynamically applicable"
        );
    }

    assert!(
        field_requires_restart("unsupported_future_setting"),
        "unknown logging settings must not accidentally receive live-apply semantics"
    );
}

#[test]
fn dynamic_limit_change_combined_with_static_change_stays_restart_required() {
    let old = minimal_valid_config().logging;
    let mut new = old.clone();
    new.retention_ttl_secs += 1;
    new.replay_capacity += 1;
    new.queue_capacity += 1;

    assert!(
        logging_changes_require_restart(&old, &new),
        "a static logging change must not be hidden by an earlier dynamic change"
    );
}

#[test]
fn dynamic_logging_change_does_not_mask_later_nested_static_change() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");

    let base_config = minimal_valid_config();
    match state.apply(base_config.clone(), 0) {
        ApplyResult::Applied { revision, .. } => assert_eq!(revision, 1),
        other => panic!("expected Applied for baseline, got {other:?}"),
    }

    let mut changed = base_config;
    changed.logging.retention_ttl_secs += 3600;
    changed.logging.artifact.byte_limit_bytes += 1024;

    match state.apply(changed, 1) {
        ApplyResult::AppliedWithRestartRequired { revision, .. } => {
            assert_eq!(revision, 2);
        }
        other => panic!(
            "expected AppliedWithRestartRequired when dynamic and nested static fields change, got {other:?}"
        ),
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn static_logging_changes_return_restart_required_from_config_state() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");

    // Apply baseline config first.
    let base_config = minimal_valid_config();
    match state.apply(base_config.clone(), 0) {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(revision, 1);
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
        }
        other => panic!("expected Applied for baseline, got {other:?}"),
    }

    let changes: [MeshConfigChange; 5] = [
        ("enabled", |config: &mut MeshConfig| {
            config.logging.enabled = !config.logging.enabled;
        }),
        ("queue_capacity", |config: &mut MeshConfig| {
            config.logging.queue_capacity += 1;
        }),
        ("cleanup_cadence_secs", |config: &mut MeshConfig| {
            config.logging.cleanup_cadence_secs += 1;
        }),
        ("retention_max_rows", |config: &mut MeshConfig| {
            config.logging.retention_max_rows += 1;
        }),
        ("artifact.capture_mode", |config: &mut MeshConfig| {
            config.logging.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
        }),
    ];
    for (index, (change_name, change)) in changes.into_iter().enumerate() {
        let mut changed = state.config().clone();
        change(&mut changed);
        match state.apply(changed, index as u64 + 1) {
            ApplyResult::AppliedWithRestartRequired { revision, .. } => {
                assert_eq!(revision, index as u64 + 2, "{change_name}");
            }
            other => {
                panic!("expected AppliedWithRestartRequired for {change_name}, got {other:?}")
            }
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}
