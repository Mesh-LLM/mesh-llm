use super::*;
#[test]
fn config_sync_state_load() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");

    std::fs::write(
        &config_path,
        "version = 1\n\n[gpu]\nassignment = \"auto\"\n",
    )
    .expect("write config");

    let state = ConfigState::load(&config_path).expect("load");
    assert_eq!(state.revision(), 0);
    assert_eq!(state.config().version, Some(1));
    assert_eq!(state.config().gpu.assignment, GpuAssignment::Auto);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_apply_success() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");

    let mut state = ConfigState::load(&config_path).expect("load");
    assert_eq!(state.revision(), 0);

    let result = state.apply(minimal_valid_config(), 0);
    match result {
        ApplyResult::Applied {
            revision,
            hash: _,
            apply_mode,
            diagnostics,
        } => {
            assert_eq!(revision, 1);
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
            assert!(diagnostics.is_empty());
        }
        other => panic!("expected Applied, got {other:?}"),
    }

    assert!(config_path.exists(), "config file not written");

    let sidecar = revision_sidecar_path(&config_path);
    let sidecar_contents = std::fs::read_to_string(&sidecar).expect("read sidecar");
    assert_eq!(sidecar_contents.trim(), "1");

    assert_eq!(state.revision(), 1);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_apply_preserves_additive_defaults_sections() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    std::fs::write(
        &config_path,
        r#"version = 1

[defaults.throughput]
parallel = 2

[defaults.model_fit]
flash_attention = "auto"

[defaults.request_defaults]
reasoning_format = "deepseek"
"#,
    )
    .expect("write baseline config");

    let mut state = ConfigState::load(&config_path).expect("load baseline config");
    let mut config = minimal_valid_config();
    config.extra = toml::from_str(
        r#"[defaults.throughput]
parallel = 6

[defaults.model_fit]
flash_attention = "disabled"

[defaults.request_defaults]
reasoning_format = "qwen"
"#,
    )
    .expect("parse additive defaults table");

    let result = state.apply(config, 0);
    match result {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(revision, 1);
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
        }
        other => panic!("expected additive defaults to be written, got {other:?}"),
    }

    let written = std::fs::read_to_string(&config_path).expect("read written config");
    let written: toml::Value = toml::from_str(&written).expect("written TOML parses");
    assert_eq!(
        written
            .get("defaults")
            .and_then(|defaults| defaults.get("throughput"))
            .and_then(|throughput| throughput.get("parallel"))
            .and_then(toml::Value::as_integer),
        Some(6)
    );
    assert_eq!(
        written
            .get("defaults")
            .and_then(|defaults| defaults.get("model_fit"))
            .and_then(|model_fit| model_fit.get("flash_attention"))
            .and_then(toml::Value::as_str),
        Some("disabled")
    );
    assert_eq!(
        written
            .get("defaults")
            .and_then(|defaults| defaults.get("request_defaults"))
            .and_then(|request_defaults| request_defaults.get("reasoning_format"))
            .and_then(toml::Value::as_str),
        Some("qwen")
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_conflict() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");

    let mut state = ConfigState::load(&config_path).expect("load");

    let result = state.apply(minimal_valid_config(), 0);
    assert!(
        matches!(result, ApplyResult::Applied { revision: 1, .. }),
        "first apply failed: {result:?}"
    );

    let result2 = state.apply(minimal_valid_config(), 0);
    match result2 {
        ApplyResult::RevisionConflict { current_revision } => {
            assert_eq!(current_revision, 1);
        }
        other => panic!("expected RevisionConflict, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn prepared_apply_keeps_config_and_hash_atomic_until_persistence_completes() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    let initial_hash = *state.config_hash();

    let pending = match state.prepare_apply(minimal_valid_config(), 0) {
        ConfigApplyPreparation::Pending(pending) => pending,
        other => panic!("expected pending apply, got {other:?}"),
    };

    assert_eq!(state.revision(), 0);
    assert_eq!(*state.config_hash(), initial_hash);
    assert!(
        !config_path.exists(),
        "preparing an apply must not persist while state is locked"
    );

    let persistence = pending.persist();
    assert!(
        matches!(&persistence, ConfigPersistence::Persisted),
        "persistence should succeed: {persistence:?}"
    );
    assert!(config_path.exists(), "persistence should write config.toml");
    assert_eq!(state.revision(), 0);
    assert_eq!(
        *state.config_hash(),
        initial_hash,
        "in-memory config and hash remain unchanged until commit"
    );

    let result = state.finish_apply(*pending, persistence);
    assert!(
        matches!(result, ApplyResult::Applied { revision: 1, .. }),
        "commit should publish the persisted config atomically: {result:?}"
    );
    assert_ne!(*state.config_hash(), initial_hash);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn finish_apply_rejects_a_pending_revision_after_another_apply_wins() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    let initial_hash = *state.config_hash();

    let pending = match state.prepare_apply(minimal_valid_config(), 0) {
        ConfigApplyPreparation::Pending(pending) => pending,
        other => panic!("expected pending apply, got {other:?}"),
    };
    let persistence = pending.persist();

    // Simulate a concurrent apply committing between preparation and this
    // pending apply's commit.
    state.revision = 1;

    let result = state.finish_apply(*pending, persistence);
    assert!(matches!(
        result,
        ApplyResult::RevisionConflict {
            current_revision: 1
        }
    ));
    assert_eq!(*state.config_hash(), initial_hash);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_concurrent_applies() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).unwrap();

    let r1 = state.apply(minimal_valid_config(), 0);
    assert!(
        matches!(r1, ApplyResult::Applied { revision: 1, .. }),
        "first apply must succeed: {r1:?}"
    );

    let r2 = state.apply(minimal_valid_config(), 0);
    assert!(
        matches!(
            r2,
            ApplyResult::RevisionConflict {
                current_revision: 1
            }
        ),
        "second apply with stale revision must conflict: {r2:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_revision_monotonic() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).unwrap();

    let make_config = |model: &str| MeshConfig {
        version: Some(1),
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![crate::plugin::ModelConfigEntry {
            model: model.to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        plugins: vec![],
        logging: Default::default(),
        extra: Default::default(),
    };

    assert_eq!(state.revision(), 0);
    state.apply(make_config("model-a.gguf"), 0);
    assert_eq!(state.revision(), 1);
    state.apply(make_config("model-b.gguf"), 1);
    assert_eq!(state.revision(), 2);
    state.apply(make_config("model-c.gguf"), 2);
    assert_eq!(state.revision(), 3);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_hash_changes_on_different_config() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).unwrap();
    let initial_hash = *state.config_hash();

    let config_with_model = MeshConfig {
        version: Some(1),
        gpu: GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![crate::plugin::ModelConfigEntry {
            model: "test.gguf".to_string(),
            mmproj: None,
            ctx_size: None,
            gpu_id: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        plugins: vec![],
        logging: Default::default(),
        extra: Default::default(),
    };
    state.apply(config_with_model, 0);
    let new_hash = *state.config_hash();
    assert_ne!(
        initial_hash, new_hash,
        "hash must change when config changes"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_apply_preserves_nested_sections_and_updates_hash() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");

    let first = representative_nested_config();
    let first_result = state.apply(first.clone(), 0);
    let first_hash = match first_result {
        ApplyResult::Applied {
            revision,
            hash,
            apply_mode,
            diagnostics,
        } => {
            assert_eq!(revision, 1);
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
            assert!(
                diagnostics
                    .iter()
                    .all(|diagnostic| diagnostic.severity != ConfigDiagnosticSeverity::Error),
                "unexpected error diagnostics: {diagnostics:?}"
            );
            hash
        }
        other => panic!("expected Applied, got {other:?}"),
    };
    assert_representative_nested_fields(state.config());

    let persisted = ConfigState::load(&config_path).expect("reload persisted config");
    assert_representative_nested_fields(persisted.config());

    let mut changed = first;
    changed
        .models
        .first_mut()
        .expect("model")
        .advanced
        .get_or_insert_with(Default::default)
        .server
        .get_or_insert_with(Default::default)
        .alias = Some("model-alias-updated".to_string());

    let second_result = state.apply(changed, 1);
    match second_result {
        ApplyResult::Applied { revision, hash, .. } => {
            assert_eq!(revision, 2);
            assert_ne!(first_hash, hash, "nested field change must change hash");
        }
        other => panic!("expected Applied, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_load_propagates_invalid_toml_error() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    std::fs::write(&config_path, "this is [not valid toml !!!\n").expect("write bad toml");
    let result = ConfigState::load(&config_path);
    assert!(result.is_err(), "load must return Err on malformed TOML");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_load_nested_validation_error_is_stable() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    std::fs::write(
        &config_path,
        r#"version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults]
reasoning_format = "mystery"
"#,
    )
    .expect("write invalid config");

    let error = match ConfigState::load(&config_path) {
        Ok(_) => panic!("load must fail"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains(
            "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
        ),
        "unexpected error: {message}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
