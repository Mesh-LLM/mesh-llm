use super::*;
#[test]
fn config_sync_load_malformed_nested_toml_still_errors() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    std::fs::write(
        &config_path,
        r#"version = 1

[[models]]
model = "Qwen3-8B-Q4_K_M"

[models.request_defaults
temperature = 0.2
"#,
    )
    .expect("write malformed config");

    let result = ConfigState::load(&config_path);
    assert!(
        result.is_err(),
        "load must return Err on malformed nested TOML"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_noop_apply_skips_disk_write() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");

    let config_with_model = MeshConfig {
        version: Some(1),
        gpu: crate::plugin::GpuConfig {
            assignment: GpuAssignment::Auto,
            parallel: None,
        },
        mesh_requirements: Default::default(),
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models: vec![crate::plugin::ModelConfigEntry {
            model: "noop-test.gguf".to_string(),
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

    let r1 = state.apply(config_with_model.clone(), 0);
    let rev_after_first = match r1 {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(
                apply_mode,
                ConfigApplyMode::Staged,
                "first apply must save to disk"
            );
            revision
        }
        other => panic!("expected Applied, got {other:?}"),
    };

    let r2 = state.apply(config_with_model.clone(), rev_after_first);
    match r2 {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(
                apply_mode,
                ConfigApplyMode::Noop,
                "no-op apply must not save to disk"
            );
            assert_eq!(
                revision, rev_after_first,
                "revision must not change on no-op"
            );
        }
        other => panic!("expected Applied with Noop apply_mode, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_telemetry_only_change_is_persisted_locally() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");

    let base = minimal_valid_config();
    let r1 = state.apply(base.clone(), 0);
    let rev_after_first = match r1 {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
            revision
        }
        other => panic!("expected Applied, got {other:?}"),
    };

    let mut telemetry_only = base;
    telemetry_only.telemetry.enabled = Some(true);
    telemetry_only.telemetry.endpoint = Some("https://otel.example.com".to_string());

    let r2 = state.apply(telemetry_only, rev_after_first);
    match r2 {
        ApplyResult::Applied {
            revision,
            apply_mode,
            ..
        } => {
            assert_eq!(
                apply_mode,
                ConfigApplyMode::Staged,
                "local-only telemetry changes must still be written to config.toml"
            );
            assert_eq!(revision, rev_after_first + 1);
        }
        other => panic!("expected Applied with Staged apply_mode, got {other:?}"),
    }

    let persisted = std::fs::read_to_string(&config_path).expect("persisted config");
    assert!(persisted.contains("https://otel.example.com"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_sidecar_path_derived_from_filename() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let sidecar = revision_sidecar_path(&config_path);
    let expected = dir.join("config.toml.revision");
    assert_eq!(
        sidecar, expected,
        "sidecar path must be config filename + .revision suffix"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_sidecar_migration_fallback() {
    let dir = test_dir();
    let legacy_path = dir.join("config-revision");
    std::fs::write(&legacy_path, "42\n").expect("write legacy revision");

    let config_path = dir.join("config.toml");
    let new_sidecar = revision_sidecar_path(&config_path);
    assert_ne!(
        new_sidecar, legacy_path,
        "new sidecar must differ from legacy"
    );

    let revision = read_revision(&new_sidecar);
    assert_eq!(
        revision, 42,
        "must fall back to legacy config-revision file"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_sync_state_apply_persists_integrated_fixture_sections_and_hashes_changes() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    let config: MeshConfig = toml::from_str(FULL_SURFACE_VALID_FIXTURE).expect("fixture parses");

    let first = state.apply(config.clone(), 0);
    let first_hash = match first {
        ApplyResult::Applied {
            revision,
            hash,
            apply_mode,
            diagnostics,
        } => {
            assert_eq!(revision, 1);
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
            let has_errors = diagnostics
                .iter()
                .any(|d| d.severity == mesh_llm_config::ConfigDiagnosticSeverity::Error);
            assert!(!has_errors, "unexpected error diagnostics: {diagnostics:?}");
            hash
        }
        other => panic!("expected Applied, got {other:?}"),
    };

    let persisted = std::fs::read_to_string(&config_path).expect("persisted config");
    assert!(persisted.contains("[models.skippy]"));
    assert!(persisted.contains("prefill_chunk_schedule = \"128,256,384\""));
    assert!(persisted.contains("reasoning_budget = 256"));

    let reloaded = ConfigState::load(&config_path).expect("reload config");
    assert_eq!(reloaded.config().models.len(), 2);
    assert_eq!(
        reloaded.config().models[0]
            .advanced
            .as_ref()
            .and_then(|advanced| advanced.server.as_ref())
            .and_then(|server| server.alias.as_deref()),
        Some("model-alias")
    );

    let mut changed = config;
    changed
        .defaults
        .as_mut()
        .and_then(|defaults| defaults.request_defaults.as_mut())
        .expect("request defaults")
        .temperature = Some(0.6);
    let second = state.apply(changed, 1);
    match second {
        ApplyResult::Applied { revision, hash, .. } => {
            assert_eq!(revision, 2);
            assert_ne!(first_hash, hash, "request-default change must update hash");
        }
        other => panic!("expected Applied, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}
