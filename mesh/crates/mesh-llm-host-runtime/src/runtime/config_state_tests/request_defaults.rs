use super::*;

/// Applying request defaults swaps every model tracked in the process-wide
/// registry, so tests that apply configs must not interleave.
static GLOBAL_REGISTRY: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialized() -> std::sync::MutexGuard<'static, ()> {
    GLOBAL_REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn config_with_temperature(temperature: &str) -> MeshConfig {
    toml::from_str(&format!(
        "version = 1\n[defaults.request_defaults]\ntemperature = {temperature}\n"
    ))
    .expect("test config should parse")
}

#[test]
fn request_defaults_change_applies_live_without_restart() {
    let _serialized = serialized();
    let dir = test_dir();
    let mut state = ConfigState::load(&dir.join("config.toml")).expect("load");

    let result = state.apply_with_live_logging(config_with_temperature("0.0"), 0);

    match result {
        ApplyResult::Applied {
            apply_mode,
            revision,
            ..
        } => {
            assert_eq!(apply_mode, ConfigApplyMode::Live);
            assert_eq!(revision, 1);
        }
        other => panic!("expected a live apply, got {other:?}"),
    }
    assert_eq!(
        state
            .config()
            .defaults
            .as_ref()
            .and_then(|defaults| defaults.request_defaults.as_ref())
            .and_then(|request_defaults| request_defaults.temperature),
        Some(0.0)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unrelated_config_change_stays_staged() {
    let _serialized = serialized();
    let dir = test_dir();
    let mut state = ConfigState::load(&dir.join("config.toml")).expect("load");
    assert!(matches!(
        state.apply_with_live_logging(config_with_temperature("0.0"), 0),
        ApplyResult::Applied { .. }
    ));
    let mut unrelated = config_with_temperature("0.0");
    unrelated.gpu.assignment = GpuAssignment::Pinned;

    let result = state.apply_with_live_logging(unrelated, 1);

    match result {
        ApplyResult::Applied { apply_mode, .. } => {
            assert_eq!(apply_mode, ConfigApplyMode::Staged);
        }
        other => panic!("expected a staged apply, got {other:?}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

fn registered_model(id: &str) -> skippy_serving::RegisteredRequestDefaultsProbe {
    super::live_request_defaults::track_model(id, None, None);
    skippy_serving::register_request_defaults_probe(
        id,
        skippy_serving::EmbeddedOpenAiRequestDefaults {
            temperature: Some(0.8),
            ..Default::default()
        },
    )
}

#[test]
fn applied_request_defaults_reach_a_serving_model() {
    let _serialized = serialized();
    let dir = test_dir();
    let mut state = ConfigState::load(&dir.join("config.toml")).expect("load");
    let model = registered_model("config-state-live-applied");

    state.apply_with_live_logging(config_with_temperature("0.0"), 0);

    assert_eq!(model.snapshot().temperature, Some(0.0));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn failed_persist_puts_the_previous_request_defaults_back() {
    let _serialized = serialized();
    let dir = test_dir();
    let state = ConfigState::load(&dir.join("config.toml")).expect("load");
    let model = registered_model("config-state-live-rollback");
    let preparation = state.prepare_apply(config_with_temperature("0.0"), 0);

    let result = apply_prepared_config_with_live_logging(
        preparation,
        |_| ConfigPersistence::PersistError("disk full".to_string()),
        |_, persistence| match persistence {
            ConfigPersistence::PersistError(error) => ApplyResult::PersistError(error),
            _ => unreachable!("persist was forced to fail"),
        },
    );

    assert!(matches!(result, ApplyResult::PersistError(_)));
    assert_eq!(
        model.snapshot().temperature,
        None,
        "the unpersisted default must not stay in force"
    );

    std::fs::remove_dir_all(&dir).ok();
}
