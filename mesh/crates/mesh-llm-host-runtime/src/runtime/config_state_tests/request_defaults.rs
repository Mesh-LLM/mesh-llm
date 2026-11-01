use super::*;

fn config_with_temperature(temperature: &str) -> MeshConfig {
    toml::from_str(&format!(
        "version = 1\n[defaults.request_defaults]\ntemperature = {temperature}\n"
    ))
    .expect("test config should parse")
}

#[test]
fn request_defaults_change_applies_live_without_restart() {
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
