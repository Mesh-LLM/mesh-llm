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

fn registered_model(
    state: &ConfigState,
    id: &str,
) -> skippy_serving::RegisteredRequestDefaultsProbe {
    let live = state.live_request_defaults();
    live.track_model(id, None, None);
    skippy_serving::register_request_defaults_probe(
        live.scope(),
        id,
        skippy_serving::EmbeddedOpenAiRequestDefaults {
            temperature: Some(0.8),
            ..Default::default()
        },
    )
}

#[test]
fn applied_request_defaults_reach_a_serving_model() {
    let dir = test_dir();
    let mut state = ConfigState::load(&dir.join("config.toml")).expect("load");
    let model = registered_model(&state, "config-state-live-applied");

    state.apply_with_live_logging(config_with_temperature("0.0"), 0);

    assert_eq!(model.snapshot().temperature, Some(0.0));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn failed_persist_puts_the_previous_request_defaults_back() {
    let dir = test_dir();
    let state = ConfigState::load(&dir.join("config.toml")).expect("load");
    let model = registered_model(&state, "config-state-live-rollback");
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
        Some(0.8),
        "the backend goes back to what it was serving with, not to a re-resolved config"
    );

    std::fs::remove_dir_all(&dir).ok();
}

fn applied_mode(result: ApplyResult) -> ConfigApplyMode {
    match result {
        ApplyResult::Applied { apply_mode, .. } => apply_mode,
        other => panic!("expected an applied config, got {other:?}"),
    }
}

#[test]
fn a_context_window_change_saved_with_a_live_default_stays_staged() {
    let dir = test_dir();
    let mut state = ConfigState::load(&dir.join("config.toml")).expect("load");
    let model = registered_model(&state, "config-state-mixed-ctx");
    let mut mixed = config_with_temperature("0.0");
    mixed
        .defaults
        .get_or_insert_with(Default::default)
        .model_fit
        .get_or_insert_with(Default::default)
        .ctx_size = Some(8192);

    let mode = applied_mode(state.apply_with_live_logging(mixed, 0));

    assert_eq!(
        mode,
        ConfigApplyMode::Staged,
        "the context window only takes effect after a reload"
    );
    assert_eq!(
        model.snapshot().temperature,
        Some(0.0),
        "the live part still took effect"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn request_defaults_that_do_not_resolve_for_every_model_stay_staged() {
    let dir = test_dir();
    let mut state = ConfigState::load(&dir.join("config.toml")).expect("load");
    let model = registered_model(&state, "config-state-unresolvable");
    let broken: MeshConfig = toml::from_str(
        "version = 1\n[defaults.request_defaults]\nchat_template_file = \"/nonexistent/live-defaults.jinja\"\n",
    )
    .expect("test config should parse");

    let mode = applied_mode(state.apply_with_live_logging(broken, 0));

    assert_eq!(mode, ConfigApplyMode::Staged);
    assert_eq!(model.snapshot().temperature, Some(0.8));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_apply_on_one_node_state_does_not_reach_another_nodes_models() {
    let dir = test_dir();
    let mut first = ConfigState::load(&dir.join("first.toml")).expect("load");
    let second = ConfigState::load(&dir.join("second.toml")).expect("load");
    let own = registered_model(&first, "config-state-shared-name");
    let other = registered_model(&second, "config-state-shared-name");

    first.apply_with_live_logging(config_with_temperature("0.0"), 0);

    assert_eq!(own.snapshot().temperature, Some(0.0));
    assert_eq!(
        other.snapshot().temperature,
        Some(0.8),
        "a model another node loaded under the same id keeps its defaults"
    );

    std::fs::remove_dir_all(&dir).ok();
}
