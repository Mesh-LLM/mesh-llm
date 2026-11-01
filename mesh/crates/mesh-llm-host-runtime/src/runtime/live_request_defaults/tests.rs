use super::*;
use std::cell::RefCell;

fn config(toml_text: &str) -> MeshConfig {
    toml::from_str(toml_text).expect("test config should parse")
}

/// Run an apply that records what it would swap into each backend.
fn recorded_apply(
    registry: &Mutex<Vec<TrackedModel>>,
    config: &MeshConfig,
    loaded: &[&str],
) -> (bool, Vec<(String, Option<f32>)>) {
    let calls = RefCell::new(Vec::new());
    let applied = apply_with(registry, config, |model, defaults| {
        calls
            .borrow_mut()
            .push((model.to_string(), defaults.temperature));
        usize::from(loaded.contains(&model))
    });
    (applied, calls.into_inner())
}

#[test]
fn global_request_default_reaches_a_loaded_model() {
    let registry = Mutex::new(Vec::new());
    track_model_in(&registry, "live-defaults-global", None, None);

    let (applied, calls) = recorded_apply(
        &registry,
        &config("version = 1\n[defaults.request_defaults]\ntemperature = 0.0\n"),
        &["live-defaults-global"],
    );

    assert!(applied);
    assert!(calls.contains(&("live-defaults-global".to_string(), Some(0.0))));
}

#[test]
fn model_entry_overrides_the_global_request_default() {
    let registry = Mutex::new(Vec::new());
    track_model_in(&registry, "live-defaults-entry", Some("entry-model"), None);
    track_model_in(&registry, "live-defaults-bare", Some("other-model"), None);

    let (_, calls) = recorded_apply(
        &registry,
        &config(
            r#"
version = 1
[defaults.request_defaults]
temperature = 0.0

[[models]]
model = "entry-model"

[models.request_defaults]
temperature = 0.3
"#,
        ),
        &["live-defaults-entry", "live-defaults-bare"],
    );

    assert!(calls.contains(&("live-defaults-entry".to_string(), Some(0.3))));
    assert!(calls.contains(&("live-defaults-bare".to_string(), Some(0.0))));
}

#[test]
fn removing_the_config_default_clears_it_for_loaded_models() {
    let registry = Mutex::new(Vec::new());
    track_model_in(&registry, "live-defaults-cleared", None, None);

    let (_, calls) = recorded_apply(
        &registry,
        &config("version = 1\n"),
        &["live-defaults-cleared"],
    );

    assert!(calls.contains(&("live-defaults-cleared".to_string(), None)));
}

#[test]
fn unloaded_models_are_forgotten() {
    let registry = Mutex::new(Vec::new());
    track_model_in(&registry, "live-defaults-unloaded", None, None);
    let global = config("version = 1\n[defaults.request_defaults]\ntemperature = 0.0\n");

    let (applied, first) = recorded_apply(&registry, &global, &[]);
    let (_, second) = recorded_apply(&registry, &global, &[]);

    assert!(applied);
    assert!(
        first
            .iter()
            .any(|(model, _)| model == "live-defaults-unloaded")
    );
    assert!(
        second
            .iter()
            .all(|(model, _)| model != "live-defaults-unloaded"),
        "a model with no running backend must not be resolved again"
    );
}

#[test]
fn unresolvable_defaults_keep_the_model_on_its_current_defaults() {
    let registry = Mutex::new(Vec::new());
    track_model_in(&registry, "live-defaults-unresolvable", None, None);
    let broken = config(
        "version = 1\n[defaults.request_defaults]\nchat_template_file = \"/nonexistent/live-defaults.jinja\"\n",
    );

    let (applied, calls) = recorded_apply(&registry, &broken, &["live-defaults-unresolvable"]);

    assert!(!applied);
    assert!(
        calls
            .iter()
            .all(|(model, _)| model != "live-defaults-unresolvable")
    );
    let (applied_again, calls) = recorded_apply(
        &registry,
        &config("version = 1\n"),
        &["live-defaults-unresolvable"],
    );
    assert!(applied_again);
    assert!(
        calls
            .iter()
            .any(|(model, _)| model == "live-defaults-unresolvable"),
        "the model stays tracked so a later valid config still reaches it"
    );
}

#[test]
fn request_defaults_change_detection_ignores_unrelated_settings() {
    let base = config("version = 1\n[defaults.request_defaults]\ntemperature = 0.0\n");
    let unrelated = config(
        "version = 1\n[gpu]\nassignment = \"auto\"\n[defaults.request_defaults]\ntemperature = 0.0\n",
    );
    let global_changed = config("version = 1\n[defaults.request_defaults]\ntemperature = 0.5\n");
    let model_changed = config(
        "version = 1\n[defaults.request_defaults]\ntemperature = 0.0\n[[models]]\nmodel = \"m\"\n[models.request_defaults]\ntop_p = 0.9\n",
    );

    assert!(!request_defaults_changed(&base, &unrelated));
    assert!(request_defaults_changed(&base, &global_changed));
    assert!(request_defaults_changed(&base, &model_changed));
}

#[test]
fn adding_a_model_entry_without_request_defaults_is_not_a_change() {
    let base = config("version = 1\n");
    let with_model = config("version = 1\n[[models]]\nmodel = \"m\"\n");

    assert!(!request_defaults_changed(&base, &with_model));
}
