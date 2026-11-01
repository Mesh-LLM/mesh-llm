use super::*;
use std::cell::RefCell;

fn config(toml_text: &str) -> MeshConfig {
    toml::from_str(toml_text).expect("test config should parse")
}

fn temperature(value: &str) -> MeshConfig {
    config(&format!(
        "version = 1\n[defaults.request_defaults]\ntemperature = {value}\n"
    ))
}

/// A set of backends that "are loaded", recording what each swap carried.
struct FakeBackends {
    loaded: Vec<&'static str>,
    calls: RefCell<Vec<(String, Option<f32>)>>,
}

impl FakeBackends {
    fn new(loaded: &[&'static str]) -> Self {
        Self {
            loaded: loaded.to_vec(),
            calls: RefCell::new(Vec::new()),
        }
    }

    fn replace(&self, model: &str, defaults: EmbeddedOpenAiRequestDefaults) -> usize {
        self.calls
            .borrow_mut()
            .push((model.to_string(), defaults.temperature));
        usize::from(self.loaded.contains(&model))
    }

    fn last_for(&self, model: &str) -> Option<Option<f32>> {
        self.calls
            .borrow()
            .iter()
            .rev()
            .find(|(id, _)| id == model)
            .map(|(_, value)| *value)
    }

    fn calls_for(&self, model: &str) -> usize {
        self.calls
            .borrow()
            .iter()
            .filter(|(id, _)| id == model)
            .count()
    }
}

#[test]
fn global_request_default_reaches_a_loaded_model() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["a"]);
    live.track_model("a", None, None);

    assert!(live.apply(&temperature("0.0"), |m, d| backends.replace(m, d)));

    assert_eq!(backends.last_for("a"), Some(Some(0.0)));
}

#[test]
fn model_entry_overrides_the_global_request_default() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["entry", "bare"]);
    live.track_model("entry", Some("entry-model"), None);
    live.track_model("bare", Some("other-model"), None);

    live.apply(
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
        |m, d| backends.replace(m, d),
    );

    assert_eq!(backends.last_for("entry"), Some(Some(0.3)));
    assert_eq!(backends.last_for("bare"), Some(Some(0.0)));
}

#[test]
fn removing_the_config_default_clears_it_for_loaded_models() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["a"]);
    live.track_model("a", None, None);

    live.apply(&config("version = 1\n"), |m, d| backends.replace(m, d));

    assert_eq!(backends.last_for("a"), Some(None));
}

#[test]
fn a_model_that_was_serving_and_unloaded_is_forgotten() {
    let live = LiveRequestDefaults::new();
    live.track_model("a", None, None);
    let serving = FakeBackends::new(&["a"]);
    live.apply(&temperature("0.0"), |m, d| serving.replace(m, d));

    let unloaded = FakeBackends::new(&[]);
    live.apply(&temperature("0.5"), |m, d| unloaded.replace(m, d));
    live.apply(&temperature("0.7"), |m, d| unloaded.replace(m, d));

    assert_eq!(
        unloaded.calls_for("a"),
        1,
        "once its backend is gone the model is not resolved again"
    );
}

#[test]
fn an_apply_during_load_does_not_drop_the_model_and_the_load_catches_up() {
    let live = LiveRequestDefaults::new();
    live.track_model("a", None, None);
    let still_loading = FakeBackends::new(&[]);

    live.apply(&temperature("0.0"), |m, d| still_loading.replace(m, d));
    let after_load = FakeBackends::new(&["a"]);
    live.sync_after_load("a", |m, d| after_load.replace(m, d));

    assert_eq!(
        after_load.last_for("a"),
        Some(Some(0.0)),
        "the backend registered with load-time defaults, so the load must sync to the last apply"
    );
}

#[test]
fn a_load_from_a_stale_config_copy_syncs_to_the_last_applied_config() {
    let live = LiveRequestDefaults::new();
    live.apply(&temperature("0.0"), |_, _| 0);
    live.track_model("late", None, None);
    let backends = FakeBackends::new(&["late"]);

    live.sync_after_load("late", |m, d| backends.replace(m, d));

    assert_eq!(backends.last_for("late"), Some(Some(0.0)));
}

#[test]
fn sync_does_nothing_before_any_apply() {
    let live = LiveRequestDefaults::new();
    live.track_model("a", None, None);
    let backends = FakeBackends::new(&["a"]);

    live.sync_after_load("a", |m, d| backends.replace(m, d));

    assert_eq!(backends.calls_for("a"), 0);
}

#[test]
fn unresolvable_defaults_keep_the_model_on_its_current_defaults() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["a"]);
    live.track_model("a", None, None);
    let broken = config(
        "version = 1\n[defaults.request_defaults]\nchat_template_file = \"/nonexistent/live-defaults.jinja\"\n",
    );

    assert!(!live.apply(&broken, |m, d| backends.replace(m, d)));
    assert_eq!(backends.calls_for("a"), 0);

    assert!(live.apply(&config("version = 1\n"), |m, d| backends.replace(m, d)));
    assert_eq!(
        backends.last_for("a"),
        Some(None),
        "the model stays tracked so a later valid config still reaches it"
    );
}

#[test]
fn defaults_the_serving_path_would_reject_are_not_swapped_in() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["a"]);
    live.track_model("a", None, None);

    let applied = live.apply(&temperature("1000.0"), |m, d| backends.replace(m, d));

    assert!(!applied);
    assert_eq!(backends.calls_for("a"), 0);
}

#[test]
fn request_defaults_change_detection_ignores_unrelated_settings() {
    let base = temperature("0.0");
    let unrelated = config(
        "version = 1\n[gpu]\nassignment = \"auto\"\n[defaults.request_defaults]\ntemperature = 0.0\n",
    );
    let global_changed = temperature("0.5");
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
