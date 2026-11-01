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
    /// The temperature each loaded backend is serving with right now.
    serving: RefCell<std::collections::HashMap<String, Option<f32>>>,
}

impl FakeBackends {
    fn new(loaded: &[&'static str]) -> Self {
        Self {
            loaded: loaded.to_vec(),
            calls: RefCell::new(Vec::new()),
            serving: RefCell::new(
                loaded
                    .iter()
                    .map(|id| (id.to_string(), Some(0.8)))
                    .collect(),
            ),
        }
    }

    fn serving_temperature(&self, model: &str) -> Option<Option<f32>> {
        self.serving.borrow().get(model).copied()
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

impl RequestDefaultsBackends for FakeBackends {
    type Previous = (String, Option<f32>);

    fn replace(&self, model: &str, defaults: EmbeddedOpenAiRequestDefaults) -> usize {
        self.calls
            .borrow_mut()
            .push((model.to_string(), defaults.temperature));
        if self.loaded.contains(&model) {
            self.serving
                .borrow_mut()
                .insert(model.to_string(), defaults.temperature);
        }
        usize::from(self.loaded.contains(&model))
    }

    fn capture(&self, model: &str) -> Self::Previous {
        (
            model.to_string(),
            self.serving.borrow().get(model).copied().flatten(),
        )
    }

    fn restore(&self, (model, temperature): Self::Previous) {
        if self.loaded.contains(&model.as_str()) {
            self.serving.borrow_mut().insert(model, temperature);
        }
    }
}

#[test]
fn global_request_default_reaches_a_loaded_model() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["a"]);
    live.track_model("a", None, None);

    assert!(live.apply(&temperature("0.0"), &backends).all_applied);

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
        &backends,
    );

    assert_eq!(backends.last_for("entry"), Some(Some(0.3)));
    assert_eq!(backends.last_for("bare"), Some(Some(0.0)));
}

#[test]
fn removing_the_config_default_clears_it_for_loaded_models() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["a"]);
    live.track_model("a", None, None);

    live.apply(&config("version = 1\n"), &backends);

    assert_eq!(backends.last_for("a"), Some(None));
}

#[test]
fn a_model_that_was_serving_and_unloaded_is_forgotten() {
    let live = LiveRequestDefaults::new();
    live.track_model("a", None, None);
    let serving = FakeBackends::new(&["a"]);
    live.apply(&temperature("0.0"), &serving);

    let unloaded = FakeBackends::new(&[]);
    live.apply(&temperature("0.5"), &unloaded);
    live.apply(&temperature("0.7"), &unloaded);

    assert_eq!(
        unloaded.calls_for("a"),
        1,
        "once its backend is gone the model is not resolved again"
    );
}

#[test]
fn an_apply_during_load_does_not_drop_the_model_and_the_load_catches_up() {
    let live = LiveRequestDefaults::new();
    let generation = live.track_model("a", None, None);
    let still_loading = FakeBackends::new(&[]);

    live.apply(&temperature("0.0"), &still_loading);
    let after_load = FakeBackends::new(&["a"]);
    live.sync_after_load("a", generation, &after_load);

    assert_eq!(
        after_load.last_for("a"),
        Some(Some(0.0)),
        "the backend registered with load-time defaults, so the load must sync to the last apply"
    );
}

#[test]
fn a_load_from_a_stale_config_copy_syncs_to_the_last_applied_config() {
    let live = LiveRequestDefaults::new();
    live.apply(&temperature("0.0"), &FakeBackends::new(&[]));
    let generation = live.track_model("late", None, None);
    let backends = FakeBackends::new(&["late"]);

    live.sync_after_load("late", generation, &backends);

    assert_eq!(backends.last_for("late"), Some(Some(0.0)));
}

#[test]
fn sync_does_nothing_before_any_apply() {
    let live = LiveRequestDefaults::new();
    let generation = live.track_model("a", None, None);
    let backends = FakeBackends::new(&["a"]);

    live.sync_after_load("a", generation, &backends);

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

    assert!(!live.apply(&broken, &backends).all_applied);
    assert_eq!(backends.calls_for("a"), 0);

    assert!(live.apply(&config("version = 1\n"), &backends).all_applied);
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

    let applied = live.apply(&temperature("1000.0"), &backends);

    assert!(!applied.all_applied);
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

/// Backends whose swap is interrupted by the model reloading: the reload
/// tracks a fresh entry before the apply prunes.
struct ReloadingBackends<'a> {
    live: &'a LiveRequestDefaults,
    generation: &'a std::cell::Cell<u64>,
    inner: &'a FakeBackends,
}

impl RequestDefaultsBackends for ReloadingBackends<'_> {
    type Previous = <FakeBackends as RequestDefaultsBackends>::Previous;

    fn replace(&self, model: &str, defaults: EmbeddedOpenAiRequestDefaults) -> usize {
        self.generation.set(self.live.track_model("a", None, None));
        self.inner.replace(model, defaults)
    }

    fn capture(&self, model: &str) -> Self::Previous {
        self.inner.capture(model)
    }

    fn restore(&self, previous: Self::Previous) {
        self.inner.restore(previous);
    }
}

#[test]
fn a_reload_tracked_during_an_apply_is_not_pruned_with_the_old_backend() {
    let live = LiveRequestDefaults::new();
    live.track_model("a", None, None);
    let serving = FakeBackends::new(&["a"]);
    live.apply(&temperature("0.0"), &serving);

    // The model reloads while an apply is mid-flight: its old backend is gone
    // and the reload tracks a fresh entry before the apply prunes.
    let reloaded = FakeBackends::new(&[]);
    let reload_generation = std::cell::Cell::new(0);
    let reloading = ReloadingBackends {
        live: &live,
        generation: &reload_generation,
        inner: &reloaded,
    };
    live.apply(&temperature("0.5"), &reloading);
    let after_load = FakeBackends::new(&["a"]);
    live.sync_after_load("a", reload_generation.get(), &after_load);

    assert_eq!(after_load.last_for("a"), Some(Some(0.5)));
}

#[test]
fn an_older_load_does_not_sync_or_register_a_newer_loads_entry() {
    let live = LiveRequestDefaults::new();
    live.apply(&temperature("0.0"), &FakeBackends::new(&[]));
    let older = live.track_model("a", None, None);
    let newer = live.track_model("a", None, None);
    let backends = FakeBackends::new(&["a"]);

    live.sync_after_load("a", older, &backends);

    assert_eq!(
        backends.calls_for("a"),
        0,
        "the older load's generation is no longer tracked"
    );
    assert!(
        live.snapshot(|entry| entry.generation == newer && !entry.registered)
            .len()
            == 1,
        "the newer entry stays unregistered until its own load syncs"
    );
    live.sync_after_load("a", newer, &backends);
    assert_eq!(backends.last_for("a"), Some(Some(0.0)));
}

#[test]
fn one_unresolvable_model_leaves_every_backend_and_the_last_config_alone() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["good", "broken"]);
    live.track_model("good", Some("good-model"), None);
    live.track_model("broken", Some("broken-model"), None);
    live.apply(&temperature("0.2"), &backends);
    let before = backends.calls.borrow().len();
    let config = config(
        r#"
version = 1
[defaults.request_defaults]
temperature = 0.0

[[models]]
model = "broken-model"

[models.request_defaults]
chat_template_file = "/nonexistent/live-defaults.jinja"
"#,
    );

    let applied = live.apply(&config, &backends);

    assert!(!applied.all_applied);
    assert_eq!(
        backends.calls.borrow().len(),
        before,
        "no backend is swapped when any model's defaults fail to resolve"
    );
    let late = FakeBackends::new(&["late"]);
    let generation = live.track_model("late", None, None);
    live.sync_after_load("late", generation, &late);
    assert_eq!(
        late.last_for("late"),
        Some(Some(0.2)),
        "a later load syncs to the last config that was actually applied"
    );
}

#[test]
fn undo_puts_back_what_the_backends_were_serving_and_the_last_config() {
    let live = LiveRequestDefaults::new();
    let backends = FakeBackends::new(&["a"]);
    live.track_model("a", None, None);
    live.apply(&temperature("0.2"), &backends);
    // The backend serves 0.2 now, whatever its saved config would resolve to.
    let applied = live.apply(&temperature("0.0"), &backends);
    assert_eq!(backends.serving_temperature("a"), Some(Some(0.0)));

    live.undo(applied, &backends);

    assert_eq!(backends.serving_temperature("a"), Some(Some(0.2)));
    let late = FakeBackends::new(&["late"]);
    let generation = live.track_model("late", None, None);
    live.sync_after_load("late", generation, &late);
    assert_eq!(late.last_for("late"), Some(Some(0.2)));
}

#[test]
fn one_owner_never_inherits_the_config_another_owner_applied() {
    let first = LiveRequestDefaults::new();
    let second = LiveRequestDefaults::new();
    assert_ne!(first.scope(), second.scope());
    first.apply(&temperature("0.0"), &FakeBackends::new(&[]));
    let backends = FakeBackends::new(&["a"]);
    let generation = second.track_model("a", None, None);

    second.sync_after_load("a", generation, &backends);

    assert_eq!(
        backends.calls_for("a"),
        0,
        "a node that starts later keeps the defaults its own config resolved"
    );
}
