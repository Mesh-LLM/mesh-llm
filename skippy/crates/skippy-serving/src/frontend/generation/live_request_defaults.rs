//! Process-wide registry of the request defaults each loaded model serves with.
//!
//! A backend resolves its configured defaults once, when the model loads. The
//! config surface promises that request defaults apply live, so the backend
//! reads them through a shared cell that [`replace_live_request_defaults`] can
//! swap while requests are in flight. A request already past its defaults
//! snapshot keeps the values it started with.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, Weak};

use super::request_defaults::EmbeddedOpenAiRequestDefaults;

type DefaultsCell = RwLock<Arc<EmbeddedOpenAiRequestDefaults>>;

static REGISTRY: Mutex<Option<HashMap<String, Vec<Weak<DefaultsCell>>>>> = Mutex::new(None);

/// The request defaults a single loaded backend reads on every request.
#[derive(Clone, Debug)]
pub(in crate::frontend) struct LiveRequestDefaults {
    cell: Arc<DefaultsCell>,
}

impl LiveRequestDefaults {
    /// Start tracking `initial` for `model_id` so later calls to
    /// [`replace_live_request_defaults`] reach this backend.
    pub(in crate::frontend) fn register(
        model_id: &str,
        initial: EmbeddedOpenAiRequestDefaults,
    ) -> Self {
        let cell = Arc::new(RwLock::new(Arc::new(initial)));
        let mut registry = REGISTRY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entries = registry
            .get_or_insert_with(HashMap::new)
            .entry(model_id.to_string())
            .or_default();
        entries.retain(|entry| entry.strong_count() > 0);
        entries.push(Arc::downgrade(&cell));
        Self { cell }
    }

    /// The defaults in force right now; hold the result for the whole request.
    pub(in crate::frontend) fn snapshot(&self) -> Arc<EmbeddedOpenAiRequestDefaults> {
        self.cell
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// Swap the request defaults of every loaded backend serving `model_id`.
///
/// Returns how many backends were updated; zero means the model is not loaded
/// (anymore), so callers can drop whatever they kept to refresh it.
pub fn replace_live_request_defaults(
    model_id: &str,
    defaults: EmbeddedOpenAiRequestDefaults,
) -> usize {
    let cells: Vec<Arc<DefaultsCell>> = {
        let mut registry = REGISTRY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entries) = registry.as_mut().and_then(|map| map.get_mut(model_id)) else {
            return 0;
        };
        entries.retain(|entry| entry.strong_count() > 0);
        let cells = entries.iter().filter_map(Weak::upgrade).collect::<Vec<_>>();
        if cells.is_empty()
            && let Some(map) = registry.as_mut()
        {
            map.remove(model_id);
        }
        cells
    };
    let shared = Arc::new(defaults);
    for cell in &cells {
        *cell
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = shared.clone();
    }
    cells.len()
}

/// A registered backend's defaults, for tests that exercise the swap from
/// outside this crate.
#[cfg(feature = "test-support")]
pub struct RegisteredRequestDefaultsProbe(LiveRequestDefaults);

#[cfg(feature = "test-support")]
impl RegisteredRequestDefaultsProbe {
    pub fn snapshot(&self) -> Arc<EmbeddedOpenAiRequestDefaults> {
        self.0.snapshot()
    }
}

/// Register `initial` for `model_id` as a loaded backend would.
#[cfg(feature = "test-support")]
pub fn register_request_defaults_probe(
    model_id: &str,
    initial: EmbeddedOpenAiRequestDefaults,
) -> RegisteredRequestDefaultsProbe {
    RegisteredRequestDefaultsProbe(LiveRequestDefaults::register(model_id, initial))
}

#[cfg(test)]
impl From<EmbeddedOpenAiRequestDefaults> for LiveRequestDefaults {
    /// A cell no `replace_live_request_defaults` call can reach, for backends
    /// that tests build directly.
    fn from(defaults: EmbeddedOpenAiRequestDefaults) -> Self {
        Self {
            cell: Arc::new(RwLock::new(Arc::new(defaults))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temperature(value: f32) -> EmbeddedOpenAiRequestDefaults {
        EmbeddedOpenAiRequestDefaults {
            temperature: Some(value),
            ..EmbeddedOpenAiRequestDefaults::default()
        }
    }

    #[test]
    fn replace_reaches_every_live_backend_of_the_model() {
        let first = LiveRequestDefaults::register("live-defaults-reach", temperature(0.8));
        let second = LiveRequestDefaults::register("live-defaults-reach", temperature(0.8));
        let other = LiveRequestDefaults::register("live-defaults-other", temperature(0.8));

        assert_eq!(
            replace_live_request_defaults("live-defaults-reach", temperature(0.0)),
            2
        );

        assert_eq!(first.snapshot().temperature, Some(0.0));
        assert_eq!(second.snapshot().temperature, Some(0.0));
        assert_eq!(other.snapshot().temperature, Some(0.8));
    }

    #[test]
    fn snapshot_taken_before_a_replace_keeps_its_values() {
        let live = LiveRequestDefaults::register("live-defaults-snapshot", temperature(0.7));
        let before = live.snapshot();

        replace_live_request_defaults("live-defaults-snapshot", temperature(0.1));

        assert_eq!(before.temperature, Some(0.7));
        assert_eq!(live.snapshot().temperature, Some(0.1));
    }

    #[test]
    fn replace_reports_zero_once_the_backend_is_dropped() {
        let live = LiveRequestDefaults::register("live-defaults-dropped", temperature(0.5));
        drop(live);

        assert_eq!(
            replace_live_request_defaults("live-defaults-dropped", temperature(0.0)),
            0
        );
        assert_eq!(
            replace_live_request_defaults("live-defaults-never-loaded", temperature(0.0)),
            0
        );
    }
}
