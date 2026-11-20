//! Host-side consumers for the model-open observations of one native load.
//!
//! Skippy owns the native event queue and its drain thread. While a model
//! open blocks, that thread hands each native record to the host's
//! `NativeModelOpenEventReporter`, so formatting, output events and
//! runtime-event ingress all run off the native thread. After the open
//! returns and Skippy has drained the queue a last time, its summary is
//! handed to the caller's reconciler together with the authoritative load
//! outcome. Skippy joins the thread before the load returns, so the
//! reconciler has run by the time the caller settles the load. The
//! reconciler reports losses and contradictions; the load result itself
//! passes through untouched.
//!
//! No summary arrives when the load unwinds or when Skippy cannot start its
//! drain thread; the reconciler then does not run.

mod observation;

use skippy_api::serving::{ModelOpenEvents, ModelOpenObservation as SkippyObservation};

pub(crate) use observation::{ModelOpenObservation, ModelOpenReturn};

use super::NativeModelOpenEventReporter;

/// Settles what the host saw of a model open against its native return.
pub(crate) type ModelOpenReconciler =
    Box<dyn FnOnce(&ModelOpenObservation, ModelOpenReturn) + Send>;

/// Host consumers for one native model open: the per-event sink and the
/// post-return reconciler.
pub(crate) struct NativeModelOpenEvents {
    pub(crate) reporter: NativeModelOpenEventReporter,
    pub(crate) reconcile: ModelOpenReconciler,
}

/// The load request's event option for `events`. Without host consumers the
/// event-capable load path is still used, with no queue or drain thread.
pub(crate) fn model_open_events(events: Option<NativeModelOpenEvents>) -> ModelOpenEvents {
    let Some(events) = events else {
        return ModelOpenEvents::Enabled(None);
    };
    let mut observer = ModelOpenObserver::new(events);
    ModelOpenEvents::observe(move |observation| match observation {
        SkippyObservation::Event(event) => observer.event(*event.source),
        SkippyObservation::Summary(summary) => observer.summary(
            &ModelOpenObservation::of(&summary),
            ModelOpenReturn::of(summary.outcome),
        ),
        // Observations added by a later Skippy are not reported.
        _ => {}
    })
}

/// The host consumers behind one load's observer.
struct ModelOpenObserver {
    reporter: NativeModelOpenEventReporter,
    reconcile: Option<ModelOpenReconciler>,
}

impl ModelOpenObserver {
    fn new(events: NativeModelOpenEvents) -> Self {
        Self {
            reporter: events.reporter,
            reconcile: Some(events.reconcile),
        }
    }

    /// A panicking reporter loses its event, not the rest of the stream:
    /// Skippy stops delivery, the summary included, when the observer
    /// itself panics.
    fn event(&mut self, event: skippy_runtime::RuntimeEvent) {
        let reporter = &mut self.reporter;
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reporter(event))).is_err() {
            tracing::warn!("native model-open event reporter panicked; event skipped");
        }
    }

    fn summary(&mut self, observation: &ModelOpenObservation, returned: ModelOpenReturn) {
        if let Some(reconcile) = self.reconcile.take() {
            reconcile(observation, returned);
        }
    }
}

#[cfg(test)]
mod tests;
