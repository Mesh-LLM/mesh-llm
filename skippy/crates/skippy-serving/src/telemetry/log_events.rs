//! Log-based events: OTLP log records with an event name, correlated with the
//! span active when they happen, as OpenTelemetry's event conventions define
//! them. An event marks a moment: a checkpoint, a state change or an outcome
//! inside a longer operation. Work with a duration stays a span, and facts
//! about a whole operation stay span attributes.
//!
//! [`Event`] types defined here carry their name, default severity and
//! attributes together, so one name always means one structure. The point
//! events [`Telemetry::emit`] records on spans are also sent here, under
//! their existing names. Events go to the OTLP endpoint traces use and, when
//! `SKIPPY_TELEMETRY_OTLP_LOGS_FILE` names a file, to that file as OTLP/JSON
//! lines. With neither, nothing is collected.

use std::{
    collections::BTreeMap,
    env,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use serde_json::Value;
use skippy_protocol::StageConfig;

use super::{
    Telemetry, TelemetryLevel, now_unix_nanos,
    trace_context::{self, Frame},
};

mod catalogue;
mod otlp_logs;

pub(crate) use catalogue::*;
use otlp_logs::LogSinks;

/// Where events are appended, one OTLP/JSON request per line.
pub(super) const LOGS_FILE_ENV: &str = "SKIPPY_TELEMETRY_OTLP_LOGS_FILE";

/// OpenTelemetry severity numbers; semantic conventions define no severity
/// text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Severity {
    Info,
    Warn,
    Error,
}

impl Severity {
    fn number(self) -> i32 {
        match self {
            Self::Info => 9,
            Self::Warn => 13,
            Self::Error => 17,
        }
    }
}

/// An event this crate defines. The name identifies the attributes, which
/// [`Event::attributes`] builds from the type's fields.
pub(crate) trait Event {
    const NAME: &'static str;
    const SEVERITY: Severity;
    fn attributes(&self) -> BTreeMap<String, Value>;
}

struct EventLog {
    config: Arc<StageConfig>,
    sinks: LogSinks,
}

static LOG: OnceLock<EventLog> = OnceLock::new();

/// Start collecting events for the process, if there is anywhere to send
/// them. The first handle with a destination serves the whole process.
pub(super) fn start(telemetry: &Telemetry, endpoint: Option<&str>) {
    if telemetry.level == TelemetryLevel::Off || LOG.get().is_some() {
        return;
    }
    let Some(sinks) = LogSinks::open(
        env::var_os(LOGS_FILE_ENV).filter(|path| !path.is_empty()),
        endpoint,
    ) else {
        return;
    };
    let _ = LOG.set(EventLog {
        config: telemetry.config.clone(),
        sinks,
    });
}

pub(super) fn finish() {
    if let Some(log) = LOG.get() {
        log.sinks.flush();
    }
}

/// Whether events reach anything; callers skip building attributes when not.
pub(crate) fn enabled() -> bool {
    LOG.get().is_some()
}

static SERVING: AtomicBool = AtomicBool::new(false);

/// The model is loaded and requests may run: native work from here on
/// stalls serving rather than startup.
pub(crate) fn mark_serving() {
    SERVING.store(true, Ordering::Release);
}

pub(crate) fn serving() -> bool {
    SERVING.load(Ordering::Acquire)
}

/// Record `event` under the span active on this thread or task.
pub(crate) fn record<E: Event>(event: &E) {
    if enabled() {
        record_under(&trace_context::current_frame_or_root(), event);
    }
}

/// Record `event` under `frame`, which need not be active: work reported
/// after the fact, such as a GPU command, belongs to the call that caused it.
pub(crate) fn record_under<E: Event>(frame: &Frame, event: &E) {
    if let Some(log) = LOG.get() {
        log.send(frame, E::NAME, E::SEVERITY, event.attributes());
    }
}

impl EventLog {
    fn send(
        &self,
        frame: &Frame,
        name: &str,
        severity: Severity,
        attributes: BTreeMap<String, Value>,
    ) {
        let time = u64::try_from(now_unix_nanos()).unwrap_or_default();
        self.sinks.send(
            &self.config,
            otlp_logs::record(frame.context(), name, severity, attributes, time),
        );
    }
}

/// Send a point event recorded through [`Telemetry::emit`] as a log-based
/// event too, under the frame it was recorded on.
pub(super) fn record_point_event(frame: &Frame, name: &str, attributes: &BTreeMap<String, Value>) {
    if let Some(log) = LOG.get() {
        log.send(frame, name, point_event_severity(name), attributes.clone());
    }
}

/// Point events report outcomes at info, except the failures among them.
fn point_event_severity(name: &str) -> Severity {
    match name {
        "stage.scheduler_worker_panic" | "stage.connection_worker_panic" => Severity::Error,
        "stage.binary_connection_error"
        | "stage.connection_admission_rejected"
        | "stage.openai_downstream_lane_return_failed"
        | "stage.openai_linear_proposal_discard_failed" => Severity::Warn,
        _ => Severity::Info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severities_are_opentelemetry_numbers() {
        assert_eq!(Severity::Info.number(), 9);
        assert_eq!(Severity::Warn.number(), 13);
        assert_eq!(Severity::Error.number(), 17);
    }

    #[test]
    fn failures_among_point_events_are_not_info() {
        assert_eq!(
            point_event_severity("stage.scheduler_worker_panic"),
            Severity::Error
        );
        assert_eq!(
            point_event_severity("stage.binary_connection_error"),
            Severity::Warn
        );
        assert_eq!(
            point_event_severity("stage.openai_server_start"),
            Severity::Info
        );
    }
}
