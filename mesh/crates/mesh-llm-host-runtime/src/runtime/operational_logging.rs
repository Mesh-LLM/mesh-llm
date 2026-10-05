//! Bounded operational audit vocabulary for runtime, model, configuration,
//! discovery, and local-serving boundaries.
//!
//! Most events carry only a static level and code. Existing model lifecycle
//! owners may add the shared, sanitized typed context; arbitrary errors,
//! configuration values, paths, endpoints, and process metadata remain out of
//! scope.

#[cfg(test)]
use crate::logging::LoggingService;
use crate::logging::{OperationalAuditContext, OperationalAuditRecord, OperationalAuditSeverity};
use mesh_llm_config::{ConfigDiagnostic, ConfigDiagnosticSeverity};

const OPERATIONAL_AUDIT_INFO: &str = "info";
const OPERATIONAL_AUDIT_WARNING: &str = "warning";

const OPERATIONAL_AUDIT_SOURCE: &str = "runtime";

fn operational_audit_record(code: &'static str, level: &'static str) -> OperationalAuditRecord {
    let severity = match level {
        OPERATIONAL_AUDIT_INFO => OperationalAuditSeverity::Info,
        OPERATIONAL_AUDIT_WARNING => OperationalAuditSeverity::Warning,
        _ => OperationalAuditSeverity::Error,
    };
    OperationalAuditRecord::builder(OPERATIONAL_AUDIT_SOURCE, code)
        .severity(severity)
        .build()
}

/// Static runtime and model lifecycle outcomes that are safe to publish locally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeOperationalEvent {
    StartupStarted,
    StartupFailed,
    Ready,
    ShutdownStarted,
    ShutdownCompleted,
    ModelLoadStarted,
    ModelReady,
    ModelLoadFailed,
    ModelUnloadStarted,
    ModelUnloadFailed,
    ModelUnloaded,
    ModelExited,
}

/// Static native Skippy runtime transitions. These deliberately identify the
/// embedded native layer rather than re-emitting the host model lifecycle.
/// They carry no model reference, native detail, path, endpoint, or error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeSkippyOperationalEvent {
    RuntimeStartupStarted,
    RuntimeReady,
    RuntimeStartupFailed,
    RuntimeShutdownStarted,
    ModelOpenStarted,
    ModelOpenFinished,
    ModelOpenFailed,
}

impl NativeSkippyOperationalEvent {
    const fn level(self) -> &'static str {
        match self {
            Self::RuntimeStartupFailed | Self::ModelOpenFailed => OPERATIONAL_AUDIT_WARNING,
            Self::RuntimeStartupStarted
            | Self::RuntimeReady
            | Self::RuntimeShutdownStarted
            | Self::ModelOpenStarted
            | Self::ModelOpenFinished => OPERATIONAL_AUDIT_INFO,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::RuntimeStartupStarted => "skippy_native_runtime_startup_started",
            Self::RuntimeReady => "skippy_native_runtime_ready",
            Self::RuntimeStartupFailed => "skippy_native_runtime_startup_failed",
            Self::RuntimeShutdownStarted => "skippy_native_runtime_shutdown_started",
            Self::ModelOpenStarted => "skippy_native_model_open_started",
            Self::ModelOpenFinished => "skippy_native_model_open_finished",
            Self::ModelOpenFailed => "skippy_native_model_open_failed",
        }
    }
}

impl RuntimeOperationalEvent {
    const fn level(self) -> &'static str {
        match self {
            Self::StartupStarted
            | Self::Ready
            | Self::ShutdownStarted
            | Self::ShutdownCompleted
            | Self::ModelLoadStarted
            | Self::ModelReady
            | Self::ModelUnloadStarted
            | Self::ModelUnloaded => OPERATIONAL_AUDIT_INFO,
            Self::StartupFailed
            | Self::ModelLoadFailed
            | Self::ModelUnloadFailed
            | Self::ModelExited => OPERATIONAL_AUDIT_WARNING,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::StartupStarted => "runtime_startup_started",
            Self::StartupFailed => "runtime_startup_failed",
            Self::Ready => "runtime_ready",
            Self::ShutdownStarted => "runtime_shutdown_started",
            Self::ShutdownCompleted => "runtime_shutdown_completed",
            Self::ModelLoadStarted => "runtime_model_load_started",
            Self::ModelReady => "runtime_model_ready",
            Self::ModelLoadFailed => "runtime_model_load_failed",
            Self::ModelUnloadStarted => "runtime_model_unload_started",
            Self::ModelUnloadFailed => "runtime_model_unload_failed",
            Self::ModelUnloaded => "runtime_model_unloaded",
            Self::ModelExited => "runtime_model_exited",
        }
    }
}

/// Sanitized aggregate of configuration diagnostics. Only severity is
/// considered; diagnostic text, paths, sources, and codes never leave the
/// configuration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConfigDiagnosticsOutcome {
    Clean,
    Info,
    Warning,
    Error,
}

impl ConfigDiagnosticsOutcome {
    pub(crate) fn from_diagnostics(diagnostics: &[ConfigDiagnostic]) -> Self {
        if diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        {
            return Self::Error;
        }
        if diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Warning)
        {
            return Self::Warning;
        }
        if diagnostics.is_empty() {
            Self::Clean
        } else {
            Self::Info
        }
    }

    const fn level(self) -> &'static str {
        match self {
            Self::Clean | Self::Info => OPERATIONAL_AUDIT_INFO,
            Self::Warning | Self::Error => OPERATIONAL_AUDIT_WARNING,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::Clean => "runtime_config_diagnostics_clean",
            Self::Info => "runtime_config_diagnostics_info",
            Self::Warning => "runtime_config_diagnostics_warning",
            Self::Error => "runtime_config_diagnostics_error",
        }
    }
}

/// Static configuration apply outcomes that are safe to publish locally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConfigOperationalEvent {
    ApplyStarted,
    ApplyAccepted,
    ApplyRejected,
    Diagnostics(ConfigDiagnosticsOutcome),
}

impl ConfigOperationalEvent {
    const fn level(self) -> &'static str {
        match self {
            Self::ApplyStarted | Self::ApplyAccepted => OPERATIONAL_AUDIT_INFO,
            Self::ApplyRejected => OPERATIONAL_AUDIT_WARNING,
            Self::Diagnostics(outcome) => outcome.level(),
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::ApplyStarted => "runtime_config_apply_started",
            Self::ApplyAccepted => "runtime_config_apply_accepted",
            Self::ApplyRejected => "runtime_config_apply_rejected",
            Self::Diagnostics(outcome) => outcome.code(),
        }
    }
}

/// Static discovery decisions and join outcomes that are safe to publish
/// locally. They deliberately do not distinguish discovery sources, meshes,
/// tokens, peers, or errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DiscoveryOperationalEvent {
    DecisionJoin,
    DecisionStartNew,
    JoinStarted,
    JoinSucceeded,
    JoinFailed,
    DiscoveryFailed,
}

impl DiscoveryOperationalEvent {
    const fn level(self) -> &'static str {
        match self {
            Self::DecisionJoin
            | Self::DecisionStartNew
            | Self::JoinStarted
            | Self::JoinSucceeded => OPERATIONAL_AUDIT_INFO,
            Self::JoinFailed | Self::DiscoveryFailed => OPERATIONAL_AUDIT_WARNING,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::DecisionJoin => "runtime_discovery_decision_join",
            Self::DecisionStartNew => "runtime_discovery_decision_start_new",
            Self::JoinStarted => "runtime_discovery_join_started",
            Self::JoinSucceeded => "runtime_discovery_join_succeeded",
            Self::JoinFailed => "runtime_discovery_join_failed",
            Self::DiscoveryFailed => "runtime_discovery_failed",
        }
    }
}

/// Static local-serving state transitions that are safe to publish locally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LocalServingOperationalEvent {
    TargetAdded,
    TargetRemoved,
    Ready,
    Unavailable,
}

impl LocalServingOperationalEvent {
    const fn level(self) -> &'static str {
        OPERATIONAL_AUDIT_INFO
    }

    const fn code(self) -> &'static str {
        match self {
            Self::TargetAdded => "runtime_local_target_added",
            Self::TargetRemoved => "runtime_local_target_removed",
            Self::Ready => "runtime_local_serving_ready",
            Self::Unavailable => "runtime_local_serving_unavailable",
        }
    }
}

/// Record one runtime lifecycle result through the process-local logging state.
/// Logging is optional and intentionally never affects startup, readiness, or
/// shutdown progress.
pub(crate) fn record_runtime_operational_event(event: RuntimeOperationalEvent) {
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

/// Record an existing runtime lifecycle boundary with the shared typed
/// correlation context. Context is sanitized and bounded before replay.
pub(crate) fn record_runtime_operational_event_with_context(
    event: RuntimeOperationalEvent,
    context: OperationalAuditContext,
) {
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let record = operational_audit_record(event.code(), event.level()).with_context(context);
    let _ = state.write_operational_audit(record);
}

/// Record a native Skippy lifecycle transition through the same bounded,
/// fail-open operational audit seam as other runtime boundaries.
pub(crate) fn record_native_skippy_operational_event(event: NativeSkippyOperationalEvent) {
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

/// Record one configuration boundary result through the process-local logging
/// state. Logging is optional and intentionally never affects config apply
/// behavior.
pub(crate) fn record_config_operational_event(event: ConfigOperationalEvent) {
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

/// Record one discovery boundary result through the process-local logging
/// state. Logging is optional and intentionally never affects discovery or
/// joining behavior.
pub(crate) fn record_discovery_operational_event(event: DiscoveryOperationalEvent) {
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

/// Record one local-serving state transition through the process-local logging
/// state. Logging is optional and intentionally never affects routing or
/// readiness behavior.
pub(crate) fn record_local_serving_operational_event(event: LocalServingOperationalEvent) {
    let Some(state) = crate::logging_runtime_state() else {
        return;
    };
    let _ = state.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
fn record_runtime_operational_event_with_service(
    service: &LoggingService,
    event: RuntimeOperationalEvent,
) {
    let _ = service.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
fn record_native_skippy_operational_event_with_service(
    service: &LoggingService,
    event: NativeSkippyOperationalEvent,
) {
    let _ = service.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
fn record_config_operational_event_with_service(
    service: &LoggingService,
    event: ConfigOperationalEvent,
) {
    let _ = service.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
fn record_discovery_operational_event_with_service(
    service: &LoggingService,
    event: DiscoveryOperationalEvent,
) {
    let _ = service.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
fn record_local_serving_operational_event_with_service(
    service: &LoggingService,
    event: LocalServingOperationalEvent,
) {
    let _ = service.write_operational_audit(operational_audit_record(event.code(), event.level()));
}

#[cfg(test)]
mod tests;
