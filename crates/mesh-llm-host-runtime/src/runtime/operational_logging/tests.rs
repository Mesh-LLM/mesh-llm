use super::{
    ConfigDiagnosticsOutcome, ConfigOperationalEvent, DiscoveryOperationalEvent,
    LocalServingOperationalEvent, NativeSkippyOperationalEvent, RuntimeOperationalEvent,
    record_config_operational_event, record_config_operational_event_with_service,
    record_discovery_operational_event, record_discovery_operational_event_with_service,
    record_local_serving_operational_event, record_local_serving_operational_event_with_service,
    record_native_skippy_operational_event, record_native_skippy_operational_event_with_service,
    record_runtime_operational_event, record_runtime_operational_event_with_service,
};
use crate::logging::{LoggingService, ServiceConfig};
use mesh_llm_config::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSeverity, ConfigDiagnosticSource,
};

fn recorded_audits(service: &LoggingService) -> Vec<serde_json::Value> {
    service
        .bus_ref()
        .drain()
        .into_iter()
        .map(|entry| {
            let audit: serde_json::Value =
                serde_json::from_str(&entry.payload).expect("audit payload");
            serde_json::json!({
                "kind": "audit",
                "level": audit["severity"],
                "message": audit["code"],
            })
        })
        .collect()
}

#[test]
fn runtime_lifecycle_audits_are_ordered_and_static() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    let events = [
        RuntimeOperationalEvent::StartupStarted,
        RuntimeOperationalEvent::Ready,
        RuntimeOperationalEvent::ShutdownStarted,
    ];

    for event in events {
        record_runtime_operational_event_with_service(&service, event);
    }

    assert_eq!(
        recorded_audits(&service),
        vec![
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_startup_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_ready",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_shutdown_started",
            }),
        ]
    );
}

#[test]
fn model_lifecycle_success_and_unload_audits_are_ordered_and_static() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    let events = [
        RuntimeOperationalEvent::ModelLoadStarted,
        RuntimeOperationalEvent::ModelReady,
        RuntimeOperationalEvent::ModelUnloaded,
    ];

    for event in events {
        record_runtime_operational_event_with_service(&service, event);
    }

    assert_eq!(
        recorded_audits(&service),
        vec![
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_model_load_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_model_ready",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_model_unloaded",
            }),
        ]
    );
}

#[test]
fn native_skippy_lifecycle_audits_are_ordered_static_and_path_free() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    let events = [
        NativeSkippyOperationalEvent::RuntimeStartupStarted,
        NativeSkippyOperationalEvent::ModelOpenStarted,
        NativeSkippyOperationalEvent::ModelOpenFinished,
        NativeSkippyOperationalEvent::RuntimeReady,
        NativeSkippyOperationalEvent::RuntimeShutdownStarted,
    ];

    for event in events {
        record_native_skippy_operational_event_with_service(&service, event);
    }

    let audits = recorded_audits(&service);
    assert_eq!(
        audits,
        vec![
            serde_json::json!({ "kind": "audit", "level": "info", "message": "skippy_native_runtime_startup_started" }),
            serde_json::json!({ "kind": "audit", "level": "info", "message": "skippy_native_model_open_started" }),
            serde_json::json!({ "kind": "audit", "level": "info", "message": "skippy_native_model_open_finished" }),
            serde_json::json!({ "kind": "audit", "level": "info", "message": "skippy_native_runtime_ready" }),
            serde_json::json!({ "kind": "audit", "level": "info", "message": "skippy_native_runtime_shutdown_started" }),
        ]
    );

    let serialized = serde_json::to_string(&audits).expect("serialized native audits");
    for raw_value in [
        "/private/models/native-secret.gguf",
        "prompt=never-persist-this",
        "native detail: bearer private-token",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "native operational audits must not include {raw_value}"
        );
    }
}

#[test]
fn runtime_startup_failure_audit_excludes_runtime_metadata() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    record_runtime_operational_event_with_service(&service, RuntimeOperationalEvent::StartupFailed);

    let audits = recorded_audits(&service);
    assert_eq!(
        audits,
        vec![serde_json::json!({
            "kind": "audit",
            "level": "warning",
            "message": "runtime_startup_failed",
        })]
    );

    let serialized = serde_json::to_string(&audits).expect("serialized audit payloads");
    for raw_value in [
        "/private/models/private-model.gguf",
        "model=private/model:secret",
        "SIGTERM pid=12345",
        "native load error: private detail",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "runtime metadata must not enter the audit payload"
        );
    }
}

#[test]
fn model_load_failure_audit_excludes_model_metadata() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    record_runtime_operational_event_with_service(
        &service,
        RuntimeOperationalEvent::ModelLoadFailed,
    );

    let audits = recorded_audits(&service);
    assert_eq!(
        audits,
        vec![serde_json::json!({
            "kind": "audit",
            "level": "warning",
            "message": "runtime_model_load_failed",
        })]
    );

    let serialized = serde_json::to_string(&audits).expect("serialized audit payloads");
    for raw_value in [
        "/private/models/private-model.gguf",
        "model=private/model:secret",
        "Private-Mistral-7B-Instruct-Q4_K_M",
        "runtime-12345",
        "SIGTERM pid=12345",
        "native load error: private detail",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "model metadata must not enter the audit payload"
        );
    }
}

#[test]
fn config_apply_outcomes_emit_ordered_static_audits_without_config_metadata() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    let diagnostic = ConfigDiagnostic::new(
        ConfigDiagnosticCode::InvalidValue,
        ConfigDiagnosticSeverity::Error,
        ConfigDiagnosticSource::Validation,
        "webhook.url=https://secret.example/hook?token=private-token",
    );
    let events = [
        ConfigOperationalEvent::ApplyStarted,
        ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Clean),
        ConfigOperationalEvent::ApplyAccepted,
        ConfigOperationalEvent::ApplyStarted,
        ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::from_diagnostics(&[
            diagnostic,
        ])),
        ConfigOperationalEvent::ApplyRejected,
    ];

    for event in events {
        record_config_operational_event_with_service(&service, event);
    }

    let audits = recorded_audits(&service);
    assert_eq!(
        audits,
        vec![
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_config_apply_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_config_diagnostics_clean",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_config_apply_accepted",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_config_apply_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "warning",
                "message": "runtime_config_diagnostics_error",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "warning",
                "message": "runtime_config_apply_rejected",
            }),
        ]
    );

    let serialized = serde_json::to_string(&audits).expect("serialized audit payloads");
    for raw_value in [
        "webhook.url=https://secret.example/hook?token=private-token",
        "/private/mesh/config.toml",
        "Bearer private-token",
        "Private-Mistral-7B-Instruct-Q4_K_M",
        "invalid value: private detail",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "config metadata must not enter the audit payload"
        );
    }
}

#[test]
fn config_diagnostic_outcomes_are_severity_only() {
    let diagnostic = |severity| {
        ConfigDiagnostic::new(
            ConfigDiagnosticCode::InvalidValue,
            severity,
            ConfigDiagnosticSource::Validation,
            "private configuration detail",
        )
    };

    assert_eq!(
        ConfigDiagnosticsOutcome::from_diagnostics(&[]),
        ConfigDiagnosticsOutcome::Clean
    );
    assert_eq!(
        ConfigDiagnosticsOutcome::from_diagnostics(&[diagnostic(ConfigDiagnosticSeverity::Info)]),
        ConfigDiagnosticsOutcome::Info
    );
    assert_eq!(
        ConfigDiagnosticsOutcome::from_diagnostics(&[diagnostic(
            ConfigDiagnosticSeverity::Warning
        )]),
        ConfigDiagnosticsOutcome::Warning
    );
    assert_eq!(
        ConfigDiagnosticsOutcome::from_diagnostics(&[
            diagnostic(ConfigDiagnosticSeverity::Info),
            diagnostic(ConfigDiagnosticSeverity::Warning),
            diagnostic(ConfigDiagnosticSeverity::Error),
        ]),
        ConfigDiagnosticsOutcome::Error
    );
}

#[test]
fn discovery_decisions_and_join_outcomes_are_ordered_and_metadata_free() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    let events = [
        DiscoveryOperationalEvent::DecisionJoin,
        DiscoveryOperationalEvent::JoinStarted,
        DiscoveryOperationalEvent::JoinSucceeded,
        DiscoveryOperationalEvent::DecisionJoin,
        DiscoveryOperationalEvent::JoinStarted,
        DiscoveryOperationalEvent::JoinFailed,
        DiscoveryOperationalEvent::DiscoveryFailed,
        DiscoveryOperationalEvent::DecisionStartNew,
    ];

    for event in events {
        record_discovery_operational_event_with_service(&service, event);
    }

    let audits = recorded_audits(&service);
    assert_eq!(
        audits,
        vec![
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_discovery_decision_join",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_discovery_join_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_discovery_join_succeeded",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_discovery_decision_join",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_discovery_join_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "warning",
                "message": "runtime_discovery_join_failed",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "warning",
                "message": "runtime_discovery_failed",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_discovery_decision_start_new",
            }),
        ]
    );

    let serialized = serde_json::to_string(&audits).expect("serialized audit payloads");
    for raw_value in [
        "private-lab-mesh",
        "mesh-secret-bootstrap-token",
        "wss://relay.private.example",
        "peer=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "join failed: private transport detail",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "discovery metadata must not enter the audit payload"
        );
    }
}

#[test]
fn local_serving_readiness_transitions_are_ordered_and_metadata_free() {
    let service = LoggingService::new_disabled(ServiceConfig::default());
    let events = [
        LocalServingOperationalEvent::TargetAdded,
        LocalServingOperationalEvent::Ready,
        LocalServingOperationalEvent::TargetRemoved,
        LocalServingOperationalEvent::Unavailable,
    ];

    for event in events {
        record_local_serving_operational_event_with_service(&service, event);
    }

    let audits = recorded_audits(&service);
    assert_eq!(
        audits,
        vec![
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_local_target_added",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_local_serving_ready",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_local_target_removed",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "runtime_local_serving_unavailable",
            }),
        ]
    );

    let serialized = serde_json::to_string(&audits).expect("serialized audit payloads");
    for raw_value in [
        "Private-Mistral-7B-Instruct-Q4_K_M",
        "/private/models/private-model.gguf",
        "127.0.0.1:41731",
        "runtime-12345",
        "local serving error: private detail",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "local-serving metadata must not enter the audit payload"
        );
    }
}

#[test]
fn runtime_operational_vocabulary_is_bounded_and_identifier_only() {
    let events = [
        RuntimeOperationalEvent::StartupStarted,
        RuntimeOperationalEvent::StartupFailed,
        RuntimeOperationalEvent::Ready,
        RuntimeOperationalEvent::ShutdownStarted,
        RuntimeOperationalEvent::ModelLoadStarted,
        RuntimeOperationalEvent::ModelReady,
        RuntimeOperationalEvent::ModelLoadFailed,
        RuntimeOperationalEvent::ModelUnloaded,
    ];

    for event in events {
        let code = event.code();
        assert!(code.len() <= 48, "audit code must stay bounded: {code}");
        assert!(
            code.bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
            "audit code must be a static identifier: {code}"
        );
        assert!(matches!(event.level(), "info" | "warning"));
    }
}

#[test]
fn config_operational_vocabulary_is_bounded_and_identifier_only() {
    let events = [
        ConfigOperationalEvent::ApplyStarted,
        ConfigOperationalEvent::ApplyAccepted,
        ConfigOperationalEvent::ApplyRejected,
        ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Clean),
        ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Info),
        ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Warning),
        ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Error),
    ];

    for event in events {
        let code = event.code();
        assert!(code.len() <= 48, "audit code must stay bounded: {code}");
        assert!(
            code.bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
            "audit code must be a static identifier: {code}"
        );
        assert!(matches!(event.level(), "info" | "warning"));
    }
}

#[test]
fn discovery_and_local_serving_vocabularies_are_bounded_and_identifier_only() {
    let discovery_events = [
        DiscoveryOperationalEvent::DecisionJoin,
        DiscoveryOperationalEvent::DecisionStartNew,
        DiscoveryOperationalEvent::JoinStarted,
        DiscoveryOperationalEvent::JoinSucceeded,
        DiscoveryOperationalEvent::JoinFailed,
        DiscoveryOperationalEvent::DiscoveryFailed,
    ];
    let local_serving_events = [
        LocalServingOperationalEvent::TargetAdded,
        LocalServingOperationalEvent::TargetRemoved,
        LocalServingOperationalEvent::Ready,
        LocalServingOperationalEvent::Unavailable,
    ];

    for (level, code) in discovery_events
        .into_iter()
        .map(|event| (event.level(), event.code()))
        .chain(
            local_serving_events
                .into_iter()
                .map(|event| (event.level(), event.code())),
        )
    {
        assert!(code.len() <= 48, "audit code must stay bounded: {code}");
        assert!(
            code.bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
            "audit code must be a static identifier: {code}"
        );
        assert!(matches!(level, "info" | "warning"));
    }
}

#[test]
fn native_skippy_operational_vocabulary_is_bounded_and_identifier_only() {
    let events = [
        NativeSkippyOperationalEvent::RuntimeStartupStarted,
        NativeSkippyOperationalEvent::RuntimeReady,
        NativeSkippyOperationalEvent::RuntimeStartupFailed,
        NativeSkippyOperationalEvent::RuntimeShutdownStarted,
        NativeSkippyOperationalEvent::ModelOpenStarted,
        NativeSkippyOperationalEvent::ModelOpenFinished,
        NativeSkippyOperationalEvent::ModelOpenFailed,
    ];

    for event in events {
        let code = event.code();
        assert!(code.len() <= 48, "audit code must stay bounded: {code}");
        assert!(
            code.bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
            "audit code must be a static identifier: {code}"
        );
        assert!(matches!(event.level(), "info" | "warning"));
    }
}

#[tokio::test]
#[serial_test::serial]
async fn runtime_boundary_producers_are_fail_open_without_startable_logging() {
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
    record_runtime_operational_event(RuntimeOperationalEvent::StartupFailed);
    record_native_skippy_operational_event(NativeSkippyOperationalEvent::ModelOpenFailed);
    record_config_operational_event(ConfigOperationalEvent::ApplyRejected);
    record_discovery_operational_event(DiscoveryOperationalEvent::JoinFailed);
    record_local_serving_operational_event(LocalServingOperationalEvent::Unavailable);
}

#[test]
fn every_runtime_operational_event_emits_exactly_its_allowed_code() {
    let cases = [
        (
            RuntimeOperationalEvent::StartupStarted,
            "info",
            "runtime_startup_started",
        ),
        (
            RuntimeOperationalEvent::StartupFailed,
            "warning",
            "runtime_startup_failed",
        ),
        (RuntimeOperationalEvent::Ready, "info", "runtime_ready"),
        (
            RuntimeOperationalEvent::ShutdownStarted,
            "info",
            "runtime_shutdown_started",
        ),
        (
            RuntimeOperationalEvent::ModelLoadStarted,
            "info",
            "runtime_model_load_started",
        ),
        (
            RuntimeOperationalEvent::ModelReady,
            "info",
            "runtime_model_ready",
        ),
        (
            RuntimeOperationalEvent::ModelLoadFailed,
            "warning",
            "runtime_model_load_failed",
        ),
        (
            RuntimeOperationalEvent::ModelUnloaded,
            "info",
            "runtime_model_unloaded",
        ),
    ];

    for (event, level, code) in cases {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        record_runtime_operational_event_with_service(&service, event);
        assert_eq!(
            recorded_audits(&service),
            vec![serde_json::json!({
                "kind": "audit",
                "level": level,
                "message": code,
            })],
            "runtime hook {event:?} must emit exactly one {code} audit"
        );
    }
}

#[test]
fn every_native_skippy_operational_event_emits_exactly_its_allowed_code() {
    let cases = [
        (
            NativeSkippyOperationalEvent::RuntimeStartupStarted,
            "info",
            "skippy_native_runtime_startup_started",
        ),
        (
            NativeSkippyOperationalEvent::RuntimeReady,
            "info",
            "skippy_native_runtime_ready",
        ),
        (
            NativeSkippyOperationalEvent::RuntimeStartupFailed,
            "warning",
            "skippy_native_runtime_startup_failed",
        ),
        (
            NativeSkippyOperationalEvent::RuntimeShutdownStarted,
            "info",
            "skippy_native_runtime_shutdown_started",
        ),
        (
            NativeSkippyOperationalEvent::ModelOpenStarted,
            "info",
            "skippy_native_model_open_started",
        ),
        (
            NativeSkippyOperationalEvent::ModelOpenFinished,
            "info",
            "skippy_native_model_open_finished",
        ),
        (
            NativeSkippyOperationalEvent::ModelOpenFailed,
            "warning",
            "skippy_native_model_open_failed",
        ),
    ];

    for (event, level, code) in cases {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        record_native_skippy_operational_event_with_service(&service, event);
        assert_eq!(
            recorded_audits(&service),
            vec![serde_json::json!({
                "kind": "audit",
                "level": level,
                "message": code,
            })],
            "native hook {event:?} must emit exactly one {code} audit"
        );
    }
}

#[test]
fn every_config_operational_event_emits_exactly_its_allowed_code() {
    let cases = [
        (
            ConfigOperationalEvent::ApplyStarted,
            "info",
            "runtime_config_apply_started",
        ),
        (
            ConfigOperationalEvent::ApplyAccepted,
            "info",
            "runtime_config_apply_accepted",
        ),
        (
            ConfigOperationalEvent::ApplyRejected,
            "warning",
            "runtime_config_apply_rejected",
        ),
        (
            ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Clean),
            "info",
            "runtime_config_diagnostics_clean",
        ),
        (
            ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Info),
            "info",
            "runtime_config_diagnostics_info",
        ),
        (
            ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Warning),
            "warning",
            "runtime_config_diagnostics_warning",
        ),
        (
            ConfigOperationalEvent::Diagnostics(ConfigDiagnosticsOutcome::Error),
            "warning",
            "runtime_config_diagnostics_error",
        ),
    ];

    for (event, level, code) in cases {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        record_config_operational_event_with_service(&service, event);
        assert_eq!(
            recorded_audits(&service),
            vec![serde_json::json!({
                "kind": "audit",
                "level": level,
                "message": code,
            })],
            "config hook {event:?} must emit exactly one {code} audit"
        );
    }
}

#[test]
fn every_discovery_operational_event_emits_exactly_its_allowed_code() {
    let cases = [
        (
            DiscoveryOperationalEvent::DecisionJoin,
            "info",
            "runtime_discovery_decision_join",
        ),
        (
            DiscoveryOperationalEvent::DecisionStartNew,
            "info",
            "runtime_discovery_decision_start_new",
        ),
        (
            DiscoveryOperationalEvent::JoinStarted,
            "info",
            "runtime_discovery_join_started",
        ),
        (
            DiscoveryOperationalEvent::JoinSucceeded,
            "info",
            "runtime_discovery_join_succeeded",
        ),
        (
            DiscoveryOperationalEvent::JoinFailed,
            "warning",
            "runtime_discovery_join_failed",
        ),
        (
            DiscoveryOperationalEvent::DiscoveryFailed,
            "warning",
            "runtime_discovery_failed",
        ),
    ];

    for (event, level, code) in cases {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        record_discovery_operational_event_with_service(&service, event);
        assert_eq!(
            recorded_audits(&service),
            vec![serde_json::json!({
                "kind": "audit",
                "level": level,
                "message": code,
            })],
            "discovery hook {event:?} must emit exactly one {code} audit"
        );
    }
}

#[test]
fn every_local_serving_operational_event_emits_exactly_its_allowed_code() {
    let cases = [
        (
            LocalServingOperationalEvent::TargetAdded,
            "info",
            "runtime_local_target_added",
        ),
        (
            LocalServingOperationalEvent::TargetRemoved,
            "info",
            "runtime_local_target_removed",
        ),
        (
            LocalServingOperationalEvent::Ready,
            "info",
            "runtime_local_serving_ready",
        ),
        (
            LocalServingOperationalEvent::Unavailable,
            "info",
            "runtime_local_serving_unavailable",
        ),
    ];

    for (event, level, code) in cases {
        let service = LoggingService::new_disabled(ServiceConfig::default());
        record_local_serving_operational_event_with_service(&service, event);
        assert_eq!(
            recorded_audits(&service),
            vec![serde_json::json!({
                "kind": "audit",
                "level": level,
                "message": code,
            })],
            "local-serving hook {event:?} must emit exactly one {code} audit"
        );
    }
}
