//! Configuration-to-service logging contract tests.

use super::*;

#[test]
fn default_policy_preserves_the_documented_summary_and_event_buffer_limits() {
    let config = mesh_llm_config::LoggingConfig::default();
    let policy = crate::logging::policy::build_policy(&config);
    let service_config = ServiceConfig::from_policy(&policy);

    assert_eq!(service_config.summary_line_limit, 2_048);
    assert_eq!(service_config.event_buffer_size, 10_000);
    assert_eq!(service_config.replay_capacity, 128);
}

#[test]
fn configured_summary_limit_bounds_the_safe_presentation_projection() {
    let service = LoggingService::new_disabled(ServiceConfig {
        summary_line_limit: 5,
        event_buffer_size: 2,
        ..ServiceConfig::default()
    });
    service
        .enqueue_event(
            RequestId::new(),
            ReplayChannel::Requests,
            serde_json::to_string(&LifecycleEvent::Failed {
                error: "Bearer secret-token-must-never-reach-presentation".into(),
                status_code: None,
            })
            .expect("lifecycle event serializes"),
        )
        .expect("enqueue stays fail-open");

    let records = service.bus_ref().drain();
    let record: serde_json::Value = serde_json::from_str(&records[0].payload).expect("bus JSON");
    let summary = record["presentation_summary"]
        .as_str()
        .expect("canonical records include a presentation summary");

    // The message body is bounded by the configured summary_line_limit...
    let body = summary
        .split(" request_id=")
        .next()
        .expect("correlation suffix is appended after the bounded body");
    assert!(
        body.chars().count() <= 5,
        "message body must be bounded to the configured limit, got {body:?}"
    );
    // ...while the correlation metadata survives the truncation so operator
    // correlation is not lost at small limits.
    assert!(
        summary.contains(" request_id="),
        "correlation request_id must survive truncation"
    );
    assert!(
        summary.contains(" event_id="),
        "correlation event_id must survive truncation"
    );
    // Secrets never reach the presentation projection.
    assert!(!summary.contains("secret-token"));
    assert!(!summary.contains("Bearer"));
}

#[test]
fn event_buffer_size_controls_replay_capacity_and_drop_oldest_eviction() {
    let small = LoggingService::new_disabled(ServiceConfig {
        event_buffer_size: 2,
        ..ServiceConfig::default()
    });
    let large = LoggingService::new_disabled(ServiceConfig {
        event_buffer_size: 3,
        ..ServiceConfig::default()
    });
    let event = serde_json::to_string(&LifecycleEvent::Admitted {
        model: None,
        method: None,
    })
    .expect("lifecycle event serializes");

    for _ in 0..3 {
        small
            .enqueue_event(RequestId::new(), ReplayChannel::Requests, event.clone())
            .expect("small buffer enqueue");
        large
            .enqueue_event(RequestId::new(), ReplayChannel::Requests, event.clone())
            .expect("large buffer enqueue");
    }

    assert_eq!(small.bus_ref().capacity(), 2);
    assert_eq!(small.bus_ref().len(), 2);
    assert_eq!(small.bus_ref().evictions.load(AtomicOrdering::Relaxed), 1);
    assert_eq!(large.bus_ref().capacity(), 3);
    assert_eq!(large.bus_ref().len(), 3);
    assert_eq!(large.bus_ref().evictions.load(AtomicOrdering::Relaxed), 0);
}

#[test]
fn event_buffer_size_bounds_initial_and_dynamic_replay_targets() {
    let service = LoggingService::new_disabled(ServiceConfig {
        event_buffer_size: 3,
        replay_capacity: 8,
        ..ServiceConfig::default()
    });

    assert_eq!(service.bus_ref().capacity(), 3);
    assert_eq!(service.dynamic_limits().replay_capacity, 3);

    service.apply_dynamic_limits(LoggingDynamicLimits {
        retention_ttl_secs: 7_200,
        replay_capacity: 9,
    });
    assert_eq!(service.bus_ref().capacity(), 3);
    assert_eq!(service.dynamic_limits().replay_capacity, 3);
}
