use super::*;
use std::sync::{Arc, Barrier};
use std::thread;

#[test]
fn routing_metric_groups_declare_explicit_layer_and_scope() {
    let mut names = ROUTING_METRIC_GROUPS
        .iter()
        .map(|group| group.name)
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "mesh_models[].routing_metrics",
            "mesh_models[].routing_metrics.targets[]",
            "routing_metrics",
            "routing_metrics.local_node",
            "routing_metrics.pressure",
        ]
    );
    assert!(
        ROUTING_METRIC_GROUPS
            .iter()
            .all(|group| !group.description.is_empty())
    );
    assert!(
        ROUTING_METRIC_GROUPS
            .iter()
            .all(|group| !group.api_surface.is_empty())
    );
    assert!(ROUTING_METRIC_GROUPS.iter().all(|group| matches!(
        group.layer,
        MetricLayer::Runtime | MetricLayer::Information | MetricLayer::Strategy
    )));
    assert!(ROUTING_METRIC_GROUPS.iter().all(|group| matches!(
        group.scope,
        MetricScope::LocalOnly | MetricScope::PeerAdvertised | MetricScope::MeshDerived
    )));
    assert!(
        ROUTING_METRIC_GROUPS
            .iter()
            .all(|group| group.scope == MetricScope::LocalOnly)
    );
}

#[test]
fn routing_metrics_enforces_model_and_target_bounds() {
    let metrics = RoutingMetrics::with_config(Duration::from_secs(3600), 2, 2);
    metrics.record_attempt(
        Some("alpha"),
        AttemptTarget::Remote("peer-a".into()),
        Duration::from_millis(1),
        Duration::from_millis(10),
        AttemptOutcome::Success,
        Some(8),
    );
    metrics.record_attempt(
        Some("alpha"),
        AttemptTarget::Remote("peer-b".into()),
        Duration::from_millis(2),
        Duration::from_millis(12),
        AttemptOutcome::Success,
        Some(9),
    );
    metrics.record_attempt(
        Some("alpha"),
        AttemptTarget::Remote("peer-c".into()),
        Duration::from_millis(3),
        Duration::from_millis(15),
        AttemptOutcome::Timeout,
        None,
    );
    metrics.record_attempt(
        Some("beta"),
        AttemptTarget::Local("127.0.0.1:9001".into()),
        Duration::from_millis(1),
        Duration::from_millis(11),
        AttemptOutcome::Success,
        Some(7),
    );
    metrics.record_attempt(
        Some("gamma"),
        AttemptTarget::Endpoint("http://example.com".into()),
        Duration::from_millis(4),
        Duration::from_millis(20),
        AttemptOutcome::Unavailable,
        None,
    );

    let model_snapshots = metrics.model_snapshots();
    assert_eq!(model_snapshots.len(), 2);
    assert!(model_snapshots.contains_key("beta"));
    assert!(model_snapshots.contains_key("gamma"));
    assert_eq!(model_snapshots["beta"].targets.len(), 1);
}

#[test]
fn endpoint_target_label_credentials_are_redacted_in_snapshot() {
    let metrics = RoutingMetrics::with_config(Duration::from_secs(3600), 8, 8);
    metrics.record_attempt(
        Some("gamma"),
        AttemptTarget::Endpoint("https://alice:s3cret@endpoint.example/v1?api_key=abc123".into()),
        Duration::from_millis(4),
        Duration::from_millis(20),
        AttemptOutcome::Success,
        Some(5),
    );

    let snapshot = &metrics.model_snapshots()["gamma"];
    let target = &snapshot.targets[0];
    assert_eq!(target.kind, "endpoint");
    assert!(
        !target.target.contains("alice:s3cret"),
        "endpoint userinfo leaked into metrics: {}",
        target.target
    );
    assert!(
        !target.target.contains("abc123"),
        "endpoint api_key leaked into metrics: {}",
        target.target
    );
    assert!(
        target.target.contains("endpoint.example"),
        "endpoint host lost from metrics: {}",
        target.target
    );
}

#[test]
fn uppercase_scheme_endpoint_label_is_still_redacted_in_snapshot() {
    // URL schemes are case-insensitive (RFC 3986); an uppercase scheme
    // must not bypass credential redaction of the metrics label.
    let metrics = RoutingMetrics::with_config(Duration::from_secs(3600), 8, 8);
    metrics.record_attempt(
        Some("gamma"),
        AttemptTarget::Endpoint("HTTPS://alice:s3cret@endpoint.example/v1?api_key=abc123".into()),
        Duration::from_millis(4),
        Duration::from_millis(20),
        AttemptOutcome::Success,
        Some(5),
    );

    let snapshot = &metrics.model_snapshots()["gamma"];
    let target = &snapshot.targets[0];
    assert!(
        !target.target.contains("alice:s3cret"),
        "endpoint userinfo leaked for uppercase scheme: {}",
        target.target
    );
    assert!(
        !target.target.contains("abc123"),
        "endpoint api_key leaked for uppercase scheme: {}",
        target.target
    );
}

#[test]
fn non_endpoint_target_labels_pass_through_verbatim() {
    let metrics = RoutingMetrics::with_config(Duration::from_secs(3600), 8, 8);
    metrics.record_attempt(
        Some("beta"),
        AttemptTarget::Local("127.0.0.1:9001".into()),
        Duration::from_millis(1),
        Duration::from_millis(11),
        AttemptOutcome::Success,
        Some(7),
    );

    let snapshot = &metrics.model_snapshots()["beta"];
    assert_eq!(snapshot.targets[0].target, "127.0.0.1:9001");
}

#[test]
fn routing_metrics_prunes_stale_entries_on_snapshot() {
    let metrics = RoutingMetrics::with_config(Duration::from_secs(1), 8, 8);
    metrics.record_attempt(
        Some("stale"),
        AttemptTarget::Remote("peer-a".into()),
        Duration::from_millis(1),
        Duration::from_millis(10),
        AttemptOutcome::Success,
        Some(3),
    );
    metrics.age_model_for_test("stale", Duration::from_secs(2));

    let snapshots = metrics.model_snapshots();
    assert!(snapshots.is_empty());
}

#[test]
fn routing_metrics_aggregates_success_retry_and_pressure() {
    let metrics = RoutingMetrics::new();
    metrics.observe_inflight(3);
    metrics.record_attempt(
        Some("glm"),
        AttemptTarget::Remote("peer-a".into()),
        Duration::from_millis(5),
        Duration::from_millis(20),
        AttemptOutcome::Timeout,
        None,
    );
    metrics.record_attempt(
        Some("glm"),
        AttemptTarget::Remote("peer-b".into()),
        Duration::from_millis(25),
        Duration::from_millis(40),
        AttemptOutcome::Success,
        Some(12),
    );
    metrics.record_request(
        Some("glm"),
        2,
        RequestOutcome::Success(RequestService::Remote),
    );

    metrics.record_attempt(
        Some("qwen"),
        AttemptTarget::Local("127.0.0.1:9338".into()),
        Duration::from_millis(2),
        Duration::from_millis(16),
        AttemptOutcome::Rejected,
        None,
    );
    metrics.record_request(
        Some("qwen"),
        1,
        RequestOutcome::Rejected(RequestService::Local),
    );

    let status = metrics.status_snapshot(1);
    assert_eq!(status.request_count, 2);
    assert_eq!(status.successful_requests, 1);
    assert_eq!(status.retry_count, 1);
    assert_eq!(status.failover_count, 1);
    assert_eq!(status.attempt_timeout_count, 1);
    assert_eq!(status.attempt_reject_count, 1);
    assert_eq!(status.local_node.peak_inflight_requests, 3);
    assert_eq!(status.pressure.fronted_request_count, 2);
    assert_eq!(status.pressure.remotely_served_request_count, 1);
    assert_eq!(status.pressure.locally_served_request_count, 1);

    let model = metrics.model_snapshots().remove("glm").unwrap();
    assert_eq!(model.request_count, 1);
    assert_eq!(model.successful_requests, 1);
    assert_eq!(model.retry_count, 1);
    assert_eq!(model.failover_count, 1);
    assert_eq!(model.attempt_timeout_count, 1);
    assert_eq!(model.targets.len(), 2);
    assert!(model.avg_tokens_per_second.is_some());
}

#[test]
fn zero_attempt_rejection_does_not_count_as_local_service() {
    let metrics = RoutingMetrics::new();
    metrics.record_request(None, 0, RequestOutcome::Rejected(RequestService::Local));
    let status = metrics.status_snapshot(0);
    assert_eq!(status.request_count, 1);
    assert_eq!(status.pressure.locally_served_request_count, 0);
}

#[test]
fn routing_metrics_tracks_unattributed_requests_in_global_status_only() {
    let metrics = RoutingMetrics::new();
    metrics.record_attempt(
        None,
        AttemptTarget::Remote("peer-a".into()),
        Duration::from_millis(3),
        Duration::from_millis(14),
        AttemptOutcome::Unavailable,
        None,
    );
    metrics.record_request(None, 1, RequestOutcome::Unavailable);

    let status = metrics.status_snapshot(0);
    let model_snapshots = metrics.model_snapshots();
    assert_eq!(status.request_count, 1);
    assert_eq!(status.attempt_unavailable_count, 1);
    assert_eq!(status.local_node.remote_attempt_count, 1);
    assert!(model_snapshots.is_empty());
}

#[test]
fn routing_metrics_ignores_auto_model_for_per_model_state() {
    let metrics = RoutingMetrics::new();
    metrics.record_attempt(
        Some("auto"),
        AttemptTarget::Local("127.0.0.1:9337".into()),
        Duration::from_millis(1),
        Duration::from_millis(5),
        AttemptOutcome::Success,
        Some(2),
    );
    metrics.record_request(
        Some("auto"),
        1,
        RequestOutcome::Success(RequestService::Local),
    );

    let status = metrics.status_snapshot(0);
    assert_eq!(status.request_count, 1);
    assert_eq!(status.successful_requests, 1);
    assert!(metrics.model_snapshots().is_empty());
}

#[test]
fn routing_metrics_advertises_only_local_hosted_model_throughput() {
    let metrics = RoutingMetrics::new();
    metrics.record_attempt(
        Some("qwen"),
        AttemptTarget::Local("127.0.0.1:9337".into()),
        Duration::from_millis(2),
        Duration::from_millis(1_000),
        AttemptOutcome::Success,
        Some(42),
    );
    metrics.record_attempt(
        Some("remote-only"),
        AttemptTarget::Remote("peer-a".into()),
        Duration::from_millis(2),
        Duration::from_millis(1_000),
        AttemptOutcome::Success,
        Some(200),
    );
    metrics.record_attempt(
        Some("failed-local"),
        AttemptTarget::Local("127.0.0.1:9338".into()),
        Duration::from_millis(2),
        Duration::from_millis(1_000),
        AttemptOutcome::Timeout,
        None,
    );

    let hosted = vec![
        "qwen".to_string(),
        "remote-only".to_string(),
        "failed-local".to_string(),
    ];
    let hints = metrics.advertisable_model_throughput(&hosted);

    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].model_name, "qwen");
    assert_eq!(hints[0].avg_tokens_per_second_milli, 42_000);
    assert_eq!(hints[0].throughput_samples, 1);
}

const TEST_NOW_UNIX_MS: u64 = 1_800_000_000_000;

fn record_at(
    metrics: &RoutingMetrics,
    now: Instant,
    target: &AttemptTarget,
    attempt_time: Duration,
    outcome: AttemptOutcome,
    completion_tokens: Option<u64>,
) {
    metrics.record_attempt_at(
        now,
        Some("qwen"),
        target.clone(),
        Duration::from_millis(2),
        attempt_time,
        outcome,
        completion_tokens,
    );
}

/// The reproducer from #2771: one success at 100 tokens/s, then a timeout
/// every ten minutes. The timeouts keep the target active past the
/// one-hour TTL, but its throughput credit ends with the evidence window
/// for local ranking, peer ranking and the advertised hint alike, while
/// the cumulative counters still record the sample.
#[test]
fn throughput_credit_expires_while_failures_keep_the_target_active() {
    let metrics = RoutingMetrics::new();
    let target = AttemptTarget::Local("127.0.0.1:9337".into());
    let hosted = vec!["qwen".to_string()];
    let start = Instant::now();
    let second = Duration::from_secs(1);
    record_at(
        &metrics,
        start,
        &target,
        second,
        AttemptOutcome::Success,
        Some(100),
    );
    for minutes in (10..=70).step_by(10) {
        let now = start + Duration::from_secs(minutes * 60);
        record_at(
            &metrics,
            now,
            &target,
            second,
            AttemptOutcome::Timeout,
            None,
        );
    }

    let at_limit = start + throughput_window::THROUGHPUT_EVIDENCE_MAX_AGE;
    assert_eq!(metrics.tps_for_model_at("qwen", at_limit), Some((100.0, 1)));
    let hint = metrics
        .throughput_hint_for_target_at("qwen", target.clone(), at_limit)
        .expect("the sample is still inside the window");
    assert_eq!(
        (hint.avg_tokens_per_second_milli, hint.throughput_samples),
        (100_000, 1)
    );
    // Read at the edge of the window, the advertised hint dates its sample
    // a full window back on the wall clock.
    let advertised = metrics.advertisable_model_throughput_at(&hosted, at_limit, TEST_NOW_UNIX_MS);
    assert_eq!(
        advertised[0].measurement,
        Some(ThroughputMeasurement {
            observed_at_unix_ms: TEST_NOW_UNIX_MS
                - duration_millis(throughput_window::THROUGHPUT_EVIDENCE_MAX_AGE),
            window_ms: duration_millis(throughput_window::THROUGHPUT_EVIDENCE_MAX_AGE),
            kind: ThroughputMeasurementKind::EndToEndAttempt,
        })
    );

    let after = start + Duration::from_secs(70 * 60);
    assert_eq!(metrics.tps_for_model_at("qwen", after), None);
    assert!(
        metrics
            .throughput_hint_for_target_at("qwen", target, after)
            .is_none()
    );
    assert!(
        metrics
            .advertisable_model_throughput_at(&hosted, after, TEST_NOW_UNIX_MS)
            .is_empty()
    );

    let snapshot = &metrics.model_snapshots()["qwen"];
    assert_eq!(snapshot.throughput_samples, 1);
    assert_eq!(snapshot.attempt_timeout_count, 7);
}

/// Attempts that carry no valid sample never refresh the evidence.
#[test]
fn attempts_without_a_valid_sample_do_not_refresh_throughput_evidence() {
    let metrics = RoutingMetrics::new();
    let target = AttemptTarget::Local("127.0.0.1:9337".into());
    let start = Instant::now();
    let second = Duration::from_secs(1);
    record_at(
        &metrics,
        start,
        &target,
        second,
        AttemptOutcome::Success,
        Some(100),
    );

    let late = start + Duration::from_secs(14 * 60);
    record_at(
        &metrics,
        late,
        &target,
        second,
        AttemptOutcome::Success,
        None,
    );
    record_at(
        &metrics,
        late,
        &target,
        second,
        AttemptOutcome::Success,
        Some(0),
    );
    record_at(
        &metrics,
        late,
        &target,
        Duration::ZERO,
        AttemptOutcome::Success,
        Some(50),
    );
    for outcome in [
        AttemptOutcome::Rejected,
        AttemptOutcome::Unavailable,
        AttemptOutcome::ContextOverflow,
    ] {
        record_at(&metrics, late, &target, second, outcome, Some(50));
    }

    let after = start + throughput_window::THROUGHPUT_EVIDENCE_MAX_AGE + second;
    assert_eq!(metrics.tps_for_model_at("qwen", after), None);
    assert!(
        metrics
            .throughput_hint_for_target_at("qwen", target, after)
            .is_none()
    );
}

/// A new valid sample restores the credit on its own: the expired ones
/// neither weigh on the average nor count toward the sample threshold.
#[test]
fn a_new_sample_restores_throughput_evidence_without_expired_samples() {
    let metrics = RoutingMetrics::new();
    let target = AttemptTarget::Local("127.0.0.1:9337".into());
    let hosted = vec!["qwen".to_string()];
    let start = Instant::now();
    let second = Duration::from_secs(1);
    for _ in 0..3 {
        record_at(
            &metrics,
            start,
            &target,
            second,
            AttemptOutcome::Success,
            Some(100),
        );
    }

    let later = start + Duration::from_secs(20 * 60);
    record_at(
        &metrics,
        later,
        &target,
        second,
        AttemptOutcome::Success,
        Some(40),
    );

    assert_eq!(metrics.tps_for_model_at("qwen", later), Some((40.0, 1)));
    let hint = metrics
        .throughput_hint_for_target_at("qwen", target, later)
        .expect("the new sample is evidence");
    assert_eq!(
        (hint.avg_tokens_per_second_milli, hint.throughput_samples),
        (40_000, 1)
    );
    let advertised = metrics.advertisable_model_throughput_at(&hosted, later, TEST_NOW_UNIX_MS);
    assert_eq!(advertised.len(), 1);
    assert_eq!(advertised[0].throughput_samples, 1);
    assert_eq!(metrics.model_snapshots()["qwen"].throughput_samples, 4);
}

/// A serving instance on another port is another target: it starts with
/// no evidence of its own.
#[test]
fn throughput_evidence_stays_with_its_target() {
    let metrics = RoutingMetrics::new();
    let old_target = AttemptTarget::Local("127.0.0.1:9337".into());
    let new_target = AttemptTarget::Local("127.0.0.1:9338".into());
    let now = Instant::now();
    record_at(
        &metrics,
        now,
        &old_target,
        Duration::from_secs(1),
        AttemptOutcome::Success,
        Some(100),
    );

    assert!(
        metrics
            .throughput_hint_for_target_at("qwen", old_target, now)
            .is_some()
    );
    assert!(
        metrics
            .throughput_hint_for_target_at("qwen", new_target, now)
            .is_none()
    );
}

/// An average over samples ten minutes apart is valid only for what is
/// left of its oldest sample's window, so a receiver never credits samples
/// older than the window.
#[test]
fn advertised_window_ends_with_the_oldest_sample_in_the_average() {
    let metrics = RoutingMetrics::new();
    let target = AttemptTarget::Local("127.0.0.1:9337".into());
    let hosted = vec!["qwen".to_string()];
    let start = Instant::now();
    let second = Duration::from_secs(1);
    record_at(
        &metrics,
        start,
        &target,
        second,
        AttemptOutcome::Success,
        Some(100),
    );
    let later = start + Duration::from_secs(10 * 60);
    record_at(
        &metrics,
        later,
        &target,
        second,
        AttemptOutcome::Success,
        Some(40),
    );

    let advertised = metrics.advertisable_model_throughput_at(&hosted, later, TEST_NOW_UNIX_MS);
    assert_eq!(advertised[0].throughput_samples, 2);
    assert_eq!(
        advertised[0].measurement,
        Some(ThroughputMeasurement {
            observed_at_unix_ms: TEST_NOW_UNIX_MS,
            window_ms: 5 * 60 * 1000,
            kind: ThroughputMeasurementKind::EndToEndAttempt,
        })
    );
}

/// A span that already equals the evidence age leaves nothing to advertise:
/// the hint survives, undated, instead of claiming a bogus window.
#[test]
fn an_exact_evidence_age_span_is_advertised_undated() {
    let metrics = RoutingMetrics::new();
    let target = AttemptTarget::Local("127.0.0.1:9337".into());
    let hosted = vec!["qwen".to_string()];
    let start = Instant::now();
    let second = Duration::from_secs(1);
    record_at(
        &metrics,
        start,
        &target,
        second,
        AttemptOutcome::Success,
        Some(100),
    );
    let latest = start + throughput_window::THROUGHPUT_EVIDENCE_MAX_AGE;
    record_at(
        &metrics,
        latest,
        &target,
        second,
        AttemptOutcome::Success,
        Some(40),
    );

    let advertised = metrics.advertisable_model_throughput_at(&hosted, latest, TEST_NOW_UNIX_MS);
    assert_eq!(advertised[0].throughput_samples, 2);
    assert_eq!(
        advertised[0].measurement, None,
        "no window remains, so the hint is kept but never fresh"
    );
}

/// Under a millisecond of evidence left is advertised undated too, never as a
/// 1 ms window that would outlast the oldest sample.
#[test]
fn a_sub_millisecond_remainder_is_advertised_undated() {
    let metrics = RoutingMetrics::new();
    let target = AttemptTarget::Local("127.0.0.1:9337".into());
    let hosted = vec!["qwen".to_string()];
    let start = Instant::now();
    let second = Duration::from_secs(1);
    record_at(
        &metrics,
        start,
        &target,
        second,
        AttemptOutcome::Success,
        Some(100),
    );
    let latest =
        start + throughput_window::THROUGHPUT_EVIDENCE_MAX_AGE - Duration::from_micros(500);
    record_at(
        &metrics,
        latest,
        &target,
        second,
        AttemptOutcome::Success,
        Some(40),
    );

    let advertised = metrics.advertisable_model_throughput_at(&hosted, latest, TEST_NOW_UNIX_MS);
    assert_eq!(advertised[0].throughput_samples, 2);
    assert_eq!(advertised[0].measurement, None);
}

/// What this node keeps as evidence and what it accepts from a peer are
/// the same 15 minutes.
#[test]
fn local_and_advertised_evidence_windows_agree() {
    assert_eq!(
        duration_millis(throughput_window::THROUGHPUT_EVIDENCE_MAX_AGE),
        mesh_llm_membership::advertised_throughput::MAX_ADVERTISED_THROUGHPUT_WINDOW_MS
    );
}

#[test]
fn throughput_hint_for_target_ignores_expired_metrics() {
    let metrics = RoutingMetrics::new();
    let target = AttemptTarget::Local("127.0.0.1:9337".into());
    metrics.record_attempt(
        Some("qwen"),
        target.clone(),
        Duration::from_millis(2),
        Duration::from_millis(1_000),
        AttemptOutcome::Success,
        Some(42),
    );

    assert!(
        metrics
            .throughput_hint_for_target("qwen", target.clone())
            .is_some()
    );
    metrics.age_model_for_test("qwen", METRICS_TTL + Duration::from_secs(1));

    assert!(metrics.throughput_hint_for_target("qwen", target).is_none());
}

#[test]
fn sanitize_model_throughput_hints_drops_invalid_and_clamps_values() {
    let hints = sanitize_model_throughput_hints([
        ModelThroughputHint {
            model_name: "  qwen  ".to_string(),
            avg_tokens_per_second_milli: MAX_ADVERTISED_TPS_MILLI + 1,
            throughput_samples: MAX_ADVERTISED_THROUGHPUT_SAMPLES + 1,
            measurement: None,
        },
        ModelThroughputHint {
            model_name: "qwen".to_string(),
            avg_tokens_per_second_milli: 42_000,
            throughput_samples: 7,
            measurement: None,
        },
        ModelThroughputHint {
            model_name: "".to_string(),
            avg_tokens_per_second_milli: 42_000,
            throughput_samples: 7,
            measurement: None,
        },
        ModelThroughputHint {
            model_name: "x".repeat(MAX_ADVERTISED_MODEL_NAME_BYTES + 1),
            avg_tokens_per_second_milli: 42_000,
            throughput_samples: 7,
            measurement: None,
        },
        ModelThroughputHint {
            model_name: "empty-speed".to_string(),
            avg_tokens_per_second_milli: 0,
            throughput_samples: 7,
            measurement: None,
        },
        ModelThroughputHint {
            model_name: "empty-samples".to_string(),
            avg_tokens_per_second_milli: 42_000,
            throughput_samples: 0,
            measurement: None,
        },
    ]);

    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].model_name, "qwen");
    assert_eq!(
        hints[0].avg_tokens_per_second_milli,
        MAX_ADVERTISED_TPS_MILLI
    );
    assert_eq!(
        hints[0].throughput_samples,
        MAX_ADVERTISED_THROUGHPUT_SAMPLES
    );
}

#[test]
fn observe_inflight_tracks_peak_monotonically() {
    let metrics = RoutingMetrics::new();
    metrics.observe_inflight(3);
    metrics.observe_inflight(1);
    metrics.observe_inflight(5);
    metrics.observe_inflight(2);

    let status = metrics.status_snapshot(0);
    assert_eq!(status.local_node.peak_inflight_requests, 5);
}

#[test]
fn routing_metrics_concurrent_updates_preserve_totals() {
    let metrics = RoutingMetrics::with_config_and_shards(Duration::from_secs(3600), 64, 8, 8);
    let metrics = Arc::new(metrics);
    let threads = 8usize;
    let per_thread = 250usize;
    let barrier = Arc::new(Barrier::new(threads));
    let mut handles = Vec::new();

    for thread_idx in 0..threads {
        let metrics = metrics.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            let model = format!("model-{}", thread_idx % 4);
            barrier.wait();
            for _ in 0..per_thread {
                metrics.observe_inflight((thread_idx + 1) as u64);
                metrics.record_attempt(
                    Some(&model),
                    AttemptTarget::Remote(format!("peer-{thread_idx}")),
                    Duration::from_millis(2),
                    Duration::from_millis(10),
                    AttemptOutcome::Success,
                    Some(4),
                );
                metrics.record_request(
                    Some(&model),
                    1,
                    RequestOutcome::Success(RequestService::Remote),
                );
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let status = metrics.status_snapshot(0);
    assert_eq!(status.request_count, (threads * per_thread) as u64);
    assert_eq!(status.successful_requests, (threads * per_thread) as u64);
    assert_eq!(
        status.local_node.remote_attempt_count,
        (threads * per_thread) as u64
    );

    let total_model_requests: u64 = metrics
        .model_snapshots()
        .values()
        .map(|snapshot| snapshot.request_count)
        .sum();
    assert_eq!(total_model_requests, (threads * per_thread) as u64);
}
