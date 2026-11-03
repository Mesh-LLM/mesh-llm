use super::tests::{survey_config, test_source};
use super::*;
use std::collections::BTreeSet;
use std::sync::Arc;

#[test]
fn the_resource_carries_a_distinct_per_process_instance_id() {
    let resource = telemetry_resource("mesh-llm-test");
    let instance_key = opentelemetry::Key::from_static_str("service.instance.id");
    let id = resource
        .get(&instance_key)
        .expect("the resource must carry a per-process instance id")
        .as_str()
        .to_string();
    assert!(!id.is_empty());

    let other = telemetry_resource("mesh-llm-test");
    assert_ne!(
        id,
        other
            .get(&instance_key)
            .expect("second build must also carry an instance id")
            .as_str()
            .to_string(),
        "two builds must not share one collector stream identity"
    );

    assert_eq!(
        resource
            .get(&opentelemetry::Key::from_static_str("service.name"))
            .expect("service.name must be preserved")
            .as_str(),
        "mesh-llm-test"
    );
    assert!(
        resource
            .get(&opentelemetry::Key::from_static_str("service.version"))
            .is_some()
    );
}

#[test]
fn a_failed_export_is_reported_instead_of_dropped() {
    let mut config = survey_config();
    config.plugins.clear();
    // Nothing listens on the discard port: every export attempt fails.
    config.telemetry.endpoint = Some("http://127.0.0.1:9".into());
    let settings = SurveySettings::from_config_with_env(&config, |_| None).expect("settings");

    let reported = Arc::new(Mutex::new(Vec::<String>::new()));
    let reporter = reported.clone();
    let mut recorder = SurveyRecorder::otlp_with_hook(
        &settings,
        Some(Arc::new(move |error| {
            reporter.lock().unwrap().push(error.to_string());
        })),
    )
    .expect("recorder");

    // Give the reader something to export so the flush actually attempts an
    // export against the dead endpoint.
    let attrs = RequestAttributes::from_request(
        Some("mesh-llm-test-model"),
        1,
        RequestOutcome::Success(RequestService::Local),
        test_source(),
    );
    recorder.record(SurveyEvent::ModelRequest { attrs });
    let _ = recorder._provider.force_flush();

    let reported = reported.lock().unwrap();
    assert!(
        !reported.is_empty(),
        "a failed OTLP export must be reported, not dropped silently"
    );
}

#[test]
fn token_histograms_declare_boundaries_beyond_sdk_defaults() {
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader};

    let queue = Arc::new(SurveyEventQueue::new(8));
    let telemetry = SurveyTelemetry {
        inner: Some(Arc::new(SurveyTelemetryInner {
            queue: queue.clone(),
            hardware: hardware::HardwareSurvey::default(),
            source: test_source(),
            prompt_shape_metrics: true,
        })),
    };
    let routing_sink: &dyn RoutingTelemetrySink = &telemetry;
    routing_sink.record_prompt_shape(
        Some("mesh-llm-test-model"),
        Some(1024),
        Some(8),
        RequestOutcome::Success(RequestService::Endpoint),
    );

    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    let mut recorder = SurveyRecorder::new(provider);
    for event in queue.drain() {
        recorder.record(event);
    }
    recorder._provider.force_flush().expect("metric flush");

    let exported = exporter.get_finished_metrics().expect("exported metrics");
    let mut token_bounds: Option<Vec<f64>> = None;
    for metric in exported
        .iter()
        .flat_map(|resource| resource.scope_metrics())
        .flat_map(|scope| scope.metrics())
    {
        let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = metric.data() else {
            continue;
        };
        if metric.name() != "mesh_llm_prompt_tokens" {
            continue;
        }
        if let Some(point) = histogram.data_points().next() {
            token_bounds = Some(point.bounds().collect());
        }
    }

    let bounds = token_bounds.expect("the prompt token histogram must be exported");
    let highest = bounds.iter().cloned().fold(0.0, f64::max);
    assert!(
        highest >= 32_768.0,
        "token histograms must declare context-sized buckets instead of the \
         SDK defaults (highest bound {highest})"
    );
}

#[test]
fn route_attempts_export_a_request_duration_histogram() {
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader};

    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    let mut recorder = SurveyRecorder::new(provider);
    recorder.record(SurveyEvent::RouteAttempt {
        attrs: RouteAttemptAttributes::from_attempt(
            Some("mesh-llm-test-model"),
            &AttemptTarget::Local("127.0.0.1:9337".into()),
            AttemptOutcome::Success,
            test_source(),
        ),
        attempt_ms: 123.0,
    });
    recorder._provider.force_flush().expect("metric flush");

    let exported = exporter.get_finished_metrics().expect("exported metrics");
    let mut names = BTreeSet::new();
    let mut duration_sum: Option<f64> = None;
    for metric in exported
        .iter()
        .flat_map(|resource| resource.scope_metrics())
        .flat_map(|scope| scope.metrics())
    {
        names.insert(metric.name().to_string());
        if metric.name() == "mesh_llm_model_request_duration_ms"
            && let opentelemetry_sdk::metrics::data::AggregatedMetrics::F64(
                opentelemetry_sdk::metrics::data::MetricData::Histogram(histogram),
            ) = metric.data()
            && let Some(point) = histogram.data_points().next()
        {
            duration_sum = Some(point.sum());
        }
    }

    assert!(
        names.contains("mesh_llm_model_request_duration_ms"),
        "route attempts must export a request duration histogram \
         (exported metrics: {names:?})"
    );
    assert_eq!(
        duration_sum,
        Some(123.0),
        "the duration histogram must record the attempt duration"
    );
}
