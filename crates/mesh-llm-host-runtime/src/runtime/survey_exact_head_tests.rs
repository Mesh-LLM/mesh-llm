use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn attribute_keys(attributes: Vec<KeyValue>) -> BTreeSet<String> {
    attributes
        .into_iter()
        .map(|attribute| attribute.key.to_string())
        .collect()
}

fn lifecycle_spec(configured_model_selector: &str) -> SurveyModelSpec<'_> {
    SurveyModelSpec {
        model: "shared-served-alias",
        configured_model_selector: Some(configured_model_selector),
        model_path: None,
        launch_kind: SurveyLaunchKind::Startup,
        pinned_gpu: None,
        backend: Some("skippy"),
        context_length: Some(16_384),
    }
}

fn attribute_map(attributes: Vec<KeyValue>) -> BTreeMap<String, String> {
    attributes
        .into_iter()
        .map(|attribute| (attribute.key.to_string(), attribute.value.to_string()))
        .collect()
}

fn model_selector_id(attributes: &SurveyAttributes) -> String {
    attribute_map(attributes.key_values(None))
        .remove("mesh_llm.model_selector_id")
        .expect("configured model selector identity")
}

#[test]
fn external_model_values_never_become_telemetry_attributes() {
    let source = SurveyTelemetrySource {
        node_id: "source-node".into(),
        node_role: "client".into(),
    };
    let lifecycle = SurveyAttributes::from_disabled_spec(SurveyModelSpec {
        model: "https://user:secret@example.test/private/model.gguf?token=leaked",
        configured_model_selector: None,
        model_path: None,
        launch_kind: SurveyLaunchKind::Startup,
        pinned_gpu: None,
        backend: None,
        context_length: None,
    });
    let request = RequestAttributes::from_request(
        Some("org/private-model:variant"),
        2,
        RequestOutcome::Success(RequestService::Remote),
        source.clone(),
    );
    let attempt = RouteAttemptAttributes::from_attempt(
        Some("/private/models/secret.gguf"),
        &AttemptTarget::Endpoint("https://private-endpoint.example/v1".into()),
        AttemptOutcome::Rejected,
        source,
    );

    for keys in [
        attribute_keys(lifecycle.key_values(None)),
        attribute_keys(request.key_values()),
        attribute_keys(attempt.key_values()),
    ] {
        assert!(!keys.contains("mesh_llm.model"));
    }
}

#[test]
fn configured_model_selector_identity_is_stable_bounded_and_private() {
    // Given: one canonical selector containing values that must never be exported raw.
    let selector = "org/private-repo@main:secret-model.gguf#sensitive-profile";
    let first = SurveyAttributes::from_disabled_spec(lifecycle_spec(selector));
    let second = SurveyAttributes::from_disabled_spec(lifecycle_spec(selector));

    // When: lifecycle attributes are generated repeatedly for that selector.
    let first_attributes = attribute_map(first.key_values(None));
    let second_attributes = attribute_map(second.key_values(None));
    let identity = first_attributes
        .get("mesh_llm.model_selector_id")
        .expect("configured model selector identity");

    // Then: the identity is deterministic, fixed-length lowercase hex and leaks no input.
    assert_eq!(first_attributes, second_attributes);
    assert_eq!(identity.len(), 39);
    assert!(identity.starts_with("sha256:"));
    assert!(
        identity[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    for forbidden in [
        "org",
        "private-repo",
        "secret-model.gguf",
        "sensitive-profile",
        selector,
    ] {
        assert!(!format!("{first_attributes:?}").contains(forbidden));
    }
}

#[test]
fn lifecycle_events_share_one_configured_model_selector_identity() {
    // Given: one retained loaded-model telemetry handle for a canonical selector.
    let queue = Arc::new(SurveyEventQueue::new(8));
    let telemetry = SurveyTelemetry {
        inner: Some(Arc::new(SurveyTelemetryInner {
            queue: queue.clone(),
            hardware: hardware::HardwareSurvey::default(),
            source: SurveyTelemetrySource {
                node_id: "source-node".into(),
                node_role: "worker".into(),
            },
            prompt_shape_metrics: false,
        })),
    };
    let loaded = telemetry.model(lifecycle_spec("org/model-a@main:model-a.gguf"));

    // When: launch, unload, and unexpected-exit events use that retained handle.
    telemetry.record_launch_success(&loaded, Duration::from_millis(5));
    telemetry.record_unload(&loaded);
    telemetry.record_unexpected_exit(&loaded);

    // Then: every lifecycle event carries the same pseudonymous selector identity.
    let identities = queue
        .drain()
        .into_iter()
        .map(|event| match event {
            SurveyEvent::LaunchSuccess { attrs, .. }
            | SurveyEvent::Unload { attrs, .. }
            | SurveyEvent::UnexpectedExit { attrs, .. } => model_selector_id(&attrs),
            _ => panic!("unexpected telemetry event"),
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(identities.len(), 1);
}

#[test]
fn unloading_one_model_preserves_another_loaded_model_gauge_series() {
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader};

    // Given: two configured selectors whose other bounded lifecycle attributes collide.
    let first = SurveyAttributes::from_disabled_spec(lifecycle_spec(
        "org/model-a@main:shared-name.gguf#profile-a",
    ));
    let second = SurveyAttributes::from_disabled_spec(lifecycle_spec(
        "org/model-b@main:shared-name.gguf#profile-b",
    ));
    let first_id = model_selector_id(&first);
    let second_id = model_selector_id(&second);
    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    let mut recorder = SurveyRecorder::new(provider);

    // When: both models load and only the first model unloads.
    recorder.record(SurveyEvent::LaunchSuccess {
        attrs: first.clone(),
        duration_ms: 1.0,
    });
    recorder.record(SurveyEvent::LaunchSuccess {
        attrs: second,
        duration_ms: 1.0,
    });
    recorder.record(SurveyEvent::Unload {
        attrs: first,
        uptime_s: 1.0,
    });
    recorder._provider.force_flush().expect("metric flush");

    // Then: selectors are distinct series and unloading one leaves the other at one.
    assert_ne!(first_id, second_id);
    let exported = exporter.get_finished_metrics().expect("exported metrics");
    let mut loaded_by_selector = BTreeMap::new();
    for metric in exported
        .iter()
        .flat_map(|resource| resource.scope_metrics())
        .flat_map(|scope| scope.metrics())
    {
        if metric.name() != "mesh_llm_model_loaded" {
            continue;
        }
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = metric.data() else {
            continue;
        };
        for point in gauge.data_points() {
            let attributes = point
                .attributes()
                .map(|attribute| (attribute.key.to_string(), attribute.value.to_string()))
                .collect::<BTreeMap<_, _>>();
            loaded_by_selector.insert(
                attributes["mesh_llm.model_selector_id"].clone(),
                point.value(),
            );
        }
    }
    assert_eq!(loaded_by_selector.get(&first_id), Some(&0));
    assert_eq!(loaded_by_selector.get(&second_id), Some(&1));
}
