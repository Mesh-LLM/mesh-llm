use super::tests::{survey_config, test_source};
use super::*;
use std::collections::BTreeMap;
use std::sync::Arc;

fn prompt_shape_telemetry(enabled: bool) -> (SurveyTelemetry, Arc<SurveyEventQueue>) {
    let queue = Arc::new(SurveyEventQueue::new(4));
    (
        SurveyTelemetry {
            inner: Some(Arc::new(SurveyTelemetryInner {
                queue: queue.clone(),
                hardware: hardware::HardwareSurvey::default(),
                source: test_source(),
                prompt_shape_metrics: enabled,
            })),
        },
        queue,
    )
}

#[test]
fn prompt_shape_metrics_emit_only_reviewed_counts_when_enabled() {
    let (telemetry, queue) = prompt_shape_telemetry(true);

    telemetry.record_prompt_shape(
        Some("/private/models/model.gguf"),
        Some(21),
        Some(8),
        RequestOutcome::Success(RequestService::Remote),
    );

    let events = queue.drain();
    assert_eq!(events.len(), 1);
    let SurveyEvent::PromptShape {
        attrs,
        prompt_tokens,
        completion_tokens,
    } = &events[0]
    else {
        panic!("expected prompt shape event");
    };
    assert_eq!(*prompt_tokens, Some(21));
    assert_eq!(*completion_tokens, Some(8));
    let exported = format!("{:?}", attrs.key_values());
    assert!(!exported.contains("model.gguf"));
    assert!(!exported.contains("source_node"));
    assert!(exported.contains("remote"));
    assert!(exported.contains("success"));
}

#[test]
fn prompt_shape_metrics_do_not_emit_when_disabled() {
    let (telemetry, queue) = prompt_shape_telemetry(false);

    telemetry.record_prompt_shape(
        Some("model"),
        Some(21),
        Some(8),
        RequestOutcome::Success(RequestService::Local),
    );

    assert!(queue.drain().is_empty());
}

#[test]
fn prompt_shape_histograms_reach_exporter_with_reviewed_attributes() {
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader};

    let mut config = survey_config();
    config.telemetry.prompt_shape_metrics = true;
    let settings = SurveySettings::from_config_with_env(&config, |_| None).expect("settings");
    let queue = Arc::new(SurveyEventQueue::new(settings.queue_size));
    let telemetry = SurveyTelemetry {
        inner: Some(Arc::new(SurveyTelemetryInner {
            queue: queue.clone(),
            hardware: hardware::HardwareSurvey::default(),
            source: test_source(),
            prompt_shape_metrics: settings.prompt_shape_metrics,
        })),
    };
    let routing_sink: &dyn RoutingTelemetrySink = &telemetry;
    routing_sink.record_prompt_shape(
        Some("https://user:secret@example.test/private/model.gguf?token=leaked"),
        Some(21),
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
    let mut observed = BTreeMap::new();
    for metric in exported
        .iter()
        .flat_map(|resource| resource.scope_metrics())
        .flat_map(|scope| scope.metrics())
    {
        let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = metric.data() else {
            continue;
        };
        let Some(point) = histogram.data_points().next() else {
            continue;
        };
        observed.insert(
            metric.name().to_string(),
            (
                point.sum(),
                point
                    .attributes()
                    .map(|attr| (attr.key.to_string(), attr.value.to_string()))
                    .collect::<BTreeMap<_, _>>(),
            ),
        );
    }

    assert_eq!(observed["mesh_llm_prompt_tokens"].0, 21);
    assert_eq!(observed["mesh_llm_completion_tokens"].0, 8);
    for (_, attrs) in observed.values() {
        assert_eq!(attrs.len(), 2);
        assert_eq!(
            attrs.get("mesh_llm.route_service").map(String::as_str),
            Some("endpoint")
        );
        assert_eq!(
            attrs.get("mesh_llm.request_outcome").map(String::as_str),
            Some("success")
        );
        let serialized = format!("{attrs:?}");
        assert!(!serialized.contains("secret"));
        assert!(!serialized.contains("example.test"));
        assert!(!serialized.contains("token=leaked"));
        assert!(!serialized.contains("/private"));
    }
}

#[tokio::test]
async fn prompt_shape_metrics_reach_the_otlp_http_boundary_without_private_labels() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("OTLP test listener");
    let address = listener.local_addr().expect("OTLP listener address");
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("OTLP connection");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer).await.expect("OTLP request read");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
            .await
            .expect("OTLP response");
        let body_start = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("OTLP request headers")
            + 4;
        request.split_off(body_start)
    });

    let mut config = survey_config();
    config.telemetry.endpoint = Some(format!("http://{address}/v1/metrics"));
    config.telemetry.export_interval_secs = Some(1);
    config.telemetry.prompt_shape_metrics = true;
    let telemetry =
        SurveyTelemetry::start(&config, hardware::HardwareSurvey::default(), test_source());
    let sink = telemetry.routing_sink().expect("routing telemetry sink");
    sink.record_prompt_shape(
        Some("https://user:secret@example.test/private/model.gguf?token=leaked"),
        Some(21),
        Some(8),
        RequestOutcome::Success(RequestService::Endpoint),
    );

    let request = tokio::time::timeout(Duration::from_secs(8), server)
        .await
        .expect("prompt shape metric must reach OTLP")
        .expect("OTLP server task");
    let exported = String::from_utf8_lossy(&request);
    assert!(exported.contains("mesh_llm_prompt_tokens"));
    assert!(exported.contains("mesh_llm_completion_tokens"));
    for forbidden in [
        "user",
        "secret",
        "example.test",
        "token=leaked",
        "/private",
        "model.gguf",
    ] {
        assert!(
            !exported.contains(forbidden),
            "OTLP payload leaked {forbidden:?}: {exported}"
        );
    }
}
