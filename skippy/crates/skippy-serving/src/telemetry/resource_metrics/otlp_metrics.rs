//! OTLP encoding of resource gauges and their two destinations: a JSON-lines
//! file in the OpenTelemetry file-exporter format, and the OTLP gRPC endpoint
//! traces go to.

use std::{
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use opentelemetry_proto::tonic::{
    collector::metrics::v1::{
        ExportMetricsServiceRequest, metrics_service_client::MetricsServiceClient,
    },
    common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value},
    metrics::v1::{
        Gauge, Metric, NumberDataPoint, ResourceMetrics, ScopeMetrics, metric, number_data_point,
    },
    resource::v1::Resource,
};
use skippy_protocol::StageConfig;
use tokio::{sync::mpsc, time::timeout};

use super::super::{
    otlp_encoding::key_values,
    resource_attrs::resource_attrs,
    sink::{OpenSpan, SpanRecord, SpanStart, TraceSink},
    trace_context::{Frame, SpanEvent},
};

const EXPORT_TIMEOUT: Duration = Duration::from_secs(5);
/// Samples queued for the gRPC exporter before new ones are dropped.
const GRPC_QUEUE: usize = 64;

pub(super) fn gauge(
    name: &str,
    unit: &str,
    description: &str,
    points: Vec<NumberDataPoint>,
) -> Metric {
    Metric {
        name: name.to_string(),
        description: description.to_string(),
        unit: unit.to_string(),
        data: Some(metric::Data::Gauge(Gauge {
            data_points: points,
        })),
        ..Default::default()
    }
}

pub(super) fn point(
    time_unix_nano: u64,
    value: f64,
    attributes: &[(&str, &str)],
) -> NumberDataPoint {
    NumberDataPoint {
        attributes: attributes
            .iter()
            .map(|(key, value)| KeyValue {
                key: (*key).to_string(),
                value: Some(AnyValue {
                    value: Some(any_value::Value::StringValue((*value).to_string())),
                }),
                ..Default::default()
            })
            .collect(),
        time_unix_nano,
        value: Some(number_data_point::Value::AsDouble(value)),
        ..Default::default()
    }
}

fn export_request(config: &StageConfig, metrics: Vec<Metric>) -> ExportMetricsServiceRequest {
    let attributes = resource_attrs(config, "skippy");
    ExportMetricsServiceRequest {
        resource_metrics: vec![ResourceMetrics {
            resource: Some(Resource {
                attributes: key_values(attributes),
                dropped_attributes_count: 0,
                entity_refs: vec![],
            }),
            scope_metrics: vec![ScopeMetrics {
                scope: Some(InstrumentationScope {
                    name: "skippy-serving".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    attributes: vec![],
                    dropped_attributes_count: 0,
                }),
                metrics,
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    }
}

pub(super) struct MetricSinks {
    file: Option<Mutex<BufWriter<File>>>,
    grpc: Option<mpsc::Sender<ExportMetricsServiceRequest>>,
}

impl MetricSinks {
    /// The configured destinations; `None` when there are none.
    pub(super) fn open(path: Option<OsString>, endpoint: Option<&str>) -> Option<Self> {
        let file =
            path.and_then(
                |path| match OpenOptions::new().create(true).append(true).open(&path) {
                    Ok(file) => Some(Mutex::new(BufWriter::new(file))),
                    Err(error) => {
                        tracing::warn!("telemetry metrics file {path:?} not opened: {error}");
                        None
                    }
                },
            );
        let grpc = endpoint.and_then(|endpoint| {
            let handle = tokio::runtime::Handle::try_current().ok()?;
            let (tx, rx) = mpsc::channel(GRPC_QUEUE);
            handle.spawn(export_loop(endpoint.to_string(), rx));
            Some(tx)
        });
        (file.is_some() || grpc.is_some()).then_some(Self { file, grpc })
    }

    pub(super) fn send(&self, config: &StageConfig, metrics: Vec<Metric>) {
        let request = export_request(config, metrics);
        if let Some(file) = self.file.as_ref() {
            let mut file = file.lock().unwrap_or_else(PoisonError::into_inner);
            let written = serde_json::to_writer(&mut *file, &request)
                .map_err(std::io::Error::from)
                .and_then(|()| file.write_all(b"\n"))
                .and_then(|()| file.flush());
            if let Err(error) = written {
                tracing::warn!("telemetry metrics file write failed: {error}");
            }
        }
        if let Some(grpc) = self.grpc.as_ref() {
            // A full queue means the endpoint is not keeping up; drop the sample.
            let _ = grpc.try_send(request);
        }
    }

    pub(super) fn flush(&self) {
        if let Some(file) = self.file.as_ref() {
            let _ = file.lock().unwrap_or_else(PoisonError::into_inner).flush();
        }
    }
}

/// The metric destinations as one of `Telemetry`'s sinks: samples only, so
/// spans and events pass it by.
pub(in crate::telemetry) struct OtlpMetricsSink {
    sinks: MetricSinks,
    config: Arc<StageConfig>,
}

impl OtlpMetricsSink {
    pub(in crate::telemetry) fn configured(
        config: Arc<StageConfig>,
        path: Option<OsString>,
        endpoint: Option<&str>,
    ) -> Option<Self> {
        MetricSinks::open(path, endpoint).map(|sinks| Self { sinks, config })
    }
}

impl TraceSink for OtlpMetricsSink {
    fn create_span_with(&self, _: &SpanStart) -> OpenSpan {
        OpenSpan::NONE
    }

    fn end_span(&self, _: OpenSpan, _: &SpanRecord) {}

    fn discard_span(&self, _: OpenSpan) {}

    fn record_span(&self, _: &SpanRecord) {}

    fn record_event(&self, _: &SpanEvent, _: &Frame) {}

    fn record_metrics(&self, metrics: &[Metric]) {
        self.sinks.send(&self.config, metrics.to_vec());
    }

    fn folds_events_into_spans(&self) -> bool {
        false
    }

    fn flush(&self, _: Duration) {
        self.sinks.flush();
    }
}

/// Export samples in order, reconnecting after a failure; a sample that
/// fails is dropped rather than retried, since the next one supersedes it.
async fn export_loop(endpoint: String, mut rx: mpsc::Receiver<ExportMetricsServiceRequest>) {
    let mut client = None;
    while let Some(request) = rx.recv().await {
        if client.is_none() {
            client = timeout(
                EXPORT_TIMEOUT,
                MetricsServiceClient::connect(endpoint.clone()),
            )
            .await
            .ok()
            .and_then(Result::ok);
        }
        let Some(connected) = client.as_mut() else {
            continue;
        };
        if !matches!(
            timeout(EXPORT_TIMEOUT, connected.export(request)).await,
            Ok(Ok(_))
        ) {
            client = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn stage_config() -> StageConfig {
        serde_json::from_value(json!({
            "run_id": "run",
            "topology_id": "topology",
            "model_id": "org/model:Q4_K_M",
            "stage_id": "stage-0",
            "stage_index": 0,
            "layer_start": 0,
            "layer_end": 4,
            "load_mode": "runtime-slice",
            "execution_contract": "",
            "bind_addr": "127.0.0.1:0",
        }))
        .expect("minimal stage config")
    }

    #[test]
    fn a_sample_is_one_otlp_json_line_under_the_skippy_service() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("metrics.jsonl");
        let sinks =
            MetricSinks::open(Some(path.clone().into_os_string()), None).expect("file sink");
        sinks.send(
            &stage_config(),
            vec![gauge(
                "skippy.gpu.busy",
                "1",
                "busy",
                vec![point(5, 0.25, &[("device", "MTL0")])],
            )],
        );
        sinks.flush();

        let text = std::fs::read_to_string(&path).expect("metrics file");
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 1);
        let request: Value = serde_json::from_str(lines[0]).expect("json line");
        let resource = &request["resourceMetrics"][0];
        let service = resource["resource"]["attributes"]
            .as_array()
            .expect("attributes")
            .iter()
            .find(|attribute| attribute["key"] == "service.name")
            .map(|attribute| attribute["value"]["stringValue"].clone());
        assert_eq!(service, Some(json!("skippy")));
        let metric = &resource["scopeMetrics"][0]["metrics"][0];
        assert_eq!(metric["name"], "skippy.gpu.busy");
        let point = &metric["gauge"]["dataPoints"][0];
        assert_eq!(point["asDouble"], json!(0.25));
        assert_eq!(point["timeUnixNano"], json!("5"));
        assert_eq!(
            point["attributes"][0]["value"]["stringValue"],
            json!("MTL0")
        );
    }

    #[test]
    fn no_destination_means_no_sinks() {
        assert!(MetricSinks::open(None, None).is_none());
    }
}
