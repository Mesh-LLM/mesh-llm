//! OTLP encoding of events and their two destinations: a JSON-lines file in
//! the OpenTelemetry file-exporter format, and the OTLP gRPC endpoint traces
//! go to. The gRPC exporter batches whatever is queued into one request.

use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    sync::{Mutex, PoisonError},
    time::Duration,
};

use opentelemetry_proto::tonic::{
    collector::logs::v1::{ExportLogsServiceRequest, logs_service_client::LogsServiceClient},
    common::v1::InstrumentationScope,
    logs::v1::{LogRecord, ResourceLogs, ScopeLogs},
    resource::v1::Resource,
};
use serde_json::Value;
use skippy_protocol::StageConfig;
use tokio::{sync::mpsc, time::timeout};

use super::super::{
    otlp_encoding::key_values, resource_attrs::resource_attrs, trace_context::SpanContext,
};
use super::Severity;

const EXPORT_TIMEOUT: Duration = Duration::from_secs(5);
/// Events queued for the gRPC exporter before new ones are dropped.
const GRPC_QUEUE: usize = 1024;
/// Events one gRPC request carries at most.
const GRPC_BATCH: usize = 256;
/// W3C trace flags: sampled.
const SAMPLED: u32 = 1;

/// One event as an OTLP log record correlated with `context`.
pub(super) fn record(
    context: SpanContext,
    name: &str,
    severity: Severity,
    attributes: BTreeMap<String, Value>,
    time_unix_nano: u64,
) -> LogRecord {
    LogRecord {
        time_unix_nano,
        observed_time_unix_nano: time_unix_nano,
        severity_number: severity.number(),
        attributes: key_values(attributes),
        flags: SAMPLED,
        trace_id: context.trace_id.to_vec(),
        span_id: context.span_id.to_vec(),
        event_name: name.to_string(),
        ..Default::default()
    }
}

fn export_request(config: &StageConfig, records: Vec<LogRecord>) -> ExportLogsServiceRequest {
    let attributes = resource_attrs(config, "skippy");
    ExportLogsServiceRequest {
        resource_logs: vec![ResourceLogs {
            resource: Some(Resource {
                attributes: key_values(attributes),
                ..Default::default()
            }),
            scope_logs: vec![ScopeLogs {
                scope: Some(InstrumentationScope {
                    name: "skippy-serving".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    ..Default::default()
                }),
                log_records: records,
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    }
}

pub(super) struct LogSinks {
    file: Option<Mutex<BufWriter<File>>>,
    grpc: Option<mpsc::Sender<(std::sync::Arc<StageConfig>, LogRecord)>>,
}

impl LogSinks {
    /// The configured destinations; `None` when there are none.
    pub(super) fn open(path: Option<OsString>, endpoint: Option<&str>) -> Option<Self> {
        let file =
            path.and_then(
                |path| match OpenOptions::new().create(true).append(true).open(&path) {
                    Ok(file) => Some(Mutex::new(BufWriter::new(file))),
                    Err(error) => {
                        tracing::warn!("telemetry events file {path:?} not opened: {error}");
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

    pub(super) fn send(&self, config: &std::sync::Arc<StageConfig>, record: LogRecord) {
        if let Some(grpc) = self.grpc.as_ref() {
            // A full queue means the endpoint is not keeping up; drop the event.
            let _ = grpc.try_send((config.clone(), record.clone()));
        }
        if let Some(file) = self.file.as_ref() {
            let request = export_request(config, vec![record]);
            let mut file = file.lock().unwrap_or_else(PoisonError::into_inner);
            let written = serde_json::to_writer(&mut *file, &request)
                .map_err(std::io::Error::from)
                .and_then(|()| file.write_all(b"\n"))
                .and_then(|()| file.flush());
            if let Err(error) = written {
                tracing::warn!("telemetry events file write failed: {error}");
            }
        }
    }

    pub(super) fn flush(&self) {
        if let Some(file) = self.file.as_ref() {
            let _ = file.lock().unwrap_or_else(PoisonError::into_inner).flush();
        }
    }
}

/// Export queued events in batches, reconnecting after a failure; a batch
/// that fails is dropped rather than retried.
async fn export_loop(
    endpoint: String,
    mut rx: mpsc::Receiver<(std::sync::Arc<StageConfig>, LogRecord)>,
) {
    let mut client = None;
    while let Some((config, first)) = rx.recv().await {
        let mut records = vec![first];
        while records.len() < GRPC_BATCH {
            match rx.try_recv() {
                Ok((_, record)) => records.push(record),
                Err(_) => break,
            }
        }
        if client.is_none() {
            client = timeout(EXPORT_TIMEOUT, LogsServiceClient::connect(endpoint.clone()))
                .await
                .ok()
                .and_then(Result::ok);
        }
        let Some(connected) = client.as_mut() else {
            continue;
        };
        let request = export_request(&config, records);
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

    fn stage_config() -> std::sync::Arc<StageConfig> {
        std::sync::Arc::new(
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
            .expect("minimal stage config"),
        )
    }

    #[test]
    fn an_event_is_a_named_log_record_on_its_span() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("events.jsonl");
        let sinks = LogSinks::open(Some(path.clone().into_os_string()), None).expect("file sink");
        let context = SpanContext {
            trace_id: [0xab; 16],
            span_id: [0xcd; 8],
        };
        sinks.send(
            &stage_config(),
            record(
                context,
                "skippy.kv.eviction",
                Severity::Warn,
                BTreeMap::from([("skippy.kv.tier".to_string(), json!("l1"))]),
                42,
            ),
        );
        sinks.flush();

        let text = std::fs::read_to_string(&path).expect("events file");
        let request: Value =
            serde_json::from_str(text.lines().next().expect("a line")).expect("json");
        let record = &request["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0];
        assert_eq!(record["eventName"], "skippy.kv.eviction");
        assert_eq!(record["severityNumber"], json!(13));
        assert_eq!(record["timeUnixNano"], json!("42"));
        assert_eq!(record["traceId"], json!("ab".repeat(16)));
        assert_eq!(record["spanId"], json!("cd".repeat(8)));
        assert_eq!(record["attributes"][0]["key"], "skippy.kv.tier");
        assert_eq!(record["attributes"][0]["value"]["stringValue"], "l1");
        assert!(record.get("body").is_none() || record["body"].is_null());
    }

    #[test]
    fn no_destination_means_no_sinks() {
        assert!(LogSinks::open(None, None).is_none());
    }
}
