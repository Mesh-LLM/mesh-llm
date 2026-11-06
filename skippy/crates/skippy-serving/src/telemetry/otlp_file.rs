//! OTLP/JSON trace file for offline analysis without a collector.
//!
//! `SKIPPY_TELEMETRY_OTLP_FILE=<path>` appends one `ExportTraceServiceRequest`
//! per line in the OTLP/JSON encoding, the OpenTelemetry file-exporter format
//! that collectors and trace tools ingest directly. One process-wide writer
//! keeps lines whole when several `Telemetry` handles share the file.

use std::{
    ffi::OsString,
    fs::OpenOptions,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, mpsc as std_mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};

use opentelemetry_proto::tonic::{
    collector::trace::v1::ExportTraceServiceRequest, trace::v1::Span,
};
use skippy_protocol::StageConfig;

use super::{EXPORT_BATCH_MAX, resource_spans_by_component};

const PATH_ENV: &str = "SKIPPY_TELEMETRY_OTLP_FILE";
const QUEUE_CAPACITY: usize = 8192;

static SINK: OnceLock<Option<OtlpFileSink>> = OnceLock::new();

/// The process-wide sink named by `SKIPPY_TELEMETRY_OTLP_FILE`, opened once.
pub(super) fn configured() -> Option<&'static OtlpFileSink> {
    SINK.get_or_init(|| {
        let path = path_from_value(std::env::var_os(PATH_ENV))?;
        OtlpFileSink::open(&path)
            .inspect_err(|error| {
                tracing::warn!(
                    "telemetry OTLP file {} is unavailable: {error}",
                    path.display()
                );
            })
            .ok()
    })
    .as_ref()
}

fn path_from_value(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

pub(super) struct OtlpFileSink {
    tx: Option<std_mpsc::SyncSender<FileItem>>,
    worker: Option<JoinHandle<()>>,
}

struct FileSpan {
    config: Arc<StageConfig>,
    span: Span,
}

enum FileItem {
    Span(Box<FileSpan>),
    /// Acknowledged once every span queued before it is on disk.
    Flush(std_mpsc::SyncSender<()>),
}

impl OtlpFileSink {
    fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let (tx, rx) = std_mpsc::sync_channel(QUEUE_CAPACITY);
        let worker = thread::Builder::new()
            .name("skippy-telemetry-otlp-file".to_string())
            .spawn(move || write_loop(&rx, BufWriter::new(file)))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
        })
    }

    /// Queue one span; false when the queue is full or the writer stopped.
    pub(super) fn try_send(&self, config: &Arc<StageConfig>, span: Span) -> bool {
        self.tx.as_ref().is_some_and(|tx| {
            tx.try_send(FileItem::Span(Box::new(FileSpan {
                config: config.clone(),
                span,
            })))
            .is_ok()
        })
    }

    /// Block until spans queued so far are written, or `timeout` passes.
    pub(super) fn flush(&self, timeout: Duration) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        let (ack, written) = std_mpsc::sync_channel(1);
        if tx.send(FileItem::Flush(ack)).is_ok() {
            let _ = written.recv_timeout(timeout);
        }
    }
}

impl Drop for OtlpFileSink {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Write each drained batch as one flushed line, so a killed server leaves
/// only whole requests behind.
fn write_loop(rx: &std_mpsc::Receiver<FileItem>, mut out: impl Write) {
    while let Ok(first) = rx.recv() {
        let mut batch = Vec::new();
        let mut acks = Vec::new();
        for item in std::iter::once(first).chain(rx.try_iter().take(EXPORT_BATCH_MAX - 1)) {
            match item {
                FileItem::Span(span) => batch.push(*span),
                FileItem::Flush(ack) => acks.push(ack),
            }
        }
        if !batch.is_empty()
            && let Err(error) = write_request(&mut out, &export_request(batch))
        {
            tracing::warn!("telemetry OTLP file write failed; file sink stopped: {error}");
            return;
        }
        acks.into_iter().for_each(|ack| {
            let _ = ack.send(());
        });
    }
}

/// Group consecutive spans of the same stage, then split each group by
/// component, one resource each.
fn export_request(batch: Vec<FileSpan>) -> ExportTraceServiceRequest {
    let mut groups: Vec<(Arc<StageConfig>, Vec<Span>)> = Vec::new();
    for FileSpan { config, span } in batch {
        match groups.last_mut() {
            Some((current, spans)) if Arc::ptr_eq(current, &config) => spans.push(span),
            _ => groups.push((config, vec![span])),
        }
    }
    ExportTraceServiceRequest {
        resource_spans: groups
            .into_iter()
            .flat_map(|(config, spans)| resource_spans_by_component(&config, spans))
            .collect(),
    }
}

fn write_request(out: &mut impl Write, request: &ExportTraceServiceRequest) -> io::Result<()> {
    serde_json::to_writer(&mut *out, request)?;
    out.write_all(b"\n")?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, ffi::OsString, sync::Arc, time::Duration};

    use serde_json::{Value, json};
    use skippy_protocol::StageConfig;

    use super::{FileItem, FileSpan, OtlpFileSink, path_from_value, write_loop};
    use crate::telemetry::{
        TelemetryCounters, TelemetryEvent, event_to_span,
        trace_context::{Frame, SpanEvent},
    };

    fn config(stage_id: &str) -> Arc<StageConfig> {
        Arc::new(
            serde_json::from_value(json!({
                "run_id": "run",
                "topology_id": "topology",
                "model_id": "org/model:Q4_K_M",
                "stage_id": stage_id,
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

    fn record(frame: &Frame) -> TelemetryEvent {
        let attributes = BTreeMap::from([("skippy.scheduler.step_ms".to_string(), json!(2.5))]);
        TelemetryEvent::span(
            frame,
            "stage.scheduler_iteration",
            attributes,
            1_000,
            3_500_000,
        )
    }

    fn file_span(config: &Arc<StageConfig>, record: &TelemetryEvent) -> FileSpan {
        FileSpan {
            config: config.clone(),
            span: event_to_span(record, &TelemetryCounters::default()),
        }
    }

    fn written_lines(spans: Vec<FileSpan>) -> Vec<Value> {
        let (tx, rx) = std::sync::mpsc::sync_channel(spans.len());
        spans
            .into_iter()
            .for_each(|span| tx.send(FileItem::Span(Box::new(span))).expect("queue span"));
        drop(tx);
        let mut out = Vec::new();
        write_loop(&rx, &mut out);
        String::from_utf8(out)
            .expect("utf8 output")
            .lines()
            .map(|line| serde_json::from_str(line).expect("one JSON request per line"))
            .collect()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn writes_otlp_json_with_hex_ids_and_string_nanos() {
        let stage = config("stage-0");
        let frame = Frame::child();
        let lines = written_lines(vec![
            file_span(&stage, &record(&frame)),
            file_span(&stage, &record(&Frame::child())),
        ]);

        assert_eq!(lines.len(), 1);
        let spans = &lines[0]["resourceSpans"][0]["scopeSpans"][0]["spans"];
        assert_eq!(spans.as_array().map(Vec::len), Some(2));
        let span = &spans[0];
        assert_eq!(span["name"], "stage.scheduler_iteration");
        assert_eq!(span["spanId"], hex(&frame.context().span_id));
        assert_eq!(span["traceId"], hex(&frame.context().trace_id));
        assert_eq!(span["startTimeUnixNano"], "1000");
        assert_eq!(span["endTimeUnixNano"], "3500000");
        assert_eq!(
            lines[0]["resourceSpans"][0]["scopeSpans"][0]["scope"]["name"],
            "skippy-serving"
        );
    }

    #[test]
    fn parents_links_and_events_are_encoded() {
        let parent = Frame::child();
        let _entered = parent.enter();
        let child = Frame::child();
        let linked = Frame::child();
        let mut traced = record(&child);
        traced.links = vec![linked.context()];
        traced.events = vec![SpanEvent {
            name: "stage.openai_kv_lookup_decision".to_string(),
            time_unix_nanos: 2_000,
            attributes: BTreeMap::from([("skippy.kv.decision".to_string(), json!("hit"))]),
        }];

        let lines = written_lines(vec![file_span(&config("stage-0"), &traced)]);
        let span = &lines[0]["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["parentSpanId"], hex(&parent.context().span_id));
        assert_eq!(span["links"][0]["spanId"], hex(&linked.context().span_id));
        assert_eq!(span["events"][0]["name"], "stage.openai_kv_lookup_decision");
        assert_eq!(span["events"][0]["timeUnixNano"], "2000");
    }

    #[test]
    fn separate_stages_keep_separate_resources() {
        let (first, second) = (config("stage-0"), config("stage-1"));
        let lines = written_lines(vec![
            file_span(&first, &record(&Frame::child())),
            file_span(&second, &record(&Frame::child())),
            file_span(&first, &record(&Frame::child())),
        ]);

        assert_eq!(lines[0]["resourceSpans"].as_array().map(Vec::len), Some(3));
    }

    #[test]
    fn each_component_gets_its_own_service() {
        let stage = config("stage-0");
        let call = Frame::child();
        let mut native = record(&call);
        native.name = "llama.graph_compute".to_string();
        let lines = written_lines(vec![
            file_span(&stage, &record(&Frame::child())),
            file_span(&stage, &native),
        ]);

        let resources = lines[0]["resourceSpans"].as_array().expect("resources");
        let services = resources
            .iter()
            .map(|resource| {
                let attributes = resource["resource"]["attributes"]
                    .as_array()
                    .expect("attributes");
                let value = |key: &str| {
                    attributes
                        .iter()
                        .find(|attribute| attribute["key"] == key)
                        .map(|attribute| attribute["value"]["stringValue"].clone())
                };
                assert_eq!(value("service.namespace"), Some(json!("skippy")));
                (
                    value("service.name").expect("service.name"),
                    resource["scopeSpans"][0]["spans"][0]["name"].clone(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            services,
            [
                (json!("llama"), json!("llama.graph_compute")),
                (
                    json!("skippy-scheduler"),
                    json!("stage.scheduler_iteration")
                ),
            ]
        );
    }

    #[test]
    fn empty_path_disables_the_sink() {
        assert_eq!(path_from_value(None), None);
        assert_eq!(path_from_value(Some(OsString::new())), None);
        assert_eq!(
            path_from_value(Some(OsString::from("/tmp/trace.jsonl"))),
            Some("/tmp/trace.jsonl".into())
        );
    }

    #[test]
    fn flush_returns_once_queued_spans_are_on_disk() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("trace.otlp.jsonl");
        std::fs::write(&path, "{\"resourceSpans\":[]}\n").expect("seed existing line");
        let sink = OtlpFileSink::open(&path).expect("open sink");
        let frame = Frame::child();
        let FileSpan { config, span } = file_span(&config("stage-0"), &record(&frame));
        assert!(sink.try_send(&config, span));

        sink.flush(Duration::from_secs(5));

        let text = std::fs::read_to_string(&path).expect("read trace file");
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2, "appends after existing content");
        let request: Value = serde_json::from_str(lines[1]).expect("JSON request");
        assert_eq!(
            request["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["spanId"],
            hex(&frame.context().span_id)
        );
    }
}
