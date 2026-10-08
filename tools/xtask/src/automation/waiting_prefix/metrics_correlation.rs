//! Correlate measured requests with a finalized metrics-server export.
//! Client-observed timing remains a separate measurement.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const DECODE: &str = "stage.openai_decode_token";
const SUMMARY: &str = "stage.openai_generation_summary";

#[derive(Deserialize)]
struct Run {
    run_id: String,
    status: String,
    finished_at_unix_nanos: Option<i64>,
}

#[derive(Deserialize)]
struct Loss {
    dropped_events: u64,
    export_errors: u64,
}

#[derive(Deserialize)]
struct Span {
    run_id: String,
    request_id: Option<String>,
    stage_id: Option<String>,
    trace_id: String,
    span_id: String,
    name: String,
    start_time_unix_nanos: i64,
    end_time_unix_nanos: i64,
}

#[derive(Deserialize)]
struct Report {
    run: Run,
    counts: BTreeMap<String, u64>,
    telemetry_loss: Loss,
    spans: Vec<Span>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct Timing {
    pub request_id: String,
    pub server_ttft_ms: f64,
    pub server_request_latency_ms: f64,
}

#[derive(Default)]
struct Request {
    first: Option<i64>,
    last: i64,
    decode: Option<i64>,
    summary: Option<i64>,
}

impl Request {
    fn observe(&mut self, span: &Span) -> DynResult<()> {
        self.first = Some(self.first.map_or(span.start_time_unix_nanos, |first| {
            first.min(span.start_time_unix_nanos)
        }));
        self.last = self.last.max(span.end_time_unix_nanos);
        if span.name == DECODE {
            self.decode = Some(self.decode.map_or(span.start_time_unix_nanos, |first| {
                first.min(span.start_time_unix_nanos)
            }));
        }
        if span.name == SUMMARY && self.summary.replace(span.end_time_unix_nanos).is_some() {
            return Err("duplicate measured generation summary in collector report".into());
        }
        Ok(())
    }

    fn timing(self, request_id: &str) -> DynResult<Timing> {
        let first = self.first.ok_or("missing measured request spans")?;
        let decode = self.decode.ok_or("missing measured decode-token span")?;
        let summary = self.summary.ok_or("missing measured generation summary")?;
        if decode < first || decode > summary {
            return Err("measured decode falls outside the generation interval".into());
        }
        Ok(Timing {
            request_id: request_id.to_owned(),
            server_ttft_ms: (decode - first) as f64 / 1_000_000.0,
            server_request_latency_ms: (self.last - first) as f64 / 1_000_000.0,
        })
    }
}

/// IDs come from the measured generation-summary slice, after cache seeding.
/// Never infer this slice from collector export order or a request count.
pub(super) fn correlate(bytes: &[u8], run_id: &str, measured: &[String]) -> DynResult<Vec<Timing>> {
    correlate_report(bytes, run_id, measured, true)
}

/// Check delivery before shutdown without claiming finalization.
pub(super) fn ready(bytes: &[u8], run_id: &str, measured: &[String]) -> DynResult<()> {
    correlate_report(bytes, run_id, measured, false).map(|_| ())
}

fn correlate_report(
    bytes: &[u8],
    run_id: &str,
    measured: &[String],
    finalized: bool,
) -> DynResult<Vec<Timing>> {
    if bytes.len() > 64 * 1024 * 1024 {
        return Err("collector report exceeds 64 MiB".into());
    }
    let report: Report = serde_json::from_slice(bytes)?;
    if run_id.is_empty()
        || report.run.run_id != run_id
        || !matches!(report.run.status.as_str(), "running" | "completed")
        || (finalized
            && (report.run.status != "completed"
                || !report
                    .run
                    .finished_at_unix_nanos
                    .is_some_and(|time| time > 0)))
        || report.telemetry_loss.dropped_events != 0
        || report.telemetry_loss.export_errors != 0
        || report.counts.get("spans").copied() != Some(report.spans.len() as u64)
    {
        return Err("collector export is mismatched, unfinished or incomplete".into());
    }
    let ids: BTreeSet<&str> = measured.iter().map(String::as_str).collect();
    if ids.is_empty() || ids.len() != measured.len() || ids.contains("") {
        return Err("measured collector IDs must be nonempty and unique".into());
    }
    let mut requests = BTreeMap::<&str, Request>::new();
    let mut spans = BTreeSet::new();
    for span in &report.spans {
        if span.run_id != run_id
            || span.start_time_unix_nanos < 0
            || span.end_time_unix_nanos < span.start_time_unix_nanos
            || span.trace_id.is_empty()
            || span.span_id.is_empty()
            || !spans.insert((&span.trace_id, &span.span_id))
        {
            return Err("collector span has invalid identity or timestamps".into());
        }
        let Some(id) = span.request_id.as_deref().filter(|id| ids.contains(id)) else {
            continue;
        };
        if span.stage_id.as_deref() != Some("stage-0") {
            return Err("measured span belongs to another stage".into());
        }
        requests.entry(id).or_default().observe(span)?;
    }
    ids.into_iter()
        .map(|id| {
            requests
                .remove(id)
                .ok_or("missing measured collector request")?
                .timing(id)
        })
        .collect()
}

#[cfg(test)]
#[path = "metrics_correlation_tests.rs"]
mod tests;
