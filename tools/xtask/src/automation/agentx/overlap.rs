//! Client HTTP overlap is evidence of in-flight requests, not GPU decode slots.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader},
    path::Path,
};

pub(super) fn read(output: &Path) -> Value {
    match measure(&output.join("upstream/profile_export.jsonl")) {
        Ok((requests, peak)) => json!({"source":"client_http_trace", "requests":requests,
            "peak_in_flight_requests":peak,"gpu_decode_concurrency":null}),
        Err(error) => json!({"source":"client_http_trace", "peak_in_flight_requests":null,
            "gpu_decode_concurrency":null,"unavailable_reason":error.to_string()}),
    }
}

fn measure(path: &Path) -> DynResult<(usize, i64)> {
    let mut boundaries = BTreeMap::<u64, i64>::new();
    let mut requests = 0;
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let record: Value = serde_json::from_str(&line?)?;
        let phase = record["metadata"]["benchmark_phase"]
            .as_str()
            .ok_or("missing AgentX benchmark phase")?;
        if phase != "profiling" {
            continue;
        }
        let trace = &record["trace_data"];
        let start = trace["request_send_start_ns"]
            .as_u64()
            .ok_or("missing HTTP start")?;
        let end = trace["response_receive_end_ns"]
            .as_u64()
            .ok_or("missing HTTP end")?;
        if end <= start {
            return Err("invalid HTTP interval".into());
        }
        *boundaries.entry(start).or_default() += 1;
        *boundaries.entry(end).or_default() -= 1;
        requests += 1;
    }
    if requests == 0 {
        return Err("no profiling HTTP intervals".into());
    }
    let (mut active, mut peak) = (0, 0);
    for delta in boundaries.values() {
        active += delta;
        peak = peak.max(active);
    }
    Ok((requests, peak))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measures_overlap_excluding_warmup_and_touching_intervals() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let rows = [
            ("warmup", 0, 100),
            ("profiling", 1, 5),
            ("profiling", 2, 4),
            ("profiling", 5, 9),
        ]
        .map(|(phase, start, end)| {
            json!({"metadata":{"benchmark_phase":phase},
                "trace_data":{"request_send_start_ns":start,"response_receive_end_ns":end}})
            .to_string()
        });
        std::fs::write(file.path(), rows.join("\n")).unwrap();
        assert_eq!(measure(file.path()).unwrap(), (3, 2));
        std::fs::write(file.path(), "{}").unwrap();
        assert!(measure(file.path()).is_err());
    }

    #[test]
    fn reads_pinned_agentx_export_with_fewer_active_requests_than_offered_sessions() {
        // Reduced AgentX 1.0.7 export from an offered-c4 smoke. Preserve the
        // real phase and interval fields; omit request identities and payloads.
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            include_str!("../../../tests/fixtures/agentx/http-overlap.jsonl"),
        )
        .unwrap();
        assert_eq!(measure(file.path()).unwrap(), (3, 3));
    }

    #[test]
    fn rejects_missing_phase_and_incomplete_http_intervals() {
        let file = tempfile::NamedTempFile::new().unwrap();
        for record in [
            json!({"metadata":{"credit_phase":"profiling"}}),
            json!({"metadata":{"benchmark_phase":"profiling"},
                "trace_data":{"request_send_start_ns":1}}),
            json!({"metadata":{"benchmark_phase":"profiling"},
                "trace_data":{"request_send_start_ns":5,"response_receive_end_ns":5}}),
        ] {
            std::fs::write(file.path(), record.to_string()).unwrap();
            assert!(measure(file.path()).is_err());
        }
    }
}
