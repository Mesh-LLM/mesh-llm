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
        if record["metadata"]["credit_phase"] != "profiling" {
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
            json!({"metadata":{"credit_phase":phase},
                "trace_data":{"request_send_start_ns":start,"response_receive_end_ns":end}})
            .to_string()
        });
        std::fs::write(file.path(), rows.join("\n")).unwrap();
        assert_eq!(measure(file.path()).unwrap(), (3, 2));
        std::fs::write(file.path(), "{}").unwrap();
        assert!(measure(file.path()).is_err());
    }
}
