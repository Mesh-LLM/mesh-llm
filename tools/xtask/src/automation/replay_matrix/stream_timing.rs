//! Preserve terminal overhead separately without relabelling historical metrics.
use serde_json::{Value, json};

pub(super) fn summarize(records: &[Value], summary: &mut Value) {
    let successful: Vec<_> = records
        .iter()
        .filter(|r| r.get("error").is_none())
        .collect();
    for field in ["output_stream_seconds", "terminal_seconds"] {
        let samples: Option<Vec<f64>> = successful.iter().map(|r| r[field].as_f64()).collect();
        let Some(mut samples) = samples.filter(|v| !v.is_empty()) else {
            summary[field] = Value::Null;
            continue;
        };
        samples.sort_by(f64::total_cmp);
        summary[field] = json!(samples.iter().sum::<f64>());
        if field == "terminal_seconds" {
            let middle = samples.len() / 2;
            summary["terminal_p50_seconds"] = json!(if samples.len().is_multiple_of(2) {
                (samples[middle - 1] + samples[middle]) / 2.0
            } else {
                samples[middle]
            });
        }
    }
    summary["output_stream_tokens_per_second"] = summary["output_stream_seconds"]
        .as_f64()
        .filter(|v| *v > 0.0)
        .and_then(|seconds| {
            summary["completion_tokens"]
                .as_f64()
                .map(|tokens| tokens / seconds)
        })
        .map_or(Value::Null, |rate| json!(rate));
    let mut reasons = serde_json::Map::new();
    for (name, value) in summary.as_object().into_iter().flatten() {
        if value.is_null() {
            reasons.insert(
                name.clone(),
                "No complete observations or positive measurement interval.".into(),
            );
        }
    }
    for name in [
        "native_prefill_tokens_per_second",
        "native_decode_tokens_per_second",
    ] {
        summary[name] = Value::Null;
        reasons.insert(name.into(), "Not exposed by the common OpenAI stream; client timing is not native execution timing.".into());
    }
    summary["unavailable_metrics"] = Value::Object(reasons);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separates_terminal_delay_and_does_not_invent_missing_historical_timing() {
        let mut summary = json!({"completion_tokens":100});
        let records = vec![
            json!({"output_stream_seconds":1,"terminal_seconds":5}),
            json!({"output_stream_seconds":3,"terminal_seconds":7}),
        ];
        summarize(&records, &mut summary);
        assert_eq!(summary["output_stream_tokens_per_second"], 25.0);
        assert_eq!(summary["terminal_p50_seconds"], 6.0);
        assert_eq!(summary["terminal_seconds"], 12.0);
        summarize(&[json!({})], &mut summary);
        assert!(summary["output_stream_tokens_per_second"].is_null());
    }
}
