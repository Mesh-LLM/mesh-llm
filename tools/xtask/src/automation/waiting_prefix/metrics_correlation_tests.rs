use super::*;
use serde_json::{Value, json};

fn span(id: &str, name: &str, start: i64, end: i64, ordinal: usize) -> Value {
    json!({"run_id":"owned-run", "request_id":id, "stage_id":"stage-0",
        "trace_id":"trace", "span_id":ordinal.to_string(), "name":name,
        "start_time_unix_nanos":start, "end_time_unix_nanos":end})
}

fn report() -> Value {
    json!({"run":{"run_id":"owned-run","status":"completed","finished_at_unix_nanos":9000000},
        "counts":{"spans":4},"telemetry_loss":{"dropped_events":0,"export_errors":0},
        "spans":[span("measured", "stage.openai_tokenize",1000000,1100000,1),
            span("seed", SUMMARY,0,900000,2),
            span("measured", DECODE,2500000,2600000,3),
            span("measured", SUMMARY,1000000,5000000,4)]})
}

fn parse(value: &Value, ids: &[&str]) -> DynResult<Vec<Timing>> {
    correlate(
        &serde_json::to_vec(value)?,
        "owned-run",
        &ids.iter().map(|id| (*id).to_owned()).collect::<Vec<_>>(),
    )
}

#[test]
fn excludes_seed_by_identity_and_ignores_export_order() {
    let mut value = report();
    value["spans"].as_array_mut().unwrap().reverse();
    let rows = parse(&value, &["measured"]).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].request_id, "measured");
    assert_eq!(rows[0].server_ttft_ms, 1.5);
    assert_eq!(rows[0].server_request_latency_ms, 4.0);
}

#[test]
fn requires_exact_completed_run_and_loss_free_export() {
    for (section, key, replacement) in [
        ("run", "run_id", json!("other")),
        ("run", "status", json!("running")),
        ("run", "finished_at_unix_nanos", Value::Null),
        ("telemetry_loss", "dropped_events", json!(1)),
        ("telemetry_loss", "export_errors", json!(1)),
        ("counts", "spans", json!(3)),
    ] {
        let mut value = report();
        value[section][key] = replacement;
        assert!(parse(&value, &["measured"]).is_err(), "{section}.{key}");
    }
}

#[test]
fn rejects_missing_duplicate_and_empty_measured_ids() {
    for ids in [
        vec![],
        vec![""],
        vec!["missing"],
        vec!["measured", "measured"],
    ] {
        assert!(parse(&report(), &ids).is_err());
    }
    for position in [2, 3] {
        let mut value = report();
        value["spans"][position]["request_id"] = json!("seed");
        assert!(parse(&value, &["measured"]).is_err());
    }
}

#[test]
fn rejects_span_aliases_cross_stage_cross_run_and_invalid_time() {
    for (position, key, replacement) in [
        (2, "span_id", json!("1")),
        (2, "stage_id", json!("stage-1")),
        (2, "run_id", json!("other")),
        (2, "end_time_unix_nanos", json!(0)),
        (2, "start_time_unix_nanos", json!(-1)),
        (2, "start_time_unix_nanos", json!(6000000)),
    ] {
        let mut value = report();
        value["spans"][position][key] = replacement;
        assert!(parse(&value, &["measured"]).is_err(), "{key}");
    }
}

#[test]
fn rejects_duplicate_generation_summaries() {
    let mut value = report();
    value["spans"]
        .as_array_mut()
        .unwrap()
        .push(span("measured", SUMMARY, 1000000, 5000000, 5));
    value["counts"]["spans"] = json!(5);
    assert!(parse(&value, &["measured"]).is_err());
}

#[test]
fn running_report_can_prove_delivery_but_cannot_be_a_final_result() {
    let mut value = report();
    value["run"]["status"] = json!("running");
    value["run"]["finished_at_unix_nanos"] = Value::Null;
    let bytes = serde_json::to_vec(&value).unwrap();
    let ids = vec!["measured".to_owned()];
    assert!(ready(&bytes, "owned-run", &ids).is_ok());
    assert!(correlate(&bytes, "owned-run", &ids).is_err());
    assert_eq!(value["run"]["status"], "running");
}
