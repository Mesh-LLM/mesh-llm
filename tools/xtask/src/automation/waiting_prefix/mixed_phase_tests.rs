use super::*;
use serde_json::json;
fn event(name: &str, start: u64, end: u64) -> Vec<u8> {
    serde_json::to_vec(&json!({"event":name,"start_time_unix_nanos":start,"end_time_unix_nanos":end,"attributes":{"skippy.otel_dropped_events":0,"skippy.otel_export_errors":0,"skippy.scheduler.token_count":7,"llama_stage.prefill_token_count":40,"llama_stage.prefill_chunk_count":2,"llama_stage.prefill_max_chunk_size":32,"skippy.kv.chain_cache_errors":0,"skippy.kv.stage0_cache_errors":0}})).unwrap()
}
fn worker() -> Value {
    json!({"measured_start_unix_nanos":100,"measured_end_unix_nanos":200,"local_clock_consistent":true,"error":null,"input_sha256":"a".repeat(64)})
}
#[test]
fn mixed_phase_uses_producer_window_for_delayed_and_shutdown_tail_events() {
    let mut observation = Observation::default();
    // Publication order differs from producer time; warmup arrives last.
    observation.observe(&event("stage.scheduler_feature_iteration", 130, 131));
    observation.observe(&event("stage.openai_prefill", 110, 111));
    observation.observe(&event("stage.openai_prefill", 150, 151));
    observation.observe(&event("stage.scheduler_feature_iteration", 210, 211));
    observation.observe(&event("stage.openai_prefill", 10, 11));
    let measured = observation.measured(&worker(), 2, true).unwrap();
    assert_eq!(measured.scheduler.len(), 1);
    assert_eq!(measured.prefills.len(), 2);
    assert_eq!(
        measured.phase_provenance,
        "owned-local-producer-timestamps-with-complete-shutdown-tail"
    );
    assert!(observation.measured(&worker(), 2, false).is_err());
    assert!(observation.measured(&worker(), 3, true).is_err());
}
#[test]
fn mixed_phase_refuses_crossing_clock_drop_counter_and_timestamp_gaps() {
    for mode in ["crossing", "drop", "missing", "clock"] {
        let mut observation = Observation::default();
        observation.observe(&event("stage.openai_prefill", 10, 11));
        observation.observe(&event("stage.openai_prefill", 110, 111));
        observation.observe(&event("stage.openai_prefill", 150, 151));
        let mut line: Value =
            serde_json::from_slice(&event("stage.scheduler_feature_iteration", 130, 131)).unwrap();
        let mut worker = worker();
        match mode {
            "crossing" => line["start_time_unix_nanos"] = json!(99),
            "drop" => line["attributes"]["skippy.otel_dropped_events"] = json!(1),
            "missing" => {
                line.as_object_mut().unwrap().remove("end_time_unix_nanos");
            }
            "clock" => worker["local_clock_consistent"] = json!(false),
            _ => unreachable!(),
        }
        observation.observe(&serde_json::to_vec(&line).unwrap());
        assert!(observation.measured(&worker, 2, true).is_err());
    }
}
