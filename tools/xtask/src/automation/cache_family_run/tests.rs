use super::metrics;
use serde_json::{Value, json};
fn measurement() -> Value {
    json!({"cohort":"openai-concurrent","status":"completed","requested_requests":2,"concurrency":2,"output_tokens":32,"prompt_sha256":"a".repeat(64),"makespan_seconds":2.0,"rows":[
        {"request_id":0,"status":"completed","excluded_warmup":false,"elapsed_ms":100.0,"ttft_ms":10.0,"tpot_ms":5.0,"tokens_predicted":4,"evidence":{"content_sha256":"b".repeat(64),"first_generated_sha256":"c".repeat(64)}},
        {"request_id":1,"status":"completed","excluded_warmup":false,"elapsed_ms":200.0,"ttft_ms":30.0,"tpot_ms":15.0,"tokens_predicted":6,"evidence":{"content_sha256":"d".repeat(64),"first_generated_sha256":"e".repeat(64)}}]})
}
#[test]
fn cache_matrix_goodput_uses_observed_tokens_seconds_and_both_slo_dimensions() {
    let m = metrics::goodput(&measurement(), 20, 10).unwrap();
    assert_eq!(m["throughput_rps"], 1.0);
    assert_eq!(m["goodput_rps"], 0.5);
    assert_eq!(m["output_tokens_per_second"], 5.0);
    assert_eq!(m["observed_output_tokens"], 10);
    assert_eq!(m["ttft_p99_ms"], 30.0);
    assert_eq!(m["latency_p50_ms"], 200.0);
}
#[test]
fn cache_matrix_goodput_preserves_partial_observed_rows_without_certifying_complete() {
    let mut input = measurement();
    input["status"] = json!("incomplete");
    input["rows"][1] = json!({"request_id":1,"status":"not-launched","excluded_warmup":false});
    let m = metrics::goodput(&input, 20, 10).unwrap();
    assert_eq!(m["complete"], false);
    let directory = tempfile::tempdir().unwrap();
    let mut prior = vec![
        json!({"cohort":"skippy-old","status":"completed","client_metrics":{"complete":true}}),
    ];
    let refused = super::child::run(
        "cache-family-cell",
        &Value::Null,
        &directory.path().join("new"),
        std::time::Instant::now(),
        &crate::process::Cancellation::default(),
    );
    assert!(refused.is_err());
    super::pipeline::child_refused(&mut prior, "skippy-new", &json!({"concurrency":2}), true);
    assert_eq!(prior[0]["status"], "completed");
    assert_eq!(prior[1]["status"], "not-launched");
    assert!(prior[1]["client_metrics"].is_null());
    assert!(!directory.path().join("new").exists());
    directory.close().unwrap();

    assert_eq!(m["failed_or_unlaunched_requests"], 1);
    assert_eq!(m["output_tokens_per_second"], 2.0);
    input["rows"][0]["ttft_ms"] = Value::Null;
    assert_eq!(
        metrics::goodput(&input, 20, 10).unwrap()["goodput_rps"],
        0.0
    );
}
#[test]
fn cache_matrix_parity_requires_full_unique_matched_protocol_workload_and_both_hashes() {
    let left = measurement();
    let mut right = left.clone();
    assert_eq!(
        metrics::accepted_parity(&left, &right, true, true).unwrap()["matches"],
        true
    );
    for (old_accepted, new_accepted) in [(false, true), (true, false), (false, false)] {
        assert!(metrics::accepted_parity(&left, &right, old_accepted, new_accepted).is_err());
    }
    assert_eq!(metrics::parity(&left, &right).unwrap()["matches"], true);
    right["rows"][1]["evidence"]["first_generated_sha256"] = json!("f".repeat(64));
    let p = metrics::parity(&left, &right).unwrap();
    assert_eq!(p["matches"], false);
    assert_eq!(p["first_generated_mismatch_request_ids"], json!([1]));
    right["rows"][1]["status"] = json!("failed");
    let p = metrics::parity(&left, &right).unwrap();
    assert_eq!(p["complete"], false);
    assert_eq!(p["unavailable_request_ids"], json!([1]));
    right = left.clone();
    right["rows"][1]["request_id"] = json!(0);
    assert!(metrics::parity(&left, &right).is_err());
    right = left.clone();
    right["output_tokens"] = json!(128);
    assert!(metrics::parity(&left, &right).is_err());
    right = left.clone();
    right["prompt_sha256"] = json!("f".repeat(64));
    assert!(metrics::parity(&left, &right).is_err());
    right = left.clone();
    right["cohort"] = json!("native-concurrent");
    assert!(metrics::parity(&left, &right).is_err());
}
#[test]
fn cache_matrix_metrics_refuse_nonfinite_rate_overflow_and_incomplete_rosters() {
    let mut input = measurement();
    input["makespan_seconds"] = json!(f64::MIN_POSITIVE);
    input["rows"][0]["tokens_predicted"] = json!(u64::MAX);
    assert!(metrics::goodput(&input, 20, 10).is_err());
    input = measurement();
    input["requested_requests"] = json!(3);
    assert!(metrics::goodput(&input, 20, 10).is_err());
    input = measurement();
    input["makespan_seconds"] = json!(0);
    assert!(metrics::goodput(&input, 20, 10).is_err());
    input = measurement();
    input["rows"][0]["elapsed_ms"] = json!(-1);
    assert!(metrics::goodput(&input, 20, 10).is_err());
}

#[test]
fn cache_matrix_report_projection_uses_only_observed_fields_and_never_promotes_unmeasured_rows() {
    let cell = json!({"case":{"family":"Qwen3 dense","model_id":"declared","payload":"resident-kv","stage_load_mode":"runtime-slice","prefix_tokens":64,"resident_kv_bytes_per_token":1024},"use_case":null});
    let unmeasured = super::reporting::row(&cell, None, &[], false);
    assert_eq!(unmeasured["skippy"]["status"], "unmeasured");
    assert_eq!(unmeasured["llama_server"]["status"], "unmeasured");
    let correctness = json!({"status":"completed","rows":[{"evidence":{"skippy":{"status":"pass","benchmark_prompt_text":"private prompt must not be projected","benchmark_prompt_token_count":65,"cache_hit_import_ms":[1.0],"cache_hit_decode_ms":[3.0],"cache_hit_total_ms":4.0,"recompute_total_ms":8.0}}}]});
    let row = super::reporting::row(
        &cell,
        Some(&correctness),
        &[
            json!({"cohort":"native-serial","status":"completed","warm_statistics":{"warm_mean_ms":null,"warm_median_ms":null}}),
        ],
        true,
    );
    assert_eq!(row["skippy"]["status"], "pass");
    assert!(row["skippy"].get("benchmark_prompt_text").is_none());
    assert_eq!(row["llama_server"]["status"], "ok");
    assert!(row["llama_server"]["warm_mean_ms"].is_null());
    let text =
        crate::automation::cache_family_report::producer(&json!([row]), &json!({"use_cases":[]}))
            .unwrap();
    assert!(text.contains("Qwen3 dense"));
    assert!(!text.contains("private prompt"));
    assert!(!text.contains("2.00x"));
    assert!(!text.contains("--runtime-lane-count 1"));
    assert!(!text.contains("--parallel 1"));
    assert!(!text.contains("Prompt sources are checked in"));
    assert!(text.contains("owning plan/profile"));
    assert!(text.contains("selected corpus"));
}

#[test]
fn cache_matrix_package_report_keeps_unavailable_baseline_and_observed_correctness() {
    let cell = json!({"case":{"family":"deepseek3","model_id":"package","payload":"resident-kv","stage_load_mode":"layer-package","prefix_tokens":4},"use_case":{"key":"default","label":"Default"}});
    let correctness = json!({"status":"completed","rows":[{"evidence":{"skippy":{"status":"pass","cache_hit_total_ms":6.0,"benchmark_prompt_token_count":5}}}]});
    let row = super::reporting::row(
        &cell,
        Some(&correctness),
        &[json!({"cohort":"native-serial","status":"unavailable","reason":"package-only preset"})],
        true,
    );
    assert_eq!(row["skippy"]["status"], "pass");
    assert_eq!(row["llama_server"]["status"], "unavailable");
    assert!(row["llama_server"]["warm_mean_ms"].is_null());
    assert_eq!(row["prefix_tokens"], 4);
    assert_eq!(row["benchmark_prompt_token_count"], 5);
}

#[test]
fn cache_matrix_terminal_failures_downgrade_admission_without_erasing_observations() {
    for (cancelled, expired, finish_ok) in [
        (false, false, true),
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ] {
        let mut receipt = json!({"status":"completed","rows":[{"result":{"status":"completed"}}],"report_error":"earlier diagnostic"});
        super::finalize(&mut receipt, cancelled, expired, finish_ok);
        assert_eq!(
            receipt["status"],
            if cancelled || expired || !finish_ok {
                "incomplete"
            } else {
                "completed"
            }
        );
        assert_eq!(receipt["rows"][0]["result"]["status"], "completed");
        assert_eq!(receipt["report_error"], "earlier diagnostic");
        if cancelled || expired || !finish_ok {
            assert_eq!(receipt["terminal_refusal"]["cancelled"], cancelled);
            assert_eq!(receipt["terminal_refusal"]["deadline_expired"], expired);
            assert_eq!(
                receipt["terminal_refusal"]["interrupt_finish_failed"],
                !finish_ok
            );
        }
    }
}

#[test]
fn cache_sweep_missing_stage_does_not_claim_owning_host_was_not_launched() {
    let mut observed = vec![json!({"cohort":"skippy-old","status":"completed"})];
    super::pipeline::child_refused(
        &mut observed,
        "skippy-new",
        &json!({"concurrency":4}),
        false,
    );
    assert_eq!(observed[0]["status"], "completed");
    assert_eq!(observed[1]["status"], "not-run");
    assert_eq!(observed[1]["launch_observation"], "unknown");
    assert!(observed[1]["client_metrics"].is_null());
}
