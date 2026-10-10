use super::*;
use serde_json::{Value, json};
fn input() -> contract::Input {
    serde_json::from_value(json!({"schema_version":1,"arms":[{"name":"off","base_url":"http://127.0.0.1:1","declared_stages":2,"declared_mtp_capable":true},{"name":"suffix","base_url":"http://127.0.0.1:2","declared_stages":2,"declared_mtp_capable":true}],"model":"model","execution_timeout_ms":10000})).unwrap()
}
fn response(source: &str, hits: u64) -> Value {
    json!({"choices":[{"message":{"content":"same"},"finish_reason":"stop"}],"timings":{"predicted_n":2,"predicted_per_second":50.0,"draft_n":2,"draft_n_accepted":1,"native_mtp_ngram_proposer":source,"native_mtp_hybrid_ngram_tokens":2,"native_mtp_hybrid_accepted_tail_tokens":1,"native_mtp_ngram_proposer_attempts":hits,"native_mtp_ngram_proposer_hits":hits,"native_mtp_ngram_proposer_match_length_max":2,"native_mtp_ngram_proposer_candidates_examined":1,"native_mtp_ngram_proposer_appended_tokens":2,"native_mtp_ngram_proposer_rebuilds":1,"native_mtp_ngram_proposer_sync_us":4,"native_mtp_ngram_proposer_lookup_us":5}})
}
#[test]
fn suffix_native_schema_defaults_and_named_arm_negative_gates() {
    let mut i = input();
    i.validate().unwrap();
    assert_eq!(
        (i.warmups, i.runs, i.max_tokens, i.seed),
        (2, 5, 256, 20260721)
    );
    i.arms[1].name = "off".into();
    assert!(i.validate().is_err());
    i = input();
    i.arms[0].base_url = "https://user:password@example.org".into();
    assert!(i.validate().is_err());
    i = input();
    i.arms[0].declared_stages = 1;
    assert!(i.validate().is_err());
}
#[test]
fn suffix_observed_timing_projection_and_activation_are_not_chunk_estimates() {
    let i = input();
    let w = contract::Workload {
        name: "edit".into(),
        prompt: "prompt".into(),
    };
    let s = sample::decode(&response("suffix", 1), &i.arms[1], &w, 0, &i, 0.5).unwrap();
    assert_eq!(s.output_sha256, evidence::digest(b"same"));
    assert_eq!(s.wall_tok_s, 4.0);
    summary::activation(std::slice::from_ref(&s), Some("suffix")).unwrap();
    let mut v = response("suffix", 0);
    assert!(
        summary::activation(
            &[sample::decode(&v, &i.arms[1], &w, 0, &i, 0.5).unwrap()],
            Some("suffix")
        )
        .is_err()
    );
    v["timings"].as_object_mut().unwrap().remove("predicted_n");
    assert!(sample::decode(&v, &i.arms[1], &w, 0, &i, 0.5).is_err());
    v = response("suffix", 1);
    v["timings"]["native_mtp_hybrid_accepted_tail_tokens"] = json!(3);
    assert!(sample::decode(&v, &i.arms[1], &w, 0, &i, 0.5).is_err());
}
#[test]
fn suffix_summary_retains_paired_hash_mismatch_and_hybrid_statistics() {
    let i = input();
    let w = contract::Workload {
        name: "edit".into(),
        prompt: "prompt".into(),
    };
    let a = sample::decode(&response("off", 0), &i.arms[0], &w, 0, &i, 1.0).unwrap();
    let mut v = response("suffix", 1);
    v["choices"][0]["message"]["content"] = json!("different");
    let b = sample::decode(&v, &i.arms[1], &w, 0, &i, 0.5).unwrap();
    let sum = summary::summarize(&[a, b], "off").unwrap();
    assert_eq!(sum["output_hash_mismatches"].as_array().unwrap().len(), 1);
    assert_eq!(sum["rows"][1]["ngram_acceptance"], 0.5);
    assert!(summary::markdown(&sum).contains("Output hash mismatches: 1"));
}

#[test]
fn suffix_advertised_identity_excludes_dynamic_created_timestamp_and_binds_model_roster() {
    let a = json!({"data":[{"id":"model","created":1,"owned_by":"skippy-runtime"}]});
    let b = json!({"data":[{"id":"model","created":2,"owned_by":"skippy-runtime"}]});
    assert_eq!(
        execution::models_identity(&a, "model").unwrap(),
        execution::models_identity(&b, "model").unwrap()
    );
    let c = json!({"data":[{"id":"other"}]});
    assert!(execution::models_identity(&c, "model").is_err());
    let d = json!({"data":[{"id":"model"},{"id":"other"}]});
    assert_ne!(
        execution::models_identity(&a, "model").unwrap(),
        execution::models_identity(&d, "model").unwrap()
    );
}
