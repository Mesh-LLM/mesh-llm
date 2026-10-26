//! Model-qualification pure helpers: usage contracts, cached-token reads,
//! suite model capability groups, and prefix sizing.
use super::*;
use serde_json::json;

#[test]
fn usage_tokens_accepts_positive_counts() {
    let response = json!({"usage": {"prompt_tokens": 5, "completion_tokens": 2}});
    assert_eq!(prompt_tokens(&response), Ok(5));
}

#[test]
fn usage_tokens_rejects_zero_decode() {
    let response = json!({"usage": {"prompt_tokens": 5, "completion_tokens": 0}});
    assert!(prompt_tokens(&response).is_err());
    let missing = json!({});
    assert!(prompt_tokens(&missing).is_err());
}

#[test]
fn cached_tokens_defaults_to_zero_and_reads_details() {
    assert_eq!(cached_tokens(&json!({})), 0);
    assert_eq!(
        cached_tokens(&json!({"usage": {"prompt_tokens_details": {"cached_tokens": 12}}})),
        12
    );
    assert_eq!(
        cached_tokens(&json!({"usage": {"prompt_tokens_details": {"cached_tokens": -1}}})),
        0
    );
}

#[test]
fn prefix_is_long_enough_for_a_reported_cache_hit() {
    // Roughly four bytes per token: the ~160-fact prefix must exceed the
    // ~500-token measured cache boundary by a comfortable margin.
    assert!(prefix().len() > 1100);
    assert!(prefix().starts_with("Reference facts: fact-0"));
    assert!(prefix().ends_with("fact-159"));
}

#[test]
fn suite_model_tags_cover_the_contract() {
    let spec = |artifact: &str| ModelSpec {
        artifact_id: artifact.to_owned(),
        path: std::path::PathBuf::from("/dev/null"),
        sha256: "a".repeat(64),
        model_id: artifact.to_owned(),
    };
    let mut known = BTreeMap::new();
    known.insert(
        "dense-model".to_owned(),
        BTreeSet::from(["dense".to_owned()]),
    );
    known.insert(
        "hybrid-model".to_owned(),
        BTreeSet::from(["hybrid".to_owned()]),
    );
    known.insert("moe-model".to_owned(), BTreeSet::from(["moe".to_owned()]));
    assert!(validate_suite_models("dense", &[spec("dense-model")], &known).is_ok());
    assert!(validate_suite_models("dense", &[spec("hybrid-model")], &known).is_err());
    assert!(validate_suite_models("recurrent", &[spec("hybrid-model")], &known).is_ok());
    assert!(validate_suite_models("moe", &[spec("moe-model")], &known).is_ok());
    assert!(
        validate_suite_models(
            "kv-cache",
            &[spec("dense-model"), spec("hybrid-model")],
            &known
        )
        .is_ok()
    );
    assert!(validate_suite_models("kv-cache", &[spec("dense-model")], &known).is_err());
    assert!(validate_suite_models("kv-cache", &[spec("hybrid-model")], &known).is_err());
    assert!(validate_suite_models("dense", &[spec("foreign-model")], &known).is_err());
}

#[test]
fn every_suite_driver_covers_the_contract_cases() {
    for (suite, cases) in contract::SUITE_CASES {
        if let Some((_, _restart)) = suite_driver(suite) {
            assert!(!cases.is_empty());
        } else {
            assert!(
                matches!(suite, "packaging-runtime" | "system-one" | "decisions"),
                "{suite}"
            );
        }
    }
}

#[test]
fn hardware_record_requires_available_row_and_expected_device() {
    let row = contract::parse_strict(
        br#"{"id":"linux-cuda","backend":"cuda","target":"x86_64-unknown-linux-gnu"}"#,
    )
    .unwrap();
    let availability = contract::parse_strict(
        br#"{"state":"available","row_id":"linux-cuda","runner":"gpu-nvidia"}"#,
    )
    .unwrap();
    let profile = json!({
        "gpus": [{"display_name": "NVIDIA Test GPU"}],
        "cuda": {"driver_version": "575.0", "toolkit_majors": [12, 13], "gpu_arches": ["sm_86", "sm_120"]},
    });
    let record = build_hardware(&row, &availability, &profile, "CUDA0").expect("hardware record");
    assert_eq!(record["actual_backend"], "cuda");
    assert_eq!(record["device"], "NVIDIA Test GPU");
    assert_eq!(record["driver"], "575.0");
    assert_eq!(record["runtime_version"], "CUDA 13");
    assert_eq!(record["device_architecture"], "sm_86");
    assert!(build_hardware(&row, &availability, &profile, "CPU").is_err());
    let unavailable =
        contract::parse_strict(br#"{"state":"hardware-unavailable","row_id":"linux-cuda"}"#)
            .unwrap();
    assert!(build_hardware(&row, &unavailable, &profile, "CUDA0").is_err());
}

#[test]
fn cpu_row_hardware_record_is_exact() {
    let row = contract::parse_strict(
        br#"{"id":"linux-cpu","backend":"cpu","target":"x86_64-unknown-linux-gnu"}"#,
    )
    .unwrap();
    let availability = contract::parse_strict(
        br#"{"state":"available","row_id":"linux-cpu","runner":"linux-hosted"}"#,
    )
    .unwrap();
    let record = build_hardware(&row, &availability, &json!({}), "CPU").expect("cpu record");
    assert_eq!(record["device"], "CPU");
    assert_eq!(record["runner"], "linux-hosted");
    assert!(record.get("driver").is_none());
}

#[test]
fn metal_runtime_version_falls_back_without_cosmetic_tokens() {
    let versioned = json!({"gpus": [{"display_name": "Metal 3.1 Support"}]});
    assert_eq!(
        metal_runtime_version(&versioned).as_deref(),
        Some("Metal 3.1")
    );
    let labeled = json!({"gpus": [{"display_name": "Metal Support: Metal"}]});
    assert_eq!(
        metal_runtime_version(&labeled).as_deref(),
        Some("Metal Support: Metal")
    );
}
