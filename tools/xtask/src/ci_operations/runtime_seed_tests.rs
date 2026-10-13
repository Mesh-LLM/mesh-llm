use super::runtime_seed_stats::Snapshot;
use super::runtime_seed_types::Cache;
use serde_json::json;

fn stats() -> serde_json::Value {
    serde_json::from_str(include_str!("../../../../ci/runtime-seed-evidence/34272984200-1/runtime-seed-evidence-3-warm-1/raw-stats.json")).expect("retained stats")
}

#[test]
fn every_required_scalar_rejects_missing_boolean_and_negative_counts() {
    for field in [
        "compile_requests",
        "requests_unsupported_compiler",
        "requests_not_compile",
        "requests_not_cacheable",
        "requests_executed",
        "cache_timeouts",
        "cache_read_errors",
        "non_cacheable_compilations",
        "forced_recaches",
        "cache_write_errors",
        "cache_writes",
        "compilations",
        "compile_fails",
        "dist_errors",
    ] {
        for corruption in [serde_json::Value::Null, json!(true), json!(-1), json!({})] {
            let mut value = stats();
            value["stats"][field] = corruption;
            assert!(
                Snapshot::parse(&serde_json::to_vec(&value).expect("JSON")).is_err(),
                "{field}"
            );
        }
        let mut value = stats();
        value["stats"].as_object_mut().expect("stats").remove(field);
        assert!(
            Snapshot::parse(&serde_json::to_vec(&value).expect("JSON")).is_err(),
            "{field}"
        );
    }
}

#[test]
fn raw_stats_strip_locations_and_preserve_language_counters() {
    let mut value = stats();
    value["cache_location"] = "private-location".into();
    value["stats"]["unconsumed_duration"] = json!({"secs":1});
    let parsed = Snapshot::parse(&serde_json::to_vec(&value).expect("JSON")).expect("stats");
    let retained = serde_json::to_value(&parsed).expect("serialize");
    assert!(retained.get("cache_location").is_none());
    assert!(retained["stats"].get("unconsumed_duration").is_none());
    assert_eq!(retained["stats"]["cache_misses"]["counts"]["C/C++"], 603);
    assert_eq!(
        parsed.measurement().expect("measurement").native_requests,
        603
    );
}

#[test]
fn malformed_language_maps_and_cache_errors_reject() {
    for field in ["cache_hits", "cache_misses", "cache_errors"] {
        for part in ["counts", "adv_counts"] {
            let mut value = stats();
            value["stats"][field][part] = json!({"C/C++":true});
            assert!(Snapshot::parse(&serde_json::to_vec(&value).expect("JSON")).is_err());
        }
    }
    for field in ["cache_read_errors", "cache_write_errors"] {
        let mut value = stats();
        value["stats"][field] = 1.into();
        assert!(
            Snapshot::parse(&serde_json::to_vec(&value).expect("JSON"))
                .expect("stats")
                .measurement()
                .is_err()
        );
    }
}

#[test]
fn cache_admission_rejects_wrong_ref_size_id_and_version() {
    let original = json!({"id":123, "key":"fixture", "version":"b".repeat(64), "ref":"refs/heads/main", "size_in_bytes":1024});
    let cache: Cache = serde_json::from_value(original.clone()).expect("cache");
    assert!(cache.validate().is_ok());
    for (field, corruption) in [
        ("id", json!(0)),
        ("version", json!("wrong")),
        ("ref", json!("refs/heads/feature")),
        ("size_in_bytes", json!(0)),
        ("size_in_bytes", json!(2147483649u64)),
    ] {
        let mut value = original.clone();
        value[field] = corruption;
        assert!(
            serde_json::from_value::<Cache>(value)
                .expect("cache")
                .validate()
                .is_err()
        );
    }
}
