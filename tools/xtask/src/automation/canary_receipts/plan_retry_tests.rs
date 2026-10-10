use super::*;
use serde_json::json;
const PLAN: &[u8] = include_bytes!("../../../tests/migration_canary_receipts/fixtures/plan.json");
const GIB: u64 = 1024 * 1024 * 1024;
fn families(names: &[&str]) -> BTreeSet<Family> {
    names
        .iter()
        .map(|name| (*name).to_owned().try_into().unwrap())
        .collect()
}
fn document() -> Value {
    let mut document: Value = serde_json::from_slice(PLAN).unwrap();
    let weights = b"pinned synthetic GGUF fixture";
    for model in document["selected_models"].as_array_mut().unwrap() {
        model["artifact"] = json!({"files":["fixture-model.gguf"],"file_integrity":{"fixture-model.gguf":{"size_bytes":weights.len(),"sha256":super::super::Digest::of_bytes(weights)}}});
        model["resources"] = json!({"estimated_model_bytes":8*GIB});
        if model["family"] == "hybrid" {
            model["resources"]["minimum_runner_memory_gib"] = json!(256);
        }
    }
    for row in document["github_matrix"]["include"].as_array_mut().unwrap() {
        row["id"] = json!(format!("family-{}", row["families"].as_str().unwrap()));
        row["estimated_work_bytes"] = json!(8 * GIB);
    }
    document
}
fn parse(document: &Value) -> SourceFamilyPlan {
    SourceFamilyPlan::parse(&serde_json::to_vec(document).unwrap()).unwrap()
}

#[test]
fn retry_matrix_preserves_original_order_identity_and_unknown_row_fields() {
    let mut document = document();
    let rows = document["github_matrix"]["include"].as_array_mut().unwrap();
    rows.reverse();
    for (index, row) in rows.iter_mut().enumerate() {
        row["future_contract"] = json!({"version":9,"opaque":[index,true,"preserve"]});
        row["placement"] = json!({"historical_source_owner":"retain unknown nested field"});
    }
    let original = rows.clone();
    let bytes = serde_json::to_vec(&document).unwrap();
    let plan = SourceFamilyPlan::parse(&bytes).unwrap();
    let projected = plan.retry_matrix(&families(&["dense", "hybrid"])).unwrap();
    assert_eq!(plan.canonical_bytes, bytes);
    for (row, original) in projected["include"]
        .as_array()
        .unwrap()
        .iter()
        .zip(original)
    {
        for (key, value) in original.as_object().unwrap() {
            assert_eq!(&row[key], value, "original source field {key}");
        }
        assert_eq!(row["resident_model_bytes"], 8 * GIB);
        assert_eq!(row["estimated_peak_bytes"], 16 * GIB);
        assert!(matches!(
            row["memory_tier"].as_str(),
            Some("accelerator-memory-128plus" | "accelerator-memory-256plus")
        ));
    }
    assert_eq!(
        plan.retry_matrix(&BTreeSet::new()).unwrap(),
        json!({"include":[]})
    );
}

#[test]
fn controller_placement_overrides_stale_source_claims_and_removes_absent_minimum() {
    let mut document = document();
    for row in document["github_matrix"]["include"].as_array_mut().unwrap() {
        row["memory_tier"] = json!("arbitrary-untrusted-runner");
        row["resident_model_bytes"] = json!(1);
        row["runtime_allowance_bytes"] = json!(1);
        row["estimated_peak_bytes"] = json!(1);
        row["minimum_runner_memory_gib"] = json!(999);
    }
    let projected = parse(&document)
        .retry_matrix(&families(&["dense", "hybrid"]))
        .unwrap();
    for row in projected["include"].as_array().unwrap() {
        assert_eq!(row["resident_model_bytes"], 8 * GIB);
        assert_eq!(row["runtime_allowance_bytes"], 8 * GIB);
        assert_eq!(row["estimated_peak_bytes"], 16 * GIB);
        if row["families"] == "hybrid" {
            assert_eq!(row["memory_tier"], "accelerator-memory-256plus");
            assert_eq!(row["minimum_runner_memory_gib"], 256);
        } else {
            assert_eq!(row["memory_tier"], "accelerator-memory-128plus");
            assert!(row.get("minimum_runner_memory_gib").is_none());
        }
    }
}

#[test]
fn retry_matrix_filters_exact_family_without_reassigning_original_shard_or_identity() {
    let document = document();
    let expected = document["github_matrix"]["include"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["families"] == "hybrid")
        .unwrap();
    let plan = parse(&document);
    let projected = plan.retry_matrix(&families(&["hybrid"])).unwrap();
    let rows = projected["include"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    for key in ["families", "id", "shard_index", "estimated_work_bytes"] {
        assert_eq!(rows[0][key], expected[key]);
    }
    assert_eq!(rows[0]["memory_tier"], "accelerator-memory-256plus");
    assert!(plan.retry_matrix(&families(&["not-planned"])).is_err());
}

#[test]
fn invalid_pinned_resource_contract_cannot_emit_a_retry_runner() {
    for fault in [
        "missing-integrity",
        "zero-size",
        "overflow",
        "minimum",
        "capacity",
    ] {
        let mut document = document();
        let model = &mut document["selected_models"][0];
        match fault {
            "missing-integrity" => model["artifact"]["file_integrity"] = json!({}),
            "zero-size" => {
                model["artifact"]["file_integrity"]["fixture-model.gguf"]["size_bytes"] = json!(0)
            }
            "overflow" => model["resources"]["estimated_model_bytes"] = json!(u64::MAX),
            "minimum" => model["resources"]["minimum_runner_memory_gib"] = json!(129),
            "capacity" => model["resources"]["estimated_model_bytes"] = json!(256 * GIB),
            _ => unreachable!(),
        }
        assert!(
            parse(&document)
                .retry_matrix(&families(&["dense"]))
                .is_err(),
            "{fault}"
        );
    }
}

#[test]
fn retained_unknown_fields_follow_json_last_value_semantics() {
    let mut document = document();
    let row = document["github_matrix"]["include"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["families"] == "dense")
        .unwrap();
    row["future"] = json!("first");
    let bytes = serde_json::to_string(&document).unwrap().replace(
        "\"future\":\"first\"",
        "\"future\":\"first\",\"future\":\"final\"",
    );
    let plan = SourceFamilyPlan::parse(bytes.as_bytes()).unwrap();
    assert_eq!(
        plan.retry_matrix(&families(&["dense"])).unwrap()["include"][0]["future"],
        "final"
    );
}
