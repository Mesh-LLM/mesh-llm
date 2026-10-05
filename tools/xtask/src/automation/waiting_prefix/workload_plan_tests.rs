use super::*;
use serde_json::json;

const CATALOG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../skippy/evals/skippy-scheduler-fixtures.json"
));
const CAPACITY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../skippy/evals/skippy-capacity-acceptance.json"
));

fn prepared() -> (Value, Vec<u8>) {
    let mut catalog: Value = serde_json::from_slice(CATALOG).unwrap();
    let prompts = (0..2).flat_map(|request| (0..8).map(move |family| json!({"family":format!("family-{family}"),"prompt":format!("work {family} {request}")}))).collect::<Vec<_>>();
    let bytes = serde_json::to_vec(&json!({"metadata":{},"prompts":prompts})).unwrap();
    catalog["profiles"]["agentic-eviction-pressure"]["corpus"]["prompt_manifest_sha256"] =
        hash(&bytes).into();
    (catalog, bytes)
}

fn plan(
    catalog: &Value,
    profile: &str,
    contract: Option<&[u8]>,
    prompts: Option<&[u8]>,
) -> DynResult<Plan> {
    let model = &catalog["profiles"][profile]["model"];
    resolve(
        &serde_json::to_vec(catalog)?,
        profile,
        model["id"].as_str().unwrap(),
        model["sha256"].as_str().unwrap(),
        contract,
        prompts,
    )
}

#[test]
fn capacity_contract_applies_only_cache_override_and_keeps_seed_outside_measured_counts() {
    let (catalog, prompts) = prepared();
    let resolved = plan(
        &catalog,
        "agentic-eviction-pressure",
        Some(CAPACITY),
        Some(&prompts),
    )
    .unwrap();
    assert_eq!(resolved.workload.cache_entries, 16);
    assert_eq!(resolved.workload.families, 8);
    assert_eq!(resolved.workload.rounds, 4);
    assert_eq!(resolved.requests_per_round, 16);
    assert_eq!(resolved.successful_requests_per_binary, 64);
    let seed = resolved.cache_seed.unwrap();
    assert_eq!(seed.families, 8);
    assert_eq!(seed.prefix_blocks, 192);
    assert_eq!(seed.output_tokens, 16);
    assert_eq!(
        resolved.acceptance_contract_sha256.as_deref(),
        Some(hash(CAPACITY).as_str())
    );
    assert_eq!(
        resolved.prompt_manifest_sha256.as_deref(),
        Some(hash(&prompts).as_str())
    );
}

#[test]
fn warm_profile_preserves_complete_workload_and_no_external_manifest() {
    let catalog: Value = serde_json::from_slice(CATALOG).unwrap();
    let resolved = plan(&catalog, "warm-affinity", None, None).unwrap();
    assert_eq!(resolved.requests_per_round, 6);
    assert_eq!(resolved.successful_requests_per_binary, 24);
    assert_eq!(resolved.workload.cache_entries, 1);
    assert!(resolved.cache_seed.is_none());
    assert!(resolved.acceptance_contract_name.is_none());
    assert!(plan(&catalog, "warm-affinity", None, Some(b"{}")).is_err());
    let model = &catalog["profiles"]["warm-affinity"]["model"];
    assert!(
        resolve(
            CATALOG,
            "warm-affinity",
            "wrong",
            model["sha256"].as_str().unwrap(),
            None,
            None
        )
        .is_err()
    );
}

#[test]
fn malformed_overrides_and_seed_never_change_the_workload() {
    let (catalog, prompts) = prepared();
    for (pointer, value) in [
        ("/workload_overrides/cache_entries", json!(0)),
        ("/workload_overrides/cache_entries", json!(-1)),
        ("/workload_overrides/cache_entries", json!(1.5)),
        ("/workload_overrides/cache_entries", json!(true)),
        ("/cache_seed/families", json!(0)),
        ("/cache_seed/prefix_blocks", json!(0)),
        ("/cache_seed/output_tokens", json!(0)),
        ("/cache_seed/stagger_ms", json!(0)),
        ("/cache_seed/stagger_ms", json!(-2)),
        ("/name", json!(" ")),
        ("/schema_version", json!(2)),
    ] {
        let mut contract: Value = serde_json::from_slice(CAPACITY).unwrap();
        *contract.pointer_mut(pointer).unwrap() = value;
        assert!(
            plan(
                &catalog,
                "agentic-eviction-pressure",
                Some(&serde_json::to_vec(&contract).unwrap()),
                Some(&prompts)
            )
            .is_err(),
            "{pointer}"
        );
    }
    let mut contract: Value = serde_json::from_slice(CAPACITY).unwrap();
    contract["workload_overrides"]["families"] = json!(1);
    assert!(
        plan(
            &catalog,
            "agentic-eviction-pressure",
            Some(&serde_json::to_vec(&contract).unwrap()),
            Some(&prompts)
        )
        .is_err()
    );
    contract = serde_json::from_slice(CAPACITY).unwrap();
    contract["cache_seed"]["unreviewed"] = json!(1);
    assert!(
        plan(
            &catalog,
            "agentic-eviction-pressure",
            Some(&serde_json::to_vec(&contract).unwrap()),
            Some(&prompts)
        )
        .is_err()
    );
}

#[test]
fn contract_profile_and_successes_are_bound_to_the_whole_selected_workload() {
    let (catalog, prompts) = prepared();
    for (pointer, value) in [
        ("/workload_profile", json!("warm-affinity")),
        (
            "/hardware_acceptance/successful_requests_per_binary",
            json!(16),
        ),
    ] {
        let mut contract: Value = serde_json::from_slice(CAPACITY).unwrap();
        *contract.pointer_mut(pointer).unwrap() = value;
        assert!(
            plan(
                &catalog,
                "agentic-eviction-pressure",
                Some(&serde_json::to_vec(&contract).unwrap()),
                Some(&prompts)
            )
            .is_err()
        );
    }
}

#[test]
fn pinned_manifest_bytes_and_complete_families_are_both_required() {
    let (mut catalog, prompts) = prepared();
    let mut corrupt = prompts.clone();
    corrupt.push(b'\n');
    assert!(plan(&catalog, "agentic-eviction-pressure", None, Some(&corrupt)).is_err());
    assert!(plan(&catalog, "agentic-eviction-pressure", None, None).is_err());
    let mut short: Value = serde_json::from_slice(&prompts).unwrap();
    short["prompts"].as_array_mut().unwrap().pop();
    let short = serde_json::to_vec(&short).unwrap();
    catalog["profiles"]["agentic-eviction-pressure"]["corpus"]["prompt_manifest_sha256"] =
        hash(&short).into();
    assert!(plan(&catalog, "agentic-eviction-pressure", None, Some(&short)).is_err());
}
