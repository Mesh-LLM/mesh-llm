//! Explicit manual workload admission; no fixture or acceptance claim is invented.
use super::{
    prompt_manifest, synthetic_prompts,
    workload_plan::{Plan, Workload},
};
use crate::command::DynResult;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(super) fn resolve(
    mut workload: Workload,
    model_id: &str,
    model_sha256: &str,
    manifest: Option<&[u8]>,
) -> DynResult<Plan> {
    workload.validate()?;
    if model_id.trim().is_empty()
        || model_sha256.len() != 64
        || !model_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("manual workload requires model identity and full SHA-256".into());
    }
    let (prompts, metadata, digest) = match manifest {
        Some(bytes) => {
            let document = prompt_manifest(bytes)?;
            (
                document.prompts,
                document.metadata,
                Some(hex::encode(Sha256::digest(bytes))),
            )
        }
        None => (
            synthetic_prompts::interleaved(
                workload.families,
                workload.requests_per_family,
                workload.prefix_blocks,
            )?,
            Map::new(),
            None,
        ),
    };
    if prompts.len() > 10000 {
        return Err("manual workload exceeds 10000 requests".into());
    }
    let requests = u64::try_from(prompts.len())?;
    if workload.admission_concurrency == 0 {
        workload.admission_concurrency = requests;
    }
    if workload.admission_concurrency < requests || workload.admission_concurrency > workload.lanes
    {
        return Err(
            "manual admission must cover the complete prompt roster and fit the runtime lanes"
                .into(),
        );
    }
    let mut counts = BTreeMap::new();
    for prompt in prompts {
        *counts.entry(prompt.family).or_insert(0_u64) += 1;
    }
    let successes = requests
        .checked_mul(workload.rounds)
        .ok_or("manual success count overflow")?;
    Ok(Plan {
        schema_version: 1,
        workload_profile: "manual".into(),
        fixture_catalog_sha256: None,
        model: json!({"id":model_id,"sha256":model_sha256,"custody":"supplied-local-model"}),
        workload,
        requests_per_round: requests,
        successful_requests_per_binary: successes,
        acceptance_contract_name: None,
        acceptance_contract_sha256: None,
        hardware_acceptance: Value::Null,
        cache_seed: None,
        prompt_manifest_sha256: digest,
        prompt_manifest_metadata: metadata,
        family_request_counts: counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn workload() -> Workload {
        serde_json::from_value(json!({"rounds":2,"families":2,"requests_per_family":2,"prefix_blocks":2,"output_tokens":2,"ctx_size":512,"lanes":4,"admission_concurrency":0,"cache_entries":1,"stagger_ms":0.0})).unwrap()
    }
    #[test]
    fn manual_plan_admits_actual_irregular_manifest_roster_without_fixture_or_acceptance() {
        let bytes=br#"{"metadata":{"source":"owned"},"prompts":[{"family":"a","prompt":"one"},{"family":"b","prompt":"two"},{"family":"a","prompt":"three"}]}"#;
        let plan = resolve(workload(), "fixture", &"a".repeat(64), Some(bytes)).unwrap();
        assert_eq!(plan.requests_per_round, 3);
        assert_eq!(plan.successful_requests_per_binary, 6);
        assert_eq!(
            plan.family_request_counts,
            BTreeMap::from([("a".into(), 2), ("b".into(), 1)])
        );
        assert_eq!(plan.prompt_manifest_metadata["source"], "owned");
        assert_eq!(
            plan.prompt_manifest_sha256,
            Some(hex::encode(Sha256::digest(bytes)))
        );
        assert!(plan.fixture_catalog_sha256.is_none() && plan.hardware_acceptance.is_null());
        assert_eq!(plan.workload.admission_concurrency, 3);
        let mut short = workload();
        short.lanes = 2;
        assert!(resolve(short, "fixture", &"a".repeat(64), Some(bytes)).is_err());
        assert!(
            resolve(
                workload(),
                "fixture",
                &"a".repeat(64),
                Some(br#"{"prompts":[]}"#)
            )
            .is_err()
        );
    }
    #[test]
    fn manual_synthetic_plan_keeps_interleaved_shape_and_complete_admission() {
        let plan = resolve(workload(), "fixture", &"a".repeat(64), None).unwrap();
        assert_eq!(plan.requests_per_round, 4);
        assert_eq!(plan.workload.admission_concurrency, 4);
        assert_eq!(
            plan.family_request_counts,
            BTreeMap::from([("family-0".into(), 2), ("family-1".into(), 2)])
        );
        assert!(plan.prompt_manifest_sha256.is_none());
    }
}
