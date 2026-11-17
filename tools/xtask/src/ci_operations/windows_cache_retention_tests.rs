use super::*;
use serde_json::{Value, json};

fn key(identity: &str, hash: char) -> String {
    format!(
        "{}{}-{}",
        identity::NAMESPACE,
        identity,
        hash.to_string().repeat(64)
    )
}

fn cache(id: u64, key: &str, bytes: u64) -> Value {
    json!({"id":id,"key":key,"version":"version-a","ref":"refs/heads/main",
        "size_in_bytes":bytes,"created_at":format!("2026-11-{:02}T00:00:00Z",id),
        "last_accessed_at":"2026-11-30T00:00:00Z"})
}

fn policy(keys: &[String], bytes: u64, history: usize) -> Value {
    json!({"schema_version":1,"target_ref":"refs/heads/main","current_keys":keys,
        "active_families_complete":true,"max_bytes":bytes,"history_per_family":history})
}

fn evaluate(rows: &[Value], policy: &Value) -> DynResult<Plan> {
    plan_json(
        &serde_json::to_vec(&json!({"total_count":rows.len(),"actions_caches":rows}))?,
        &serde_json::to_vec(policy)?,
    )
}

fn ids(rows: &[Cache]) -> Vec<u64> {
    rows.iter().map(|cache| cache.id).collect()
}

#[test]
fn preserves_every_active_family_before_any_history() {
    let cuda12 = key(
        "cuda-61_75_80_86_89_90-cuda-12.9.2-Jimver-v0.2.36-github-win22-image-X64",
        'a',
    );
    let cuda13 = key(
        "cuda-75_80_86_89_90_100_103_120-cuda-13.1.2-Jimver-v0.2.36-github-win22-image-X64",
        'a',
    );
    let old = cuda12.replace(&"a".repeat(64), &"b".repeat(64));
    let plan = evaluate(
        &[
            cache(1, &cuda13, 6),
            cache(2, &cuda12, 4),
            cache(3, &old, 4),
        ],
        &policy(&[cuda12, cuda13], 10, 2),
    )
    .unwrap();
    assert_eq!(ids(&plan.retain), [1, 2]);
    assert_eq!(ids(&plan.delete), [3]);
    assert_eq!(plan.retained_bytes, 10);
}

#[test]
fn history_obeys_count_and_inclusive_byte_boundaries() {
    let current = key("cpu--cpu-github-win22-image-X64", 'a');
    let older = current.replace(&"a".repeat(64), &"b".repeat(64));
    let oldest = current.replace(&"a".repeat(64), &"c".repeat(64));
    let rows = [
        cache(3, &current, 5),
        cache(2, &older, 3),
        cache(1, &oldest, 1),
    ];
    for (bytes, history, expected) in [
        (8, 2, vec![2, 3]),
        (7, 2, vec![1, 3]),
        (20, 1, vec![3]),
        (9, 3, vec![1, 2, 3]),
    ] {
        let plan = evaluate(
            &rows,
            &policy(std::slice::from_ref(&current), bytes, history),
        )
        .unwrap();
        assert_eq!(ids(&plan.retain), expected);
    }
}

#[test]
fn keeps_newest_version_for_current_key_and_counts_other_versions_as_history() {
    let current = key("cpu--cpu-epoch", 'a');
    let mut newest = cache(3, &current, 2);
    newest["version"] = json!("version-b");
    let rows = [
        cache(1, &current, 2),
        newest,
        cache(2, &current.replace(&"a".repeat(64), &"b".repeat(64)), 2),
    ];
    let plan = evaluate(&rows, &policy(&[current], 4, 2)).unwrap();
    assert_eq!(ids(&plan.retain), [2, 3]);
    assert_eq!(ids(&plan.delete), [1]);
}

#[test]
fn uses_newest_family_member_when_current_key_is_cold() {
    let current = key("cpu--cpu-epoch", 'a');
    let older = current.replace(&"a".repeat(64), &"b".repeat(64));
    let plan = evaluate(
        &[cache(1, &older, 2), cache(2, &older, 2)],
        &policy(&[current], 2, 1),
    )
    .unwrap();
    assert_eq!(ids(&plan.retain), [2]);
}

#[test]
fn obsolete_epochs_are_removed_only_from_exact_owned_main_scope() {
    let current = key("cpu--cpu-current-epoch", 'a');
    let stale = key("cpu--cpu-inactive-epoch", 'a');
    let mut pr = cache(3, &stale, 4);
    pr["ref"] = json!("refs/pull/2794/merge");
    let mut branch = cache(4, &stale, 4);
    branch["ref"] = json!("refs/heads/feature");
    let plan = evaluate(
        &[
            cache(1, &current, 3),
            cache(2, &stale, 4),
            pr,
            branch,
            cache(5, "other-cache", 4),
        ],
        &policy(&[current], 3, 1),
    )
    .unwrap();
    assert_eq!(ids(&plan.delete), [2]);
    assert_eq!(ids(&plan.untouched), [3, 4, 5]);
    assert_eq!(
        (
            plan.inventory_bytes,
            plan.retained_bytes,
            plan.deleted_bytes,
            plan.untouched_bytes
        ),
        (19, 3, 4, 12)
    );
}

#[test]
fn refuses_insufficient_minimum_budget_and_incomplete_or_pr_policy() {
    let current = key("cpu--cpu-epoch", 'a');
    let rows = [cache(1, &current, 5)];
    for (field, value) in [
        ("max_bytes", json!(4)),
        ("active_families_complete", json!(false)),
        ("target_ref", json!("refs/pull/2794/merge")),
        ("schema_version", json!(2)),
        ("history_per_family", json!(0)),
        ("current_keys", json!([])),
    ] {
        let mut invalid = policy(std::slice::from_ref(&current), 5, 1);
        invalid[field] = value;
        assert!(evaluate(&rows, &invalid).is_err(), "{field}");
    }
}

#[test]
fn rejects_duplicate_ids_overflow_bad_dates_and_malformed_owned_keys() {
    let current = key("cpu--cpu-epoch", 'a');
    let policy = policy(std::slice::from_ref(&current), 10, 1);
    let valid = cache(1, &current, 1);
    assert!(evaluate(&[valid.clone(), valid.clone()], &policy).is_err());
    assert!(
        evaluate(
            &[cache(1, &current, u64::MAX), cache(2, &current, 1)],
            &policy
        )
        .is_err()
    );
    for (field, value) in [
        ("created_at", json!("2026-02-30T00:00:00Z")),
        ("last_accessed_at", json!("2026-11-01T00:00:00")),
        ("id", json!(0)),
        ("size_in_bytes", json!(-1)),
        ("version", json!("")),
        ("key", json!(format!("{}cuda-75-bad", identity::NAMESPACE))),
    ] {
        let mut invalid = valid.clone();
        invalid[field] = value;
        assert!(evaluate(&[invalid], &policy).is_err(), "{field}");
    }
}

#[test]
fn rejects_truncated_and_oversized_inputs_without_a_plan() {
    let policy = serde_json::to_vec(&policy(&[key("cpu--cpu-epoch", 'a')], 10, 1)).unwrap();
    for inventory in [
        br#"{"total_count":1,"actions_caches":[]}"#.to_vec(),
        br#"{"total_count":0,"actions_caches":[]"#.to_vec(),
        vec![b' '; input::INPUT_LIMIT + 1],
        vec![0xff],
    ] {
        assert!(plan_json(&inventory, &policy).is_err());
    }
}

#[test]
fn full_prefix_separates_toolkits_architectures_rocm_variants_and_epochs() {
    let identities = [
        "cuda-75-cuda-12.6.3-Jimver-v0.2.36-epoch1",
        "cuda-75-cuda-13.1.2-Jimver-v0.2.36-epoch1",
        "cuda-75_80-cuda-12.6.3-Jimver-v0.2.36-epoch1",
        "cuda-75-cuda-12.6.3-Jimver-v0.2.36-epoch2",
        "rocm-gfx1100-rocm-AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe-epoch1",
        "rocm-gfx90a_gfx942-rocm-AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe-epoch1",
        "rocm-gfx1100-rocm-AMD-Software-PRO-Edition-26.Q1-WinSvr2022-For-HIP.exe-epoch1",
        "vulkan--vulkan-1.4.328.1-jakoch-v1.5.2-epoch1",
        "cpu--cpu-epoch1",
    ];
    let families = identities
        .iter()
        .map(|identity| family(&key(identity, 'a')).unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(families.len(), 9);
}

#[test]
fn stable_plan_order_ignores_inventory_order_and_breaks_timestamp_ties_by_id() {
    let current = key("cpu--cpu-epoch", 'a');
    let first = cache(1, &current, 2);
    let mut second = cache(2, &current, 2);
    second["created_at"] = first["created_at"].clone();
    let policy = policy(&[current], 2, 1);
    let forward = evaluate(&[first.clone(), second.clone()], &policy).unwrap();
    let reverse = evaluate(&[second, first], &policy).unwrap();
    assert_eq!(ids(&forward.retain), [2]);
    assert_eq!(
        serde_json::to_value(forward).unwrap(),
        serde_json::to_value(reverse).unwrap()
    );
}

#[test]
fn cli_reads_supplied_files_and_reports_planned_retention() {
    let temp = tempfile::tempdir().unwrap();
    let inventory = temp.path().join("inventory.json");
    let policy_path = temp.path().join("policy.json");
    let current = key("cpu--cpu-epoch", 'a');
    std::fs::write(
        &inventory,
        serde_json::to_vec(&json!({"total_count":1,"actions_caches":[cache(1,&current,2)]}))
            .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &policy_path,
        serde_json::to_vec(&policy(&[current], 2, 1)).unwrap(),
    )
    .unwrap();
    let args = vec![
        "--inventory".into(),
        inventory.display().to_string(),
        "--policy".into(),
        policy_path.display().to_string(),
    ];
    let report = run(&args);
    assert_eq!(report.code, 0);
    let plan: Value = serde_json::from_str(&report.stdout).unwrap();
    assert_eq!(plan["retain"][0]["id"], 1);
}
