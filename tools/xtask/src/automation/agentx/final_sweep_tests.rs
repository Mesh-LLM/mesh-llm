use super::{
    final_sweep::{self, COHORT_SHA, ORDER},
    prepared_launch,
    prepared_tools::{Tools, Tree},
};
use serde_json::{Value, json};

pub(super) fn arms() -> Vec<Value> {
    ORDER.into_iter().enumerate().map(|(index,engine)|json!({"index":index+1,"engine":engine,"status":"passed",
        "cells":([1,2,4].map(|c|json!({"offered_sessions":c,"output_tok_s":20.0,
            "cohort":{"selected_cohort_sha256":COHORT_SHA},"phase_outcome":{"errors":0,"cancelled":0},
            "completed_work":{"requests":69,"output_tokens":18713,"source_turn_output_sha256":"a".repeat(64)}})))})).collect()
}
#[test]
fn complete_balanced_work_rejects_partial_reordered_cancelled_or_changed_cells() {
    let complete = arms();
    final_sweep::validate_work(&complete).unwrap();
    assert!(final_sweep::validate_work(&complete[..5]).is_err());
    for (pointer, value) in [
        ("/engine", json!("skippy")),
        ("/cells/0/offered_sessions", json!(4)),
        ("/cells/0/completed_work/output_tokens", json!(18712)),
        ("/cells/0/cohort/selected_cohort_sha256", json!("changed")),
        (
            "/cells/0/completed_work/source_turn_output_sha256",
            json!("b".repeat(64)),
        ),
        ("/cells/0/phase_outcome/cancelled", json!(1)),
        ("/cells/0/output_tok_s", json!(-1)),
    ] {
        let mut changed = complete.clone();
        *changed[0].pointer_mut(pointer).unwrap() = value;
        assert!(final_sweep::validate_work(&changed).is_err(), "{pointer}");
    }
}
fn tools() -> Tools {
    serde_json::from_str(include_str!(
        "../../../../../skippy/evals/agentx/tools.example.json"
    ))
    .unwrap()
}
#[test]
fn product_auto_capacity_is_preserved_and_diagnostic_override_is_verified() {
    let mut tools = tools();
    let mut args = vec!["serve".into(), "--n-gpu-layers".into(), "99".into()];
    prepared_launch::arguments(&mut args, "skippy", &tools);
    assert!(!args.iter().any(|v| v == "--generation-concurrency"));
    let config = json!({"stage":{"lane_count":4,"ctx_size":131072,"n_batch":2048,"n_ubatch":512,
        "cache_type_k":"f16","cache_type_v":"f16","flash_attn_type":"enabled","kv_cache":{"l2_max_bytes":8589934592u64}},
        "frontend":{"prefill_chunk_size":64,"prefill_chunk_policy":"fixed","generation_concurrency":4},
        "options":{"kv-cache-disk":"off"},"constraints":{"unified_kv":"required"},"execution":{"threads":20,"threads_batch":20}});
    prepared_launch::validate_config(&config, &tools).unwrap();
    tools.diagnostic_generation_lanes = Some(1);
    assert!(prepared_launch::validate_config(&config, &tools).is_err());
    tools.diagnostic_generation_lanes = None;
    let mut wrong = config;
    wrong["stage"]["kv_cache"]["l2_max_bytes"] = json!(1073741824);
    assert!(prepared_launch::validate_config(&wrong, &tools).is_err());
}
#[test]
fn prepared_tree_identity_detects_mutation_and_refuses_directory_links() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("code"), b"first").unwrap();
    let tree = Tree {
        path: dir.path().into(),
        sha256: crate::product::digest::tree_sha256(dir.path())
            .map_err(|e| e.error)
            .unwrap(),
    };
    tree.verify().unwrap();
    std::fs::write(dir.path().join("code"), b"changed").unwrap();
    assert!(tree.verify().is_err());
    #[cfg(unix)]
    {
        let target = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(target.path(), dir.path().join("editable")).unwrap();
        assert!(tree.verify().is_err());
    }
}

#[test]
fn manifest_cannot_pair_wrong_source_backend_or_upstream_with_pinned_tools() {
    let tools = tools();
    let manifest = json!({"runtime":{"backend":{"kind":"cuda"},"skippy_abi":"0.1.76"},
        "build":{"llama_patched_sha":tools.native_source_commit,"llama_patch_digest":tools.native_patch_digest,"llama_upstream_sha":tools.standard_upstream}});
    tools.validate_manifest(&manifest).unwrap();
    for pointer in [
        "/build/llama_patched_sha",
        "/build/llama_patch_digest",
        "/build/llama_upstream_sha",
        "/runtime/backend/kind",
        "/runtime/skippy_abi",
    ] {
        let mut wrong = manifest.clone();
        *wrong.pointer_mut(pointer).unwrap() = json!("different");
        assert!(tools.validate_manifest(&wrong).is_err(), "{pointer}");
    }
}

#[cfg(unix)]
#[test]
fn ordinary_venv_alias_is_pinned_without_permitting_external_editable_packages() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("lib")).unwrap();
    std::fs::create_dir(dir.path().join("other")).unwrap();
    std::fs::write(dir.path().join("lib/package"), b"package").unwrap();
    let alias = dir.path().join("lib64");
    std::os::unix::fs::symlink("lib", &alias).unwrap();
    let mut tree = Tree {
        path: dir.path().into(),
        sha256: String::new(),
    };
    tree.sha256 = tree.observe().unwrap();
    tree.verify().unwrap();
    std::fs::remove_file(&alias).unwrap();
    std::os::unix::fs::symlink("other", &alias).unwrap();
    assert!(tree.verify().is_err());
    std::fs::remove_file(&alias).unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), &alias).unwrap();
    assert!(tree.observe().is_err());
}
