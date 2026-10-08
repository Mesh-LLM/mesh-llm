use super::{contract::Input, planning};
use serde_json::json;
use std::path::Path;
fn input(root: &Path) -> Input {
    serde_json::from_value(json!({"schema_version":1,"cache_root":root,"cases":[],"use_cases":[],"corpus":null,"prefix_sweep":[],"prefix_tokens":null,"n_gpu_layers":null,"cache_hit_repeats":null,"runtime_lane_count":null,"serving_ctx_size":null,"concurrency":[1,2,4,8,16,32,64],"concurrent_requests":64,"concurrent_output_tokens":32,"llama_parallel":1,"llama_repeats":3,"ttft_slo_ms":2000,"tpot_slo_ms":100,"skip_llama_server":false,"old_server":null,"new_server":null,"model_sha256":{}})).unwrap()
}
#[test]
fn cache_catalog_preserves_full_current_presets_revision_and_missing_rows() {
    let root = tempfile::tempdir().unwrap();
    let p = planning::plan(&input(root.path())).unwrap();
    assert_eq!(p["cell_count"], 14);
    assert!(p["execution"].is_null());
    let rows = p["cells"].as_array().unwrap();
    assert!(
        rows.iter()
            .all(|c| c["model_observation"]["status"] == "missing-model")
    );
    let llama = rows.iter().find(|c| c["key"] == "llama").unwrap();
    assert_eq!(
        llama["case"]["revision"],
        "7d1f70022fcab2038000074bd0342e03e1d8b755"
    );
    assert_eq!(llama["case"]["layer_end"], 16);
    assert_eq!(llama["case"]["activation_width"], 2048);
    let deep = rows.iter().find(|c| c["key"] == "deepseek3").unwrap();
    assert_eq!(deep["case"]["ctx_size"], 32);
    assert_eq!(deep["case"]["prefix_tokens"], 4);
    assert_eq!(deep["case"]["state_layer_start"], 3);
    assert_eq!(deep["case"]["state_layer_end"], 4);
    assert_eq!(deep["case"]["stage_load_mode"], "layer-package");
    assert_eq!(deep["tasks"]["serial_baseline"]["planned"], false);
    assert_eq!(
        rows.iter().find(|c| c["key"] == "falcon_h1").unwrap()["case"]["payload"],
        "kv-recurrent"
    );
}
fn corpus(root: &Path) -> (std::path::PathBuf, String) {
    use sha2::{Digest, Sha256};
    let path = root.join("corpus.json");
    let b=serde_json::to_vec(&json!({"version":1,"use_cases":[{"key":"tool","label":"Tool","prefix_tokens":256,"prompt":"literal prompt","source":{"dataset":"owner/source","row_idx":7}},{"key":"code","label":"Code","prefix_tokens":32,"prompt":"code prompt","source":{}}]})).unwrap();
    std::fs::write(&path, &b).unwrap();
    (path, hex::encode(Sha256::digest(&b)))
}
#[test]
fn cache_matrix_prefix_precedence_and_usecase_provenance_are_explicit() {
    let root = tempfile::tempdir().unwrap();
    let mut i = input(root.path());
    i.cases = vec!["llama".into()];
    i.use_cases = vec!["all".into()];
    let (path, sha256) = corpus(root.path());
    i.corpus = Some(super::contract::Pin { path, sha256 });
    i.prefix_sweep = vec![64, 1024];
    i.n_gpu_layers = Some(-1);
    let p = planning::plan(&i).unwrap();
    assert_eq!(p["cell_count"], 4);
    assert_eq!(p["cells"][0]["case"]["prefix_tokens"], 64);
    assert_eq!(p["cells"][2]["case"]["ctx_size"], 1152);
    assert_eq!(p["cells"][0]["use_case"]["source"]["row_idx"], 7);
    assert_eq!(p["cells"][0]["case"]["n_gpu_layers"], -1);
    i.prefix_tokens = Some(16);
    let p = planning::plan(&i).unwrap();
    assert_eq!(p["cell_count"], 2);
    assert_eq!(p["cells"][1]["case"]["prefix_tokens"], 16);
    i.prefix_tokens = None;
    i.prefix_sweep.clear();
    assert_eq!(
        planning::plan(&i).unwrap()["cells"][0]["case"]["prefix_tokens"],
        256
    );
}
#[test]
fn cache_plan_refuses_unknown_catalog_mixed_all_and_corrupt_corpus_pin() {
    let root = tempfile::tempdir().unwrap();
    let mut i = input(root.path());
    i.cases = vec!["qwen3moe".into()];
    assert!(planning::plan(&i).is_err());
    i.cases.clear();
    i.use_cases = vec!["all".into(), "tool".into()];
    assert!(planning::plan(&i).is_err());
    let (path, _) = corpus(root.path());
    i.use_cases = vec!["all".into()];
    i.corpus = Some(super::contract::Pin {
        path,
        sha256: "a".repeat(64),
    });
    assert!(planning::plan(&i).is_err());
    i.corpus = None;
    i.use_cases.clear();
    i.prefix_sweep = vec![1, 1];
    assert!(planning::plan(&i).is_err());
}
#[test]
fn cache_plan_present_bytes_remain_unqualified_and_skip_applies_all_baselines() {
    let root = tempfile::tempdir().unwrap();
    let mut i = input(root.path());
    i.cases = vec!["llama".into()];
    let c = planning::catalog()
        .unwrap()
        .into_iter()
        .find(|c| c.key == "llama")
        .unwrap();
    let path = root.path().join(c.snapshot_relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, b"not admitted GGUF").unwrap();
    i.skip_llama_server = true;
    i.model_sha256.insert("llama".into(), "a".repeat(64));
    let p = planning::plan(&i).unwrap();
    let cell = &p["cells"][0];
    assert_eq!(cell["model_observation"]["status"], "present-unqualified");
    assert_eq!(cell["tasks"]["correctness"]["planned"], true);
    assert_eq!(cell["tasks"]["serial_baseline"]["planned"], false);
    assert_eq!(cell["tasks"]["concurrent_baseline"]["planned"], false);
    assert_eq!(
        cell["model_observation"]["custody"],
        "path_metadata_only_not_byte_or_model_admission"
    );
}
