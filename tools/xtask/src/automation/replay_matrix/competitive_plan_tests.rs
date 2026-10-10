use super::*;
fn fixture() -> Value {
    serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../skippy/evals/skippy-competitive-benchmark.json"
    )))
    .unwrap()
}
fn plan(document: &Value, keys: &[&str], workloads: &[&str]) -> DynResult<Value> {
    config::admit(document)?;
    build(
        document,
        &serde_json::to_vec(document)?,
        &["cuda", "metal"],
        keys,
        workloads,
    )
}
#[test]
fn competitive_plan_keeps_complete_models_platforms_and_reviewed_concurrency_ladder() {
    let document = fixture();
    let output = plan(&document, &[], &["synthetic", "thoughtworks"]).unwrap();
    let cells = output["cells"].as_array().unwrap();
    let models = document["models"].as_array().unwrap();
    let outputs = document["synthetic"]["output_tokens"].as_array().unwrap();
    assert_eq!(
        cells.len(),
        2 * models.len() * 2 * config::LADDER.len() * (outputs.len() + 1)
    );
    for platform in ["cuda", "metal"] {
        for model in models {
            for concurrency in config::LADDER {
                assert!(cells.iter().any(|cell| cell["platform"] == platform
                    && cell["model"] == model["key"]
                    && cell["concurrency"] == concurrency
                    && cell["workload"] == "thoughtworks"));
                for output in outputs {
                    for arm in ["llama", "mesh"] {
                        assert!(cells.iter().any(|cell| cell["platform"] == platform
                            && cell["model"] == model["key"]
                            && cell["concurrency"] == concurrency
                            && cell["output_tokens"] == *output
                            && cell["arm"] == arm
                            && cell["workload"] == "synthetic"));
                    }
                }
            }
        }
    }
}
#[test]
fn competitive_trace_model_overrides_and_fallback_share_fixed_runtime_shape() {
    let mut document = fixture();
    document["models"][0]["thoughtworks_context_size"] = json!(65536);
    document["models"][0]["thoughtworks_active_lanes"] = json!(7);
    document["models"][1]
        .as_object_mut()
        .unwrap()
        .remove("thoughtworks_context_size");
    document["models"][1]
        .as_object_mut()
        .unwrap()
        .remove("thoughtworks_active_lanes");
    let output = plan(&document, &[], &["thoughtworks"]).unwrap();
    for cell in output["cells"].as_array().unwrap() {
        if cell["model"] == document["models"][0]["key"] {
            assert_eq!(cell["context_size"], 65536);
            assert_eq!(cell["active_lanes"], 7);
        } else if cell["model"] == document["models"][1]["key"] {
            assert_eq!(
                cell["context_size"],
                document["thoughtworks"]["context_size"]
            );
            assert_eq!(
                cell["active_lanes"],
                document["thoughtworks"]["active_lanes"]
            );
        }
    }
}
#[test]
fn competitive_config_refuses_nonpositive_boolean_shapes_and_reasonless_exclusions() {
    for field in ["thoughtworks_context_size", "thoughtworks_active_lanes"] {
        for value in [json!(0), json!(-1), json!(true), json!(false), json!(1.5)] {
            let mut document = fixture();
            document["models"][0][field] = value;
            assert!(config::admit(&document).is_err(), "{field}");
        }
    }
    let mut document = fixture();
    document["models"][0]["comparison_support"] = json!({"vllm":{"available":false}});
    assert!(config::admit(&document).is_err());
    document["models"][0]["comparison_support"]["vllm"]["reason"] =
        json!("pinned unsupported architecture");
    assert!(config::admit(&document).is_ok());
    document["models"][0]["comparison_support"]["vllm"]["available"] = json!("false");
    assert!(config::admit(&document).is_err());
}
#[test]
fn competitive_trace_alternates_arms_and_keeps_complete_prompt_waves() {
    let document = fixture();
    let key = document["models"][0]["key"].as_str().unwrap();
    let output = plan(&document, &[key], &["thoughtworks"]).unwrap();
    let cells = output["cells"].as_array().unwrap();
    for platform in ["cuda", "metal"] {
        let selected: Vec<_> = cells
            .iter()
            .filter(|cell| cell["platform"] == platform)
            .collect();
        for (index, concurrency) in config::LADDER.into_iter().enumerate() {
            let pair = &selected[index * 2..index * 2 + 2];
            let expected = if index % 2 == 0 {
                ["llama", "mesh"]
            } else {
                ["mesh", "llama"]
            };
            for (cell, arm) in pair.iter().zip(expected) {
                assert_eq!(cell["arm"], arm);
                let prompts = cell["prompt_count"].as_u64().unwrap();
                assert!(prompts >= concurrency);
                assert_eq!(prompts % concurrency, 0);
            }
        }
    }
}
#[test]
fn competitive_cli_filters_source_binds_output_and_refuses_unknown_selection() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("config with spaces.json");
    let bytes = serde_json::to_vec_pretty(&fixture()).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let args = vec![
        "--config".into(),
        path.to_str().unwrap().into(),
        "--platform".into(),
        "rocm".into(),
        "--workload".into(),
        "thoughtworks".into(),
        "--model".into(),
        "llama32-dense".into(),
    ];
    let accepted = report(&args);
    assert_eq!(accepted.code, 0, "{}", accepted.stderr);
    assert!(accepted.stderr.is_empty());
    let output: Value = serde_json::from_str(&accepted.stdout).unwrap();
    assert_eq!(output["schema_version"], 2);
    assert_eq!(output["config_hash_kind"], "source_bytes_sha256");
    assert_eq!(output["config_sha256"], hex::encode(Sha256::digest(&bytes)));
    assert_eq!(output["platforms"], json!(["rocm"]));
    assert_eq!(output["models"], json!(["llama32-dense"]));
    assert!(
        output["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|cell| cell["workload"] == "thoughtworks")
    );
    let mut refused = args.clone();
    refused.extend(["--model".into(), "unknown".into()]);
    assert_ne!(report(&refused).code, 0);
    assert!(report(&refused).stdout.is_empty());
    refused = args;
    refused.extend(["--platform".into(), "unsupported".into()]);
    assert_ne!(report(&refused).code, 0);
    assert!(report(&refused).stdout.is_empty());
    temporary.close().unwrap();
}
