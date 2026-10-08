use super::*;
fn input() -> Input {
    serde_json::from_value(json!({"schema_version":1,"case_key":"llama","model_id":"fixture","correctness":"/fixture","correctness_sha256":"a".repeat(64),"stage_server":"/fixture","stage_server_sha256":"a".repeat(64),"model":"/fixture","model_sha256":"a".repeat(64),"native_build":"/fixture","native_build_sha256":"a".repeat(64),"ctx_size":512,"prefix_tokens":8,"cache_hit_repeats":2,"runtime_lane_count":1,"source_port":19061,"restore_port":19062,"n_gpu_layers":-1,"prompt":null,"topologies":["one-stage"],"borrow_resident_hits":false,"cache_decoded_result_hits":false,"execution_seconds":30,"cell_seconds":8,"settings":{}})).unwrap()
}
#[test]
fn cache_topology_covers_actual_layer_ranges_and_recurrent_payload() {
    let ranges = [
        Topology::SplitStage0,
        Topology::SplitMiddle,
        Topology::SplitFinal,
    ]
    .map(|t| t.range(8).unwrap());
    assert_eq!(ranges, [(0, 2, 0), (2, 5, 1), (5, 8, 2)]);
    assert_eq!(Topology::OneStage.range(8).unwrap(), (0, 8, 0));
    assert_eq!(catalog::family("falcon_h1").unwrap().1, "kv-recurrent");
    assert_eq!(
        catalog::family("deepseek3").unwrap(),
        ("DeepSeek3", "resident-kv")
    );
    assert!(catalog::family("qwen3moe").is_err());
}
#[test]
fn cache_admission_refuses_unowned_settings_ports_and_duplicate_topologies() {
    let mut i = input();
    i.validate().unwrap();
    i.settings.insert("HF_TOKEN".into(), "private".into());
    assert!(i.validate().is_err());
    i.settings.clear();
    i.restore_port = i.source_port;
    assert!(i.validate().is_err());
    i.restore_port = 19062;
    i.topologies.push(Topology::OneStage);
    assert!(i.validate().is_err());
}
fn value() -> Value {
    json!({"mode":"state-handoff","status":"pass","matches":true,"predicted_token_matches":true,"cache_hit_matches":true,"model_identity":{"model_id":"fixture"},"state_payload_kind":"resident-kv","stage_index":0,"layer_start":0,"layer_end":6,"requested_prefix_token_count":8,"benchmark_prompt_token_count":9,"benchmark_prompt_text":"observed prompt","activation_width":16,"cache_hit_repeats":2,"cache_hit_import_ms":[1.0,2.0],"cache_hit_decode_ms":[3.0,4.0],"recompute_total_ms":8.0,"cache_hit_total_ms":5.0})
}
#[test]
fn cache_observed_report_requires_full_roster_and_correlated_actual_prompt() {
    let receipt = admission::Receipt {
        request_sha256: "a".repeat(64),
        admitted: input(),
        layers: 6,
        activation_width: 16,
        model_identity: json!({}),
    };
    report::accept(&value(), &receipt, Topology::OneStage).unwrap();
    for (field, bad) in [
        ("benchmark_prompt_token_count", json!(8)),
        ("cache_hit_decode_ms", json!([3.0])),
        ("cache_hit_total_ms", json!(9)),
        ("cache_hit_matches", json!(false)),
        ("model_identity", json!({"model_id":"other"})),
    ] {
        let mut v = value();
        v[field] = bad;
        assert!(
            report::accept(&v, &receipt, Topology::OneStage).is_err(),
            "{field}"
        );
    }
}
#[test]
fn cache_precancel_retains_partial_receipt_without_launch() {
    let root = tempfile::tempdir().unwrap();
    let c = Cancellation::default();
    c.cancel();
    let mut v = execute(&input(), root.path(), &c).unwrap();
    assert_eq!(v["status"], "failed-or-incomplete");
    assert_eq!(v["completed_topologies"], 0);
    assert!(!root.path().join("cache-correctness-stage.json").exists());
    super::finalize(&mut v, true, false, true);
    super::publish(&root.path().join("cache-correctness-stage.json"), &v).unwrap();
    let observed: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.path().join("cache-correctness-stage.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(observed["completed_topologies"], 0);
    assert_eq!(observed["status"], "failed-or-incomplete");
    assert_eq!(observed["terminal_refusal"]["cancelled"], true);
    assert!(!root.path().join("topology-00").exists());
}
#[test]
fn cache_single_file_metadata_refuses_unbound_shards_and_duplicate_fields() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("shape.gguf");
    for entries in [
        vec![],
        vec![("split.count", 1_u16), ("split.no", 0)],
        vec![("split.count", 2)],
        vec![("split.count", 1), ("split.count", 1)],
        vec![("split.no", 1)],
    ] {
        let mut b = b"GGUF".to_vec();
        b.extend(3_u32.to_le_bytes());
        b.extend(0_u64.to_le_bytes());
        b.extend((entries.len() as u64).to_le_bytes());
        for (key, value) in &entries {
            b.extend((key.len() as u64).to_le_bytes());
            b.extend(key.as_bytes());
            b.extend(2_u32.to_le_bytes());
            b.extend(value.to_le_bytes());
        }
        std::fs::write(&path, b).unwrap();
        let accepted = entries.is_empty() || entries == vec![("split.count", 1), ("split.no", 0)];
        assert_eq!(
            crate::automation::replay_matrix::model_preflight::require_single_file(&path).is_ok(),
            accepted
        );
    }
}

#[test]
fn cache_observed_report_refuses_finite_sample_sum_overflow() {
    let receipt = admission::Receipt {
        request_sha256: "a".repeat(64),
        admitted: input(),
        layers: 6,
        activation_width: 16,
        model_identity: json!({}),
    };
    for decode in [json!([1e308, 1e308]), json!([0.0, 0.0])] {
        let mut report = value();
        report["cache_hit_import_ms"] = json!([1e308, 1e308]);
        report["cache_hit_decode_ms"] = decode;
        assert!(report::accept(&report, &receipt, Topology::OneStage).is_err());
    }
}

#[test]
fn cache_artifact_package_profile_preserves_exact_range_context_and_refuses_plain_deepseek() {
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    let mut i = input();
    i.model = root.join("package");
    i.correctness = root.join("correctness");
    i.stage_server = root.join("stage");
    i.native_build = root.join("build");
    i.case_key = "deepseek3".into();
    i.ctx_size = 32;
    i.prefix_tokens = 4;
    i.topologies = vec![Topology::PackageStage1];
    assert!(i.validate().is_err());
    i.artifact = Some(artifact::Artifact {
        kind: artifact::Kind::LayerPackage,
        tool: root.join("tool"),
        tool_sha256: "a".repeat(64),
        shard_pins: BTreeMap::new(),
    });
    i.validate().unwrap();
    assert_eq!(i.range(Topology::PackageStage1, 61).unwrap(), (3, 4, 1));
    i.ctx_size = 33;
    assert!(i.validate().is_err());
}
#[test]
fn cache_artifact_minimax_requires_all_three_pins_and_keeps_existing_single_file_refusal() {
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    let mut i = input();
    i.correctness = root.join("correctness");
    i.stage_server = root.join("stage");
    i.native_build = root.join("build");
    i.case_key = "minimax_m27".into();
    i.model = root.join("MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf");
    i.model_sha256 = "a".repeat(64);
    let pins = (1..=3)
        .map(|n| {
            (
                format!("MiniMax-M2.7-UD-Q2_K_XL-{n:05}-of-00003.gguf"),
                "a".repeat(64),
            )
        })
        .collect();
    i.artifact = Some(artifact::Artifact {
        kind: artifact::Kind::CompleteShards,
        tool: root.join("tool"),
        tool_sha256: "a".repeat(64),
        shard_pins: pins,
    });
    i.validate().unwrap();
    i.artifact.as_mut().unwrap().shard_pins.pop_last();
    assert!(i.validate().is_err());
}

#[test]
fn cache_family_correctness_terminal_admission_retains_observed_rows_process_and_prior_failure() {
    for (cancelled, expired, finish_ok) in [
        (false, false, true),
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ] {
        let mut receipt = json!({"status":"completed","rows":[{"status":"pass"}],"measurement":{"status":"completed"},"process":{"cleanup_complete":true},"error":"prior classified failure"});
        super::finalize(&mut receipt, cancelled, expired, finish_ok);
        assert_eq!(
            receipt["status"],
            if cancelled || expired || !finish_ok {
                "failed-or-incomplete"
            } else {
                "completed"
            }
        );
        assert_eq!(receipt["rows"][0]["status"], "pass");
        assert_eq!(receipt["measurement"]["status"], "completed");
        assert_eq!(receipt["process"]["cleanup_complete"], true);
        assert_eq!(receipt["error"], "prior classified failure");
        if cancelled || expired || !finish_ok {
            assert_eq!(receipt["terminal_refusal"]["cancelled"], cancelled);
            assert_eq!(receipt["terminal_refusal"]["deadline_expired"], expired);
            assert_eq!(
                receipt["terminal_refusal"]["interrupt_finish_failed"],
                !finish_ok
            );
        }
    }
}
