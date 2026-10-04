use super::*;

#[test]
fn invalid_manifest_and_selection_reject_before_emitting_any_plan() {
    let changes: &[ManifestChange] = &[
        ("duplicate family", |doc| {
            doc["models"][1]["family"] = json!("zeta")
        }),
        ("invalid label", |doc| {
            doc["models"][0]["family"] = json!("Bad Label")
        }),
        ("missing core lane", |doc| {
            doc["policy"]["profiles"]["full"]["required_lanes"] = json!(["single-step"])
        }),
        ("missing target artifact", |doc| {
            doc["models"][0]
                .as_object_mut()
                .expect("row")
                .remove("artifact");
        }),
        ("unsafe file path", |doc| {
            doc["models"][0]["artifact"]["files"] = json!(["../escape.gguf"]);
            doc["models"][0]["artifact"]["file_integrity"] = json!({"../escape.gguf": {"size_bytes": 1, "blob_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}});
        }),
        ("zero weight", |doc| {
            doc["models"][0]["resources"]["estimated_model_bytes"] = json!(0)
        }),
        ("boolean weight", |doc| {
            doc["models"][0]["resources"]["estimated_model_bytes"] = json!(true)
        }),
        ("non-chat oracle missing evidence", |doc| {
            doc["models"][0]["class"] = json!("embedding");
            doc["models"][0]["profile"] = json!("workload-oracle");
        }),
        ("non-chat class requires workload", |doc| {
            doc["models"][0]["class"] = json!("embedding")
        }),
    ];
    for (name, change) in changes {
        let path = with_manifest(*change);
        let output = run(&["--manifest", path.to_str().expect("UTF-8")]);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{name}");
        assert!(
            String::from_utf8_lossy(&output.stderr).starts_with("family battery plan failed: "),
            "{name}"
        );
        fs::remove_file(path).expect("cleanup fixture");
    }
    for families in ["zeta,zeta", "bad label", "unknown"] {
        let output = run(&[
            "--manifest",
            "tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json",
            "--families",
            families,
        ]);
        assert_eq!(output.status.code(), Some(2), "{families}");
        assert!(output.stdout.is_empty());
    }
    let empty_path = with_manifest(|manifest| manifest["models"] = json!([]));
    let empty = run(&["--manifest", empty_path.to_str().expect("UTF-8")]);
    assert_eq!(empty.status.code(), Some(2));
    assert!(empty.stdout.is_empty());
    assert!(String::from_utf8_lossy(&empty.stderr).contains("models must be a non-empty array"));
    fs::remove_file(empty_path).expect("cleanup fixture");
}

#[test]
fn github_without_output_preserves_stdout_and_validates_manifest_first() {
    let github = temp_path("github-not-written.txt");
    for existing in [false, true] {
        if existing {
            fs::write(&github, b"existing=kept\n").expect("seed output");
        }
        let output = run_historical(&[
            "--families",
            "llama,qwen3-dense",
            "--github-output",
            github.to_str().expect("path"),
        ]);
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(output.stdout, fixture("real-reversed", "stdout"));
        assert_eq!(
            output.stderr,
            b"family battery plan failed: --github-output requires --output\n"
        );
        let invalid = run(&[
            "--manifest",
            "tools/xtask/tests/fixtures/family_evidence/malformed-manifest.json",
            "--github-output",
            github.to_str().expect("path"),
        ]);
        assert_eq!(invalid.status.code(), Some(2));
        assert_eq!(invalid.stdout, fixture("malformed-manifest", "stdout"));
        assert_eq!(invalid.stderr, fixture("malformed-manifest", "stderr"));
        if existing {
            assert_eq!(
                fs::read(&github).expect("untouched output"),
                b"existing=kept\n"
            );
            fs::remove_file(&github).expect("cleanup output");
        } else {
            assert!(!github.exists());
        }
    }
}

#[test]
fn non_chat_and_optional_semantics_are_not_skipped_for_unselected_rows() {
    let path = with_manifest(|manifest| {
        manifest["models"][4]["class"] = json!("ocr");
        manifest["models"][4]["profile"] = json!("workload-oracle");
        manifest["models"][4]["evidence"] = json!({"fixture":"image", "comparison":"exact"});
    });
    let output = run(&[
        "--manifest",
        path.to_str().expect("UTF-8"),
        "--families",
        "zeta",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("models[4].class ocr requires an mmproj_artifact")
    );
    fs::remove_file(path).expect("cleanup fixture");
}

#[test]
fn verification_rejects_extra_fields_and_missing_shards() {
    let mut plan: Value =
        serde_json::from_slice(&fixture("real-reversed", "stdout")).expect("frozen plan");
    plan["unexpected"] = json!(true);
    let path = temp_path("tampered-plan.json");
    fs::write(&path, serde_json::to_vec(&plan).expect("JSON")).expect("write plan");
    let extra = run_historical(&["--verify-plan", path.to_str().expect("UTF-8")]);
    assert_eq!(extra.status.code(), Some(2));
    assert_eq!(extra.stderr, fixture("tampered-plan", "stderr"));
    assert!(extra.stdout.is_empty());
    plan["shards"] = json!([]);
    fs::write(&path, serde_json::to_vec(&plan).expect("JSON")).expect("write plan");
    let missing = run_historical(&["--verify-plan", path.to_str().expect("UTF-8")]);
    assert_eq!(missing.status.code(), Some(2));
    assert_eq!(
        missing.stderr,
        b"family battery plan failed: plan.shards must be a nonempty list\n"
    );
    assert!(missing.stdout.is_empty());
    fs::remove_file(path).expect("cleanup fixture");
}

#[test]
fn bounded_flags_reject_unimplemented_modes_and_malformed_values() {
    for flags in [
        vec!["--check-cache"],
        vec!["--inspect-gguf", "model.gguf"],
        vec!["--shard-count", "abc"],
    ] {
        let output = run(&flags);
        assert_eq!(output.status.code(), Some(2), "{flags:?}");
        assert!(output.stdout.is_empty(), "{flags:?}");
        assert!(!output.stderr.is_empty(), "{flags:?}");
    }
    let huge = run(&[
        "--shard-count",
        "18446744073709551616000000000000000",
        "--families",
        "llama,qwen3-dense",
    ]);
    assert_eq!(
        huge.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&huge.stderr)
    );
    assert!(huge.stdout.is_empty());
}

#[test]
fn invalid_label_diagnostic_uses_json_quoting_and_keeps_one_error_line() {
    for (label, quoted) in [("bad'label\"", true), ("bad\n'label\"", false)] {
        let path = with_manifest(|manifest| manifest["models"][0]["family"] = json!(label));
        let output = run(&["--manifest", path.to_str().expect("UTF-8")]);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).expect("UTF-8 diagnostic");
        assert_eq!(stderr.lines().count(), 1);
        if quoted {
            assert!(stderr.contains("has an invalid label"));
            assert!(stderr.contains(&serde_json::to_string(label).expect("JSON label")));
        } else {
            assert!(stderr.contains("models[0].family must be a non-empty single-line string"));
        }
        fs::remove_file(path).expect("cleanup fixture");
    }
}

#[test]
fn profile_and_non_causal_policy_refusals_preserve_specific_domain_reasons() {
    let changes: &[ManifestChange] = &[
        (
            "activation_width must be an unsigned 64-bit integer >= 1",
            |doc| {
                doc["models"][0]["execution"]
                    .as_object_mut()
                    .expect("execution")
                    .remove("activation_width");
            },
        ),
        ("must require exactly the three core lanes", |doc| {
            doc["policy"]["profiles"]["full"]["required_lanes"] =
                json!(["state-handoff", "chain", "single-step"]);
        }),
        ("must require exactly the three core lanes", |doc| {
            doc["policy"]["profiles"]["package-oracle"]["required_lanes"] =
                json!(["single-step", "chain", "state-handoff", "graph-parse"]);
        }),
        (
            "workload-smoke must remain provisional and oracle-free",
            |doc| {
                doc["policy"]["profiles"]["workload-smoke"]["oracle"] = json!("local-monolithic");
            },
        ),
        (
            "workload-oracle requires certified local-monolithic smoke and oracle lanes",
            |doc| {
                doc["policy"]["profiles"]["workload-oracle"]["required_lanes"] =
                    json!(["class-specific-smoke"]);
            },
        ),
        ("models[0].class must be", |doc| {
            doc["models"][0]
                .as_object_mut()
                .expect("model")
                .remove("class");
        }),
        ("models[0].class must be", |doc| {
            doc["models"][0]["class"] = json!("guessed-from-name");
        }),
        ("requires an mmproj_artifact", |doc| {
            doc["models"][0]["class"] = json!("speech_recognition");
            doc["models"][0]["profile"] = json!("workload-smoke");
        }),
        ("must disable speculative decoding", |doc| {
            doc["models"][0]["class"] = json!("embedding");
            doc["models"][0]["profile"] = json!("workload-smoke");
            doc["models"][0]["execution"]["speculative_policy"] = json!("mtp-if-present");
        }),
        ("must not request split or MTP certification", |doc| {
            doc["models"][0]["class"] = json!("embedding");
            doc["models"][0]["profile"] = json!("workload-smoke");
            doc["models"][0]["execution"]["mtp_layers"] = json!(1);
        }),
    ];
    for (reason, change) in changes {
        let path = with_manifest(*change);
        let output = run(&["--manifest", path.to_str().expect("UTF-8")]);
        assert_eq!(output.status.code(), Some(2), "{reason}");
        assert!(output.stdout.is_empty(), "{reason}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(reason),
            "{reason}: {:?}",
            output.stderr
        );
        fs::remove_file(path).expect("cleanup fixture");
    }
}

#[test]
fn current_manifest_preserves_every_family_class_projector_and_native_head() {
    let manifest: Value = serde_json::from_slice(
        &fs::read(root().join("ci/llama-canary/family-certified.json")).expect("current manifest"),
    )
    .expect("manifest JSON");
    let output = run(&["--shard-count", "4"]);
    assert!(output.status.success(), "{:?}", output.stderr);
    let plan: Value = serde_json::from_slice(&output.stdout).expect("plan JSON");
    let source = manifest["models"].as_array().expect("source models");
    let selected = plan["selected_models"].as_array().expect("selected models");
    assert_eq!(selected.len(), source.len());
    let families: std::collections::BTreeSet<_> = selected
        .iter()
        .map(|model| model["family"].as_str().expect("family"))
        .collect();
    assert_eq!(families.len(), source.len());
    for model in source {
        let actual = selected
            .iter()
            .find(|row| row["family"] == model["family"])
            .expect("family retained");
        assert_eq!(actual["class"], model["class"]);
        assert_eq!(actual["architecture"], model["architecture"]);
        assert_eq!(actual["artifact"], model["artifact"]);
        assert_eq!(actual["mmproj_artifact"], model["mmproj_artifact"]);
        let heads = model["execution"]["mtp_layers"]
            .as_u64()
            .expect("native heads");
        assert_eq!(actual["execution"]["mtp_layers"], json!(heads));
        let lanes = actual["certification_lanes"].as_array().expect("lanes");
        if model["class"] == "causal_generation" {
            assert_eq!(
                lanes.iter().any(|lane| lane == "native-mtp-heads"),
                heads > 0
            );
            assert!(lanes.iter().any(|lane| lane == "single-step"));
            assert!(lanes.iter().any(|lane| lane == "chain"));
            assert!(lanes.iter().any(|lane| lane == "state-handoff"));
        } else {
            assert_eq!(heads, 0);
            assert_eq!(lanes.len(), 2);
            assert!(lanes[0].as_str().expect("smoke lane").ends_with("-smoke"));
            assert!(lanes[1].as_str().expect("oracle lane").ends_with("-oracle"));
        }
    }
    let sharded: Vec<_> = plan["shards"]
        .as_array()
        .expect("shards")
        .iter()
        .flat_map(|shard| shard["families"].as_array().expect("shard families").iter())
        .map(|family| family.as_str().expect("family"))
        .collect();
    assert_eq!(sharded.len(), families.len());
    assert_eq!(
        sharded
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>(),
        families
    );
}

#[test]
fn current_supplied_plan_rejects_coordinated_omission_and_oracle_promotion() {
    for omission in [true, false] {
        let output = if omission {
            run(&["--shard-count", "4"])
        } else {
            run(&["--families", "nomic-bert-embedding"])
        };
        assert!(output.status.success(), "{:?}", output.stderr);
        let mut plan: Value = serde_json::from_slice(&output.stdout).expect("generated plan");
        if omission {
            let removed = plan["selected_models"]
                .as_array_mut()
                .expect("models")
                .pop()
                .expect("row");
            plan["selected_family_count"] =
                json!(plan["selected_models"].as_array().expect("models").len());
            for shard in plan["shards"].as_array_mut().expect("shards") {
                shard["families"]
                    .as_array_mut()
                    .expect("families")
                    .retain(|family| family != &removed["family"]);
            }
        } else {
            plan["selected_models"][0]["oracle"] = json!("none");
        }
        let path = temp_path("current-tampered-plan.json");
        fs::write(&path, serde_json::to_vec(&plan).expect("JSON")).expect("write plan");
        let rejected = run(&["--verify-plan", path.to_str().expect("UTF-8")]);
        assert_eq!(rejected.status.code(), Some(2), "{:?}", rejected.stderr);
        assert!(rejected.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&rejected.stderr)
                .contains("differs from the canonical manifest and selection")
        );
        fs::remove_file(path).expect("cleanup");
    }
}

#[test]
fn current_matrix_orders_smallest_first_and_sharding_is_reproducible() {
    let first = run(&["--shard-count", "4"]);
    let second = run(&["--shard-count", "4"]);
    assert!(first.status.success(), "{:?}", first.stderr);
    assert!(second.status.success(), "{:?}", second.stderr);
    assert_eq!(first.stdout, second.stdout);
    let output = run(&["--shard-count", "256"]);
    assert!(output.status.success(), "{:?}", output.stderr);
    let plan: Value = serde_json::from_slice(&output.stdout).expect("JSON");
    let mut selected = plan["selected_models"]
        .as_array()
        .expect("models")
        .iter()
        .collect::<Vec<_>>();
    selected.sort_by_key(|model| {
        (
            model["resources"]["estimated_model_bytes"]
                .as_u64()
                .expect("bytes"),
            model["family"].as_str().expect("family"),
        )
    });
    let rows = plan["github_matrix"]["include"].as_array().expect("matrix");
    assert_eq!(rows.len(), selected.len());
    let mut indices = std::collections::BTreeSet::new();
    for (row, expected) in rows.iter().zip(selected) {
        assert_eq!(row["families"], expected["family"]);
        assert!(indices.insert(row["shard_index"].as_u64().expect("index")));
        let shard = plan["shards"]
            .as_array()
            .expect("shards")
            .iter()
            .find(|shard| shard["shard_index"] == row["shard_index"])
            .expect("matching shard");
        assert_eq!(shard["families"], json!([expected["family"]]));
    }
}

#[test]
fn obsolete_cadence_selection_is_rejected_without_emitting_a_plan() {
    let output = run(&["--cadence", "nightly"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}
