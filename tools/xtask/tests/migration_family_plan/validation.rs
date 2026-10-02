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
