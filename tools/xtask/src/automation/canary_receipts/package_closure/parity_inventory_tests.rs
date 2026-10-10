use super::*;
#[path = "parity_inventory/boundary_selection_tests.rs"]
mod boundary_selection_tests;
fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).into()).collect()
}
#[test]
fn inventory_requires_every_source_all_statuses_and_real_paired_boundaries() {
    let family = json!({"models":[]});
    let parity = json!({"candidates":[{"llama_model":"foo","status":"candidate"},{"llama_model":"bar","status":"non_causal_aux"}]});
    assert!(validate(&parity, &family, &set(&["foo", "bar"]), &set(&["foo"])).is_ok());
    assert!(
        validate(
            &parity,
            &family,
            &set(&["foo", "bar", "unclassified"]),
            &set(&["foo"])
        )
        .is_err()
    );
    assert!(validate(&parity, &family, &set(&["foo", "bar"]), &set(&[])).is_err());
    let mut changed = parity.clone();
    changed["candidates"][0]["status"] = json!("uncertified_magic");
    assert!(validate(&changed, &family, &set(&["foo"]), &set(&["foo"])).is_err());
    changed = parity;
    changed["candidates"][0]["unsupported_reason"] = json!("not actually runnable");
    assert!(validate(&changed, &family, &set(&["foo"]), &set(&["foo"])).is_err());
}
#[test]
fn pins_join_actual_certified_identity_and_reject_every_binding_change() {
    let pin = json!({"repo":"org/model","revision":"a".repeat(40),"file":"model.gguf","selector":"Q4","size_bytes":42,"blob_sha256":"b".repeat(64)});
    let family = json!({"models":[{"artifact":{"repo":"org/model","revision":"a".repeat(40),"selector":"Q4","file_integrity":{"model.gguf":{"size_bytes":42,"blob_id":"b".repeat(64)}}}}]});
    assert!(validate_pin(&pin, &family).is_ok());
    for (key, value) in [
        ("repo", json!("other/model")),
        ("revision", json!("main")),
        ("file", json!("other.gguf")),
        ("selector", json!("Q8")),
        ("size_bytes", json!(43)),
        ("blob_sha256", json!("c".repeat(64))),
    ] {
        let mut changed = pin.clone();
        changed[key] = value;
        assert!(
            validate_pin(&changed, &family).is_err(),
            "must reject {key}"
        );
    }
}

#[test]
fn actual_native_files_require_real_calls_and_exact_source_coverage() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("src/models")).unwrap();
    let path = root.path().join("src/models/foo.cpp");
    fs::write(
        &path,
        "stage->begin_block (layer); stage->end_block(layer);",
    )
    .unwrap();
    let parity = json!({"candidates":[{"llama_model":"foo","status":"certified"}]});
    let family = json!({"models":[]});
    let (sources, boundaries) = model_boundaries::inventory(root.path()).unwrap();
    assert!(validate(&parity, &family, &sources, &boundaries).is_ok());
    fs::write(
        &path,
        r#"// begin_block(layer)
        auto fake = R"x(begin_block(layer); end_block(layer))x";
        stage->end_block(layer);"#,
    )
    .unwrap();
    let (sources, boundaries) = model_boundaries::inventory(root.path()).unwrap();
    assert!(validate(&parity, &family, &sources, &boundaries).is_err());
    fs::write(
        root.path().join("src/models/new_model.cpp"),
        "void model() {}",
    )
    .unwrap();
    let (sources, boundaries) = model_boundaries::inventory(root.path()).unwrap();
    assert!(validate(&parity, &family, &sources, &boundaries).is_err());
}

#[test]
fn orphan_manifest_pins_are_not_filtered_out_of_admission() {
    let mut parity = json!({"candidates":[{"llama_model":"orphan","status":"non_causal_aux","model_pin":{"repo":"org/model","revision":"main","file":"model.gguf","selector":"Q4","size_bytes":42,"blob_sha256":"b".repeat(64)}}]});
    let family = json!({"models":[{"artifact":{"repo":"org/model","revision":"a".repeat(40),"selector":"Q4","file_integrity":{"model.gguf":{"size_bytes":42,"blob_id":"b".repeat(64)}}}}]});
    assert!(validate(&parity, &family, &set(&[]), &set(&[])).is_err());
    parity["candidates"][0]["model_pin"]["revision"] = json!("a".repeat(40));
    assert!(validate(&parity, &family, &set(&[]), &set(&[])).is_ok());
    parity["candidates"][0]["model_pin"]["size_bytes"] = json!(43);
    assert!(validate(&parity, &family, &set(&[]), &set(&[])).is_err());
}

// Append to existing parity_inventory_tests.rs; no extra target/module or process source inclusion.
#[test]
fn pin_grammar_and_missing_join_are_independently_refused_with_no_pin_rows_allowed() {
    let pin = json!({"repo":"org/model","revision":"a".repeat(40),"file":"model.gguf","selector":"Q4","size_bytes":42,"blob_sha256":"b".repeat(64)});
    let family = json!({"models":[{"artifact":{"repo":"org/model","revision":"a".repeat(40),"selector":"Q4","file_integrity":{"model.gguf":{"size_bytes":42,"blob_id":"b".repeat(64)}}}}]});
    assert!(validate_pin(&pin, &family).is_ok());
    let mut malformed = pin.clone();
    malformed["blob_sha256"] = json!("not-a-blob");
    assert!(validate_pin(&malformed, &family).is_err());
    assert!(validate_pin(&pin, &json!({"models":[]})).is_err());
    assert!(
        validate(
            &json!({"candidates":[]}),
            &json!({"models":[]}),
            &set(&[]),
            &set(&[])
        )
        .is_ok()
    );
}

#[test]
fn paired_inventory_refuses_missing_model_directory_and_records_only_both_real_hooks() {
    let root = tempfile::tempdir().unwrap();
    assert!(model_boundaries::inventory(root.path()).is_err());
    let models = root.path().join("src/models");
    fs::create_dir_all(&models).unwrap();
    for (name, source) in [
        (
            "full",
            "stage->begin_block(layer); stage->end_block(layer);",
        ),
        ("begin_only", "stage->begin_block(layer);"),
        ("end_only", "stage->end_block(layer);"),
        ("comment", "// begin_block(layer); end_block(layer);"),
    ] {
        fs::write(models.join(format!("{name}.cpp")), source).unwrap();
    }
    let (sources, registered) = model_boundaries::inventory(root.path()).unwrap();
    assert_eq!(sources, set(&["full", "begin_only", "end_only", "comment"]));
    assert_eq!(registered, set(&["full"]));
    root.close()
        .expect("all owned local inventory files removed");
}

#[test]
fn admitted_inventory_emits_every_status_and_duplicate_classification_without_cache_claims() {
    let rows: Vec<_> = STATUSES.iter().enumerate().map(|(index, status)| json!({"llama_model":format!("model_{index}"),"family":format!("family_{index}"),"status":status,"notes":"kept","repo":"org/model","include":"*.gguf"})).collect();
    let sources: BTreeSet<_> = (0..STATUSES.len())
        .map(|index| format!("model_{index}"))
        .collect();
    let boundaries = sources.clone();
    let mut parity = json!({"candidates":rows,"support_priority":{"p0":{"families":["family_0"]},"p1":{"llama_models":["model_0","model_1"]}}});
    parity["candidates"].as_array_mut().unwrap().push(json!({"llama_model":"model_0","family":"alternate","status":"candidate_multimodal","notes":"second representative"}));
    assert!(validate(&parity, &json!({"models":[]}), &sources, &boundaries).is_ok());
    let projected = classifications(&parity, &sources, &boundaries).unwrap();
    assert_eq!(projected.len(), STATUSES.len() + 1);
    for status in STATUSES {
        assert!(projected.iter().any(|row| row["status"] == status));
    }
    let family = projected
        .iter()
        .find(|row| row["family"] == "family_0")
        .unwrap();
    assert_eq!(family["priority"], "p0");
    assert_eq!(family["notes"], "kept");
    assert_eq!(family["include"], "*.gguf");
    assert_eq!(family["boundary_registered"], true);
    assert!(family.get("local_path").is_none());
    assert_eq!(
        projected
            .iter()
            .find(|row| row["family"] == "alternate")
            .unwrap()["priority"],
        "p1"
    );
    assert_eq!(
        projected
            .iter()
            .find(|row| row["family"] == "family_2")
            .unwrap()["priority"],
        "p2"
    );
}

#[test]
fn runnable_statuses_refuse_unsupported_reasons_while_auxiliary_reason_is_admitted() {
    let family = json!({"models":[]});
    for status in ["certified", "candidate", "candidate_stateful"] {
        let mut parity = json!({"candidates":[{"llama_model":"somearch","status":status}]});
        assert!(validate(&parity, &family, &set(&["somearch"]), &set(&["somearch"])).is_ok());
        parity["candidates"][0]["unsupported_reason"] = json!("leftover");
        assert!(validate(&parity, &family, &set(&["somearch"]), &set(&["somearch"])).is_err());
    }
    let auxiliary = json!({"candidates":[{"llama_model":"encoder","status":"non_causal_aux","unsupported_reason":"non-causal encoder"}]});
    // Keep the legacy no-source shape, and independently exercise the row
    // policy with an actual inventoried source and no paired boundary.
    assert!(validate(&auxiliary, &family, &set(&[]), &set(&[])).is_ok());
    assert!(validate(&auxiliary, &family, &set(&["encoder"]), &set(&[])).is_ok());
    let mut malformed = auxiliary;
    malformed["candidates"][0]["unsupported_reason"] = json!({"not":"a string"});
    assert!(validate(&malformed, &family, &set(&["encoder"]), &set(&[])).is_err());
}
