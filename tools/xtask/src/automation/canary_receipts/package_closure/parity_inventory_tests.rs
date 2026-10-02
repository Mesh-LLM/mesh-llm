use super::*;
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
