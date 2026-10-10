use super::*;
#[test]
fn artifact_dimensions_require_actual_architecture_context_layer_and_width() {
    let value = json!({"general.architecture":"deepseek2","deepseek2.block_count":61,"deepseek2.embedding_length":7168,"deepseek2.context_length":32});
    let mut metadata: BTreeMap<String, Value> = serde_json::from_value(value).unwrap();
    assert_eq!(dimensions(&metadata, 32).unwrap()["layer_count"], 61);
    assert!(dimensions(&metadata, 33).is_err());
    metadata.remove("deepseek2.embedding_length");
    assert!(dimensions(&metadata, 32).is_err());
}
#[test]
fn complete_source_pin_admission_refuses_duplicates_paths_and_noncanonical_digests() {
    let good = format!("fixture-00001-of-00003.gguf={}", "a".repeat(64));
    assert_eq!(pins(std::slice::from_ref(&good)).unwrap().len(), 1);
    for values in [
        vec![good.clone(), good],
        vec![format!("../fixture.gguf={}", "a".repeat(64))],
        vec![format!("fixture.gguf={}", "A".repeat(64))],
        vec![],
    ] {
        assert!(pins(&values).is_err());
    }
}
#[test]
fn native_source_admission_rejects_missing_shard_before_native_tensor_inspection() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("fixture-00001-of-00003.gguf");
    std::fs::write(&first, b"GGUF").unwrap();
    let first = first.canonicalize().unwrap();
    let error = source(
        &first,
        &[format!("fixture-00001-of-00003.gguf={}", "a".repeat(64))],
        32,
    )
    .unwrap_err();
    assert!(error.to_string().contains("missing sibling"));
    directory.close().unwrap();
}

#[test]
fn local_admission_cli_variants_never_request_download_cache_preparation() {
    use clap::Parser as _;
    for argv in [
        vec![
            "tool",
            "admit-source",
            "/declared/model.gguf",
            "--pin",
            "model.gguf=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--minimum-context",
            "32",
        ],
        vec![
            "tool",
            "admit-package",
            "/declared/package",
            "--manifest-sha256",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--model-id",
            "declared",
            "--layer-start",
            "3",
            "--layer-end",
            "4",
            "--minimum-context",
            "32",
        ],
    ] {
        let args = crate::cli::Args::try_parse_from(argv).unwrap();
        assert!(!args.command.requires_download_preparation());
    }
}
