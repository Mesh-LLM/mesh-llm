//! Contract helper unit tests: closed catalogs, strict JSON, and containment.
use super::*;
use serde_json::json;

fn fixture_root() -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().expect("fixture root");
    (root, PathBuf::new())
}

fn write_catalog(root: &Path, rows: serde_json::Value) {
    let ci = root.join("ci");
    std::fs::create_dir_all(&ci).unwrap();
    std::fs::write(
        ci.join("slices.yml"),
        json!({"runtime_rows": rows}).to_string(),
    )
    .unwrap();
}

fn nine_rows() -> Vec<serde_json::Value> {
    CORE_ROWS
        .iter()
        .map(|id| {
            let (platform, backend) = id.split_once('-').unwrap();
            let target = match platform {
                "linux" => "x86_64-unknown-linux-gnu",
                "macos" => "aarch64-apple-darwin",
                _ => "x86_64-pc-windows-msvc",
            };
            json!({"id": id, "platform": platform, "backend": backend, "target": target})
        })
        .collect()
}

#[test]
fn digest_formats_are_closed_lowercase_hex() {
    assert!(is_sha256(&"a".repeat(64)));
    assert!(!is_sha256(&"A".repeat(64)));
    assert!(!is_sha256(&"a".repeat(63)));
    assert!(is_git_sha(&"f".repeat(40)));
    assert!(!is_git_sha(&"g".repeat(40)));
}

#[test]
fn strict_json_rejects_duplicate_keys_at_any_depth() {
    assert!(parse_strict(br#"{"a": 1, "a": 2}"#).is_err());
    assert!(parse_strict(br#"{"outer": {"b": 1, "b": 2}}"#).is_err());
    let value = parse_strict(br#"{"a": [1, {"c": true}], "b": null}"#).expect("valid JSON");
    assert_eq!(
        value.get("a").and_then(Json::as_array).map(<[_]>::len),
        Some(2)
    );
}

#[test]
fn catalog_row_fails_closed_on_drift() {
    let (temp, _) = fixture_root();
    write_catalog(temp.path(), json!(nine_rows()));
    let row = catalog_row(temp.path(), "linux-cpu").expect("known row");
    assert_eq!(row.get("backend").and_then(Json::as_str), Some("cpu"));
    assert!(catalog_row(temp.path(), "linux-cuda").is_ok());
    let unknown = catalog_row(temp.path(), "plan9-cpu").unwrap_err();
    assert!(unknown.contains("unknown core row"), "{unknown}");
    let drifted = nine_rows().into_iter().take(8).collect::<Vec<_>>();
    write_catalog(temp.path(), json!(drifted));
    let drift = catalog_row(temp.path(), "linux-cpu").unwrap_err();
    assert!(drift.contains("catalog drift"), "{drift}");
    let mut duplicated = nine_rows();
    duplicated.push(duplicated[0].clone());
    write_catalog(temp.path(), json!(duplicated));
    assert!(
        catalog_row(temp.path(), "linux-cpu")
            .unwrap_err()
            .contains("catalog drift")
    );
}

#[cfg(unix)]
#[test]
fn tree_digest_rejects_symlinks() {
    let (temp, _) = fixture_root();
    std::fs::write(temp.path().join("file"), b"bytes").unwrap();
    let digest = tree_digest(temp.path()).expect("plain tree");
    assert!(is_sha256(&digest));
    std::os::unix::fs::symlink(temp.path().join("file"), temp.path().join("link")).unwrap();
    assert!(tree_digest(temp.path()).unwrap_err().contains("symlink"));
}

#[cfg(unix)]
#[test]
fn product_file_rejects_escape_and_symlinked_parents() {
    let (temp, _) = fixture_root();
    let root = temp.path();
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("sub/file"), b"x").unwrap();
    let file = product_file(root, &Json::String("sub/file".into())).expect("contained file");
    assert!(file.ends_with("sub/file"));
    assert!(product_file(root, &Json::String("../escape".into())).is_err());
    assert!(product_file(root, &Json::String("/absolute".into())).is_err());
    std::os::unix::fs::symlink(root.join("sub"), root.join("linked")).unwrap();
    let error = product_file(root, &Json::String("linked/file".into())).unwrap_err();
    assert!(error.contains("symlink"), "{error}");
}

#[test]
fn validate_models_checks_registry_pins_and_tags() {
    let (temp, _) = fixture_root();
    let root = temp.path();
    let model_hash = "b".repeat(64);
    write_catalog(root, json!(nine_rows()));
    let registry_dir = root.join("ci/model-artifacts");
    std::fs::create_dir_all(&registry_dir).unwrap();
    std::fs::write(
        registry_dir.join("registry.json"),
        json!({"artifacts": [
            {"id": "dense-model", "capability_tags": ["dense"],
             "artifact": {"files": [{"sha256": model_hash}]}},
            {"id": "hybrid-model", "capability_tags": ["hybrid"],
             "artifact": {"files": [{"sha256": "c".repeat(64)}]}},
        ]})
        .to_string(),
    )
    .unwrap();
    let models = Json::Object(vec![(
        "dense-model".to_owned(),
        Json::String(model_hash.clone()),
    )]);
    assert!(validate_models(root, "dense", &models).is_ok());
    let wrong_digest = Json::Object(vec![(
        "dense-model".to_owned(),
        Json::String("d".repeat(64)),
    )]);
    assert!(validate_models(root, "dense", &wrong_digest).is_err());
    let unregistered = Json::Object(vec![("foreign-model".to_owned(), Json::String(model_hash))]);
    assert!(validate_models(root, "dense", &unregistered).is_err());
    let missing_tag = Json::Object(vec![(
        "hybrid-model".to_owned(),
        Json::String("c".repeat(64)),
    )]);
    assert!(validate_models(root, "dense", &missing_tag).is_err());
    assert!(validate_models(root, "recurrent", &missing_tag).is_ok());
    let kv = Json::Object(vec![
        ("dense-model".to_owned(), Json::String("b".repeat(64))),
        ("hybrid-model".to_owned(), Json::String("c".repeat(64))),
    ]);
    assert!(validate_models(root, "kv-cache", &kv).is_ok());
    assert!(validate_models(root, "kv-cache", &models).is_err());
    let empty = Json::Object(vec![]);
    assert!(validate_models(root, "packaging-runtime", &empty).is_ok());
    assert!(validate_models(root, "packaging-runtime", &models).is_err());
}
