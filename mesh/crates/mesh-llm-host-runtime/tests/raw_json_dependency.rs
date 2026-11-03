//! The host's raw JSON contract must not rely on another crate's feature union.

#[test]
fn host_declares_its_raw_json_dependency_feature() {
    let manifest: toml::Value = toml::from_str(include_str!("../Cargo.toml")).unwrap();
    let features = manifest["dependencies"]["serde_json"]
        .get("features")
        .and_then(toml::Value::as_array);

    assert!(features.is_some_and(|features| {
        features
            .iter()
            .any(|feature| feature.as_str() == Some("raw_value"))
    }));
}
