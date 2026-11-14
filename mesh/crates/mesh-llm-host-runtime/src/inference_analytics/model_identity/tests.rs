use super::*;

#[test]
fn size_labels_bucket_by_whole_billions() {
    for (label, bucket) in [
        ("0.6B", "lt1b"),
        ("270M", "lt1b"),
        ("1.5B", "1-3b"),
        ("3.8B", "1-3b"),
        ("4B", "4-9b"),
        ("8B", "4-9b"),
        ("14B", "10-19b"),
        ("27B", "20-39b"),
        ("35B-A3B", "20-39b"),
        ("8x7B", "40-79b"),
        ("70B", "40-79b"),
        ("235B-A22B", "160-399b"),
        ("1T", "400b+"),
    ] {
        assert_eq!(parameter_bucket(label), Some(bucket), "{label}");
    }
}

#[test]
fn an_unparseable_size_label_reports_nothing() {
    for label in ["", "large", "B", "-8B", "8 parsecs"] {
        assert_eq!(parameter_bucket(label), None, "{label}");
    }
}

#[test]
fn an_unknown_name_is_redacted_with_nothing_else() {
    let identity = lookup("acme-merger-finetune-never-registered");
    assert_eq!(identity, ModelIdentity::redacted());
    let rendered = format!("{:?}", identity.apply(Properties::new()));
    assert!(!rendered.contains("acme"), "{rendered}");
}

#[test]
fn a_registered_model_is_described_by_its_identity_not_its_served_name() {
    register(
        "acme-private-stem",
        ModelIdentity {
            architecture: Label::sanitize("qwen3moe"),
            parameters: Some("20-39b"),
            moe: Some(true),
            ..ModelIdentity::redacted()
        },
    );

    let properties = lookup("acme-private-stem").apply(Properties::new());
    let rendered = format!("{properties:?}");

    assert!(!rendered.contains("acme"), "{rendered}");
    assert_eq!(
        properties.get("model").and_then(text),
        Some("redacted".to_owned())
    );
    assert_eq!(
        properties.get("model_arch").and_then(text),
        Some("qwen3moe".to_owned())
    );
    assert_eq!(
        properties.get("model_params"),
        Some(&mesh_llm_analytics::Value::Static("20-39b"))
    );
    assert_eq!(
        properties.get("model_moe"),
        Some(&mesh_llm_analytics::Value::Bool(true))
    );
}

fn text(value: &mesh_llm_analytics::Value) -> Option<String> {
    match value {
        mesh_llm_analytics::Value::Text(label) => Some(label.as_str().to_owned()),
        mesh_llm_analytics::Value::Static(text) => Some((*text).to_owned()),
        _ => None,
    }
}

#[test]
fn a_selected_device_names_its_backend() {
    assert_eq!(backend_of_selected_device("CPU"), "cpu");
    assert_eq!(backend_of_selected_device("CUDA1"), "cuda");
    assert_eq!(backend_of_selected_device("MTL0"), "metal");
    assert_eq!(backend_of_selected_device("SYCL0"), "other");
}
