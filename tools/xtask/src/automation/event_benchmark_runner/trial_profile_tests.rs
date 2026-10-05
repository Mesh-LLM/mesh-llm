use super::*;

fn values(items: &[(&str, &str)]) -> BTreeMap<OsString, OsString> {
    items
        .iter()
        .map(|(name, value)| ((*name).into(), (*value).into()))
        .collect()
}
fn value(environment: &BTreeMap<OsString, Value>, name: &str) -> String {
    match environment.get(std::ffi::OsStr::new(name)).unwrap() {
        Value::Public(value) | Value::Secret(value) => value.to_string_lossy().into_owned(),
    }
}

#[test]
fn inherits_safe_benchmark_settings_and_explicit_device_thread_profile() {
    let inherited = values(&[
        ("MESH_LLM_LIFECYCLE_LOG_PARSER", "1"),
        ("MESH_LLM_EVENT_INGRESS_CAPACITY", "64"),
        ("CUDA_VISIBLE_DEVICES", "2"),
        ("HIP_VISIBLE_DEVICES", "1"),
        ("ROCR_VISIBLE_DEVICES", "0"),
        ("OMP_NUM_THREADS", "8"),
        ("GGML_CUDA_FORCE_MMQ", "1"),
    ]);
    let (environment, evidence) = apply(Default::default(), &inherited, Mode::Production);
    for (name, expected) in &inherited {
        assert_eq!(
            value(&environment, name.to_str().unwrap()),
            expected.to_string_lossy()
        );
    }
    assert!(evidence.dropped_inherited_settings.is_empty());
    assert_eq!(
        evidence.device_and_backend_settings["CUDA_VISIBLE_DEVICES"],
        "2"
    );
    assert_eq!(evidence.device_and_backend_settings["OMP_NUM_THREADS"], "8");
    assert_eq!(value(&environment, trial_environment::GATE), "1");
}

#[test]
fn preserves_private_discovery_and_overrides_inherited_gate_and_mode() {
    let isolated = [
        (
            OsString::from("HOME"),
            Value::Public("/isolated/home".into()),
        ),
        (
            OsString::from("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"),
            Value::Public("/approved/runtime".into()),
        ),
    ]
    .into_iter()
    .collect();
    let inherited = values(&[
        ("HOME", "/operator/home"),
        ("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", "/other/runtime"),
        (trial_environment::GATE, "0"),
        (trial_environment::SELECTOR, "off"),
    ]);
    let (environment, evidence) = apply(isolated, &inherited, Mode::EventDisabled);
    assert_eq!(value(&environment, "HOME"), "/isolated/home");
    assert_eq!(
        value(&environment, "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"),
        "/approved/runtime"
    );
    assert_eq!(value(&environment, trial_environment::GATE), "1");
    assert_eq!(
        value(&environment, trial_environment::SELECTOR),
        "event-disabled"
    );
    assert_eq!(evidence.dropped_inherited_settings.len(), 3);
}

#[test]
fn records_dropped_sensitive_and_path_settings_without_values() {
    let inherited = values(&[
        ("MESH_LLM_AUTH_TOKEN", "private-token"),
        ("MESH_LLM_MODEL_PATH", "/private/model"),
        ("GGML_METAL_PATH_RESOURCES", "/private/resources"),
        ("UNRELATED", "outside-profile"),
    ]);
    let (environment, evidence) = apply(Default::default(), &inherited, Mode::Off);
    let bytes = serde_json::to_string(&evidence).unwrap();
    assert_eq!(evidence.dropped_inherited_settings.len(), 3);
    assert!(!bytes.contains("private-token"));
    assert!(!bytes.contains("/private"));
    assert!(!environment.contains_key(std::ffi::OsStr::new("UNRELATED")));
}

#[test]
fn snapshot_matches_actual_owned_mode_and_retained_settings() {
    let inherited = values(&[
        (trial_environment::SELECTOR, "off"),
        ("MESH_LLM_LIFECYCLE_LOG_PARSER", "1"),
    ]);
    let (environment, _) = apply(Default::default(), &inherited, Mode::Production);
    let actual = environment
        .into_iter()
        .map(|(key, value)| {
            (
                key,
                match value {
                    Value::Public(value) | Value::Secret(value) => value,
                },
            )
        })
        .collect();
    let snapshot = trial_environment::snapshot(&actual);
    assert_eq!(
        snapshot[trial_environment::SELECTOR].value,
        serde_json::json!("production")
    );
    assert_eq!(
        snapshot[trial_environment::GATE].value,
        serde_json::json!(true)
    );
    assert_eq!(
        snapshot["MESH_LLM_LIFECYCLE_LOG_PARSER"].value,
        serde_json::json!("1")
    );
}
