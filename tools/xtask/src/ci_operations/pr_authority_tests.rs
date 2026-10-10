use super::*;
use std::collections::BTreeMap;

fn policy(depot: bool, native_cache: bool) -> Policy {
    Policy {
        depot,
        native_cache,
        remote_cache: false,
    }
}

fn environment(directory: &std::path::Path) -> BTreeMap<String, String> {
    [
        ("INPUT_DEPOT_SELECTED", "true".to_owned()),
        ("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "false".into()),
        ("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "false".into()),
        ("GITHUB_EVENT_NAME", "pull_request".into()),
        ("DOCKER_CONFIG", directory.to_str().unwrap().into()),
    ]
    .into_iter()
    .map(|(name, value)| (name.into(), value))
    .collect()
}

fn verify(values: &BTreeMap<String, String>) -> Result<(), String> {
    check(&|name| values.get(name).cloned())
}

#[test]
fn strict_depot_admission_accepts_github_https_and_explicit_loopback_proxies() {
    for endpoint in [
        "https://actions.githubusercontent.com/cache",
        "HTTPS://results.actions.githubusercontent.com?cache=1",
        "https://actions.githubusercontent.com:443/cache",
        "http://localhost:80/cache",
        "https://127.0.0.1:443/cache",
        "http://[::1]:1234/cache",
        "http://[::ffff:127.0.0.1]:1234/cache",
        "",
    ] {
        assert!(policy(true, false).endpoint(endpoint).is_ok(), "{endpoint}");
    }
}

#[test]
fn strict_depot_admission_rejects_deceptive_authorities_and_incomplete_proxies() {
    for endpoint in [
        "http://actions.githubusercontent.com/cache",
        "https://actions.githubusercontent.com.evil/cache",
        "https://@actions.githubusercontent.com/cache",
        "https://user:secret@actions.githubusercontent.com/cache",
        "https://bad_label.actions.githubusercontent.com/cache",
        "http://localhost/cache",
        "http://localhost:0/cache",
        "http://localhost:80?query=/",
        "http://localhost:80",
        "http://localhost:70000/cache",
        "http://cache.example.invalid:80/cache",
        "http://localhost:80/with space",
        "file:///tmp/cache",
    ] {
        assert!(
            policy(true, false).endpoint(endpoint).is_err(),
            "{endpoint}"
        );
    }
}

#[test]
fn native_cache_exception_requires_depot_selection_and_explicit_authorization() {
    let endpoint = "https://cache.depot.dev/cache";
    assert!(policy(true, true).endpoint(endpoint).is_ok());
    for (depot, native) in [(true, false), (false, true), (false, false)] {
        assert!(policy(depot, native).endpoint(endpoint).is_err());
    }
    assert!(
        policy(false, false)
            .endpoint("https://hosted.example.invalid/cache")
            .is_ok()
    );
    assert!(
        policy(true, true)
            .endpoint("https://user@cache.depot.dev/cache")
            .is_err()
    );
    assert!(
        policy(true, true)
            .endpoint("ftp://cache.depot.dev/cache")
            .is_err()
    );
}

#[test]
fn policy_flags_are_required_and_cannot_enable_remote_cache_on_depot() {
    let directory = tempfile::tempdir().unwrap();
    let baseline = environment(directory.path());
    verify(&baseline).unwrap();
    for name in [
        "INPUT_DEPOT_SELECTED",
        "INPUT_ALLOW_NATIVE_GITHUB_CACHE",
        "INPUT_ALLOW_DEPOT_REMOTE_CACHE",
    ] {
        for value in [
            None,
            Some(""),
            Some("TRUE"),
            Some("true "),
            Some("secret-value"),
        ] {
            let mut values = baseline.clone();
            if let Some(value) = value {
                values.insert(name.into(), value.into());
            } else {
                values.remove(name);
            }
            let error = verify(&values).unwrap_err();
            assert!(error.starts_with(name));
            assert!(!error.contains("secret-value"));
        }
    }
    let mut values = baseline;
    values.insert("INPUT_ALLOW_DEPOT_REMOTE_CACHE".into(), "true".into());
    assert!(verify(&values).is_err());
}

#[test]
fn pr_authority_rejects_every_forbidden_variable_without_retaining_its_value() {
    let directory = tempfile::tempdir().unwrap();
    let baseline = environment(directory.path());
    for name in FORBIDDEN {
        let mut values = baseline.clone();
        values.insert((*name).into(), "private-credential-value".into());
        let error = verify(&values).unwrap_err();
        assert!(error.contains(name));
        assert!(!error.contains("private-credential-value"));
        values.insert((*name).into(), String::new());
        verify(&values).unwrap();
    }
}

#[test]
fn all_three_action_endpoints_are_checked_and_errors_do_not_disclose_urls() {
    let directory = tempfile::tempdir().unwrap();
    for name in [
        "ACTIONS_CACHE_URL",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_RUNTIME_URL",
    ] {
        let mut values = environment(directory.path());
        values.insert(
            name.into(),
            "https://private.example.invalid/secret-path".into(),
        );
        let error = verify(&values).unwrap_err();
        assert!(error.starts_with(name));
        assert!(!error.contains("private.example"));
        assert!(!error.contains("secret-path"));
    }
}

#[test]
fn original_event_controls_pr_checks_without_changing_boolean_admission() {
    let directory = tempfile::tempdir().unwrap();
    let mut values = environment(directory.path());
    values.insert("DEPOT_TOKEN".into(), "private-value".into());
    values.insert("GITHUB_EVENT_NAME".into(), "workflow_dispatch".into());
    verify(&values).unwrap();
    values.insert("INPUT_ORIGINAL_EVENT_NAME".into(), "pull_request".into());
    assert!(verify(&values).is_err());
    values.insert("INPUT_ORIGINAL_EVENT_NAME".into(), "push".into());
    verify(&values).unwrap();
    values.insert("INPUT_DEPOT_SELECTED".into(), "invalid".into());
    assert!(verify(&values).is_err());
}

#[test]
fn docker_auth_environment_and_file_keep_provider_specific_policy() {
    let directory = tempfile::tempdir().unwrap();
    let mut values = environment(directory.path());
    let config = directory.path().join("config.json");
    std::fs::write(&config, r#"{"auths":{}}"#).unwrap();
    assert!(verify(&values).is_err());
    values.insert("INPUT_DEPOT_SELECTED".into(), "false".into());
    verify(&values).unwrap();
    values.insert(
        "DOCKER_AUTH_CONFIG".into(),
        r#"{"auths":{"registry.depot.dev":{"auth":"private-value"}}}"#.into(),
    );
    let error = verify(&values).unwrap_err();
    assert!(!error.contains("private-value"));
    assert!(error.contains("depot-authentication"));
    values.remove("DOCKER_AUTH_CONFIG");
    std::fs::write(&config, b"{\"auths\":NaN}").unwrap();
    assert!(verify(&values).is_err());
}
