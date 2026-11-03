use super::*;
use crate::ci_governance::collector::request::{CHECK_NAME, CheckConclusion};

#[test]
fn response_body_is_lossless_and_malformed_status_headers_fail() {
    let response=parse_response(b"HTTP/2.0 200 OK\r\ncontent-type: application/json\r\n\r\n{\r\n \"path\":\"line\\nbreak\"\r\n}").unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.body, b"{\r\n \"path\":\"line\\nbreak\"\r\n}");
    for bytes in [
        b"HTTP/2.0 200 OK\r\n".as_slice(),
        b"junk 200\n\n{}",
        b"HTTP/2.0 999\n\n{}",
    ] {
        assert!(parse_response(bytes).is_err());
    }
}

#[test]
fn fixed_requests_reject_invalid_ids_cursors_and_arbitrary_policy_refs() {
    for request in [
        Request::Pull { number: 0 },
        Request::Files {
            number: 1,
            page: 101,
        },
        Request::Policy {
            revision: "feature".into(),
        },
        Request::Reviews {
            number: 1,
            cursor: Some("\ninvalid".into()),
        },
        Request::AppInstallation { id: 0 },
    ] {
        assert!(request.encode().is_err());
    }
    let encoded = Request::CreateGovernanceCheck {
        head_sha: "a".repeat(40),
        external_id: "governance:1".into(),
    }
    .encode()
    .unwrap();
    let body: serde_json::Value = serde_json::from_slice(encoded.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["name"], CHECK_NAME);
    let completed = Request::CompleteGovernanceCheck {
        id: 1,
        conclusion: CheckConclusion::Failure,
    }
    .encode()
    .unwrap();
    assert_eq!(completed.method, "PATCH");
    assert_eq!(completed.endpoint, "repos/Mesh-LLM/mesh-llm/check-runs/1");
}

#[test]
fn opaque_credential_validation_never_echoes_input() {
    assert!(serde_json::from_str::<SecretToken>("\"fixture-token\"").is_ok());
    let error = serde_json::from_str::<SecretToken>("\"fixture-token\\n\"")
        .err()
        .unwrap()
        .to_string();
    assert!(!error.contains("fixture-token"));
}

fn inert_client(jwt: bool) -> GithubClient {
    let state = tempfile::tempdir().unwrap();
    let mut environment = child_environment();
    let header = authenticate(
        state.path(),
        &mut environment,
        SecretToken::checked("fixture-credential-value".into()).unwrap(),
        jwt,
    )
    .unwrap();
    GithubClient {
        executable: "inert-fixture".into(),
        root: state.path().into(),
        state,
        environment,
        jwt_header: header,
        deadline: Instant::now() + Duration::from_secs(180),
        cancellation: Cancellation::default(),
        received: 0,
        requests: 0,
    }
}

#[test]
fn jwt_header_file_is_private_ephemeral_and_no_credentials_reach_argv() {
    let client = inert_client(true);
    let path = client.jwt_header.clone().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    let encoded = Request::MintPublisherToken {
        installation_id: 7,
        repository_id: 1,
    }
    .encode()
    .unwrap();
    let args = client.arguments(&encoded, Duration::from_secs(10)).unwrap();
    let args: Vec<_> = args
        .iter()
        .map(|value| match value {
            Value::Public(value) => value.to_string_lossy().into_owned(),
            Value::Secret(_) => panic!("secret argument forbidden"),
        })
        .collect();
    assert!(
        !args
            .iter()
            .any(|argument| argument.contains("fixture-credential-value"))
    );
    assert!(
        args.iter()
            .any(|argument| argument == "https://api.github.com/app/installations/7/access_tokens")
    );
    assert!(args.contains(&"--disable".into()));
    assert!(!args.contains(&"--location".into()));
    assert!(
        !client
            .environment
            .contains_key(std::ffi::OsStr::new("GH_TOKEN"))
    );
    let body: serde_json::Value =
        serde_json::from_slice(&std::fs::read(client.state.path().join("request.json")).unwrap())
            .unwrap();
    assert_eq!(body["repository_ids"], serde_json::json!([1]));
    drop(client);
    assert!(!path.exists());
}

#[test]
fn installation_auth_is_env_only_and_never_acquires_cached_auth() {
    let client = inert_client(false);
    assert!(matches!(
        client.environment.get(std::ffi::OsStr::new("GH_TOKEN")),
        Some(Value::Secret(_))
    ));
    for name in [
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
        "HOME",
        "GH_CONFIG_DIR",
    ] {
        assert!(!child_environment().contains_key(std::ffi::OsStr::new(name)));
    }
    for value in client
        .arguments(
            &Request::Repository.encode().unwrap(),
            Duration::from_secs(5),
        )
        .unwrap()
    {
        match value {
            Value::Public(value) => {
                assert!(!value.to_string_lossy().contains("fixture-credential-value"))
            }
            Value::Secret(_) => panic!("secret argument forbidden"),
        }
    }
}

#[test]
fn cancelled_expired_or_oversubscribed_transactions_never_launch() {
    let mut cancelled = inert_client(false);
    cancelled.cancellation.cancel();
    assert!(cancelled.send(&Request::Repository).is_err());
    let mut expired = inert_client(false);
    expired.deadline = Instant::now() - Duration::from_secs(1);
    assert!(expired.send(&Request::Repository).is_err());
    let mut exhausted = inert_client(false);
    exhausted.requests = 512;
    assert!(exhausted.send(&Request::Repository).is_err());
}

#[test]
fn tracked_activation_ignores_dirty_worktree_and_git_replacement_objects() {
    let root = tempfile::tempdir().unwrap();
    let cancellation = Cancellation::default();
    let run = |arguments: &[&str]| git_bytes(root.path(), &cancellation, arguments, 4096).unwrap();
    run(&["init", "--quiet"]);
    std::fs::create_dir_all(root.path().join("ci/governance")).unwrap();
    let path = root.path().join("ci/governance/activation.json");
    let protected = b"{\"active\":false}\n";
    std::fs::write(&path, protected).unwrap();
    run(&["add", "--", "ci/governance/activation.json"]);
    run(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "protected fixture",
    ]);
    let base = checkout_head(root.path(), &cancellation).unwrap();
    std::fs::write(&path, b"{\"active\":true}\n").unwrap();
    assert_eq!(
        checkout_activation(root.path(), &base, &cancellation).unwrap(),
        protected
    );
    run(&["add", "--", "ci/governance/activation.json"]);
    run(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "replacement fixture",
    ]);
    let replacement = checkout_head(root.path(), &cancellation).unwrap();
    run(&["checkout", "--detach", "--quiet", &base]);
    run(&["replace", &base, &replacement]);
    assert!(!run(&["replace", "--list"]).is_empty());
    assert_eq!(
        checkout_activation(root.path(), &base, &cancellation).unwrap(),
        protected
    );
    assert!(checkout_activation(root.path(), &replacement, &cancellation).is_err());
}
