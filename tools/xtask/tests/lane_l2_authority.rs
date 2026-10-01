#[path = "../src/ci_operations/authority_audit.rs"]
mod authority_audit;

use authority_audit::{AuthorityFailure, DockerFailure, attest_endpoint, audit_docker_json};

#[test]
fn rejects_all_loopback_address_forms() {
    for authority in [
        "localhost",
        "127.0.0.1",
        "127.42.3.4",
        "[::1]",
        "[0:0:0:0:0:0:0:1]",
        "[::ffff:127.9.8.7]",
        "[::ffff:7f00:1]",
    ] {
        assert_eq!(
            attest_endpoint(&format!("http://{authority}:80/cache")),
            Err(AuthorityFailure::Loopback)
        );
    }
}

#[test]
fn accepts_remote_ipv4_ipv6_and_dns() {
    for authority in ["192.0.2.1", "[2001:db8::1]", "cache.example.test"] {
        assert_eq!(
            attest_endpoint(&format!("http://{authority}:65535/cache")),
            Ok(())
        );
    }
}

#[test]
fn rejects_malformed_or_unapproved_endpoints_without_values() {
    for endpoint in [
        "",
        "https://remote:80/cache",
        "http://user:secret@remote:80/cache",
        "http://remote:0/cache",
        "http://remote:65536/cache",
        "http://remote:80",
        "http://[bogus]:80/cache",
        "http://remote:80/a b",
        "http://actions.githubusercontent.com:80/cache",
    ] {
        let failure = attest_endpoint(endpoint).unwrap_err();
        assert!(!failure.to_string().contains("secret"));
    }
}

#[test]
fn depot_rejects_even_empty_auth_sections() {
    for raw in [
        r#"{"auths":{}}"#,
        r#"{"credHelpers":{}}"#,
        r#"{"credsStore":""}"#,
    ] {
        assert_eq!(
            audit_docker_json(raw, true),
            Err(DockerFailure::Authentication)
        );
    }
}

#[test]
fn hosted_accepts_unrelated_auth_but_rejects_depot() {
    assert_eq!(
        audit_docker_json(r#"{"auths":{"ghcr.io":{}}}"#, false),
        Ok(())
    );
    assert_eq!(
        audit_docker_json(r#"{"auths":{"ORG.REGISTRY.DEPOT.DEV":{}}}"#, false),
        Err(DockerFailure::DepotAuthentication)
    );
}

#[test]
fn malformed_docker_json_fails_closed() {
    for raw in [
        "[]",
        "null",
        "{",
        r#"{"auths":[]}"#,
        r#"{"credsStore":1}"#,
        r#"{"auths":NaN}"#,
    ] {
        assert_eq!(audit_docker_json(raw, false), Err(DockerFailure::Malformed));
    }
}

#[test]
fn depot_rejects_auth_environment_even_when_inert() {
    assert_eq!(
        authority_audit::audit_docker_sources("{}", std::path::Path::new("absent-config"), true),
        Err(DockerFailure::Authentication)
    );
}

#[test]
fn missing_config_is_inert() {
    assert_eq!(
        authority_audit::audit_docker_sources("", std::path::Path::new("absent-config"), true),
        Ok(())
    );
}
