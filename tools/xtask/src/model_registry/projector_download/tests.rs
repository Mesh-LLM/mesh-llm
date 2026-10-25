use super::*;
use crate::{
    automation::tls_fixture::{Reply, Server},
    process,
};
use std::time::Duration;

#[test]
fn hf_projector_actual_tls_pins_host_certificate_and_streams_valid_gguf_without_auth() {
    let server = Server::with_host(
        vec![Reply::bytes(200, b"GGUFownedfixture".to_vec())],
        "huggingface.co",
    );
    let curl = process::curl_https::Curl::fixture(&server.ca);
    let url = url::Url::parse(&format!("{}/mmproj?signature=fixture", server.base)).unwrap();
    let pin = format!("huggingface.co:{}:127.0.0.1", url.port().unwrap());
    let directory = tempfile::tempdir().unwrap();
    let reply = transfer::exchange(
        &curl,
        &url,
        &pin,
        directory.path(),
        Duration::from_secs(5),
        &process::Cancellation::default(),
        64,
    )
    .unwrap();
    let transfer::Reply::Complete(path) = reply else {
        panic!("expected complete body");
    };
    assert_eq!(std::fs::read(path).unwrap(), b"GGUFownedfixture");
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].0.contains("Host: huggingface.co:"));
    assert!(
        !requests[0]
            .0
            .to_ascii_lowercase()
            .contains("authorization:")
    );
    drop(requests);
    drop(server);
    directory.close().unwrap();
}

#[test]
fn hf_projector_actual_tls_unknown_length_body_limit_and_incomplete_eof_refuse() {
    for reply in [
        Reply::stream("GGUFoverflow".into(), false),
        Reply::incomplete(),
    ] {
        let server = Server::with_host(vec![reply], "hf.co");
        let curl = process::curl_https::Curl::fixture(&server.ca);
        let url = url::Url::parse(&server.base).unwrap();
        let pin = format!("hf.co:{}:127.0.0.1", url.port().unwrap());
        let directory = tempfile::tempdir().unwrap();
        assert!(
            transfer::exchange(
                &curl,
                &url,
                &pin,
                directory.path(),
                Duration::from_millis(3250),
                &process::Cancellation::default(),
                4
            )
            .is_err()
        );
        assert_eq!(server.requests.lock().unwrap().len(), 1);
        drop(server);
        directory.close().unwrap();
    }
}

#[test]
fn hf_projector_worker_refuses_untrusted_host_and_existing_output_without_dns() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input");
    let output = directory.path().join("output");
    std::fs::write(&input, r#"{"schema_version":1,"host":"localhost"}"#).unwrap();
    let args = vec![
        "--input".into(),
        input.display().to_string(),
        "--output".into(),
        output.display().to_string(),
    ];
    assert!(resolver::worker(&args).is_err());
    assert!(!output.exists());
    std::fs::write(&output, b"keep").unwrap();
    assert!(resolver::worker(&args).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"keep");
    directory.close().unwrap();
}

#[test]
fn hf_projector_actual_tls_redirect_requires_new_policy_admission_and_never_auto_follows() {
    for destination in [
        "https://cdn.hf.co/mmproj?signed=fixture",
        "https://example.com/secret",
    ] {
        let server = Server::with_host(vec![Reply::redirect(destination)], "hf.co");
        let curl = process::curl_https::Curl::fixture(&server.ca);
        let url = url::Url::parse(&server.base).unwrap();
        let pin = format!("hf.co:{}:127.0.0.1", url.port().unwrap());
        let directory = tempfile::tempdir().unwrap();
        let reply = transfer::exchange(
            &curl,
            &url,
            &pin,
            directory.path(),
            Duration::from_secs(5),
            &process::Cancellation::default(),
            64,
        )
        .unwrap();
        let transfer::Reply::Redirect(location) = reply else {
            panic!("expected redirect");
        };
        assert_eq!(location, destination);
        assert_eq!(
            url_policy::trusted(&location).is_ok(),
            destination.contains("cdn.hf.co")
        );
        assert_eq!(server.requests.lock().unwrap().len(), 1);
        drop(server);
        directory.close().unwrap();
    }
}
