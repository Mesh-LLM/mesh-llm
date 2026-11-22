use super::{
    health_probe_client_builder, is_loopback_host,
    probe_openai_compatible_http_endpoint_for_protocol,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use url::Url;

const NO_CA_OR_PROXY_CHILD: &str = "MESH_LLM_TEST_PLUGIN_HEALTH_NO_CA_CHILD";

#[test]
fn is_loopback_host_classifies_loopback_and_remote_urls() {
    for url in [
        "http://127.0.0.1:8000/v1/models",
        "http://[::1]:8000/v1/models",
        "http://localhost:8000/v1/models",
        "http://LOCALHOST.:8000/v1/models",
    ] {
        assert!(
            is_loopback_host(&Url::parse(url).expect("parse loopback url")),
            "expected loopback: {url}"
        );
    }
    for url in [
        "http://192.0.2.1:8000/v1/models",
        "http://plugin.internal:8000/v1/models",
        "https://api.example.com/v1/models",
    ] {
        assert!(
            !is_loopback_host(&Url::parse(url).expect("parse remote url")),
            "expected non-loopback: {url}"
        );
    }
}

// SSL_CERT_FILE and SSL_CERT_DIR point system root loading at an empty file and
// directory on Linux, and HTTP_PROXY points at a closed port; both are set on
// the child test process only.
#[test]
fn probe_http_endpoint_reaches_loopback_without_ca_bundle_or_proxy() {
    let temp = tempfile::tempdir().expect("tempdir for empty CA bundle");
    let empty_file = temp.path().join("empty.pem");
    std::fs::write(&empty_file, "").expect("write empty cert file");
    let empty_dir = temp.path().join("certs");
    std::fs::create_dir(&empty_dir).expect("create empty cert dir");

    let closed = std::net::TcpListener::bind("127.0.0.1:0").expect("bind closed proxy port");
    let proxy = format!("http://{}", closed.local_addr().expect("closed proxy addr"));
    drop(closed);

    let output = std::process::Command::new(std::env::current_exe().expect("current test exe"))
        .args([
            "--exact",
            "plugin::health::probe_client_tests::probe_http_endpoint_without_ca_bundle_child",
            "--nocapture",
        ])
        .env(NO_CA_OR_PROXY_CHILD, "1")
        .env("SSL_CERT_FILE", &empty_file)
        .env("SSL_CERT_DIR", &empty_dir)
        .env("HTTP_PROXY", &proxy)
        .env("http_proxy", &proxy)
        .env("ALL_PROXY", &proxy)
        .env("all_proxy", &proxy)
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .output()
        .expect("spawn plugin health probe child");

    assert!(
        output.status.success(),
        "plugin health probe child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "child test filter must execute its assertion"
    );
}

#[tokio::test]
async fn probe_http_endpoint_without_ca_bundle_child() {
    if std::env::var_os(NO_CA_OR_PROXY_CHILD).is_none() {
        return;
    }

    // Non-loopback plain HTTP must also build without system CA certificates.
    let remote_http = Url::parse("http://192.0.2.1:8000/v1/models").expect("parse remote http url");
    assert!(
        health_probe_client_builder(&remote_http).build().is_ok(),
        "non-loopback plain-HTTP probe client should build without CA certs"
    );

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback probe server");
    let addr = listener.local_addr().expect("probe server addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept health probe");
        let mut buf = [0u8; 1024];
        let _ = socket.read(&mut buf).await;
        let body = r#"{"data":[{"id":"demo-model","availability":"ready","workload_class":"causal_generation"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write probe response");
        let _ = socket.shutdown().await;
    });

    let record = probe_openai_compatible_http_endpoint_for_protocol(
        &format!("http://{addr}"),
        Some("openai_compatible"),
    )
    .await;
    assert!(
        record.available,
        "loopback plain-HTTP plugin health probe should succeed without CA certs or proxy: {record:?}"
    );
    assert_eq!(record.models, vec!["demo-model".to_string()]);
    server.await.expect("probe server join");
}
