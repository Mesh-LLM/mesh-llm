use super::runtime_smoke::*;
use super::{
    CertificationGateStatus, aggregate_certification_status, certification_stage_ranges,
    runtime_smoke_gates,
};
use crate::package::inspection::{StagePackageInfo, StagePackageLayerInfo};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[test]
fn certification_ranges_split_two_stage_package() {
    let ranges = certification_stage_ranges(5).unwrap();

    assert_eq!(ranges[0].layer_start, 0);
    assert_eq!(ranges[0].layer_end, 2);
    assert!(ranges[0].include_embeddings);
    assert!(!ranges[0].include_output);
    assert_eq!(ranges[1].layer_start, 2);
    assert_eq!(ranges[1].layer_end, 5);
    assert!(!ranges[1].include_embeddings);
    assert!(ranges[1].include_output);
}

#[test]
fn certification_ranges_reject_single_layer_package() {
    let error = certification_stage_ranges(1).unwrap_err().to_string();

    assert!(error.contains("at least two transformer layers"), "{error}");
}

#[test]
fn aggregate_status_prefers_failed_over_incomplete() {
    let status = aggregate_certification_status([
        CertificationGateStatus::Passed,
        CertificationGateStatus::Incomplete,
        CertificationGateStatus::Failed,
    ]);

    assert_eq!(status, CertificationGateStatus::Failed);
}

#[test]
fn aggregate_status_allows_not_required_runtime_gates() {
    let status = aggregate_certification_status([
        CertificationGateStatus::Passed,
        CertificationGateStatus::NotRequired,
    ]);

    assert_eq!(status, CertificationGateStatus::Passed);
}

#[test]
fn models_response_requires_matching_model_id() {
    let response = json!({
        "object": "list",
        "data": [
            { "id": "other" },
            { "id": "org/repo:Q4_K_M" }
        ]
    });

    assert!(models_response_contains(&response, "org/repo:Q4_K_M"));
    assert!(!models_response_contains(&response, "missing"));
}

#[test]
fn chat_response_validator_accepts_string_and_structured_text_content() {
    let string_content = json!({
        "choices": [
            { "message": { "content": "ok" } }
        ]
    });
    let structured_content = json!({
        "choices": [
            {
                "message": {
                    "content": [
                        { "type": "text", "text": "ok" }
                    ]
                }
            }
        ]
    });

    assert!(response_has_chat_choice_content(&string_content));
    assert!(response_has_chat_choice_content(&structured_content));
}

#[test]
fn responses_response_validator_accepts_output_text_and_output_parts() {
    let output_text = json!({
        "output_text": "ok"
    });
    let output_parts = json!({
        "output": [
            {
                "content": [
                    { "type": "output_text", "text": "ok" }
                ]
            }
        ]
    });

    assert!(response_has_responses_output(&output_text));
    assert!(response_has_responses_output(&output_parts));
}

#[tokio::test]
async fn chat_smoke_rejects_success_status_without_choice_content() {
    let api_base = spawn_single_response_server(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 14\r\n\r\n{\"choices\":[]}",
        )
        .await;
    let package = fake_package_info();
    let request = fake_certification_request();

    let gate = smoke_chat_completions(&reqwest::Client::new(), &api_base, &package, &request).await;

    assert_eq!(gate.status, CertificationGateStatus::Failed);
    assert!(
        gate.details
            .as_deref()
            .is_some_and(|details| details.contains("choice content")),
        "{gate:?}"
    );
}

#[tokio::test]
async fn responses_smoke_rejects_success_status_without_output() {
    let api_base = spawn_single_response_server(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
    )
    .await;
    let package = fake_package_info();
    let request = fake_certification_request();

    let gate = smoke_responses(&reqwest::Client::new(), &api_base, &package, &request).await;

    assert_eq!(gate.status, CertificationGateStatus::Failed);
    assert!(
        gate.details
            .as_deref()
            .is_some_and(|details| details.contains("Responses output")),
        "{gate:?}"
    );
}

fn fake_certification_request() -> super::SkippyCertificationRequest {
    super::SkippyCertificationRequest {
        model_ref: "hf://meshllm/demo@abc123".to_string(),
        package_only: false,
        api_base: None,
        prompt: "Say ok.".to_string(),
        max_tokens: 2,
    }
}

fn fake_package_info() -> StagePackageInfo {
    StagePackageInfo {
        package_ref: "hf://meshllm/demo@abc123".to_string(),
        package_dir: std::path::PathBuf::from("/tmp/demo-package"),
        manifest_sha256: "a".repeat(64),
        model_id: "meshllm/demo".to_string(),
        source_model_path: "model.gguf".to_string(),
        source_model_sha256: "b".repeat(64),
        source_model_bytes: Some(42),
        layer_count: 2,
        activation_width: 4096,
        generation: None,
        projector_path: None,
        layers: vec![StagePackageLayerInfo {
            layer_index: 0,
            tensor_count: 1,
            tensor_bytes: 1,
            artifact_bytes: 1,
        }],
    }
}

async fn spawn_single_response_server(response: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 2048];
        let _ = stream.read(&mut buf).await.unwrap();
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    format!("http://{addr}")
}

async fn read_complete_http_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    loop {
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "unexpected EOF while reading certification request");
        request.extend_from_slice(&chunk[..n]);
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let body_start = header_end + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if request.len() >= body_start + content_length {
            return request;
        }
    }
}

/// Mimics a node that only recognizes `served_model_id` — anything else 404s,
/// the same way a real host does when a client asks for a model name it
/// doesn't advertise.
async fn spawn_certification_stub_server(served_model_id: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        for _ in 0..3 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_complete_http_request(&mut stream).await;
            let request = String::from_utf8_lossy(&request);
            let response = if request.starts_with("GET") {
                let body = json!({
                    "object": "list",
                    "data": [{ "id": served_model_id }]
                })
                .to_string();
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
            } else if request.contains(&format!("\"model\":\"{served_model_id}\"")) {
                let body = if request.contains("/v1/chat/completions") {
                    json!({"choices": [{"message": {"content": "ok"}}]})
                } else {
                    json!({"output_text": "ok"})
                }
                .to_string();
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
            } else {
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_string()
            };
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn runtime_smoke_gates_certify_against_served_package_ref() {
    // A package's `model_id` is the *source* model recorded in the manifest
    // (e.g. an upstream HF ref). The node advertises and routes on
    // `package_ref` instead — the ref it was actually started with. The
    // fake package below has both, and differs deliberately.
    let package = fake_package_info();
    assert_ne!(package.model_id, package.package_ref);

    let api_base = spawn_certification_stub_server(package.package_ref.clone()).await;
    let request = super::SkippyCertificationRequest {
        api_base: Some(api_base),
        ..fake_certification_request()
    };

    let gates = runtime_smoke_gates(&request, &package).await;

    for gate in &gates {
        assert_eq!(gate.status, CertificationGateStatus::Passed, "{gate:?}");
    }
}

#[tokio::test]
async fn certification_stub_reads_a_body_split_from_its_headers() {
    let model = "hf://meshllm/split-request@abc123";
    let api_base = spawn_certification_stub_server(model.to_string()).await;
    let addr = api_base.strip_prefix("http://").unwrap();
    let body = json!({"model": model, "messages": []}).to_string();
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(headers.as_bytes()).await.unwrap();
    tokio::task::yield_now().await;
    stream.write_all(body.as_bytes()).await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200 OK"));
}
