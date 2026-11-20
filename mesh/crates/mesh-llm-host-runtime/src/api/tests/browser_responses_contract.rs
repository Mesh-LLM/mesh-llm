//! Browser Chat keeps the real native Responses adapter, not just HTTP 200.

use super::*;
use crate::network::openai::ingress::api_proxy;

const CHAT_JSON: &str = r#"{"id":"chatcmpl-console","object":"chat.completion","created":1,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"console-marker"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#;
const CHAT_SSE: &str = "data: {\"id\":\"chatcmpl-console\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"console-marker\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-console\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

async fn browser_responses(streaming: bool) -> String {
    let (backend_port, backend_rx, backend) = if streaming {
        spawn_streaming_upstream(
            "text/event-stream",
            vec![(Duration::ZERO, CHAT_SSE.as_bytes().to_vec())],
        )
        .await
    } else {
        spawn_capturing_upstream(CHAT_JSON).await
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let inference_port = listener.local_addr().unwrap().port();
    let state = build_test_mesh_api_with_api_port(inference_port).await;
    let node = state.node().await;
    node.set_hosted_models(vec!["test-model".into()]).await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "test-model".into(),
        vec![election::InferenceTarget::Local(backend_port)],
    );
    let (_targets_tx, targets_rx) = tokio::sync::watch::channel(targets);
    let inference = tokio::spawn(api_proxy(
        node,
        inference_port,
        targets_rx,
        Some(listener.into()),
        false,
        affinity::AffinityRouter::new(),
    ));
    let (address, management) = spawn_management_test_server(state).await;
    let result = reqwest::Client::new()
        .post(format!("http://{address}/api/responses?trace=1"))
        .json(&json!({
            "model": "test-model",
            "input": "hello",
            "stream": streaming,
            "max_output_tokens": 32,
        }))
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .unwrap();
    let status = result.status();
    let response = result.text().await.unwrap();
    management.await.unwrap().unwrap();
    inference.abort();
    let _ = inference.await;
    backend.await.unwrap();
    assert_eq!(status, reqwest::StatusCode::OK, "{response}");
    let forwarded = String::from_utf8(backend_rx.await.unwrap()).unwrap();
    assert!(
        forwarded.starts_with("POST /v1/chat/completions?trace=1 HTTP/1.1\r\n"),
        "{forwarded}"
    );
    let body: serde_json::Value =
        serde_json::from_str(http_body_text(forwarded.as_bytes())).unwrap();
    assert_eq!(body["messages"][0]["content"], "hello");
    assert_eq!(body["stream"], streaming);
    response
}

#[tokio::test]
async fn browser_responses_json_preserves_response_contract() {
    let response = browser_responses(false).await;
    let body: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(body["object"], "response", "{response}");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["content"][0]["text"], "console-marker");
    assert!(body.get("choices").is_none());
}

#[tokio::test]
async fn browser_responses_stream_preserves_response_contract() {
    let response = browser_responses(true).await;
    let events: Vec<serde_json::Value> = response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("Responses streams contain JSON events"))
        .collect();
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.created"),
        "{response}"
    );
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.output_text.delta"
                && event["delta"] == "console-marker")
    );
    let completed = events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .expect("Responses stream must complete");
    assert_eq!(completed["response"]["object"], "response");
    assert_eq!(completed["response"]["status"], "completed");
    assert!(!response.contains("chat.completion.chunk"));
}
