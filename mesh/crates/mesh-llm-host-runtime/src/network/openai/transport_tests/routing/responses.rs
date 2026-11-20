use super::*;
use crate::network::openai::request_parse::parse_json_body_from_http_request;

#[derive(serde::Deserialize)]
struct ReleasedPeerRequestMetadata {
    model: Option<String>,
    stream: Option<bool>,
    max_completion_tokens: Option<u32>,
    max_tokens: Option<u32>,
    max_output_tokens: Option<u32>,
    n_predict: Option<u32>,
}

async fn parse_http_request(raw: &[u8]) -> BufferedHttpRequest {
    let (mut client, mut server) = tokio::io::duplex(raw.len() + 1);
    client.write_all(raw).await.unwrap();
    client.shutdown().await.unwrap();
    read_http_request(&mut server).await.unwrap()
}

fn responses_request(directive: &str) -> Vec<u8> {
    let body = serde_json::json!({
        "model": directive,
        "input": "Reply only OK",
        "max_output_tokens": 16,
        "max_tokens": 4_294_967_296_u64,
        "max_completion_tokens": "ignored",
        "n_predict": {},
    })
    .to_string();
    format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

#[tokio::test]
async fn responses_auto_plan_sends_concrete_model_despite_malformed_losing_aliases() -> Result<()> {
    let model = "selected-local-model";
    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let node = test_node_with_remote_models(&[(model, peer_id)]).await;
    let affinity = AffinityRouter::new();

    for directive in ["auto", "mesh"] {
        let ingress = responses_request(directive);
        let released_poisoned_body = serde_json::json!({
            "model": model,
            "max_tokens": 16,
            "n_predict": {},
        });
        assert!(
            serde_json::from_value::<ReleasedPeerRequestMetadata>(released_poisoned_body).is_err(),
            "{directive}: malformed n_predict must reproduce released peer atomic-parse failure"
        );
        let mut request = parse_http_request(&ingress).await;
        let plan = build_mesh_request_plan(&node, &mut request, false, &affinity)
            .await
            .map_err(|_| anyhow::anyhow!("{directive} Responses route should resolve"))?;

        assert_eq!(plan.effective_model.as_deref(), Some(model), "{directive}");
        assert_eq!(request.model_name.as_deref(), Some(model), "{directive}");
        assert_eq!(request.client_path, "/v1/responses", "{directive}");
        assert_eq!(request.path, "/v1/chat/completions", "{directive}");
        assert_eq!(request.body_json.as_ref().unwrap()["model"], model);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(request.body_bytes.as_ref().unwrap())
                .unwrap()["model"],
            model
        );
        let outgoing =
            parse_json_body_from_http_request(&request.raw).expect("peer-bound request body");
        assert_eq!(outgoing["model"], model, "{directive}");
        assert_eq!(outgoing["max_tokens"], 16, "{directive}");
        for alias in ["max_output_tokens", "max_completion_tokens", "n_predict"] {
            assert!(
                outgoing.get(alias).is_none(),
                "{directive}: forwarded {alias}"
            );
        }
        assert_ne!(outgoing["model"], "auto", "{directive}");
        assert_ne!(outgoing["model"], "mesh", "{directive}");

        // Released peers parse this metadata atomically before dispatch and do
        // not have the current marker-based recovery. The old peer-bound shape
        // above fails that parser, but the newly normalized body parses.
        let released: ReleasedPeerRequestMetadata =
            serde_json::from_value(outgoing).expect("released peer metadata parser");
        assert_eq!(released.model.as_deref(), Some(model), "{directive}");
        assert_eq!(released.stream, None, "{directive}");
        assert_eq!(released.max_tokens, Some(16), "{directive}");
        assert_eq!(released.max_completion_tokens, None, "{directive}");
        assert_eq!(released.max_output_tokens, None, "{directive}");
        assert_eq!(released.n_predict, None, "{directive}");
    }
    Ok(())
}
