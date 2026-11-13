//! The ingress step that runs plugin request transforms
//! (`request_transform.v1`, see `plugin::request_transform`).
//!
//! Transforms run once per client request, on the node the client sent it
//! to, after Responses and Messages bodies have been normalized to Chat
//! Completions and before routing. A request another node forwarded here was
//! already transformed there, so its ingress skips them.

use crate::plugin;
use anyhow::{Result, bail};
use skippy_inference_api::{MESH_COMPACT_FIELD, REQUIRES_TRANSFORM_KEY};

/// Whether this ingress runs plugin request transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RequestTransforms {
    /// The client sent the request to this node.
    Apply,
    /// Another node forwarded the request, already transformed.
    Skip,
}

/// Client paths whose bodies normalize to a Chat Completions request.
const TRANSFORMABLE_PATHS: &[&str] = &["/v1/chat/completions", "/v1/responses", "/v1/messages"];

/// The plugin manager to transform with: present only when this ingress runs
/// transforms and some plugin declares one.
pub(super) async fn request_transformer(
    plugin_manager: Option<&plugin::PluginManager>,
    transforms: RequestTransforms,
) -> Option<&plugin::PluginManager> {
    let plugin_manager = plugin_manager.filter(|_| transforms == RequestTransforms::Apply)?;
    plugin_manager
        .has_request_transforms()
        .await
        .then_some(plugin_manager)
}

pub(super) fn is_transformable_path(path: &str) -> bool {
    TRANSFORMABLE_PATHS.contains(&path.split('?').next().unwrap_or(path))
}

/// Run the transforms over a normalized request. Returns whether the body
/// changed; an error rejects the request.
pub(super) async fn apply_request_transforms(
    transformer: Option<&plugin::PluginManager>,
    (private_hint, node_limit): (bool, Option<crate::mesh::PrivateScope>),
    request_path: &str,
    client_path: &str,
    body: &mut serde_json::Value,
) -> Result<bool> {
    let transformed = match transformer {
        Some(plugin_manager)
            if request_path.split('?').next() == Some("/v1/chat/completions")
                && is_transformable_path(client_path) =>
        {
            // Transforms run before the scope is resolved; a transform's own
            // inference for a private request must stay private regardless,
            // including a bare `@private` model that ingress rejects later.
            // A nested call's node limit applies to the transform's own
            // calls too, so they keep to the scope of the request behind it.
            let private = private_hint
                || node_limit.is_some()
                || body
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(crate::network::openai::routing_scope::has_private_model_suffix);
            plugin_manager
                .transform_request(request_path, client_path, body, private, node_limit)
                .await
        }
        _ => Ok(false),
    }?;
    let reasserted = reassert_responses_ingress_policy(client_path, body);
    Ok(transformed || reasserted)
}

fn reassert_responses_ingress_policy(client_path: &str, body: &mut serde_json::Value) -> bool {
    if client_path.split('?').next() != Some("/v1/responses") {
        return false;
    }
    let Some(object) = body.as_object_mut() else {
        return false;
    };
    let compact_changed = object.insert(MESH_COMPACT_FIELD.into(), serde_json::Value::Bool(false))
        != Some(serde_json::Value::Bool(false));
    let max_tool_calls_removed = object.remove("max_tool_calls").is_some();
    compact_changed || max_tool_calls_removed
}

/// Reject a request that still holds an attachment only a transform can use
/// (an Anthropic `document` or a flat Responses `input_file`). These were
/// rejected before transforms existed, so without a plugin that converts them
/// they still are, instead of reaching a backend that would answer without
/// them.
pub(super) fn reject_parts_needing_transform(body: &serde_json::Value) -> Result<()> {
    let needs_transform = body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("content")?.as_array())
        .flatten()
        .any(|part| {
            part.get(REQUIRES_TRANSFORM_KEY)
                .and_then(serde_json::Value::as_bool)
                == Some(true)
        });
    if needs_transform {
        bail!("document attachments need a request transform plugin to convert them to text");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{BridgeFuture, PluginRpcBridge, RpcResult, proto};
    use mesh_llm_plugin::request_transform::{RequestTransformInput, RequestTransformOutput};
    use serde_json::json;
    use std::sync::Arc;
    use tokio::io::AsyncWriteExt;

    /// Replaces every user message with the client path it arrived on.
    struct PathStamp;

    impl PluginRpcBridge for PathStamp {
        fn handle_request(
            &self,
            _plugin_name: String,
            _method: String,
            params_json: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            Box::pin(async move {
                let input: RequestTransformInput = serde_json::from_str(&params_json).unwrap();
                let mut body = input.body;
                let original_content = body["messages"][0]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                match original_content.as_str() {
                    "remove-compact" => {
                        body.as_object_mut().unwrap().remove(MESH_COMPACT_FIELD);
                    }
                    "enable-compact" => {
                        body[MESH_COMPACT_FIELD] = json!(true);
                    }
                    "inject-max-tool-calls" => {
                        body["max_tool_calls"] = json!(999);
                    }
                    "weaken-parallel" => {
                        body["parallel_tool_calls"] = json!(true);
                    }
                    _ => {}
                }
                let content = if original_content == "bloat" {
                    "x".repeat(super::super::MAX_BODY_BYTES)
                } else {
                    format!("{} via {}", input.client_path, input.path)
                };
                body["messages"] = json!([{"role": "user", "content": content}]);
                Ok(RpcResult {
                    result_json: serde_json::to_string(&RequestTransformOutput::replaced(body))
                        .unwrap(),
                })
            })
        }

        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    async fn stamping_manager() -> plugin::PluginManager {
        let manager = plugin::PluginManager::for_test_bridge(&["stamp"], Arc::new(PathStamp));
        manager
            .set_test_manifests(
                [(
                    "stamp".to_string(),
                    mesh_llm_plugin::plugin_manifest![
                        mesh_llm_plugin::request_transform::request_transform()
                    ],
                )]
                .into(),
            )
            .await;
        manager
    }

    async fn read(
        path: &str,
        body: serde_json::Value,
        manager: &plugin::PluginManager,
        transforms: RequestTransforms,
    ) -> super::super::BufferedHttpRequest {
        let body = body.to_string();
        let (mut client, mut server) = tokio::io::duplex(64 * 1024);
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        client.write_all(request.as_bytes()).await.unwrap();
        super::super::read_http_request_with_limits_with_context(
            &mut server,
            super::super::HTTP_READ_LIMITS,
            Some(manager),
            transforms,
            None,
            None,
        )
        .await
        .map_err(super::super::OpenAiRequestReadError::into_error)
        .unwrap()
    }

    fn forwarded_body(request: &super::super::BufferedHttpRequest) -> serde_json::Value {
        super::super::parse_json_body_from_http_request(&request.raw).unwrap()
    }

    #[tokio::test]
    async fn client_requests_reach_routing_transformed_in_chat_shape() {
        let manager = stamping_manager().await;
        let messages = json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        });
        let request = read("/v1/messages", messages, &manager, RequestTransforms::Apply).await;
        assert_eq!(request.path, "/v1/chat/completions");
        let body = forwarded_body(&request);
        assert_eq!(
            body["messages"][0]["content"],
            "/v1/messages via /v1/chat/completions"
        );
        assert_eq!(request.body_len_bytes, body.to_string().len());
        assert_eq!(request.body_json.unwrap(), body);

        let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hello"}]});
        let request = read(
            "/v1/chat/completions",
            chat,
            &manager,
            RequestTransforms::Apply,
        )
        .await;
        assert_eq!(
            forwarded_body(&request)["messages"][0]["content"],
            "/v1/chat/completions via /v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn responses_transform_cannot_remove_or_enable_compaction() {
        let manager = stamping_manager().await;
        for attack in ["remove-compact", "enable-compact"] {
            let path = format!("/v1/responses?attack={attack}");
            let request = read(
                &path,
                json!({"model": "m", "input": attack}),
                &manager,
                RequestTransforms::Apply,
            )
            .await;
            let body = forwarded_body(&request);
            assert_eq!(body[MESH_COMPACT_FIELD], false, "attack: {attack}");
            assert_eq!(
                request.body_len_bytes,
                serde_json::to_vec(&body).unwrap().len()
            );
            assert_eq!(request.body_json.unwrap(), body);
        }
    }

    #[tokio::test]
    async fn responses_transform_cannot_reintroduce_ingress_max_tool_calls() {
        let manager = stamping_manager().await;
        let request = read(
            "/v1/responses",
            json!({
                "model": "m",
                "input": "inject-max-tool-calls",
                "max_tool_calls": 1,
            }),
            &manager,
            RequestTransforms::Apply,
        )
        .await;
        let body = forwarded_body(&request);
        assert!(body.get("max_tool_calls").is_none());
        let response = request.responses_turn.as_ref().unwrap().response(
            skippy_inference_api::responses::ResponseStatus::InProgress,
            &[],
            None,
            None,
        );
        assert_eq!(response["max_tool_calls"], 1);
        assert_eq!(
            request.body_len_bytes,
            serde_json::to_vec(&body).unwrap().len()
        );
        assert_eq!(request.body_json.unwrap(), body);
    }

    #[tokio::test]
    async fn responses_transform_cannot_weaken_captured_serial_policy() {
        let manager = stamping_manager().await;
        let request = read(
            "/v1/responses",
            json!({
                "model": "m",
                "input": "weaken-parallel",
                "parallel_tool_calls": false,
                "tools": [{"type": "function", "name": "same"}],
            }),
            &manager,
            RequestTransforms::Apply,
        )
        .await;
        let body = forwarded_body(&request);
        assert_eq!(body["parallel_tool_calls"], true);

        let response = request
            .responses_turn
            .as_ref()
            .unwrap()
            .finish_chat_completion(&json!({
                "choices": [{
                    "message": {"role": "assistant", "content": null, "tool_calls": [
                        {"id":"c0","type":"function","function":{"name":"same","arguments":"{}"}},
                        {"id":"c1","type":"function","function":{"name":"same","arguments":"{}"}}
                    ]},
                    "finish_reason": "tool_calls"
                }]
            }));
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "tool_call_limit_exceeded");
        assert_eq!(response["parallel_tool_calls"], false);
    }

    #[tokio::test]
    async fn chat_transform_compaction_override_is_unchanged() {
        let manager = stamping_manager().await;
        let request = read(
            "/v1/chat/completions",
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "enable-compact"}]
            }),
            &manager,
            RequestTransforms::Apply,
        )
        .await;

        assert_eq!(forwarded_body(&request)[MESH_COMPACT_FIELD], true);
    }

    fn responses_handshake(path: &str) -> super::super::BufferedHttpRequest {
        super::super::BufferedHttpRequest {
            raw: format!(
                "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n"
            )
            .into_bytes(),
            method: "GET".into(),
            path: path.into(),
            client_path: path.into(),
            request_id: super::super::RequestId::new(),
            body_json: None,
            body_json_attempted: true,
            body_bytes: None,
            body_len_bytes: 0,
            completion_tokens: None,
            stream: None,
            model_name: None,
            request_object_request_ids: Vec::new(),
            response_adapter: super::super::ResponseAdapter::None,
            responses_turn: None,
            correlation_id: None,
            exchange_observation_id: None,
            automatic_route: false,
        }
    }

    #[tokio::test]
    async fn responses_websocket_turn_reasserts_marker_and_wire_length() {
        let manager = stamping_manager().await;
        let session = skippy_inference_api::responses::ResponsesSession::new(
            skippy_inference_api::responses::ResponsesSessionConfig::default(),
            None,
            None,
        );
        let prepared = session
            .prepare_create(
                json!({
                    "type": "response.create",
                    "model": "m",
                    "input": "enable-compact"
                }),
                None,
            )
            .unwrap();
        let handshake = responses_handshake("/v1/responses?transport=websocket");
        let request = super::super::responses_websocket::buffer_prepared_responses_turn(
            &handshake,
            prepared,
            Some(&manager),
        )
        .await
        .unwrap();
        let body = forwarded_body(&request);

        assert_eq!(body[MESH_COMPACT_FIELD], false);
        assert_eq!(
            request.body_len_bytes,
            serde_json::to_vec(&body).unwrap().len()
        );
        assert_eq!(request.body_json.unwrap(), body);
    }

    #[tokio::test]
    async fn responses_websocket_turn_strips_reintroduced_max_tool_calls() {
        let manager = stamping_manager().await;
        let session = skippy_inference_api::responses::ResponsesSession::new(
            skippy_inference_api::responses::ResponsesSessionConfig::default(),
            None,
            None,
        );
        let prepared = session
            .prepare_create(
                json!({
                    "type": "response.create",
                    "model": "m",
                    "input": "inject-max-tool-calls",
                    "max_tool_calls": 1
                }),
                None,
            )
            .unwrap();
        let handshake = responses_handshake("/v1/responses?transport=websocket");
        let request = super::super::responses_websocket::buffer_prepared_responses_turn(
            &handshake,
            prepared,
            Some(&manager),
        )
        .await
        .unwrap();
        let body = forwarded_body(&request);
        assert!(body.get("max_tool_calls").is_none());
        let response = request.responses_turn.as_ref().unwrap().response(
            skippy_inference_api::responses::ResponseStatus::InProgress,
            &[],
            None,
            None,
        );
        assert_eq!(response["max_tool_calls"], 1);
        assert_eq!(
            request.body_len_bytes,
            serde_json::to_vec(&body).unwrap().len()
        );
        assert_eq!(request.body_json.unwrap(), body);
    }

    #[tokio::test]
    async fn forwarded_requests_are_not_transformed_again() {
        let manager = stamping_manager().await;
        let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hello"}]});
        let request = read(
            "/v1/chat/completions",
            chat.clone(),
            &manager,
            RequestTransforms::Skip,
        )
        .await;
        assert_eq!(forwarded_body(&request), chat);
    }

    #[tokio::test]
    async fn other_paths_are_not_transformed() {
        let manager = stamping_manager().await;
        let embeddings = json!({"model": "m", "input": "hello", "messages": []});
        let request = read(
            "/v1/embeddings",
            embeddings.clone(),
            &manager,
            RequestTransforms::Apply,
        )
        .await;
        assert_eq!(forwarded_body(&request), embeddings);
    }

    #[tokio::test]
    async fn a_transform_may_not_grow_the_body_past_the_forwarding_limit() {
        let manager = stamping_manager().await;
        let body =
            json!({"model": "m", "messages": [{"role": "user", "content": "bloat"}]}).to_string();
        let (mut client, mut server) = tokio::io::duplex(64 * 1024);
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        client.write_all(request.as_bytes()).await.unwrap();
        let error = super::super::read_http_request_with_limits_with_context(
            &mut server,
            super::super::HTTP_READ_LIMITS,
            Some(&manager),
            RequestTransforms::Apply,
            None,
            None,
        )
        .await
        .map_err(super::super::OpenAiRequestReadError::into_error)
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("transformed request body exceeds"),
            "{error:#}"
        );
    }

    async fn read_without_plugins(
        path: &str,
        body: serde_json::Value,
    ) -> Result<super::super::BufferedHttpRequest> {
        let body = body.to_string();
        let (mut client, mut server) = tokio::io::duplex(64 * 1024);
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        client.write_all(request.as_bytes()).await.unwrap();
        super::super::read_http_request_with_limits_with_context(
            &mut server,
            super::super::HTTP_READ_LIMITS,
            None,
            RequestTransforms::Apply,
            None,
            None,
        )
        .await
        .map_err(super::super::OpenAiRequestReadError::into_error)
    }

    fn pdf_document() -> serde_json::Value {
        json!({"type": "document", "source":
            {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}})
    }

    #[tokio::test]
    async fn documents_without_a_transform_are_rejected_as_before() {
        let messages = json!({"model": "m", "max_tokens": 16, "messages": [
            {"role": "user", "content": [pdf_document(), {"type": "text", "text": "summarize"}]}
        ]});
        let tool_result = json!({"model": "m", "max_tokens": 16, "messages": [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [pdf_document()]}]}
        ]});
        let flat_file = json!({"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0="},
            {"type": "input_text", "text": "summarize"}
        ]}]});
        for (path, body) in [
            ("/v1/messages", messages),
            ("/v1/messages", tool_result),
            ("/v1/responses", flat_file),
        ] {
            let error = read_without_plugins(path, body).await.unwrap_err();
            assert!(
                format!("{error:#}").contains("need a request transform plugin"),
                "{path}: {error:#}"
            );
        }
    }

    #[tokio::test]
    async fn files_accepted_before_transforms_still_pass_without_one() {
        let nested_file = json!({"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_file", "input_file": {"url": "data:text/plain;base64,aGVsbG8="}},
            {"type": "input_text", "text": "summarize"}
        ]}]});
        let request = read_without_plugins("/v1/responses", nested_file)
            .await
            .unwrap();
        let body = forwarded_body(&request);
        assert_eq!(body["messages"][0]["content"][0]["type"], "input_file");
        assert!(
            body["messages"][0]["content"][0]
                .get(REQUIRES_TRANSFORM_KEY)
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_transform_that_converts_documents_lets_them_through() {
        let manager = stamping_manager().await;
        let messages = json!({"model": "m", "max_tokens": 16, "messages": [
            {"role": "user", "content": [pdf_document(), {"type": "text", "text": "summarize"}]}
        ]});
        let request = read("/v1/messages", messages, &manager, RequestTransforms::Apply).await;
        assert_eq!(
            forwarded_body(&request)["messages"][0]["content"],
            "/v1/messages via /v1/chat/completions"
        );
    }

    #[test]
    fn only_inference_paths_are_transformable() {
        for path in ["/v1/chat/completions", "/v1/responses?x=1", "/v1/messages"] {
            assert!(is_transformable_path(path), "{path}");
        }
        for path in [
            "/v1/messages/count_tokens",
            "/v1/embeddings",
            "/v1/models",
            "/api/objects",
        ] {
            assert!(!is_transformable_path(path), "{path}");
        }
    }
}
