//! The ingress step that runs plugin request transforms
//! (`request_transform.v1`, see `plugin::request_transform`).
//!
//! Transforms run once per client request, on the node the client sent it
//! to, after Responses and Messages bodies have been normalized to Chat
//! Completions and before routing. A request another node forwarded here was
//! already transformed there, so its ingress skips them.

use crate::plugin;
use anyhow::{Result, bail};
use skippy_inference_api::REQUIRES_TRANSFORM_KEY;

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
    (private_hint, owner_limit): (bool, Option<crate::mesh::OwnerScope>),
    request_path: &str,
    client_path: &str,
    body: &mut serde_json::Value,
) -> Result<bool> {
    match transformer {
        Some(plugin_manager)
            if request_path.split('?').next() == Some("/v1/chat/completions")
                && is_transformable_path(client_path) =>
        {
            // Transforms run before the scope is resolved; a transform's own
            // inference for a private request must stay private regardless,
            // including a bare `@private` model that ingress rejects later.
            // A nested call's owner limit applies to the transform's own
            // calls too, so they keep to the circle of the request behind it.
            let private = private_hint
                || owner_limit.is_some()
                || body
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(crate::network::openai::routing_scope::has_private_model_suffix);
            plugin_manager
                .transform_request(request_path, client_path, body, private, owner_limit)
                .await
        }
        _ => Ok(false),
    }
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
                let content = if body["messages"][0]["content"] == "bloat" {
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
