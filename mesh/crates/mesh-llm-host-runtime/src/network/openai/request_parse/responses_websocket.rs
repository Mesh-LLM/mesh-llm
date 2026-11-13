//! Convert a shared session-prepared Responses turn into the ordinary Mesh
//! buffered-request representation.

use super::*;

/// The shared session has already normalized and expanded the Responses
/// conversation. This seam applies the same Mesh request-object and plugin
/// transforms that follow normalization on the HTTP path.
pub(in crate::network::openai) async fn buffer_prepared_responses_turn(
    handshake: &BufferedHttpRequest,
    prepared: skippy_inference_api::responses::PreparedResponsesTurn,
    plugin_manager: Option<&plugin::PluginManager>,
) -> Result<BufferedHttpRequest> {
    let ingress_body = serde_json::to_vec(prepared.ingress_body())
        .context("serialize Responses WebSocket ingress body")?;
    let (turn, generation) = prepared.into_parts();
    let mut body_json = serde_json::to_value(
        skippy_inference_api::chat::request_view::ChatCompletionRequestRef(&generation),
    )
    .context("serialize prepared Responses generation request")?;
    let request_object_request_ids =
        resolve_request_objects(plugin_manager, &mut body_json).await?;
    let body = match transform_body(plugin_manager, handshake, &mut body_json).await {
        Ok(body) => body,
        Err(error) => {
            if let Some(plugin_manager) = plugin_manager {
                release_request_objects_with(plugin_manager, &request_object_request_ids).await;
            }
            return Err(error);
        }
    };
    let request_id = RequestId::new();
    let raw = match build_raw_turn(handshake, &body, request_id) {
        Ok(raw) => raw,
        Err(error) => {
            if let Some(plugin_manager) = plugin_manager {
                release_request_objects_with(plugin_manager, &request_object_request_ids).await;
            }
            return Err(error);
        }
    };
    Ok(BufferedHttpRequest {
        raw,
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        client_path: handshake.client_path.clone(),
        request_id,
        body_json: Some(body_json),
        body_json_attempted: true,
        body_bytes: Some(ingress_body),
        body_len_bytes: body.len(),
        completion_tokens: generation.effective_max_tokens(),
        stream: Some(true),
        model_name: Some(generation.model.clone()),
        request_object_request_ids,
        response_adapter: ResponseAdapter::OpenAiResponsesStream,
        responses_turn: Some(turn),
        correlation_id: handshake.correlation_id.clone(),
        exchange_observation_id: None,
        automatic_route: false,
    })
}

async fn resolve_request_objects(
    plugin_manager: Option<&plugin::PluginManager>,
    body: &mut serde_json::Value,
) -> Result<Vec<String>> {
    match plugin_manager {
        Some(plugin_manager) => {
            resolve_request_object_references("/v1/chat/completions", body, plugin_manager).await
        }
        None => Ok(Vec::new()),
    }
}

async fn transform_body(
    plugin_manager: Option<&plugin::PluginManager>,
    handshake: &BufferedHttpRequest,
    body: &mut serde_json::Value,
) -> Result<Vec<u8>> {
    plugin_transform::apply_request_transforms(
        plugin_manager,
        (
            super::super::routing_scope::head_requests_private_scope(&handshake.raw),
            super::super::routing_scope::head_node_limit(&handshake.raw),
        ),
        "/v1/chat/completions",
        &handshake.client_path,
        body,
    )
    .await?;
    plugin_transform::reject_parts_needing_transform(body)?;
    let body =
        serde_json::to_vec(body).context("serialize transformed Responses WebSocket turn")?;
    if body.len() > MAX_BODY_BYTES {
        bail!("rewritten request body exceeds {MAX_BODY_BYTES} bytes");
    }
    Ok(body)
}

fn build_raw_turn(
    handshake: &BufferedHttpRequest,
    body: &[u8],
    request_id: RequestId,
) -> Result<Vec<u8>> {
    let header_end = handshake
        .raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|offset| offset + 4)
        .context("Responses WebSocket handshake has no HTTP header terminator")?;
    finalize_forwarded_request(
        handshake.raw.clone(),
        header_end,
        ForwardedRequestRewrite {
            strip_expect: true,
            rewritten_method: Some("POST"),
            rewritten_path: Some("/v1/chat/completions"),
            rewritten_body: Some(body),
            request_id,
            private_base: false,
            websocket_upgrade: false,
            strip_websocket_headers: true,
            ensure_json_content_type: true,
        },
    )
}
