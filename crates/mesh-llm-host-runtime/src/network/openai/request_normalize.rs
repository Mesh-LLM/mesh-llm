use crate::mesh;
use crate::plugin;
use anyhow::Result;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseAdapter {
    None,
    OpenAiChatCompletionsJson,
    OpenAiChatCompletionsStream,
    OpenAiResponsesJson,
    OpenAiResponsesStream,
}

#[derive(Debug)]
pub(super) struct RequestNormalization {
    pub(super) changed: bool,
    pub(super) rewritten_path: Option<String>,
    pub(super) response_adapter: ResponseAdapter,
}

pub(super) fn normalize_openai_compat_request(
    path: &str,
    body: &mut serde_json::Value,
) -> Result<RequestNormalization> {
    let normalized = openai_frontend::normalize_openai_compat_request(path, body)?;
    let response_adapter = match normalized.response_adapter {
        openai_frontend::ResponseAdapterMode::None => ResponseAdapter::None,
        openai_frontend::ResponseAdapterMode::OpenAiResponsesJson => {
            ResponseAdapter::OpenAiResponsesJson
        }
        openai_frontend::ResponseAdapterMode::OpenAiResponsesStream => {
            ResponseAdapter::OpenAiResponsesStream
        }
    };
    Ok(RequestNormalization {
        changed: normalized.changed,
        rewritten_path: normalized.rewritten_path,
        response_adapter,
    })
}

fn request_id_from_body(body: &serde_json::Value) -> Option<String> {
    body.get("request_id")
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn mesh_blob_token_from_url(url: &str) -> Option<String> {
    let path = url.strip_prefix("mesh://blob/")?;
    let mut parts = path.split('/').filter(|part| !part.trim().is_empty());
    let _client_id = parts.next()?;
    let token = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    Some(token.to_string())
}

fn blob_token_from_container(container: &serde_json::Value) -> Option<String> {
    container
        .get("url")
        .and_then(|value| value.as_str())
        .and_then(mesh_blob_token_from_url)
        .or_else(|| {
            ["mesh_token", "blob_token", "token"]
                .into_iter()
                .find_map(|key| {
                    container
                        .get(key)
                        .and_then(|value| value.as_str())
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(ToString::to_string)
                })
        })
}

fn data_url(mime_type: &str, bytes_base64: &str) -> String {
    format!("data:{mime_type};base64,{bytes_base64}")
}

fn audio_format_from_mime_type(mime_type: &str) -> Option<&'static str> {
    match mime_type {
        "audio/wav" | "audio/x-wav" => Some("wav"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/flac" => Some("flac"),
        "audio/ogg" | "audio/opus" => Some("ogg"),
        "audio/webm" => Some("webm"),
        _ => None,
    }
}

enum MediaRefAction {
    DataUrlContainer { container_key: &'static str },
    InputAudio,
}

fn block_media_ref_action(block: &serde_json::Value) -> Option<(MediaRefAction, String)> {
    for key in [
        "image_url",
        "audio_url",
        "image",
        "audio",
        "input_image",
        "file",
        "input_file",
    ] {
        let Some(container) = block.get(key) else {
            continue;
        };
        let Some(token) = blob_token_from_container(container) else {
            continue;
        };
        return Some((
            MediaRefAction::DataUrlContainer { container_key: key },
            token,
        ));
    }

    let input_audio = block.get("input_audio")?;
    let token = blob_token_from_container(input_audio)?;
    Some((MediaRefAction::InputAudio, token))
}

pub(super) async fn resolve_request_object_references(
    path: &str,
    body: &mut serde_json::Value,
    plugin_manager: &plugin::PluginManager,
) -> Result<Vec<String>> {
    let path_only = path.split('?').next().unwrap_or(path);
    if path_only != "/v1/chat/completions" {
        return Ok(Vec::new());
    }
    let request_id = request_id_from_body(body);
    let Some(messages) = body
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
    else {
        return Ok(Vec::new());
    };

    let mut request_ids = Vec::new();
    let mut blob_cache: HashMap<String, crate::plugins::blobstore::GetRequestObjectResponse> =
        HashMap::new();
    for message in messages.iter_mut() {
        let Some(blocks) = message
            .get_mut("content")
            .and_then(|value| value.as_array_mut())
        else {
            continue;
        };
        for block in blocks.iter_mut() {
            let Some((action, token)) = block_media_ref_action(block) else {
                continue;
            };
            let blob = if let Some(cached) = blob_cache.get(&token) {
                cached.clone()
            } else {
                let fetched = crate::plugins::blobstore::get_request_object(
                    plugin_manager,
                    crate::plugins::blobstore::GetRequestObjectRequest {
                        token: token.clone(),
                        request_id: request_id.clone(),
                    },
                )
                .await?;
                blob_cache.insert(token.clone(), fetched.clone());
                fetched
            };
            if !request_ids
                .iter()
                .any(|existing| existing == &blob.request_id)
            {
                request_ids.push(blob.request_id.clone());
            }
            match action {
                MediaRefAction::DataUrlContainer { container_key } => {
                    if let Some(container) = block
                        .get_mut(container_key)
                        .and_then(|value| value.as_object_mut())
                    {
                        container.insert(
                            "url".into(),
                            serde_json::Value::String(data_url(
                                &blob.mime_type,
                                &blob.bytes_base64,
                            )),
                        );
                        container.remove("mesh_token");
                        container.remove("blob_token");
                        container.remove("token");
                    }
                }
                MediaRefAction::InputAudio => {
                    if let Some(container) = block
                        .get_mut("input_audio")
                        .and_then(|value| value.as_object_mut())
                    {
                        container.insert(
                            "data".into(),
                            serde_json::Value::String(blob.bytes_base64.clone()),
                        );
                        if let Some(format) = audio_format_from_mime_type(&blob.mime_type) {
                            container
                                .entry("format")
                                .or_insert_with(|| serde_json::Value::String(format.to_string()));
                        }
                        container.insert(
                            "mime_type".into(),
                            serde_json::Value::String(blob.mime_type.clone()),
                        );
                        container.remove("url");
                        container.remove("mesh_token");
                        container.remove("blob_token");
                        container.remove("token");
                    }
                }
            }
        }
    }

    Ok(request_ids)
}

pub async fn release_request_objects(node: &mesh::Node, request_ids: &[String]) {
    if request_ids.is_empty() {
        return;
    }
    let Some(plugin_manager) = node.plugin_manager().await else {
        return;
    };
    for request_id in request_ids {
        if let Err(err) = crate::plugins::blobstore::complete_request(
            &plugin_manager,
            crate::plugins::blobstore::FinishRequestRequest {
                request_id: request_id.clone(),
            },
        )
        .await
        {
            tracing::warn!(
                request_id,
                error = %err,
                "blobstore: failed to release request-scoped objects"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_openai_compat_request_translates_responses_input() {
        let mut body = serde_json::json!({
            "model": "test",
            "instructions": "be concise",
            "max_output_tokens": 256,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this"},
                    {"type": "input_image", "image_url": "mesh://blob/client-1/token-1"},
                    {"type": "input_audio", "audio_url": "mesh://blob/client-1/token-2"}
                ]
            }]
        });

        let normalization = normalize_openai_compat_request("/v1/responses", &mut body).unwrap();

        assert!(normalization.changed);
        assert_eq!(
            normalization.rewritten_path.as_deref(),
            Some("/v1/chat/completions")
        );
        assert_eq!(
            normalization.response_adapter,
            ResponseAdapter::OpenAiResponsesJson
        );
        assert_eq!(body["max_tokens"], 256);
        assert!(body.get("max_output_tokens").is_none());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "be concise");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"][0]["type"], "text");
        assert_eq!(body["messages"][1]["content"][1]["type"], "image_url");
        assert_eq!(
            body["messages"][1]["content"][1]["image_url"]["url"],
            "mesh://blob/client-1/token-1"
        );
        assert_eq!(body["messages"][1]["content"][2]["type"], "input_audio");
        assert_eq!(
            body["messages"][1]["content"][2]["input_audio"]["url"],
            "mesh://blob/client-1/token-2"
        );
    }

    #[test]
    fn test_normalize_openai_compat_request_marks_streaming_responses_adapter() {
        let mut body = serde_json::json!({
            "model": "test",
            "stream": true,
            "input": "hello",
        });
        let normalization = normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
        assert_eq!(
            normalization.response_adapter,
            ResponseAdapter::OpenAiResponsesStream
        );
        assert_eq!(
            normalization.rewritten_path.as_deref(),
            Some("/v1/chat/completions")
        );
        assert_eq!(body["messages"][0]["content"], "hello");
    }
    #[test]
    fn test_mesh_blob_token_from_url_requires_client_id_segment() {
        assert_eq!(
            mesh_blob_token_from_url("mesh://blob/client-1/token-123"),
            Some("token-123".to_string())
        );
        assert_eq!(mesh_blob_token_from_url("mesh://blob/token-123"), None);
        assert_eq!(
            mesh_blob_token_from_url("mesh://blob/client-1/token-123/extra"),
            None
        );
    }
}
