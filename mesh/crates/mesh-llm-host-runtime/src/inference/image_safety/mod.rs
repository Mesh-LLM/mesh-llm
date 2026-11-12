//! Mandatory local admission for providers that declare image output.
mod artifacts;
mod json_keys;
mod response;

use std::time::Duration;

use mesh_llm_plugin::image_safety::{
    GENERATION_GUARD_CAPABILITY, GuardDecision, GuardStatus, GuardVerdict, IMAGE_OPERATION,
    ImageScreenRequest, MAX_TEXT_BYTES, PROTOCOL_VERSION, REQUEST_GUARD_CAPABILITY,
    STATUS_OPERATION, TEXT_OPERATION, TextScreenRequest,
};
use serde::{Serialize, de::DeserializeOwned};

use crate::mesh::Node;
use crate::network::openai::client_stream::ImageResponseSafetyPlan;

pub(crate) use response::screen_response;

#[cfg(test)]
pub(crate) fn validated_image_count_for_test(
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<usize> {
    artifacts::images_from_response(content_type, body).map(|images| images.len())
}

/// Deliberately carries no prompt, classifier rationale, or image bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageSafetyResponseError {
    Denied,
    Unavailable,
}

impl ImageSafetyResponseError {
    pub(crate) fn status(self) -> u16 {
        match self {
            Self::Denied => 403,
            Self::Unavailable => 503,
        }
    }
}

type SafetyResult<T> = Result<T, ImageSafetyResponseError>;
const GUARD_TIMEOUT: Duration = Duration::from_secs(120);

/// Modality selection must not interpret a different duplicate model field
/// than the backend receiving the original request bytes.
pub(crate) fn request_has_duplicate_keys(body: &[u8]) -> bool {
    json_keys::has_duplicate_keys(body)
}

/// A requester that opts into independent verification cannot trust a peer's
/// modality declaration. It screens every generation response from that peer.
pub(crate) async fn independent_requester_screening(node: &Node) -> SafetyResult<bool> {
    let Some(manager) = node.plugin_manager().await else {
        return Ok(false);
    };
    let provider = manager
        .provider_for_capability(REQUEST_GUARD_CAPABILITY)
        .await
        .map_err(|_| ImageSafetyResponseError::Unavailable)?;
    let Some(provider) = provider else {
        return Ok(false);
    };
    if !provider.available {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    // The selected mode is known while its weights are still loading. A
    // default text-only guard must not stall unrelated text traffic, while a
    // strict requester still reaches the ready-required admission below.
    Ok(read_status(node, &provider.plugin_name)
        .await?
        .screen_received_images)
}

/// Resolve and screen the effective request at each selected backend attempt.
/// Remote ingress only needs the serving guard; local ingress needs the
/// requester guard as well, including when it elects its own local provider.
pub(crate) async fn admit(
    node: &Node,
    body: &[u8],
    requesting: bool,
    serving: bool,
) -> SafetyResult<ImageResponseSafetyPlan> {
    let text = effective_text(body)?;
    let mut guard_plugins = Vec::new();
    if requesting {
        let (name, status) = required_guard(node, REQUEST_GUARD_CAPABILITY).await?;
        screen_text(node, &name, &status, &text).await?;
        if status.screen_received_images {
            guard_plugins.push(name);
        }
    }
    if serving {
        let (name, status) = required_guard(node, GENERATION_GUARD_CAPABILITY).await?;
        if !status.screen_received_images {
            return Err(ImageSafetyResponseError::Unavailable);
        }
        screen_text(node, &name, &status, &text).await?;
        if !guard_plugins.contains(&name) {
            guard_plugins.push(name);
        }
    }
    Ok(ImageResponseSafetyPlan { guard_plugins })
}

async fn required_guard(node: &Node, capability: &str) -> SafetyResult<(String, GuardStatus)> {
    let manager = node
        .plugin_manager()
        .await
        .ok_or(ImageSafetyResponseError::Unavailable)?;
    let provider = manager
        .available_provider_for_capability(capability)
        .await
        .map_err(|_| ImageSafetyResponseError::Unavailable)?
        .ok_or(ImageSafetyResponseError::Unavailable)?;
    let status = guard_status(node, &provider.plugin_name).await?;
    Ok((provider.plugin_name, status))
}

async fn guard_status(node: &Node, name: &str) -> SafetyResult<GuardStatus> {
    let status = read_status(node, name).await?;
    if !status.ready {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    Ok(status)
}

async fn read_status(node: &Node, name: &str) -> SafetyResult<GuardStatus> {
    let status: GuardStatus = invoke(node, name, STATUS_OPERATION, &serde_json::json!({})).await?;
    if status.protocol_version != PROTOCOL_VERSION
        || status.model.is_empty()
        || status.policy_revision.is_empty()
    {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    Ok(status)
}

async fn screen_text(
    node: &Node,
    name: &str,
    status: &GuardStatus,
    text: &str,
) -> SafetyResult<()> {
    let verdict = invoke(
        node,
        name,
        TEXT_OPERATION,
        &TextScreenRequest {
            text: text.to_owned(),
        },
    )
    .await?;
    check_verdict(verdict, status)
}

async fn screen_image(node: &Node, name: &str, data_url: String) -> SafetyResult<()> {
    let status = guard_status(node, name).await?;
    if !status.screen_received_images {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    let verdict = invoke(
        node,
        name,
        IMAGE_OPERATION,
        &ImageScreenRequest { data_url },
    )
    .await?;
    check_verdict(verdict, &status)
}

fn check_verdict(verdict: GuardVerdict, status: &GuardStatus) -> SafetyResult<()> {
    if verdict.model != status.model
        || verdict.policy_revision != status.policy_revision
        || verdict.reason_code.is_empty()
        || verdict.reason_code.len() > 128
        || !verdict
            .reason_code
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    match verdict.decision {
        GuardDecision::Allow => Ok(()),
        GuardDecision::Block => Err(ImageSafetyResponseError::Denied),
        GuardDecision::Unavailable => Err(ImageSafetyResponseError::Unavailable),
    }
}

async fn invoke<T: DeserializeOwned>(
    node: &Node,
    name: &str,
    operation: &str,
    input: &impl Serialize,
) -> SafetyResult<T> {
    let manager = node
        .plugin_manager()
        .await
        .ok_or(ImageSafetyResponseError::Unavailable)?;
    let input = serde_json::to_string(input).map_err(|_| ImageSafetyResponseError::Unavailable)?;
    let result = tokio::time::timeout(
        GUARD_TIMEOUT,
        manager.invoke_operation(name, operation, &input),
    )
    .await
    .map_err(|_| ImageSafetyResponseError::Unavailable)?
    .map_err(|_| ImageSafetyResponseError::Unavailable)?;
    if result.is_error || result.content_json.len() > 4096 {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    serde_json::from_str(&result.content_json).map_err(|_| ImageSafetyResponseError::Unavailable)
}

/// Only supported text input is accepted. Never truncate a conversation to fit
/// the guard: an omitted turn or tool result could change its meaning.
fn effective_text(body: &[u8]) -> SafetyResult<String> {
    let value = json_keys::parse(body).map_err(|_| ImageSafetyResponseError::Unavailable)?;
    let object = value
        .as_object()
        .ok_or(ImageSafetyResponseError::Unavailable)?;
    // Providers may consume additional fields (template arguments, assistant
    // prefills, negative prompts). Keep their complete effective input in the
    // audit instead of guessing a universal allowlist of semantic fields.
    reject_media(&value)?;
    if !["messages", "input", "prompt"]
        .iter()
        .filter_map(|key| object.get(*key))
        .any(has_text_input)
    {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    let text = serde_json::to_string(&value).map_err(|_| ImageSafetyResponseError::Unavailable)?;
    if text.len() > MAX_TEXT_BYTES {
        return Err(ImageSafetyResponseError::Unavailable);
    }
    Ok(text)
}

fn has_text_input(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(text) => !text.trim().is_empty(),
        serde_json::Value::Array(values) => values.iter().any(has_text_input),
        serde_json::Value::Object(object) => ["content", "text"]
            .iter()
            .filter_map(|key| object.get(*key))
            .any(has_text_input),
        _ => false,
    }
}

fn reject_media(value: &serde_json::Value) -> SafetyResult<()> {
    match value {
        serde_json::Value::Object(object) => {
            if object
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|kind| {
                    matches!(
                        kind,
                        "image_url"
                            | "image"
                            | "input_image"
                            | "input_audio"
                            | "audio"
                            | "audio_url"
                            | "video"
                            | "video_url"
                            | "input_video"
                            | "document"
                            | "file"
                            | "input_file"
                    )
                })
                || object.contains_key("image_url")
                || object.contains_key("file_id")
            {
                return Err(ImageSafetyResponseError::Unavailable);
            }
            for value in object.values() {
                reject_media(value)?;
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                reject_media(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests;
