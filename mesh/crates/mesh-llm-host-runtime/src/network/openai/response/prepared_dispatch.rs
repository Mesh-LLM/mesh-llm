//! Policy observes the actual prepared orchestration dispatch bytes.
use crate::{mesh, plugin};
use serde_json::json;

pub(crate) struct PreparedSubdispatch<'a> {
    pub observation_id: Option<&'a str>,
    pub exchange_id: String,
    /// The ingress request id, a separate field from the exchange id.
    pub request_id: String,
    pub bytes: &'a [u8],
    pub client_path: &'a str,
    pub encoding: &'a str,
    pub model: &'a str,
    pub provider: &'a str,
    pub target: &'a str,
    pub attempt: usize,
}

pub(crate) async fn admit(
    node: &mesh::Node,
    dispatch: PreparedSubdispatch<'_>,
) -> Option<plugin::PhaseResult> {
    let manager = node.plugin_manager().await?;
    if !manager.has_exchange_hooks().await {
        return None;
    }
    let event = prepared_event(&dispatch);
    Some(match dispatch.observation_id {
        Some(id) => manager.selected_exchange_phase(id, event).await,
        None => manager.exchange_phase(event).await,
    })
}

fn prepared_event(dispatch: &PreparedSubdispatch<'_>) -> serde_json::Value {
    let endpoint = match dispatch.client_path.split('?').next() {
        Some("/v1/responses") => "responses",
        Some("/v1/completions") => "completions",
        _ => "chat_completions",
    };
    let mut event = plugin::request_event(
        dispatch.exchange_id.clone(),
        endpoint,
        "POST",
        dispatch.client_path,
        dispatch.bytes,
        Default::default(),
        false,
    );
    event["request_id"] = json!(dispatch.request_id);
    event["phase"] = json!("backend_selected");
    event["observation_point"] = json!("backend_dispatch");
    event["model"] = json!(dispatch.model);
    event["provider"] = json!(dispatch.provider);
    event["target"] = json!(dispatch.target);
    event["attempt"] = json!(dispatch.attempt);
    event["effective_request_encoding"] = json!(dispatch.encoding);
    event["effective_request_wire_digest"] = event["request_wire_digest"].take();
    event.as_object_mut().unwrap().remove("request_wire_digest");
    event
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_invocation_admission_preserves_original_client_endpoint_and_path() {
        for (path, endpoint) in [
            ("/v1/responses?debug=1", "responses"),
            ("/v1/completions", "completions"),
            ("/v1/chat/completions", "chat_completions"),
        ] {
            let event = prepared_event(&PreparedSubdispatch {
                observation_id: None,
                exchange_id: "test-exchange".into(),
                request_id: "test-request".into(),
                bytes: br#"{"request":{"model":"virtual-model","messages":[]}}"#,
                client_path: path,
                encoding: "plugin_invocation_json",
                model: "virtual-model",
                provider: "virtual_model",
                target: "author-plugin",
                attempt: 1,
            });
            assert_eq!(event["exchange_id"], "test-exchange");
            assert_eq!(event["request_id"], "test-request");
            assert_eq!(event["endpoint"], endpoint);
            assert_eq!(event["path"], path.split('?').next().unwrap());
            assert_eq!(event["phase"], "backend_selected");
            assert_eq!(
                event["effective_request_encoding"],
                "plugin_invocation_json"
            );
            assert_eq!(event["body"]["request"]["model"], "virtual-model");
            assert!(event.get("request_wire_digest").is_none());
            assert!(event["effective_request_wire_digest"].is_object());
        }
    }
}
