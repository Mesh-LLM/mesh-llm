//! Request transforms: a plugin rewrites an inference request on the node the
//! client sent it to, before the mesh routes it.
//!
//! A plugin declares [`REQUEST_TRANSFORM_CAPABILITY`] (with
//! [`request_transform`]) and handles calls with
//! [`crate::SimplePlugin::with_request_transform`]. The host calls it with the
//! request already normalized to Chat Completions shape (Responses and
//! Anthropic Messages requests are translated first). The plugin returns a
//! replacement body, leaves the request unchanged, or fails the call, which
//! rejects the request with the error's message. Transforms never run on a
//! request another node forwarded, so each runs once per client request.
//! See `mesh/docs/plugins/request-transform.md`.

use crate::{ManifestEntry, capability};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Declared in a plugin manifest's capabilities, and listed in
/// `InitializeRequest.host_capabilities` by hosts that call transforms.
pub const REQUEST_TRANSFORM_CAPABILITY: &str = "request_transform.v1";

/// The manifest entry that opts a plugin into request transforms.
pub fn request_transform() -> ManifestEntry {
    capability(REQUEST_TRANSFORM_CAPABILITY)
}

/// What the host sends.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RequestTransformInput {
    /// The path the request is forwarded on, e.g. `/v1/chat/completions`.
    pub path: String,
    /// The path the client used, e.g. `/v1/messages`.
    pub client_path: String,
    /// The JSON body, after normalization and earlier transforms.
    pub body: Value,
}

/// What the plugin returns.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct RequestTransformOutput {
    /// The replacement body; absent leaves the request unchanged. It must keep
    /// the request's `model` and `stream`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
}

impl RequestTransformOutput {
    pub fn unchanged() -> Self {
        Self::default()
    }

    pub fn replaced(body: Value) -> Self {
        Self { body: Some(body) }
    }
}

pub type RequestTransformFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = crate::PluginResult<RequestTransformOutput>> + Send + 'a>,
>;
pub type RequestTransformHandler = std::sync::Arc<
    dyn for<'a, 'ctx> Fn(
            RequestTransformInput,
            &'a mut crate::PluginContext<'ctx>,
        ) -> RequestTransformFuture<'a>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unchanged_output_has_no_body() {
        assert_eq!(
            serde_json::to_value(RequestTransformOutput::unchanged()).unwrap(),
            json!({})
        );
        let parsed: RequestTransformOutput = serde_json::from_value(json!({})).unwrap();
        assert_eq!(parsed, RequestTransformOutput::unchanged());
        let replaced = RequestTransformOutput::replaced(json!({"model": "m"}));
        assert_eq!(
            serde_json::to_value(replaced).unwrap(),
            json!({"body": {"model": "m"}})
        );
    }

    #[test]
    fn the_manifest_entry_declares_the_capability() {
        let manifest = crate::plugin_manifest![request_transform()];
        assert_eq!(manifest.capabilities, vec![REQUEST_TRANSFORM_CAPABILITY]);
    }
}
