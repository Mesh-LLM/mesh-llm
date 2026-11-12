//! Request transforms (`request_transform.v1`): plugins that rewrite an
//! inference request at ingress, after core normalization and before routing.
//!
//! Every plugin that declares the capability is called in plugin-name order,
//! each seeing the previous one's output. A plugin that is declared but not
//! available, fails, times out, or returns a body without the request's
//! `model` and `stream` rejects the request: a transform is configured because
//! the request must not reach a model without it.

use super::{PluginManager, proto};
use anyhow::{Context, Result, bail};
use mesh_llm_plugin::request_transform::{
    REQUEST_TRANSFORM_CAPABILITY, RequestTransformInput, RequestTransformOutput,
};
use serde_json::Value;
use std::time::Duration;

/// Generous: a transform may extract a document (OCR) before inference.
const TRANSFORM_TIMEOUT: Duration = Duration::from_secs(300);
/// The plugin control frame limit (16 MiB), less room for the envelope.
const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024 - 64 * 1024;
/// The handler name sent in `InvokeServiceRequest.service_name`.
const SERVICE_NAME: &str = "request_transform";

impl PluginManager {
    /// Plugins declaring request transforms, in call order, with availability.
    /// A plugin that declared one earlier in this manager's lifetime stays
    /// listed, unavailable, while it restarts or after it failed: its live
    /// provider entry is gone then, and requests must not skip its transform.
    async fn request_transform_plugins(&self) -> Result<Vec<(String, bool)>> {
        let mut plugins: Vec<(String, bool)> = self
            .capability_providers()
            .await?
            .into_iter()
            .filter(|provider| provider.capability == REQUEST_TRANSFORM_CAPABILITY)
            .map(|provider| (provider.plugin_name, provider.available))
            .collect();
        plugins.extend(
            self.inner
                .capability_registration
                .plugins_for(REQUEST_TRANSFORM_CAPABILITY)?
                .into_iter()
                .map(|plugin_name| (plugin_name, false)),
        );
        // Sorting puts `true` after `false`; keep each plugin's best entry.
        plugins.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        plugins.dedup_by(|a, b| a.0 == b.0);
        Ok(plugins)
    }

    /// Whether any plugin declares a request transform. An error counts as
    /// yes, so the request reaches `transform_request` and is rejected there.
    pub(crate) async fn has_request_transforms(&self) -> bool {
        self.request_transform_plugins()
            .await
            .map_or(true, |plugins| !plugins.is_empty())
    }

    /// Run every request transform over `body`. Returns whether any changed it.
    pub(crate) async fn transform_request(
        &self,
        path: &str,
        client_path: &str,
        body: &mut Value,
        private: bool,
    ) -> Result<bool> {
        let plugins = self.request_transform_plugins().await?;
        let mut changed = false;
        for (plugin_name, available) in plugins {
            if !available {
                bail!("request transform plugin '{plugin_name}' is unavailable");
            }
            let input = RequestTransformInput {
                path: path.to_string(),
                client_path: client_path.to_string(),
                body: body.clone(),
            };
            let output = self
                .invoke_request_transform(&plugin_name, &input, private)
                .await?;
            let Some(replacement) = output.body else {
                continue;
            };
            check_replacement(&plugin_name, body, &replacement)?;
            *body = replacement;
            changed = true;
        }
        Ok(changed)
    }

    async fn invoke_request_transform(
        &self,
        plugin_name: &str,
        input: &RequestTransformInput,
        private: bool,
    ) -> Result<RequestTransformOutput> {
        let input_json = serde_json::to_string(input)?;
        // The input goes to the plugin in one control frame, which its reader
        // caps at 16 MiB and treats as a broken connection beyond that. An
        // inlined request object can push a small request past it.
        if input_json.len() > MAX_INPUT_BYTES {
            bail!(
                "request too large for request transform plugin '{plugin_name}' \
                 ({} bytes, limit {MAX_INPUT_BYTES})",
                input_json.len()
            );
        }
        // Transforms run before the request is scoped: nested calls keep to
        // this node's trust circle, the one the request will get. The plugin
        // is done once it replies, whatever the reply says; only a call that
        // ends without one leaves the plugin private for a while.
        let private_call = || private.then(|| self.begin_private_invocation(plugin_name, None));
        let replied = |call: Option<super::private_invocations::PrivateInvocationGuard>| {
            if let Some(call) = call {
                call.finish();
            }
        };
        if self.is_test_bridge_enabled(plugin_name) {
            let call = private_call();
            let output = self
                .mcp_request(plugin_name, "request_transform/invoke", input)
                .await;
            replied(call);
            return output;
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        let call = private_call();
        let response = plugin
            .invoke_service(
                proto::ServiceKind::RequestTransform,
                SERVICE_NAME,
                &input_json,
                Some(TRANSFORM_TIMEOUT),
            )
            .await?;
        replied(call);
        if response.is_error {
            bail!(
                "request transform plugin '{plugin_name}' failed: {}",
                response.output_json
            );
        }
        serde_json::from_str(&response.output_json).with_context(|| {
            format!("request transform plugin '{plugin_name}' returned an invalid response")
        })
    }
}

/// Fields routing and response adapters already decided from the original
/// request: the model, how the response is framed, and the output budget the
/// context check reserved.
const FIXED_FIELDS: &[&str] = &[
    "model",
    "stream",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "n_predict",
];

/// A transform rewrites content; it may not change [`FIXED_FIELDS`].
fn check_replacement(plugin_name: &str, before: &Value, after: &Value) -> Result<()> {
    if !after.is_object() {
        bail!("request transform plugin '{plugin_name}' returned a non-object body");
    }
    for key in FIXED_FIELDS {
        if before.get(key) != after.get(key) {
            bail!("request transform plugin '{plugin_name}' may not change `{key}`");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{BridgeFuture, PluginRpcBridge, RpcResult};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Appends its plugin name to the first message's content.
    struct Appender;

    impl PluginRpcBridge for Appender {
        fn handle_request(
            &self,
            plugin_name: String,
            method: String,
            params_json: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            Box::pin(async move {
                assert_eq!(method, "request_transform/invoke");
                let input: RequestTransformInput = serde_json::from_str(&params_json).unwrap();
                let output = match plugin_name.as_str() {
                    "a-append" | "b-append" => {
                        let mut body = input.body;
                        let text = body["messages"][0]["content"].as_str().unwrap().to_string();
                        body["messages"][0]["content"] = json!(format!("{text}+{plugin_name}"));
                        RequestTransformOutput::replaced(body)
                    }
                    "c-noop" => RequestTransformOutput::unchanged(),
                    "d-model" => {
                        let mut body = input.body;
                        body["model"] = json!("other");
                        RequestTransformOutput::replaced(body)
                    }
                    "f-budget" => {
                        let mut body = input.body;
                        body["max_tokens"] = json!(100_000);
                        RequestTransformOutput::replaced(body)
                    }
                    _ => {
                        return Err(proto::ErrorResponse {
                            code: -32602,
                            message: "cannot read the attachment".into(),
                            data_json: String::new(),
                        });
                    }
                };
                Ok(RpcResult {
                    result_json: serde_json::to_string(&output).unwrap(),
                })
            })
        }

        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    async fn manager(plugins: &[&str]) -> PluginManager {
        let manager = PluginManager::for_test_bridge(plugins, Arc::new(Appender));
        let manifest = mesh_llm_plugin::plugin_manifest![
            mesh_llm_plugin::request_transform::request_transform()
        ];
        manager
            .set_test_manifests(
                plugins
                    .iter()
                    .map(|name| (name.to_string(), manifest.clone()))
                    .collect::<BTreeMap<_, _>>(),
            )
            .await;
        manager
    }

    fn request() -> Value {
        json!({"model": "m", "stream": true, "messages": [{"role": "user", "content": "hi"}]})
    }

    #[tokio::test]
    async fn transforms_run_in_name_order_and_chain() {
        let manager = manager(&["b-append", "c-noop", "a-append"]).await;
        let mut body = request();
        let changed = manager
            .transform_request("/v1/chat/completions", "/v1/messages", &mut body, false)
            .await
            .unwrap();
        assert!(changed);
        assert_eq!(body["messages"][0]["content"], "hi+a-append+b-append");
    }

    #[tokio::test]
    async fn no_transform_plugins_leaves_the_request_alone() {
        let manager = manager(&[]).await;
        let mut body = request();
        assert!(
            !manager
                .transform_request(
                    "/v1/chat/completions",
                    "/v1/chat/completions",
                    &mut body,
                    false
                )
                .await
                .unwrap()
        );
        assert_eq!(body, request());
    }

    #[tokio::test]
    async fn the_manager_reports_whether_transforms_exist() {
        assert!(!manager(&[]).await.has_request_transforms().await);
        assert!(manager(&["c-noop"]).await.has_request_transforms().await);
    }

    #[tokio::test]
    async fn a_transform_plugin_that_lost_its_providers_still_rejects() {
        // A crashed plugin's provider entries are cleared until it restarts.
        let manager = manager(&["c-noop"]).await;
        manager
            .plugin_summary_producer("c-noop")
            .clear_plugin_reports("c-noop");
        manager.publish_plugin_providers("c-noop", Vec::new());
        assert!(manager.has_request_transforms().await);
        let error = manager
            .transform_request(
                "/v1/chat/completions",
                "/v1/chat/completions",
                &mut request(),
                false,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unavailable"), "{error}");
    }

    #[tokio::test]
    async fn an_unchanged_answer_keeps_the_body() {
        let manager = manager(&["c-noop"]).await;
        let mut body = request();
        assert!(
            !manager
                .transform_request(
                    "/v1/chat/completions",
                    "/v1/chat/completions",
                    &mut body,
                    false
                )
                .await
                .unwrap()
        );
        assert_eq!(body, request());
    }

    #[tokio::test]
    async fn a_failing_transform_rejects_the_request() {
        let manager = manager(&["e-fails"]).await;
        let error = manager
            .transform_request(
                "/v1/chat/completions",
                "/v1/chat/completions",
                &mut request(),
                false,
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("cannot read the attachment"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_transform_may_not_change_the_model() {
        let manager = manager(&["d-model"]).await;
        let error = manager
            .transform_request(
                "/v1/chat/completions",
                "/v1/chat/completions",
                &mut request(),
                false,
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("may not change `model`"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_transform_may_not_change_the_output_budget() {
        let error = manager(&["f-budget"])
            .await
            .transform_request(
                "/v1/chat/completions",
                "/v1/chat/completions",
                &mut request(),
                false,
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("may not change `max_tokens`"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn an_input_too_large_for_one_plugin_frame_is_rejected() {
        let mut body = request();
        body["messages"][0]["content"] = json!("x".repeat(MAX_INPUT_BYTES));
        let error = manager(&["c-noop"])
            .await
            .transform_request(
                "/v1/chat/completions",
                "/v1/chat/completions",
                &mut body,
                false,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("request too large"), "{error}");
    }
}
