//! Setup diagnostics demonstrating permissioned identity calls over the normal IPC connection.

use mesh_llm_plugin::{
    OperationRouter, PluginError, operation_with_schema, structured_tool_result,
};
use serde_json::{Value, json};
use std::sync::Arc;

pub fn router() -> OperationRouter {
    // This key belongs to the plugin process. Only its public key is sent to
    // the host; the owner's signing key is never available to this process.
    let key = Arc::new(mesh_llm_identity::OwnerKeypair::generate());
    let mut router = OperationRouter::new();
    router.add_raw(operation_with_schema("identity_probe", "Read public identity or renew delegated evidence signing authority outside inference",
        json!({"type":"object","properties":{"method":{"type":"string","enum":["ReadIdentityBundle","DelegatePluginSigningKey"]},"params":{"type":"object"}},"required":["method"]}).as_object().unwrap().clone()),
        move |request, context| {
            let key = key.clone();
            Box::pin(async move {
                let arguments: Value = request.arguments()?;
                let method = arguments["method"].as_str().ok_or_else(|| PluginError::invalid_request("identity method required"))?;
                if !matches!(method, "ReadIdentityBundle" | "DelegatePluginSigningKey") {
                    return Err(PluginError::invalid_request("unsupported identity method"));
                }
                let mut params = arguments.get("params").cloned().unwrap_or_else(|| json!({}));
                let public_key = hex::encode(key.verifying_key().as_bytes());
                if method == "DelegatePluginSigningKey" {
                    let fields = params.as_object_mut().ok_or_else(|| PluginError::invalid_request("identity params must be an object"))?;
                    if fields.contains_key("signing_public_key") {
                        return Err(PluginError::invalid_request("the exemplar owns its signing key; external keys are not accepted"));
                    }
                    fields.insert("signing_public_key".into(), json!(public_key));
                    fields.entry("scope").or_insert_with(|| json!("mesh.openai.exchange.evidence.sign.v1"));
                    fields.entry("lifetime_ms").or_insert_with(|| json!(60_000));
                }
                let result = match context.request_host::<_, Value>(method, params).await {
                    Ok(response) => json!({"response":response,"plugin_signing_public_key":public_key}),
                    Err(error) => json!({"error":error.to_string()}),
                };
                structured_tool_result(result)
            })
        });
    router
}
