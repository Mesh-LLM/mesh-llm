//! Installable lifecycle exemplar with a setup identity diagnostic operation.
mod openai_exchange_compat;
mod openai_exchange_identity;
mod openai_exchange_stream;
mod openai_exchange_virtual;
use mesh_llm_plugin::openai_exchange::{
    OpenAiAdmissionDecision, OpenAiExchangeDecision, openai_exchange_hook,
};
use mesh_llm_plugin::{
    PluginMetadata, PluginRuntime, SimplePlugin, plugin_manifest, plugin_server_info,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let plugin_id = args
        .windows(2)
        .find(|pair| pair[0] == "--plugin-id")
        .map(|pair| pair[1].as_str())
        .unwrap_or("openai-exchange-observer");
    let admission = std::env::args().any(|arg| arg == "--admission");
    let mut hook = openai_exchange_hook("observe");
    hook.required = !args.iter().any(|arg| arg == "--optional");
    hook.admission = admission;
    let bodies = std::env::args().any(|arg| arg == "--body");
    hook.read_identity_bundle = args.iter().any(|arg| arg == "--identity");
    hook.delegate_signing_key = args.iter().any(|arg| arg == "--delegate");
    if hook.delegate_signing_key {
        hook.signing_scopes = vec!["mesh.openai.exchange.evidence.sign.v1".into()];
        hook.max_delegation_ttl_secs = 60;
    }
    hook.request_body = bodies;
    hook.effective_request_body = bodies;
    hook.response_body = bodies;
    let virtual_echo = args.iter().any(|arg| arg == "--virtual-echo");
    let legacy_conformance = args.iter().any(|arg| arg == "--legacy-conformance");
    let mut builder = plugin_manifest().item(mesh_llm_plugin::operation::<serde_json::Value>(
        "identity_probe",
        "Permissioned identity setup diagnostics",
    ));
    if legacy_conformance {
        builder = builder.item(mesh_llm_plugin::operation::<serde_json::Value>(
            "legacy_echo",
            "Generation-3 compatibility fixture",
        ));
    } else {
        builder = builder.item(hook);
    }
    if virtual_echo {
        builder = builder.item(mesh_llm_plugin::virtual_model(
            "exchange-echo",
            "virtual_echo",
        ));
    }
    let manifest = builder.build();
    if std::env::args().any(|arg| arg == "--manifest") {
        println!("{}", mesh_llm_plugin::package_manifest_json(&manifest)?);
        return Ok(());
    }
    let metadata = PluginMetadata::new(
        plugin_id,
        "1.0.0",
        plugin_server_info(
            plugin_id,
            "1.0.0",
            "OpenAI exchange observer",
            "Read-only lifecycle observer and admission exemplar",
            None::<String>,
        ),
    )
    .with_manifest(manifest);
    let streams = openai_exchange_stream::StreamEvidence::default();
    let receiver = streams.clone();
    let plugin = SimplePlugin::new(metadata)
        .with_virtual_model_router(openai_exchange_virtual::router())
        .with_operation_router(openai_exchange_compat::router(legacy_conformance))
        .on_open_stream(move |request, _context| {
            let receiver = receiver.clone();
            Box::pin(async move {
                if let Ok(path) = std::env::var("MESH_LLM_EXEMPLAR_STREAM_LOG") {
                    use tokio::io::AsyncWriteExt;
                    let mut options = tokio::fs::OpenOptions::new();
                    options.create(true).append(true);
                    #[cfg(unix)]
                    options.mode(0o600);
                    let mut file = options.open(path).await.map_err(|error| {
                        mesh_llm_plugin::PluginError::invalid_request(error.to_string())
                    })?;
                    file.write_all(
                        format!("{}\n", request.metadata_json.as_deref().unwrap_or("{}"))
                            .as_bytes(),
                    )
                    .await
                    .map_err(|error| {
                        mesh_llm_plugin::PluginError::invalid_request(error.to_string())
                    })?;
                }
                if std::env::var("MESH_LLM_EXEMPLAR_FAULT").as_deref()
                    == Ok("disconnect_on_response_stream")
                    && request
                        .metadata_json
                        .as_deref()
                        .is_some_and(|json| json.contains("openai_exchange_response"))
                {
                    std::process::exit(23);
                }
                receiver.open(request).await
            })
        })
        .with_openai_exchange_handler(move |_name, event, _context| {
            let streams = streams.clone();
            Box::pin(async move {
                if event["phase"] == "exchange_finished"
                    && event.get("response_wire_commitment").is_some()
                {
                    let verified = streams.verify_terminal(&event).await;
                    eprintln!("response byte commitment independently verified: {verified}");
                }
                // Admission is deterministic and needs no body permission. The operator
                // selects the rejected model; no user-controlled claims grant authority.
                let request_body = streams
                    .request_body(&event)
                    .await
                    .or_else(|| event.get("body").cloned());
                let deny_text = std::env::var("MESH_LLM_EXEMPLAR_DENY_TEXT").ok();
                let text_denied = deny_text.as_deref().is_some_and(|text| {
                    request_body
                        .as_ref()
                        .is_some_and(|body| body.to_string().contains(text))
                });
                let deny_model = std::env::var("MESH_LLM_EXEMPLAR_DENY_MODEL").ok();
                let selected_denied = event["phase"] == "backend_selected"
                    && std::env::var("MESH_LLM_EXEMPLAR_DENY_SELECTED_MODEL")
                        .ok()
                        .as_deref()
                        .is_some_and(|model| event["model"].as_str() == Some(model));
                let denied = admission
                    && event["parse_status"] != "invalid_json"
                    && (selected_denied
                        || (event["phase"] == "request_received"
                            && (text_denied
                                || deny_model
                                    .as_deref()
                                    .is_some_and(|model| event["model"].as_str() == Some(model)))));
                if let Ok(path) = std::env::var("MESH_LLM_EXEMPLAR_EVENT_LOG") {
                    use tokio::io::AsyncWriteExt;
                    let mut options = tokio::fs::OpenOptions::new();
                    options.create(true).append(true);
                    #[cfg(unix)]
                    options.mode(0o600);
                    let mut file = options.open(path).await.map_err(|e| {
                        mesh_llm_plugin::PluginError::invalid_request(e.to_string())
                    })?;
                    let mut line = serde_json::to_vec(&event).map_err(|e| {
                        mesh_llm_plugin::PluginError::invalid_request(e.to_string())
                    })?;
                    line.push(b'\n');
                    file.write_all(&line).await.map_err(|e| {
                        mesh_llm_plugin::PluginError::invalid_request(e.to_string())
                    })?;
                }
                Ok(OpenAiExchangeDecision {
                    decision: if denied {
                        OpenAiAdmissionDecision::Deny
                    } else if admission && event["parse_status"] != "invalid_json" {
                        OpenAiAdmissionDecision::Allow
                    } else {
                        OpenAiAdmissionDecision::Abstain
                    },
                    annotations: Vec::new(),
                    response_headers: Vec::new(),
                    reason: denied.then(|| "Model denied by exemplar operator policy".into()),
                })
            })
        });
    PluginRuntime::run(plugin).await
}
