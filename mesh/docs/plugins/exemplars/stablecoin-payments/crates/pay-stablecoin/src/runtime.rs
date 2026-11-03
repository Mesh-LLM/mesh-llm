use crate::{
    binding::Wallet,
    config::Config,
    engine::Engine,
    protocol::{CHANNEL, Message},
};
use anyhow::{Context, Result, ensure};
use mesh_llm_plugin::{
    OperationRouter, PluginMetadata, PluginRuntime, SimplePlugin, operation_with_schema,
    plugin_manifest, plugin_server_info, structured_tool_result,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};
use wallet_contract::ops;

pub const NAME: &str = "pay-stablecoin";
type Slot = Arc<OnceLock<Arc<Engine>>>;

pub fn manifest() -> mesh_llm_plugin::proto::PluginManifest {
    use mesh_llm_plugin::{
        config_path, config_schema, config_setting, mesh_channel, operation, operation_binding,
    };
    let mut hook = mesh_llm_plugin::openai_exchange::openai_exchange_hook("payment_gate");
    hook.metadata = false;
    hook.required = true;
    hook.admission = true;
    hook.deadline_ms = 30_000;
    hook.request_body = true;
    hook.effective_request_body = true;
    hook.read_identity_bundle = true;
    hook.max_body_bytes = 64 * 1024;
    plugin_manifest()
        .item(config_schema(NAME).setting(config_setting("config_path", config_path())))
        .item(hook)
        .item(mesh_channel(CHANNEL))
        .operation_binding(operation_binding(
            "wallet",
            wallet_contract::CAPABILITY,
            ops::OPEN,
            [
                ops::BALANCE,
                ops::ESTIMATE,
                ops::PAY,
                ops::LOOKUP,
                ops::VERIFY,
                ops::ATTEMPTS,
            ],
            false,
        ))
        .item(operation::<Value>(
            "control",
            "Inspect payment state, balance, or retransmit durable proofs and bills",
        ))
        .build()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    config_path: std::path::PathBuf,
}

fn get(slot: &Slot) -> Result<Arc<Engine>> {
    slot.get()
        .cloned()
        .context("stablecoin payment engine is not ready")
}

pub async fn run() -> Result<()> {
    let slot: Slot = Default::default();
    let prepared = Arc::new(std::sync::Mutex::new(None::<Config>));
    let metadata = PluginMetadata::new(
        NAME,
        env!("CARGO_PKG_VERSION"),
        plugin_server_info(
            NAME,
            env!("CARGO_PKG_VERSION"),
            "Stablecoin payments",
            "Direct stablecoin admission and post-decode settlement",
            None::<String>,
        ),
    )
    .with_manifest(manifest());
    let initialize = prepared.clone();
    let startup = slot.clone();
    let channels = slot.clone();
    let hooks = slot.clone();
    let capacity = Arc::new(tokio::sync::Semaphore::new(32));
    let bodies = crate::request_stream::RequestStreams::default();
    let plugin = SimplePlugin::new(metadata)
        .with_operation_router(controls(slot))
        .on_open_stream(move |request, _| {
            let bodies = bodies.clone();
            Box::pin(async move { bodies.open(request).await })
        })
        .on_initialize(move |request, _| {
            let prepared = initialize.clone();
            Box::pin(async move {
                let settings: Settings =
                    serde_json::from_str(&request.settings_json).map_err(anyhow::Error::from)?;
                let config = Config::load(&settings.config_path)?;
                *prepared.lock().unwrap() = Some(config);
                Ok(())
            })
        })
        .on_initialized(move |context| {
            let slot = startup.clone();
            let prepared = prepared.clone();
            let context = context.owned();
            Box::pin(async move {
                let config = prepared
                    .lock()
                    .unwrap()
                    .take()
                    .context("payment settings missing")?;
                ensure!(
                    context.host_supports(mesh_llm_plugin::operation_binding::CAPABILITY),
                    "host requires operation_bindings.v1"
                );
                let identity: Value = context
                    .request_host("ReadIdentityBundle", json!({}))
                    .await?;
                let node = identity["node_endpoint_id"]
                    .as_str()
                    .context("host identity omitted node endpoint")?
                    .into();
                let wallet = Wallet::open(config.wallet_binding.clone(), context.clone()).await?;
                let engine = Engine::new(config, wallet, node, context.clone())?;
                for model in engine.config.prices.keys() {
                    context
                        .clone()
                        .require_channels_for_model(model, &[(CHANNEL, &[])])
                        .await?;
                }
                ensure!(
                    slot.set(engine.clone()).is_ok(),
                    "payment plugin initialized twice"
                );
                let weak = Arc::downgrade(&engine);
                tokio::spawn(async move {
                    let mut interval = tokio::time::interval(std::time::Duration::from_secs(3));
                    loop {
                        interval.tick().await;
                        let Some(engine) = weak.upgrade() else { break };
                        if let Err(error) = engine.retry().await {
                            eprintln!("pay-stablecoin: reconciliation failed: {error:#}");
                        }
                    }
                });
                Ok(())
            })
        })
        .on_channel_message(move |message, _context| {
            let slot = channels.clone();
            let capacity = capacity.clone();
            Box::pin(async move {
                if message.channel != CHANNEL || message.body.len() > 32 * 1024 {
                    return Ok(());
                }
                let Ok(body) = serde_json::from_slice::<Message>(&message.body) else {
                    tracing::warn!("dropping malformed stablecoin channel message");
                    return Ok(());
                };
                let respond_to_error = !matches!(body, Message::Error { .. });
                let id = message_id(&body).to_owned();
                let Ok(engine) = get(&slot) else {
                    return Ok(());
                };
                let Ok(permit) = capacity.try_acquire_owned() else {
                    tracing::warn!("stablecoin channel capacity exhausted");
                    return Ok(());
                };
                tokio::spawn(async move {
                    let _permit = permit;
                    if let Err(error) = engine.message(&message.source_peer_id, body).await {
                        eprintln!("pay-stablecoin: peer operation failed: {error:#}");
                        let reason = error.to_string().chars().take(512).collect();
                        if respond_to_error {
                            let _ = engine
                                .send(&message.source_peer_id, &Message::Error { id, reason })
                                .await;
                        }
                    }
                });
                Ok(())
            })
        })
        .with_openai_exchange_handler(move |_, event, _| {
            let slot = hooks.clone();
            Box::pin(async move {
                let engine = get(&slot)?;
                if event["phase"] == "exchange_finished" {
                    tokio::spawn(async move {
                        if let Err(error) = engine.hook(event).await {
                            eprintln!("pay-stablecoin: terminal billing failed: {error:#}");
                        }
                    });
                    return Ok(Default::default());
                }
                let decision = engine.hook(event).await.inspect_err(|error| {
                    eprintln!("pay-stablecoin: payment hook failed: {error:#}");
                })?;
                Ok(decision)
            })
        });
    PluginRuntime::run(plugin).await
}

fn message_id(message: &Message) -> &str {
    match message {
        Message::Quote { quote } => &quote.request_id,
        Message::Hello { id, .. }
        | Message::AdmissionProof { id, .. }
        | Message::Ready { id }
        | Message::Bill { id, .. }
        | Message::UsageProof { id, .. }
        | Message::Settled { id }
        | Message::Cancel { id }
        | Message::Error { id, .. } => id,
    }
}

fn controls(slot: Slot) -> OperationRouter {
    let mut router = OperationRouter::new();
    router.add_raw(operation_with_schema("control","Inspect state or reconcile durable payment messages",
        json!({"type":"object","properties":{"command":{"type":"string","enum":["status","balance","retry","resubmit"]},"request_id":{"type":"string"}},"required":["command"]}).as_object().unwrap().clone()),
        move |request,_| {
            let slot=slot.clone();
            Box::pin(async move {
                let engine=get(&slot)?;
                let input:Value=request.arguments()?;
                let value=match input["command"].as_str() {
                    Some("status")=>json!({"node":engine.node,"wallet":engine.wallet.info,"buying_enabled":engine.config.buying.enabled,"active":engine.ledger.active()?}),
                    Some("balance")=>serde_json::to_value(engine.wallet.balance().await?).map_err(anyhow::Error::from)?,
                    Some("retry")=>{engine.retry().await?;json!({"retransmitted":true})},
                    Some("resubmit")=>{
                        let id=input["request_id"].as_str().context("resubmit requires the existing request_id")?;
                        engine.resubmit(id).await?;json!({"resubmitted_same_intent":true,"request_id":id})
                    },
                    _=>return Err(mesh_llm_plugin::PluginError::invalid_request("unknown control command")),
                };
                structured_tool_result(value)
            })
        });
    router
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_sees_a_generic_binding_and_usage_hooks_only() {
        let m = manifest();
        assert_eq!(m.mesh_channels[0].name, CHANNEL);
        let hook = m.openai_exchange_hook.unwrap();
        assert!(hook.required && hook.admission && hook.read_identity_bundle);
        assert!(!hook.delegate_signing_key && !hook.response_body);
        let grants = m.grant_requests.unwrap();
        assert_eq!(
            grants.operation_bindings[0].capability,
            wallet_contract::CAPABILITY
        );
        assert!(!grants.operation_bindings[0].setup_directory);
    }

    #[test]
    fn documented_grant_fully_authorizes_the_payment_hook() {
        let hook = manifest().openai_exchange_hook.unwrap();
        let grant = serde_json::from_value(json!({
            "endpoints":["chat_completions","completions","responses"],
            "phases":["request_received","backend_selected","exchange_finished"],
            "request_body":true,"effective_request_body":true,
            "admission":true,"read_identity_bundle":true,
            "deadline_ms":30000,"max_body_bytes":65536,
            "max_queue_bytes":2097152,"max_in_flight":32,
            "failure_policy":"required"
        }))
        .unwrap();
        assert!(
            mesh_llm_plugin::openai_exchange::negotiate_openai_exchange(Some(&hook), Some(&grant))
                .unwrap()
                .is_some()
        );
    }
}
