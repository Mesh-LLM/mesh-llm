//! Standalone plugin. Money-moving behavior requires explicit isolated-state config.
use crate::{CHANNEL, Config, Engine, Message};
use anyhow::{Context, Result, ensure};
use mesh_llm_payments::service::PaymentService;
use mesh_llm_plugin::{
    PluginContext, PluginMetadata, PluginRuntime, SimplePlugin, mesh_channel, mesh_event_peer_down,
    mesh_event_peer_up, plugin_manifest, plugin_server_info,
};

use std::sync::Arc;

pub(crate) fn manifest(header: &str) -> mesh_llm_plugin::proto::PluginManifest {
    let mut hook = mesh_llm_plugin::openai_exchange::openai_exchange_hook("payment_gate");
    hook.phases = vec![
        "request_received".into(),
        "backend_selected".into(),
        "exchange_finished".into(),
    ];
    hook.required = true;
    hook.admission = true;
    hook.headers = vec![header.into()];
    hook.deadline_ms = 30_000;
    use mesh_llm_plugin::{
        config_boolean, config_path, config_schema, config_setting, config_string,
    };
    plugin_manifest()
        .item(
            config_schema("pay-lightning")
                .setting(config_setting("config_path", config_path()).required(true))
                .setting(config_setting("wallet_binding", config_string()).required(true))
                .setting(
                    config_setting("allow_wallet_provisioning", config_boolean())
                        .default_value(&false),
                ),
        )
        .item(hook)
        .item(mesh_channel(CHANNEL))
        .item(mesh_event_peer_up())
        .item(mesh_event_peer_down())
        .build()
}

/// Run the payment protocol with an in-process wallet adapter. Manifest inspection
/// is handled before reading configuration or opening any state.
pub async fn run(
    name: String,
    version: &str,
    factory: impl mesh_llm_wallet::provisioning::WalletFactory + 'static,
    config: Config,
) -> Result<()> {
    run_with_context(name, version, Arc::new(factory), config, None).await
}

pub async fn run_bound(
    name: String,
    version: &str,
    binding: String,
    allow_provisioning: bool,
    config: Config,
) -> Result<()> {
    let context = Arc::new(std::sync::OnceLock::new());
    let factory = crate::wallet_binding::InitializedWalletFactory {
        context: context.clone(),
        binding,
        allow_provisioning,
    };
    run_with_context(name, version, Arc::new(factory), config, Some(context)).await
}

async fn run_with_context(
    name: String,
    version: &str,
    factory: Arc<dyn mesh_llm_wallet::provisioning::WalletFactory>,
    config: Config,
    wallet_context: Option<Arc<std::sync::OnceLock<PluginContext<'static>>>>,
) -> Result<()> {
    config.validate()?;
    crate::diagnostics::configure(config.timing_log.clone(), config.error_log.clone())?;
    let namespace: String = name.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
    let header = format!("x-plugin-{namespace}-pay-terms");
    let manifest = manifest(&header);
    let directory = &config.data_dir;
    let service = Arc::new(PaymentService::with_factory(directory, factory)?);
    service.ledger.set_policy(&config.policy)?;
    for (model, price) in &config.prices {
        service.ledger.set_pricing(model, Some(price))?;
    }
    let mut engine = Engine::new(service, &namespace, config.cap_msat, directory)?;
    engine.buying = config.buy_prices;
    engine.buy_peers = config.buy_peers;
    let engine = Arc::new(engine);

    let metadata = payment_metadata(&name, version, manifest);
    let messages = engine.clone();
    let events = engine.clone();
    let startup = engine.clone();
    let plugin = SimplePlugin::new(metadata)
        // After the handshake: opening a wallet must not consume the host's
        // initialize-response timeout. Manifest inspection never reaches here.
        .on_initialized(move |context| {
            let engine = startup.clone();
            let mut context = context.owned();
            let wallet_context = wallet_context.clone();
            Box::pin(async move {
                for model in engine.service.ledger.pricing()?.keys() {
                    context
                        .require_channels_for_model(model, &[(CHANNEL, &[])])
                        .await?;
                }
                if let Some(wallet_context) = wallet_context {
                    ensure!(
                        context.host_supports(mesh_llm_plugin::operation_binding::CAPABILITY),
                        "host does not support operation bindings"
                    );
                    let _ = wallet_context.set(context.owned());
                }
                engine.start().await?;
                observe_receipts(Arc::downgrade(&engine));
                Ok(())
            })
        })
        .on_channel_message(move |message, context| {
            let engine = messages.clone();
            let mut context = context.owned();
            Box::pin(async move {
                let detached = serde_json::from_slice::<Message>(&message.body).is_ok_and(|body| {
                    matches!(
                        body,
                        Message::Settlement { .. }
                            | Message::Recover { .. }
                            | Message::Paid { .. }
                            | Message::Admission { .. }
                            | Message::Quote
                    )
                });
                let work = async move {
                    let result: Result<()> = async {
                        ensure!(message.channel == CHANNEL, "unexpected channel");
                        let message_body: Message = serde_json::from_slice(&message.body)?;
                        let _timing = crate::diagnostics::Timing::start(
                            "channel_receive_handler",
                            message_body.kind(),
                        );
                        let ready = match &message_body {
                            Message::Ready { id } => Some(id.clone()),
                            _ => None,
                        };
                        for reply in engine
                            .channel(&message.source_peer_id, message_body)
                            .await?
                        {
                            send(&mut context, &message.source_peer_id, &reply).await?;
                        }
                        if let Some(id) = ready
                            && engine.vouch_once(&message.source_peer_id, &id)?
                        {
                            context
                                .request_peer_block(mesh_llm_plugin::proto::PeerBlockRequest {
                                    change:
                                        mesh_llm_plugin::proto::peer_block_request::Change::Unblock
                                            as i32,
                                    peer_id: message.source_peer_id.clone(),
                                    ..Default::default()
                                })
                                .await?;
                        }
                        Ok(())
                    }
                    .await;
                    if let Err(error) = &result {
                        record_error(error);
                    }
                    result
                };
                dispatch_channel_work(detached, work).await
            })
        })
        .on_mesh_event(move |event, context| {
            let engine = events.clone();
            let mut context = context.owned();
            Box::pin(async move {
                if event.kind == mesh_llm_plugin::proto::mesh_event::Kind::PeerDown as i32
                    && let Some(peer) = &event.peer
                {
                    engine.peer_down(&peer.peer_id);
                }
                if event.kind == mesh_llm_plugin::proto::mesh_event::Kind::PeerUp as i32
                    && engine.cap_msat > 0
                    && let Some(peer) = event.peer
                    && engine.buy_peers.contains(&peer.peer_id)
                {
                    tokio::spawn(async move {
                        let result: Result<()> = async {
                            for message in engine.peer_up(&peer.peer_id).await? {
                                if let Err(error) =
                                    send(&mut context, &peer.peer_id, &message).await
                                {
                                    engine.peer_down(&peer.peer_id);
                                    return Err(error);
                                }
                            }
                            Ok(())
                        }
                        .await;
                        if let Err(error) = result {
                            record_error(&error);
                        }
                    });
                }

                Ok(())
            })
        })
        .with_openai_exchange_handler(move |_name, event, context| {
            let engine = engine.clone();
            let mut context = context.owned();
            Box::pin(async move {
                // Terminal wallet work is detached; it must not consume the hook's deadline.
                if event["phase"] == "exchange_finished" {
                    tokio::spawn(async move {
                        let result: Result<()> = async {
                            if let Some((peer, message)) = engine.terminal(&event).await? {
                                send(&mut context, &peer, &message).await?;
                            }
                            Ok(())
                        }
                        .await;
                        if let Err(error) = result {
                            record_error(&error);
                        }
                    });
                    return Ok(Default::default());
                }
                if event["phase"] == "backend_selected"
                    && let Some(peer) = event["target_peer_id"].as_str()
                    && engine.cap_msat > 0
                    && engine.buy_peers.contains(peer)
                {
                    let result: Result<_> = async {
                        let model = event["model"].as_str().context("selected model missing")?;
                        tokio::time::timeout(std::time::Duration::from_secs(28), async {
                            for message in engine.peer_up(peer).await? {
                                if let Err(error) = send(&mut context, peer, &message).await {
                                    engine.peer_down(peer);
                                    return Err(error);
                                }
                            }
                            engine
                                .buyer_wait(peer, model, std::time::Duration::from_secs(28))
                                .await
                        })
                        .await
                        .context("seller admission deadline exceeded")?
                    }
                    .await;
                    return result.map_err(|error| {
                        mesh_llm_plugin::PluginError::internal(error.to_string())
                    });
                }
                let decision = engine
                    .gate_wait(&event, std::time::Duration::from_secs(28))
                    .await
                    .map_err(|error| mesh_llm_plugin::PluginError::internal(error.to_string()))?;
                if decision.decision
                    == mesh_llm_plugin::openai_exchange::OpenAiAdmissionDecision::Allow
                {
                    // Refill in parallel with inference, never on its hook deadline.
                    tokio::spawn(async move {
                        if let Err(error) = replenish(&engine, &event, &mut context).await {
                            record_error(&error);
                        }
                    });
                }
                Ok(decision)
            })
        });
    PluginRuntime::run(plugin).await
}

pub(crate) fn observe_receipts(engine: std::sync::Weak<Engine>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(250));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let Some(engine) = engine.upgrade() else {
                break;
            };
            // One sweep at a time, outside admission and buyer payment locks.
            // A wedged wallet lookup must not retain the engine indefinitely.
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                engine.refresh_receipts(),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(error)) => record_error(&error),
                Err(error) => record_error(&anyhow::Error::new(error)),
            }
        }
    });
}

async fn replenish(
    engine: &Engine,
    event: &serde_json::Value,
    context: &mut PluginContext<'_>,
) -> Result<()> {
    if let Some((peer, replies)) = engine.replenish(event).await? {
        for reply in replies {
            send(context, &peer, &reply).await?;
        }
    }
    Ok(())
}

async fn dispatch_channel_work(
    detached: bool,
    work: impl std::future::Future<Output = Result<()>> + Send + 'static,
) -> Result<()> {
    if detached {
        // Charges are durable; submissions retain the service payment lock.
        tokio::spawn(work);
        Ok(())
    } else {
        work.await
    }
}

async fn send(context: &mut PluginContext<'_>, peer: &str, message: &Message) -> Result<()> {
    let _timing = crate::diagnostics::Timing::start("channel_send", message.kind());
    context
        .send_json_channel(CHANNEL, peer, "payment", message)
        .await
}
fn record_error(error: &anyhow::Error) {
    crate::diagnostics::record_error(error);
}

fn payment_metadata(
    name: &str,
    version: &str,
    manifest: mesh_llm_plugin::proto::PluginManifest,
) -> PluginMetadata {
    PluginMetadata::new(
        name,
        version,
        plugin_server_info(
            name,
            version,
            "Lightning payments",
            "Plugin-owned credit, ledger and settlement",
            None::<String>,
        ),
    )
    .with_manifest(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn settlement_does_not_hold_ordered_handler() {
        let (release, blocked) = tokio::sync::oneshot::channel();
        let (finished, completion) = tokio::sync::oneshot::channel();
        dispatch_channel_work(true, async move {
            blocked.await?;
            let _ = finished.send(());
            Ok(())
        })
        .await
        .unwrap();
        dispatch_channel_work(false, async { Ok(()) })
            .await
            .unwrap();
        release.send(()).unwrap();
        completion.await.unwrap();
        assert!(
            dispatch_channel_work(false, async { anyhow::bail!("failure") })
                .await
                .is_err()
        );
    }

    #[test]
    fn manifest_requests_only_required_gate_metadata_and_channel() {
        let manifest = manifest("x-plugin-7061792d6c696768746e696e67-pay-terms");
        assert!(manifest.operations.is_empty());
        let hook = manifest.openai_exchange_hook.unwrap();
        assert!(hook.required && hook.admission && hook.metadata);
        assert!(!hook.request_body && !hook.response_body && !hook.delegate_signing_key);
        assert_eq!(
            hook.phases,
            vec!["request_received", "backend_selected", "exchange_finished"]
        );
        assert_eq!(manifest.mesh_channels.len(), 1);
        assert_eq!(manifest.mesh_channels[0].name, CHANNEL);
    }
}
