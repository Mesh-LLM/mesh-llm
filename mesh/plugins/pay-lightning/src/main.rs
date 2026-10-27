//! Standalone plugin. Money-moving behavior requires explicit isolated-state config.
use anyhow::{Context, Result, ensure};
use mesh_llm_payments::{ledger::Policy, service::PaymentService};
use mesh_llm_plugin::{
    PluginContext, PluginMetadata, PluginRuntime, SimplePlugin, mesh_channel, mesh_event_peer_down,
    mesh_event_peer_up, plugin_manifest, plugin_server_info,
};
use mesh_pay_lightning::{CHANNEL, Engine, Message, parse_prices};

use std::{path::PathBuf, sync::Arc};

fn manifest(header: &str) -> mesh_llm_plugin::proto::PluginManifest {
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
    plugin_manifest()
        .item(hook)
        .item(mesh_channel(CHANNEL))
        .item(mesh_event_peer_up())
        .item(mesh_event_peer_down())
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let name = std::env::var("MESH_LLM_PLUGIN_NAME").unwrap_or_else(|_| "pay-lightning".into());
    let namespace: String = name.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
    let header = format!("x-plugin-{namespace}-pay-terms");
    let manifest = manifest(&header);
    if std::env::args().any(|arg| arg == "--manifest") {
        println!("{}", mesh_llm_plugin::package_manifest_json(&manifest)?);
        return Ok(());
    }
    let directory = PathBuf::from(
        std::env::var_os("PAY_LIGHTNING_DATA")
            .context("PAY_LIGHTNING_DATA must name isolated plugin state")?,
    );
    ensure!(
        directory.is_absolute(),
        "PAY_LIGHTNING_DATA must be absolute"
    );
    // State is plugin-owned. No implicit adoption of ~/.mesh-llm/payments.
    std::fs::create_dir_all(&directory)?;
    let wallet_executable = PathBuf::from(
        std::env::var_os("PAY_LIGHTNING_WALLET")
            .context("PAY_LIGHTNING_WALLET must name lexe-wallet executable")?,
    );
    let service = Arc::new(PaymentService::with_factory(
        &directory,
        Arc::new(mesh_pay_lightning::wallet_child::ChildWalletFactory {
            executable: wallet_executable,
            directory: directory.join("wallet"),
        }),
    )?);
    let policy: Policy = serde_json::from_str(
        &std::env::var("PAY_LIGHTNING_POLICY")
            .unwrap_or_else(|_| "{\"mode\":\"free_only\",\"daily_budget_msat\":null}".into()),
    )?;
    service.ledger.set_policy(&policy)?;
    let prices =
        parse_prices(&std::env::var("PAY_LIGHTNING_PRICES").unwrap_or_else(|_| "{}".into()))?;
    for (model, price) in prices {
        service.ledger.set_pricing(&model, Some(&price))?;
    }
    let cap = std::env::var("PAY_LIGHTNING_CAP_MSAT")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0);
    let mut engine = Engine::new(service, &namespace, cap, &directory)?;
    engine.buying =
        parse_prices(&std::env::var("PAY_LIGHTNING_BUY_PRICES").unwrap_or_else(|_| "{}".into()))?;
    engine.buy_peers = std::env::var("PAY_LIGHTNING_BUY_PEERS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    let engine = Arc::new(engine);

    let metadata = PluginMetadata::new(
        &name,
        env!("CARGO_PKG_VERSION"),
        plugin_server_info(
            &name,
            env!("CARGO_PKG_VERSION"),
            "Lightning payments",
            "Plugin-owned credit, ledger and settlement",
            None::<String>,
        ),
    )
    .with_manifest(manifest);
    let messages = engine.clone();
    let events = engine.clone();
    let startup = engine.clone();
    let plugin = SimplePlugin::new(metadata)
        // After the handshake: opening a wallet must not consume the host's
        // initialize-response timeout. Manifest inspection never reaches here.
        .on_initialized(move |_context| {
            let engine = startup.clone();
            Box::pin(async move { engine.start().await })
        })
        .on_channel_message(move |message, context| {
            let engine = messages.clone();
            let mut context = context.owned();
            Box::pin(async move {
                let result: Result<()> = async {
                    ensure!(message.channel == CHANNEL, "unexpected channel");
                    let message_body: Message = serde_json::from_slice(&message.body)?;
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
                                change: mesh_llm_plugin::proto::peer_block_request::Change::Unblock
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
            })
        })
        .on_mesh_event(move |event, context| {
            let engine = events.clone();
            let mut context = context.owned();
            Box::pin(async move {
                if event.kind == mesh_llm_plugin::proto::mesh_event::Kind::PeerUp as i32
                    && engine.cap_msat > 0
                    && let Some(peer) = event.peer
                    && engine.buy_peers.contains(&peer.peer_id)
                {
                    for message in engine.peer_up(&peer.peer_id).await? {
                        send(&mut context, &peer.peer_id, &message).await?;
                    }
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
                                send(&mut context, peer, &message).await?;
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
                engine
                    .gate_wait(&event, std::time::Duration::from_secs(28))
                    .await
                    .map_err(|error| mesh_llm_plugin::PluginError::internal(error.to_string()))
            })
        });
    PluginRuntime::run(plugin).await
}

async fn send(context: &mut PluginContext<'_>, peer: &str, message: &Message) -> Result<()> {
    context
        .send_json_channel(CHANNEL, peer, "payment", message)
        .await
}
fn record_error(error: &anyhow::Error) {
    // Operational errors stay in the plugin's isolated directory, not the host ledger.
    if let Some(path) = std::env::var_os("PAY_LIGHTNING_ERROR_LOG") {
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "{}", serde_json::json!({"error": error.to_string()}));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
