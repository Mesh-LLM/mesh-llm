// What a plugin that goes away while the node is running is reported as.
//
// The supervisor restarts a plugin whose status says it needs restarting and
// leaves an intentionally stopped plugin down, so the status a disconnect
// leaves behind decides whether a crashed plugin's background work resumes.

use super::tests::in_process_plugin;
use super::*;
use crate::plugin::InProcessPluginRunner;
use mesh_llm_plugin::{LocalStream, read_envelope, write_envelope};
use std::sync::Arc;
use tokio::sync::watch;

/// Answers `initialize`, then stays connected until the test says the plugin
/// process is going away, which is what a crash looks like from the host's
/// side of the pipe.
async fn exit_when_told(mut stream: LocalStream, mut gone: watch::Receiver<bool>) -> Result<()> {
    let incoming = read_envelope(&mut stream).await?;
    let Some(proto::envelope::Payload::InitializeRequest(_)) = incoming.payload else {
        bail!("fixture expected an initialize request");
    };
    write_envelope(
        &mut stream,
        &proto::Envelope {
            protocol_version: crate::plugin::PROTOCOL_VERSION,
            plugin_id: incoming.plugin_id,
            request_id: incoming.request_id,
            payload: Some(proto::envelope::Payload::InitializeResponse(
                proto::InitializeResponse {
                    plugin_id: "exit-fixture".into(),
                    plugin_protocol_version: crate::plugin::PROTOCOL_VERSION,
                    plugin_version: "0.0.0".into(),
                    server_info_json: serde_json::to_string(&rmcp::model::ServerConfig::default())?,
                    ..Default::default()
                },
            )),
        },
    )
    .await?;
    let shutdown = async {
        loop {
            let incoming = read_envelope(&mut stream).await?;
            if matches!(
                incoming.payload,
                Some(proto::envelope::Payload::ShutdownRequest(_))
            ) {
                return Ok(());
            }
        }
    };
    tokio::select! {
        result = shutdown => result,
        result = gone.wait_for(|gone| *gone) => Ok(result.map(|_| ())?),
    }
}

fn exiting_runner() -> (InProcessPluginRunner, watch::Sender<bool>) {
    let (sender, receiver) = watch::channel(false);
    (
        Arc::new(move |stream| Box::pin(exit_when_told(stream, receiver.clone()))),
        sender,
    )
}

/// Waits for the connection loop to react to the plugin going away.
async fn await_status(plugin: &ExternalPlugin) -> String {
    for _ in 0..1_000 {
        let status = plugin.summary().await.status;
        if status != "running" {
            return status;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    plugin.summary().await.status
}

#[tokio::test]
async fn an_unexpected_exit_is_reported_as_restarting() {
    let (runner, gone) = exiting_runner();
    let plugin = in_process_plugin("exit-fixture", runner);
    plugin.ensure_running().await.unwrap();
    assert_eq!(plugin.summary().await.status, "running");

    let _ = gone.send(true);
    let status = tokio::time::timeout(std::time::Duration::from_secs(10), await_status(&plugin))
        .await
        .expect("the connection loop must notice the exit");
    assert_eq!(
        status, "restarting",
        "a plugin that exited on its own must not be reported as stopped"
    );
    plugin.shutdown().await;
}

#[tokio::test]
async fn an_unexpected_exit_leaves_the_plugin_eligible_for_supervision() {
    let (runner, gone) = exiting_runner();
    let plugin = in_process_plugin("exit-fixture", runner);
    plugin.ensure_running().await.unwrap();

    let _ = gone.send(true);
    tokio::time::timeout(std::time::Duration::from_secs(10), await_status(&plugin))
        .await
        .expect("the connection loop must notice the exit");

    assert!(
        !plugin.is_stopping().await,
        "the supervisor skips a plugin reported as stopped, so a crashed plugin \
         would stay down until something called it"
    );
    plugin.shutdown().await;
}

#[tokio::test]
async fn an_intentional_shutdown_is_still_reported_as_stopped() {
    let (runner, _gone) = exiting_runner();
    let plugin = in_process_plugin("exit-fixture", runner);
    plugin.ensure_running().await.unwrap();
    plugin.shutdown().await;
    assert_eq!(plugin.summary().await.status, "stopped");
}
