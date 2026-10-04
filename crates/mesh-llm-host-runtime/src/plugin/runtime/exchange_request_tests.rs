use super::{ExternalPlugin, PluginRuntime, tests::plugin_for_spec};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{Duration, Instant};

async fn running_observer() -> (Arc<ExternalPlugin>, mpsc::Receiver<super::proto::Envelope>) {
    let plugin = Arc::new(plugin_for_spec(super::ExternalPluginSpec {
        name: "observer".into(),
        command: "unused".into(),
        args: Vec::new(),
        url: None,
        env: Default::default(),
        startup: Default::default(),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        installed_metadata: None,
        openai_exchange_grant: None,
    }));
    let (outbound_tx, outbound_rx) = mpsc::channel(4);
    *plugin.runtime.lock().await = Some(PluginRuntime {
        generation: 7,
        _child: None,
        connection_task: tokio::spawn(std::future::pending()),
        outbound_tx,
        pending: Arc::new(Mutex::new(HashMap::new())),
    });
    *plugin.manifest.lock().await = Some(super::proto::PluginManifest::default());
    plugin.summary.lock().await.status = "running".into();
    (plugin, outbound_rx)
}

async fn assert_running_without_pending(plugin: &ExternalPlugin) {
    let runtime = plugin.runtime.lock().await;
    let runtime = runtime.as_ref().expect("observer runtime must survive");
    assert_eq!(runtime.generation, 7);
    assert!(runtime.pending.lock().await.is_empty());
    assert!(plugin.manifest.lock().await.is_some());
    assert_eq!(plugin.summary.lock().await.status, "running");
    runtime.connection_task.abort();
}

#[tokio::test]
async fn exchange_deadline_sends_once_without_restarting_observer() {
    let (plugin, mut outbound) = running_observer().await;
    let result = plugin
        .invoke_exchange_service("observe", "{}", Instant::now() + Duration::from_millis(20))
        .await;
    assert!(result.is_err());
    assert!(outbound.try_recv().is_ok());
    assert!(outbound.try_recv().is_err(), "a hook must never retry");
    assert_running_without_pending(&plugin).await;
}

#[tokio::test]
async fn cancelled_exchange_removes_only_its_pending_request() {
    let (plugin, mut outbound) = running_observer().await;
    let pending = plugin
        .runtime
        .lock()
        .await
        .as_ref()
        .unwrap()
        .pending
        .clone();
    let (sibling_tx, _sibling_rx) = tokio::sync::oneshot::channel();
    pending.lock().await.insert(999, sibling_tx);
    let task_plugin = plugin.clone();
    let task = tokio::spawn(async move {
        task_plugin
            .invoke_exchange_service("observe", "{}", Instant::now() + Duration::from_secs(60))
            .await
    });
    outbound.recv().await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let mut requests = pending.lock().await;
    assert_eq!(requests.len(), 1);
    assert!(requests.remove(&999).is_some());
    drop(requests);
    assert_running_without_pending(&plugin).await;
}
