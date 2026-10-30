use super::{ExternalPlugin, PluginRuntime, tests::plugin_for_spec};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{Duration, Instant};

async fn running_observer() -> (Arc<ExternalPlugin>, mpsc::Receiver<super::proto::Envelope>) {
    let plugin = Arc::new(plugin_for_spec(super::ExternalPluginSpec {
        operation_bindings: Vec::new(),
        name: "observer".into(),
        command: "unused".into(),
        args: Vec::new(),
        url: None,
        env: Default::default(),
        settings: Default::default(),
        startup: Default::default(),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        installed_metadata: None,
        openai_exchange_grant: None,
    }));
    let (outbound_tx, outbound_rx) = mpsc::channel(4);
    *plugin.runtime.lock().await = Some(PluginRuntime {
        generation: 7,
        authenticated_peer: true,
        initialized_lifecycle: Some(Box::new(
            mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe"),
        )),
        status_declaration: Default::default(),
        declares_inference_observation: false,
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
        .invoke_exchange_service(
            &mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe"),
            "{}",
            Instant::now() + Duration::from_millis(20),
        )
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
            .invoke_exchange_service(
                &mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe"),
                "{}",
                Instant::now() + Duration::from_secs(60),
            )
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

#[tokio::test]
async fn cancelled_control_request_cleans_pending_without_stopping_provider() {
    let (plugin, mut outbound) = running_observer().await;
    let (generation, sender, pending) = plugin.runtime_handles().await.unwrap();
    let caller = plugin.clone();
    let task = tokio::spawn(async move {
        caller
            .request_once(
                generation,
                sender,
                pending,
                super::proto::envelope::Payload::HealthRequest(super::proto::HealthRequest {}),
                None,
            )
            .await
    });
    outbound.recv().await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_running_without_pending(&plugin).await;
}

#[tokio::test]
async fn queued_control_request_has_no_pending_entry_before_admission() {
    let (plugin, _outbound) = running_observer().await;
    let (generation, sender, pending) = plugin.runtime_handles().await.unwrap();
    for _ in 0..4 {
        sender
            .send(super::proto::Envelope::default())
            .await
            .unwrap();
    }
    let caller = plugin.clone();
    let requests = pending.clone();
    let task = tokio::spawn(async move {
        caller
            .request_once(
                generation,
                sender,
                requests,
                super::proto::envelope::Payload::HealthRequest(super::proto::HealthRequest {}),
                None,
            )
            .await
    });
    // A bounded outer timeout proves the call cannot complete while saturated.
    assert!(
        tokio::time::timeout(Duration::from_millis(20), async {
            while !task.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .is_err()
    );
    assert!(pending.lock().await.is_empty());
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_running_without_pending(&plugin).await;
}

#[tokio::test(start_paused = true)]
async fn control_deadline_includes_saturated_queue_admission() {
    let (plugin, _outbound) = running_observer().await;
    let (generation, sender, pending) = plugin.runtime_handles().await.unwrap();
    for _ in 0..4 {
        sender
            .send(super::proto::Envelope::default())
            .await
            .unwrap();
    }
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        plugin.request_once(
            generation,
            sender,
            pending.clone(),
            super::proto::envelope::Payload::HealthRequest(super::proto::HealthRequest {}),
            Some(Duration::from_millis(100)),
        ),
    )
    .await
    .expect("admission must not outlive caller's deadline");
    assert!(result.is_err());
    assert!(pending.lock().await.is_empty());
    assert!(plugin.runtime.lock().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn delayed_control_admission_does_not_reset_response_budget() {
    let (plugin, mut outbound) = running_observer().await;
    let (generation, sender, pending) = plugin.runtime_handles().await.unwrap();
    for _ in 0..4 {
        sender
            .send(super::proto::Envelope::default())
            .await
            .unwrap();
    }
    let caller = plugin.clone();
    let start = Instant::now();
    let task = tokio::spawn(async move {
        caller
            .request_once(
                generation,
                sender,
                pending,
                super::proto::envelope::Payload::HealthRequest(super::proto::HealthRequest {}),
                Some(Duration::from_millis(100)),
            )
            .await
    });
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(80)).await;
    outbound.recv().await.unwrap();
    tokio::task::yield_now().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(25), task)
            .await
            .expect("only remaining 20ms belongs to response")
            .unwrap()
            .is_err()
    );
    assert_eq!(Instant::now() - start, Duration::from_millis(100));
}

#[tokio::test]
async fn cancelled_control_request_cleans_up_after_pending_map_contention() {
    let (plugin, mut outbound) = running_observer().await;
    let (generation, sender, pending) = plugin.runtime_handles().await.unwrap();
    let caller = plugin.clone();
    let requests = pending.clone();
    let task = tokio::spawn(async move {
        caller
            .request_once(
                generation,
                sender,
                requests,
                super::proto::envelope::Payload::HealthRequest(super::proto::HealthRequest {}),
                None,
            )
            .await
    });
    outbound.recv().await.unwrap();
    let locked = pending.lock().await;
    assert_eq!(locked.len(), 1);
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    drop(locked);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if pending.lock().await.is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("deferred cleanup must release correlation state");
    assert_running_without_pending(&plugin).await;
}
