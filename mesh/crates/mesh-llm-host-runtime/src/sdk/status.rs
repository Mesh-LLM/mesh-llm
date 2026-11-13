//! Per-node native snapshots. Subscriptions coalesce updates rather than dropping state on lag.
use super::EmbeddedServeHandle;
use crate::models::discovery::MeshModel;
use crate::runtime_data::{RuntimeDataCollector, RuntimeLlamaEndpointStatus};
use anyhow::{Context, Result};
pub use mesh_llm_types::mesh::ServedModelDescriptor;
use serde::Serialize;
use tokio::sync::watch;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DownloadProgress {
    pub file: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "phase", content = "error", rename_all = "snake_case")]
pub enum NodeLifecycle {
    Loading,
    Downloading,
    Ready,
    Failed(String),
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelCapacity {
    pub model: String,
    /// Empty for an unavailable instance; zero denotes an unknown lane context.
    pub context_per_lane: Vec<u64>,
    pub last_success_unix_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeNodeStatus {
    pub state: NodeLifecycle,
    pub serving_models: Vec<String>,
    pub models: Vec<MeshModel>,
    pub local_models: Vec<ServedModelDescriptor>,
    pub capacities: Vec<ModelCapacity>,
    pub download: Option<DownloadProgress>,
}

/// Initial snapshot followed by changed snapshots; intermediate progress may coalesce.
/// Each subscription belongs to one node generation. After a terminal snapshot it
/// closes; restarting requires subscribing to the new handle. No HTTP or timer is used.
pub struct StatusEvents {
    source: Option<Box<dyn SnapshotSource>>,
    changes: watch::Receiver<crate::runtime_data::subscriptions::RuntimeDataSubscriptionState>,
    exit: watch::Receiver<Option<NodeLifecycle>>,
    initial: bool,
    finished: bool,
}

impl StatusEvents {
    pub async fn next(&mut self) -> Result<Option<NativeNodeStatus>> {
        if self.finished {
            return Ok(None);
        }
        if self.exit.has_changed().is_err() && self.exit.borrow().is_none() {
            self.finish();
            anyhow::bail!("native runtime exited without a terminal status");
        }
        if !self.initial && self.exit.borrow().is_none() {
            tokio::select! {
                biased;
                result = self.exit.changed() => {
                    if result.is_err() && self.exit.borrow().is_none() {
                        self.finish();
                        anyhow::bail!("native runtime exited without a terminal status");
                    }
                },
                result = self.changes.changed() => {
                    if result.is_err() { self.finish(); anyhow::bail!("native status source closed"); }
                },
            }
        }
        self.initial = false;
        self.changes.borrow_and_update();
        let terminal = self.exit.borrow_and_update().clone();
        let result = self
            .source
            .as_ref()
            .context("native status source closed")?
            .snapshot(terminal)
            .await;
        let status = match result {
            Ok(status) => status,
            Err(error) => {
                self.finish();
                return Err(error);
            }
        };
        if matches!(
            status.state,
            NodeLifecycle::Failed(_) | NodeLifecycle::Stopped
        ) {
            self.finish();
        }
        Ok(Some(status))
    }

    fn finish(&mut self) {
        self.finished = true;
        drop(self.source.take());
    }
}

/// A cancellable, single-consumer subscription for language bindings.
/// Closing wakes a blocked reader and releases native node references before shutdown.
pub struct StatusSubscription {
    events: tokio::sync::Mutex<Option<StatusEvents>>,
    closed: watch::Sender<bool>,
}

impl StatusSubscription {
    pub fn new(events: StatusEvents) -> Self {
        let (closed, _) = watch::channel(false);
        Self {
            events: tokio::sync::Mutex::new(Some(events)),
            closed,
        }
    }

    pub async fn next(&self) -> Result<Option<NativeNodeStatus>> {
        let mut events = self.events.lock().await;
        let mut closed = self.closed.subscribe();
        if *closed.borrow() {
            return Ok(None);
        }
        let Some(source) = events.as_mut() else {
            return Ok(None);
        };
        let result = tokio::select! {
            biased;
            _ = closed.changed() => Ok(None),
            result = source.next() => result,
        };
        if !matches!(&result, Ok(Some(_))) {
            events.take();
        }
        result
    }

    pub async fn close(&self) {
        self.closed.send_replace(true);
        self.events.lock().await.take();
    }
}

impl EmbeddedServeHandle {
    pub async fn native_status(&self) -> Result<NativeNodeStatus> {
        anyhow::ensure!(
            self.exit.has_changed().is_ok() || self.exit.borrow().is_some(),
            "native runtime exited without a terminal status"
        );
        let terminal = self.exit.borrow().clone();
        snapshot(
            self.native_node
                .as_ref()
                .context("native node unavailable")?,
            terminal,
        )
        .await
    }

    pub fn status_events(&self) -> Result<StatusEvents> {
        let node = self
            .native_node
            .as_ref()
            .context("native node unavailable")?
            .clone();
        let changes = node.runtime_data_collector().subscribe();
        Ok(StatusEvents {
            source: Some(Box::new(node)),
            changes,
            exit: self.exit.clone(),
            initial: true,
            finished: false,
        })
    }
}

#[async_trait::async_trait]
trait SnapshotSource: Send + Sync {
    async fn snapshot(&self, terminal: Option<NodeLifecycle>) -> Result<NativeNodeStatus>;
}

#[async_trait::async_trait]
impl SnapshotSource for crate::mesh::Node {
    async fn snapshot(&self, terminal: Option<NodeLifecycle>) -> Result<NativeNodeStatus> {
        snapshot(self, terminal).await
    }
}

async fn snapshot(
    node: &crate::mesh::Node,
    terminal: Option<NodeLifecycle>,
) -> Result<NativeNodeStatus> {
    if let Some(state) = terminal {
        return Ok(NativeNodeStatus {
            state,
            serving_models: Vec::new(),
            models: Vec::new(),
            local_models: Vec::new(),
            capacities: Vec::new(),
            download: None,
        });
    }
    let collector: RuntimeDataCollector = node.runtime_data_collector();
    let runtime = collector.runtime_status_snapshot();
    let serving_models = node.serving_models().await;
    let error = node
        .runtime_intents
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .rev()
        .find_map(|intent| intent.last_error.clone());
    let instances: Vec<_> = if runtime.llama_runtime_by_instance.is_empty() {
        vec![&runtime.llama_runtime]
    } else {
        runtime.llama_runtime_by_instance.values().collect()
    };
    let capacities = instances
        .into_iter()
        .filter_map(|runtime| {
            let slots = &runtime.slots;
            Some(ModelCapacity {
                model: slots.model.clone()?,
                // Retain unavailable instances so consumers cannot overstate aggregate capacity.
                context_per_lane: if slots.status == RuntimeLlamaEndpointStatus::Ready {
                    slots
                        .slots
                        .iter()
                        .map(|slot| slot.n_ctx.unwrap_or(0))
                        .collect()
                } else {
                    Vec::new()
                },
                last_success_unix_ms: slots.last_success_unix_ms,
            })
        })
        .collect::<Vec<_>>();
    let state = lifecycle(
        node.role().await == crate::mesh::NodeRole::Client
            || serving_ready(runtime.llama_ready, &serving_models, &capacities),
        runtime.download.is_some(),
        error,
    );
    Ok(NativeNodeStatus {
        state,
        serving_models,
        models: crate::models::discovery::collect(node).await?,
        local_models: node.served_model_descriptors().await,
        capacities,
        download: runtime.download,
    })
}

fn serving_ready(
    reported_ready: bool,
    serving_models: &[String],
    capacities: &[ModelCapacity],
) -> bool {
    !serving_models.is_empty()
        && (reported_ready
            || capacities.iter().any(|capacity| {
                serving_models.contains(&capacity.model)
                    && !capacity.context_per_lane.is_empty()
                    && capacity.context_per_lane.iter().all(|context| *context > 0)
            }))
}

fn lifecycle(ready: bool, downloading: bool, error: Option<String>) -> NodeLifecycle {
    if let Some(error) = error {
        NodeLifecycle::Failed(error)
    } else if ready {
        NodeLifecycle::Ready
    } else if downloading {
        NodeLifecycle::Downloading
    } else {
        NodeLifecycle::Loading
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct FakeSource(Arc<Mutex<NodeLifecycle>>);

    #[test]
    fn readiness_requires_native_serving_and_never_hides_a_load_failure() {
        assert_eq!(lifecycle(false, false, None), NodeLifecycle::Loading);
        assert_eq!(lifecycle(false, true, None), NodeLifecycle::Downloading);
        assert_eq!(lifecycle(true, false, None), NodeLifecycle::Ready);
        assert_eq!(
            lifecycle(true, true, Some("allocation failed".into())),
            NodeLifecycle::Failed("allocation failed".into())
        );
    }

    #[test]
    fn native_lanes_make_the_served_model_ready_without_console_derived_flags() {
        let models = vec!["chat".to_string()];
        let mut capacities = vec![ModelCapacity {
            model: "other".into(),
            context_per_lane: vec![8192],
            last_success_unix_ms: Some(1),
        }];
        assert!(!serving_ready(false, &models, &capacities));
        capacities[0].model = "chat".into();
        assert!(serving_ready(false, &models, &capacities));
        capacities[0].context_per_lane = vec![0];
        assert!(!serving_ready(false, &models, &capacities));
        assert!(!serving_ready(true, &[], &capacities));
    }
    #[async_trait::async_trait]
    impl SnapshotSource for FakeSource {
        async fn snapshot(&self, terminal: Option<NodeLifecycle>) -> Result<NativeNodeStatus> {
            Ok(NativeNodeStatus {
                state: terminal.unwrap_or_else(|| self.0.lock().unwrap().clone()),
                serving_models: Vec::new(),
                models: Vec::new(),
                local_models: Vec::new(),
                capacities: Vec::new(),
                download: None,
            })
        }
    }

    #[tokio::test]
    async fn subscriptions_start_with_current_state_coalesce_and_close_after_terminal() {
        let collector = RuntimeDataCollector::new();
        let state = Arc::new(Mutex::new(NodeLifecycle::Loading));
        let (exit, receiver) = watch::channel(None);
        let mut events = StatusEvents {
            source: Some(Box::new(FakeSource(state.clone()))),
            changes: collector.subscribe(),
            exit: receiver,
            initial: true,
            finished: false,
        };
        assert_eq!(
            events.next().await.unwrap().unwrap().state,
            NodeLifecycle::Loading
        );
        for phase in [NodeLifecycle::Downloading, NodeLifecycle::Ready] {
            *state.lock().unwrap() = phase;
            collector.mark_dirty(crate::runtime_data::RuntimeDataDirty::STATUS);
        }
        assert_eq!(
            events.next().await.unwrap().unwrap().state,
            NodeLifecycle::Ready
        );
        exit.send_replace(Some(NodeLifecycle::Failed("runtime failed".into())));
        assert_eq!(
            events.next().await.unwrap().unwrap().state,
            NodeLifecycle::Failed("runtime failed".into())
        );
        assert!(events.next().await.unwrap().is_none());
        assert_eq!(
            Arc::strong_count(&state),
            1,
            "terminal streams release their native source before node shutdown"
        );
    }

    #[tokio::test]
    async fn closing_a_language_subscription_wakes_its_reader_and_releases_the_node() {
        let collector = RuntimeDataCollector::new();
        let state = Arc::new(Mutex::new(NodeLifecycle::Loading));
        let (_exit, receiver) = watch::channel(None);
        let subscription = StatusSubscription::new(StatusEvents {
            source: Some(Box::new(FakeSource(state.clone()))),
            changes: collector.subscribe(),
            exit: receiver,
            initial: true,
            finished: false,
        });
        assert_eq!(
            subscription.next().await.unwrap().unwrap().state,
            NodeLifecycle::Loading
        );
        let mut pending = Box::pin(subscription.next());
        tokio::select! {
            result = &mut pending => panic!("expected a blocked reader, got {result:?}"),
            () = tokio::task::yield_now() => {},
        }
        let ((), result) = tokio::join!(subscription.close(), pending);
        assert!(result.unwrap().is_none());
        assert_eq!(Arc::strong_count(&state), 1);
        assert!(subscription.next().await.unwrap().is_none());
        subscription.close().await;
    }

    #[test]
    fn lifecycle_wire_contract_preserves_failure_reason() {
        assert_eq!(
            serde_json::to_value(NodeLifecycle::Ready).unwrap(),
            serde_json::json!({"phase": "ready"})
        );
        assert_eq!(
            serde_json::to_value(NodeLifecycle::Failed("load failed".into())).unwrap(),
            serde_json::json!({"phase": "failed", "error": "load failed"})
        );
    }

    #[tokio::test]
    async fn a_lost_runtime_publisher_is_an_error_instead_of_a_ready_snapshot() {
        let collector = RuntimeDataCollector::new();
        let (exit, receiver) = watch::channel(None);
        let mut events = StatusEvents {
            source: Some(Box::new(FakeSource(Arc::new(Mutex::new(
                NodeLifecycle::Ready,
            ))))),
            changes: collector.subscribe(),
            exit: receiver,
            initial: true,
            finished: false,
        };
        assert_eq!(
            events.next().await.unwrap().unwrap().state,
            NodeLifecycle::Ready
        );
        drop(exit);
        assert!(
            events
                .next()
                .await
                .unwrap_err()
                .to_string()
                .contains("without a terminal status")
        );
    }
}
