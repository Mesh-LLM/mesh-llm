//! A plugin process that starts after the host's own start tells the node,
//! which sends it the mesh snapshot (Mesh-LLM/capsules#48).

use super::{ExternalPlugin, PROTOCOL_VERSION, PluginMeshEvent, proto};
use anyhow::Result;
use tokio::sync::mpsc::error::SendTimeoutError;
use tokio::sync::oneshot;

/// How long a plugin that started again waits to tell the node.
const PLUGIN_STARTED_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// How long it then waits for the node to send it the mesh snapshot before
/// it takes requests.
const PLUGIN_SNAPSHOT_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

impl ExternalPlugin {
    /// Tell the node this plugin process started, and wait until the node
    /// has sent it the mesh snapshot (this node's id, the mesh id, each peer),
    /// so the request that started it arrives after this node's id. The node
    /// sends with [`Self::send_mesh_event_to_generation`], which takes no
    /// restart lock, so waiting under it here cannot hold the snapshot up.
    /// Both waits are bounded for a node that is busy or stopping. An event
    /// that could not be queued in time is queued from a task instead, so
    /// the snapshot reaches this process late rather than never.
    pub(super) async fn announce_started(&self, generation: u64) {
        let (delivered, snapshot_sent) = oneshot::channel();
        let event = PluginMeshEvent::PluginStarted {
            plugin_id: self.spec.name.clone(),
            generation,
            delivered,
        };
        match self
            .mesh_tx
            .send_timeout(event, PLUGIN_STARTED_SEND_TIMEOUT)
            .await
        {
            Ok(()) => {
                let _ = tokio::time::timeout(PLUGIN_SNAPSHOT_WAIT, snapshot_sent).await;
            }
            Err(SendTimeoutError::Timeout(event)) => {
                let mesh_tx = self.mesh_tx.clone();
                tokio::spawn(async move {
                    let _ = mesh_tx.send(event).await;
                });
            }
            Err(SendTimeoutError::Closed(_)) => {}
        }
    }

    /// Whether the installed process's start finished; `None` when none is
    /// installed.
    pub(super) async fn runtime_started(&self) -> Option<bool> {
        self.runtime
            .lock()
            .await
            .as_ref()
            .map(|runtime| runtime.started)
    }

    /// The process of `generation` finished its start, announcement included.
    pub(super) async fn mark_started(&self, generation: u64) {
        if let Some(runtime) = self.runtime.lock().await.as_mut()
            && runtime.generation == generation
        {
            runtime.started = true;
        }
    }

    /// Stop an installed process whose start never finished, so the next
    /// start is a whole one. Called under the restart lock, when no start is
    /// in progress: a start whose caller was cancelled after its process was
    /// installed leaves it uninitialized or without the mesh snapshot, and
    /// serving it would bring back Mesh-LLM/capsules#48.
    pub(super) async fn discard_unfinished_start(&self) {
        let generation = self.runtime.lock().await.as_ref().map(|r| r.generation);
        self.handle_runtime_failure(
            generation,
            format!("Plugin '{}' did not finish starting", self.spec.name),
        )
        .await;
    }

    /// Send `event` to the plugin process of `generation` only. Never starts
    /// or restarts the plugin: a snapshot meant for a process that has since
    /// exited is dropped, so it cannot start the next one.
    pub(crate) async fn send_mesh_event_to_generation(
        &self,
        generation: u64,
        event: proto::MeshEvent,
    ) -> Result<()> {
        let outbound_tx = {
            let runtime = self.runtime.lock().await;
            match runtime.as_ref() {
                Some(runtime) if runtime.generation == generation => runtime.outbound_tx.clone(),
                _ => return Ok(()),
            }
        };
        outbound_tx
            .send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.spec.name.clone(),
                request_id: 0,
                payload: Some(proto::envelope::Payload::MeshEvent(event)),
            })
            .await
            .map_err(|_| {
                anyhow::anyhow!("Plugin '{}' is not accepting mesh events", self.spec.name)
            })
    }
}
