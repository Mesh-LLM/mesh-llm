//! A plugin process that starts after the host's own start tells the node,
//! which sends it the mesh snapshot (Mesh-LLM/capsules#48).

use super::{ExternalPlugin, PROTOCOL_VERSION, PluginMeshEvent, proto};
use anyhow::{Result, bail};
use tokio::sync::mpsc::error::SendTimeoutError;
use tokio::sync::oneshot;

/// How long a plugin that started again waits to tell the node.
const PLUGIN_STARTED_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// How long it then waits for the node to send it the mesh snapshot before
/// it takes requests.
const PLUGIN_SNAPSHOT_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// How long the node waits for one process to take one mesh event. The node
/// sends from one task for every plugin, so a process that stops reading
/// must not hold it.
const MESH_EVENT_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

impl ExternalPlugin {
    /// Tell the node this plugin process started, and wait until the node
    /// has sent it the mesh snapshot (this node's id, the mesh id, each peer),
    /// so the request that started it arrives after this node's id. The node
    /// sends with [`Self::send_mesh_event_to_generation`], which takes no
    /// restart lock, so waiting under it here cannot hold the snapshot up.
    /// Both waits are bounded for a node that is busy or stopping. Fails
    /// when the snapshot did not arrive in time: the start then fails, and
    /// the next request starts a new process that asks again, so no request
    /// reaches a process without this node's id.
    pub(super) async fn announce_started(&self, generation: u64) -> Result<()> {
        let (delivered, snapshot_sent) = oneshot::channel();
        let event = PluginMeshEvent::PluginStarted {
            plugin_id: self.spec.name.clone(),
            generation,
            delivered,
        };
        if self
            .mesh_tx
            .send_timeout(event, PLUGIN_STARTED_SEND_TIMEOUT)
            .await
            .is_err()
        {
            bail!(
                "Plugin '{}' started, but the node did not take its start in time; retry the request",
                self.spec.name
            );
        }
        match tokio::time::timeout(PLUGIN_SNAPSHOT_WAIT, snapshot_sent).await {
            Ok(Ok(true)) => Ok(()),
            _ => bail!(
                "Plugin '{}' started, but did not get the mesh snapshot in time; retry the request",
                self.spec.name
            ),
        }
    }

    /// A node serves this plugin from now on: each later start waits for
    /// the node's mesh snapshot. Set before the node sends the snapshot to
    /// the plugins already running, so no start falls between the two.
    pub(crate) fn expect_mesh_snapshots(&self) {
        self.announce_starts
            .store(true, std::sync::atomic::Ordering::Release);
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
    /// exited is dropped, so it cannot start the next one. A process that
    /// does not take the event within [`MESH_EVENT_SEND_TIMEOUT`] has stopped
    /// reading: it is failed, so a request starts a new one.
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
        let envelope = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.spec.name.clone(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::MeshEvent(event)),
        };
        match outbound_tx
            .send_timeout(envelope, MESH_EVENT_SEND_TIMEOUT)
            .await
        {
            Ok(()) => Ok(()),
            Err(SendTimeoutError::Timeout(_)) => {
                let reason = format!(
                    "Plugin '{}' did not take a mesh event in time",
                    self.spec.name
                );
                self.handle_runtime_failure(Some(generation), reason.clone())
                    .await;
                bail!(reason)
            }
            Err(SendTimeoutError::Closed(_)) => {
                bail!("Plugin '{}' is not accepting mesh events", self.spec.name)
            }
        }
    }
}
