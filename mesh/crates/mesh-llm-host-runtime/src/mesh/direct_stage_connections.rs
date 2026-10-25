//! Bind authorized transport streams directly to an embedded execution ingress.

mod guarded_stream;

use std::{
    collections::HashMap,
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use iroh::EndpointId;
use mesh_llm_transport::protocol::ProtocolStream;
use prost::Message;
use skippy_serving::stage_transport::{StageConnection, StageTransport};
use tokio::{
    io::AsyncWriteExt,
    sync::{mpsc, watch},
};

use super::Node;
use crate::inference::skippy::StageLoadRequest;

pub(crate) fn endpoint(stage_id: &str) -> String {
    format!("mesh-stage://{stage_id}")
}

pub(crate) struct DirectStageConnections {
    node: Node,
    topology: String,
    run: String,
    targets: HashMap<String, (EndpointId, String)>,
    incoming: Mutex<mpsc::Receiver<StageConnection>>,
    sender: mpsc::Sender<StageConnection>,
    stopped: watch::Sender<bool>,
    ready: AtomicBool,
    runtime: tokio::runtime::Handle,
}

impl std::fmt::Debug for DirectStageConnections {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectStageConnections")
            .field("topology", &self.topology)
            .field("run", &self.run)
            .finish_non_exhaustive()
    }
}

impl Drop for DirectStageConnections {
    fn drop(&mut self) {
        self.stopped.send_replace(true);
    }
}

impl DirectStageConnections {
    fn close(&self) {
        self.stopped.send_replace(true);
        if let Ok(mut incoming) = self.incoming.lock() {
            while incoming.try_recv().is_ok() {}
        }
    }

    async fn open(&self, target: &str) -> Result<StageConnection> {
        let (peer, stage) = self
            .targets
            .get(target)
            .context("stage endpoint is outside the admitted topology")?;
        let open = skippy_protocol::proto::stage::StageTransportOpen {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: self.node.id().as_bytes().to_vec(),
            topology_id: self.topology.clone(),
            run_id: self.run.clone(),
            stage_id: stage.clone(),
        };
        skippy_protocol::validate_stage_transport_open(&open).map_err(anyhow::Error::msg)?;
        if *peer == self.node.id() {
            let (local, remote) = tokio::io::duplex(128 * 1024);
            self.node
                .deliver_direct_stage(self.node.id(), &open, ProtocolStream::new(remote))
                .await?;
            return Ok(self.connection(self.node.id(), ProtocolStream::new(local)));
        }
        let stream = match self.node.open_application_stage_stream(*peer).await? {
            Some(mut stream) => {
                let bytes = open.encode_to_vec();
                crate::protocol::ensure_control_frame_size(&bytes)?;
                stream.write_u32_le(u32::try_from(bytes.len())?).await?;
                stream.write_all(&bytes).await?;
                stream
            }
            None => {
                let (send, recv) = self
                    .node
                    .open_stage_transport_stream(*peer, &self.topology, &self.run, stage)
                    .await?;
                ProtocolStream::from_iroh(send, recv)
            }
        };
        Ok(self.connection(*peer, stream))
    }

    fn connection(&self, peer: EndpointId, stream: ProtocolStream) -> StageConnection {
        StageConnection::from_async(
            guarded_stream::GuardedStageStream::new(
                self.node.clone(),
                peer,
                stream,
                self.stopped.subscribe(),
            ),
            self.runtime.clone(),
        )
    }
}

impl StageTransport for DirectStageConnections {
    fn close(&self) {
        DirectStageConnections::close(self);
    }

    fn ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    fn connect(
        &self,
        endpoint: &str,
        deadline: Instant,
        shutdown: Option<&AtomicBool>,
    ) -> io::Result<StageConnection> {
        self.runtime.block_on(async {
            let acquisition = self.open(endpoint);
            tokio::pin!(acquisition);
            loop {
                if *self.stopped.borrow() || shutdown.is_some_and(|stop| stop.load(Ordering::Acquire)) {
                    return Err(io::ErrorKind::ConnectionAborted.into());
                }
                if Instant::now() >= deadline {
                    return Err(io::ErrorKind::TimedOut.into());
                }
                tokio::select! {
                    result = &mut acquisition => return result.map_err(io::Error::other),
                    _ = tokio::time::sleep(deadline.saturating_duration_since(Instant::now()).min(std::time::Duration::from_millis(25))) => {},
                }
            }
        })
    }

    fn try_accept(&self) -> io::Result<StageConnection> {
        if *self.stopped.borrow() {
            return Err(io::ErrorKind::ConnectionAborted.into());
        }
        self.incoming
            .lock()
            .map_err(|_| io::Error::other("stage ingress lock poisoned"))?
            .try_recv()
            .map_err(|error| match error {
                mpsc::error::TryRecvError::Empty => io::ErrorKind::WouldBlock.into(),
                mpsc::error::TryRecvError::Disconnected => io::ErrorKind::ConnectionAborted.into(),
            })
    }
}

impl Node {
    pub(crate) async fn stop_direct_stage_connections(
        &self,
        topology: &str,
        run: &str,
        stage: &str,
    ) {
        if let Ok(mut ingresses) = self.application_protocols.stage_ingresses.lock()
            && let Some(ingress) = ingresses
                .remove(&ingress_key(topology, run, stage))
                .and_then(|ingress| ingress.upgrade())
        {
            ingress.close();
        }
    }

    pub(crate) async fn direct_stage_transport(
        &self,
        load: &StageLoadRequest,
    ) -> Result<Arc<dyn StageTransport>> {
        ensure!(
            mesh_llm_transport::stage_link_delay()?.is_none(),
            "MESH_LLM_STAGE_LINK_DELAY_MS requires a direct-stream emulator; refusing an unshaped benchmark"
        );
        let mut targets = HashMap::new();
        let mut peers = std::collections::HashSet::new();
        for stage in &load.topology_stages {
            targets.insert(
                endpoint(&stage.stage_id),
                (stage.node_id, stage.stage_id.clone()),
            );
        }
        for peer in [load.upstream.as_ref(), load.downstream.as_ref()]
            .into_iter()
            .flatten()
        {
            let node = peer
                .node_id
                .context("direct stage connection requires a peer node identity")?;
            targets.insert(endpoint(&peer.stage_id), (node, peer.stage_id.clone()));
            if node != self.id() {
                peers.insert(node);
            }
        }
        for peer in peers {
            self.verify_required_stage_provider(peer).await?;
        }
        let (sender, incoming) = mpsc::channel(32);
        let transport = Arc::new(DirectStageConnections {
            node: self.clone(),
            topology: load.topology_id.clone(),
            run: load.run_id.clone(),
            targets,
            incoming: Mutex::new(incoming),
            sender,
            stopped: watch::channel(false).0,
            ready: AtomicBool::new(false),
            runtime: tokio::runtime::Handle::current(),
        });
        let key = ingress_key(&load.topology_id, &load.run_id, &load.stage_id);
        let mut ingresses = self
            .application_protocols
            .stage_ingresses
            .lock()
            .map_err(|_| anyhow::anyhow!("stage ingress registry poisoned"))?;
        ingresses.retain(|_, ingress| ingress.strong_count() > 0);
        if let Some(old) = ingresses
            .insert(key, Arc::downgrade(&transport))
            .and_then(|old| old.upgrade())
        {
            old.close();
        }
        Ok(transport)
    }

    pub(crate) async fn deliver_direct_stage(
        &self,
        peer: EndpointId,
        open: &skippy_protocol::proto::stage::StageTransportOpen,
        stream: ProtocolStream,
    ) -> Result<()> {
        let key = ingress_key(&open.topology_id, &open.run_id, &open.stage_id);
        let ingress = self
            .application_protocols
            .stage_ingresses
            .lock()
            .map_err(|_| anyhow::anyhow!("stage ingress registry poisoned"))?
            .get(&key)
            .and_then(std::sync::Weak::upgrade)
            .context("destination stage has no execution ingress")?;
        ensure!(
            ingress.ready.load(Ordering::Acquire) && !*ingress.stopped.borrow(),
            "destination stage is not ready"
        );
        ingress
            .sender
            .try_send(ingress.connection(peer, stream))
            .map_err(|_| anyhow::anyhow!("stage execution ingress is full or closed"))?;
        Ok(())
    }
}

fn ingress_key(topology: &str, run: &str, stage: &str) -> String {
    // Length prefixes keep arbitrary caller identifiers unambiguous.
    format!(
        "{}:{topology}{}:{run}{}:{stage}",
        topology.len(),
        run.len(),
        stage.len()
    )
}
