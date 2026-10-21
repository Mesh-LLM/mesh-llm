use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use anyhow::{Result, ensure};
use mesh_llm_transport::protocol::{
    IncomingProtocolStream, MeshProtocol, PeerId, ProtocolContext, ProtocolDescriptor,
    ProtocolNetwork, ProtocolPeer, ProtocolStream, REQUIRED_STAGE_PROVIDER_FEATURE,
    StageTransportPolicy, async_trait,
};
use tokio::sync::{Mutex, Semaphore, watch};

use super::Node;
mod stage_policy;

struct Registration {
    descriptor: ProtocolDescriptor,
    handler: Arc<dyn MeshProtocol>,
}

pub(crate) struct ApplicationProtocols {
    pub(super) stage_ingresses: std::sync::Mutex<
        std::collections::HashMap<
            String,
            std::sync::Weak<super::direct_stage_connections::DirectStageConnections>,
        >,
    >,
    policy: OnceLock<StageTransportPolicy>,
    registrations: OnceLock<BTreeMap<(String, u32), Registration>>,
    peers: watch::Sender<Vec<ProtocolPeer>>,
    cancelled: watch::Sender<bool>,
    tasks: Mutex<tokio::task::JoinSet<()>>,
    streams: Arc<Semaphore>,
}

impl Default for ApplicationProtocols {
    fn default() -> Self {
        Self {
            stage_ingresses: Default::default(),
            policy: OnceLock::new(),
            registrations: OnceLock::new(),
            peers: watch::channel(Vec::new()).0,
            cancelled: watch::channel(false).0,
            tasks: Mutex::new(tokio::task::JoinSet::new()),
            streams: Arc::new(Semaphore::new(32)),
        }
    }
}

impl ApplicationProtocols {
    pub(crate) fn descriptors(&self) -> Vec<crate::proto::node::MeshSubprotocol> {
        self.registrations
            .get()
            .into_iter()
            .flat_map(|registry| registry.values())
            .map(|entry| {
                let mut features = entry.descriptor.features.clone();
                if self.policy.get() == Some(&StageTransportPolicy::RequireProvider)
                    && entry.handler.stage_transport().is_some()
                {
                    features.push(REQUIRED_STAGE_PROVIDER_FEATURE.to_owned());
                }
                crate::proto::node::MeshSubprotocol {
                    name: entry.descriptor.name.clone(),
                    major: entry.descriptor.major,
                    features,
                }
            })
            .collect()
    }

    async fn spawn(&self, future: impl std::future::Future<Output = ()> + Send + 'static) {
        let mut tasks = self.tasks.lock().await;
        while tasks.try_join_next().is_some() {}
        if !*self.cancelled.borrow() {
            tasks.spawn(future);
        }
    }

    pub(crate) async fn shutdown(&self) {
        self.cancelled.send_replace(true);
        self.streams.close();
        let mut tasks = self.tasks.lock().await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while tasks.join_next().await.is_some() {}
        })
        .await;
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        self.peers.send_replace(Vec::new());
    }
}

#[derive(Clone)]
struct NodeProtocolNetwork {
    node: Node,
    descriptor: ProtocolDescriptor,
}

impl std::fmt::Debug for NodeProtocolNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeProtocolNetwork")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ProtocolNetwork for NodeProtocolNetwork {
    async fn open(&self, peer: PeerId, protocol: &ProtocolDescriptor) -> Result<ProtocolStream> {
        let state = self.node.state.lock().await;
        let remote = state
            .peers
            .get(&peer)
            .filter(|remote| remote.is_admitted())
            .ok_or_else(|| anyhow::anyhow!("application protocol peer is not admitted"))?;
        ensure!(
            remote
                .application_protocols
                .iter()
                .any(|candidate| candidate.name == protocol.name
                    && candidate.major == protocol.major),
            "peer does not advertise application protocol"
        );
        drop(state);
        let (send, recv) = self
            .node
            .open_mesh_subprotocol_stream(peer, &protocol.name, protocol.major)
            .await?;
        Ok(ProtocolStream::from_iroh(send, recv))
    }
    async fn accept_stage(&self, peer: PeerId, stream: ProtocolStream) -> Result<()> {
        ensure!(
            self.node.stage_provider_context_allowed(&self.descriptor),
            "required stage transport can only be accepted by the registered provider"
        );
        self.node
            .accept_application_stage_stream(peer, stream)
            .await
    }
}

impl Node {
    pub(crate) fn application_protocol_context(
        &self,
        descriptor: ProtocolDescriptor,
    ) -> ProtocolContext {
        ProtocolContext::new(
            self.endpoint.id(),
            descriptor.clone(),
            Arc::new(NodeProtocolNetwork {
                node: self.clone(),
                descriptor,
            }),
            self.application_protocols.peers.subscribe(),
            self.application_protocols.cancelled.subscribe(),
        )
    }

    #[cfg(test)]
    pub(crate) async fn install_application_protocols(
        &self,
        protocols: Vec<Arc<dyn MeshProtocol>>,
    ) -> Result<()> {
        self.install_application_protocols_with_policy(protocols, StageTransportPolicy::default())
            .await
    }

    pub(crate) async fn install_application_protocols_with_policy(
        &self,
        protocols: Vec<Arc<dyn MeshProtocol>>,
        policy: StageTransportPolicy,
    ) -> Result<()> {
        policy.validate(&protocols)?;
        let registry = protocols
            .into_iter()
            .map(|handler| {
                let descriptor = handler.descriptor();
                (
                    (descriptor.name.clone(), descriptor.major),
                    Registration {
                        descriptor,
                        handler,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        ensure!(
            !*self.application_protocols.cancelled.borrow(),
            "application protocols have stopped"
        );
        self.application_protocols
            .registrations
            .set(registry)
            .map_err(|_| anyhow::anyhow!("application protocols already installed"))?;
        self.application_protocols
            .policy
            .set(policy)
            .map_err(|_| anyhow::anyhow!("stage transport policy already installed"))?;
        self.refresh_application_peers().await;
        let registry = self
            .application_protocols
            .registrations
            .get()
            .expect("registry just installed");
        for entry in registry.values() {
            let handler = entry.handler.clone();
            let context = self.application_protocol_context(entry.descriptor.clone());
            self.application_protocols.spawn(async move {
                let result = handler.run(context.clone()).await;
                if let Err(error) = result {
                    tracing::warn!(protocol = %context.descriptor().name, %error, "application protocol background task stopped");
                }
            }).await;
        }
        if !registry.is_empty() {
            let node = self.clone();
            self.application_protocols
                .spawn(async move {
                    let mut changes = node.peer_change_rx.clone();
                    let mut cancelled = node.application_protocols.cancelled.subscribe();
                    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
                    loop {
                        tokio::select! {
                            _ = cancelled.changed() => break,
                            changed = changes.changed() => if changed.is_err() { break; },
                            _ = interval.tick() => {},
                        }
                        node.refresh_application_peers().await;
                    }
                })
                .await;
        }
        Ok(())
    }

    pub(crate) async fn refresh_application_peers(&self) {
        let state = self.state.lock().await;
        let mut peers = state
            .peers
            .values()
            .filter(|peer| peer.is_admitted())
            .map(|peer| ProtocolPeer {
                id: peer.id,
                protocols: peer
                    .application_protocols
                    .iter()
                    .map(|protocol| ProtocolDescriptor {
                        name: protocol.name.clone(),
                        major: protocol.major,
                        features: protocol.features.clone(),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        peers.sort_by_key(|peer| peer.id);
        self.application_protocols
            .peers
            .send_if_modified(|current| {
                if *current == peers {
                    false
                } else {
                    *current = peers;
                    true
                }
            });
    }

    pub(crate) async fn accept_application_protocol(
        &self,
        remote: PeerId,
        name: &str,
        major: u32,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        ensure!(
            self.state
                .lock()
                .await
                .peers
                .get(&remote)
                .is_some_and(|peer| peer.is_admitted()),
            "application protocol peer is not admitted"
        );
        let entry = self
            .application_protocols
            .registrations
            .get()
            .and_then(|registry| registry.get(&(name.into(), major)))
            .ok_or_else(|| anyhow::anyhow!("unsupported mesh subprotocol {name}/{major}"))?;
        let permit = self
            .application_protocols
            .streams
            .clone()
            .try_acquire_owned()?;
        let context = self.application_protocol_context(entry.descriptor.clone());
        let handler = entry.handler.clone();
        self.application_protocols.spawn(async move {
            let _permit = permit;
            let incoming = IncomingProtocolStream { peer: remote, stream: ProtocolStream::from_iroh(send, recv) };
            let result = tokio::select! {
                result = handler.accept(context.clone(), incoming) => result,
                () = context.cancelled() => return,
            };
            if let Err(error) = result {
                tracing::debug!(protocol = %context.descriptor().name, peer = %remote.fmt_short(), %error, "application protocol stream ended with error");
            }
        }).await;
        Ok(())
    }
}

impl Node {
    pub(crate) fn application_stage_cancellation(&self) -> watch::Receiver<bool> {
        self.application_protocols.cancelled.subscribe()
    }

    pub(crate) async fn application_stage_link(
        &self,
        peer: PeerId,
    ) -> Option<mesh_llm_transport::protocol::StageLinkObservation> {
        if *self.application_protocols.cancelled.borrow() {
            return None;
        }
        let registry = self.application_protocols.registrations.get()?;
        let state = self.state.lock().await;
        let remote = state
            .peers
            .get(&peer)
            .filter(|remote| remote.is_admitted())?;
        let entry = registry.values().find(|entry| {
            entry.handler.stage_transport().is_some()
                && remote.application_protocols.iter().any(|protocol| {
                    protocol.name == entry.descriptor.name
                        && protocol.major == entry.descriptor.major
                })
        })?;
        drop(state);
        entry
            .handler
            .stage_transport()?
            .stage_link(peer)
            .filter(|observation| {
                observation.samples >= 3
                    && observation.age < std::time::Duration::from_secs(30)
                    && !observation.acknowledgement_latency.is_zero()
            })
    }

    pub(crate) fn application_stage_peers(&self) -> watch::Receiver<Vec<ProtocolPeer>> {
        self.application_protocols.peers.subscribe()
    }

    pub(crate) async fn open_application_stage_stream(
        &self,
        peer: PeerId,
    ) -> Result<Option<ProtocolStream>> {
        let required = self.stage_transport_policy() == StageTransportPolicy::RequireProvider;
        let Some(registry) = self.application_protocols.registrations.get() else {
            ensure!(
                !required,
                "required stage transport provider is not installed"
            );
            return Ok(None);
        };
        for entry in registry.values() {
            let Some(provider) = entry.handler.stage_transport() else {
                continue;
            };
            let supported = if required {
                self.stage_required_peer_allowed(peer, false).await
            } else {
                self.application_protocols
                    .peers
                    .borrow()
                    .iter()
                    .any(|p| p.id == peer && p.supports(&entry.descriptor))
            };
            if !supported {
                continue;
            }
            ensure!(
                self.stage_stream_admitted(peer).await,
                "stage peer is not admitted"
            );
            ensure!(
                !required || provider.stage_available(peer),
                "required stage transport is unavailable for peer {}",
                peer.fmt_short()
            );
            let context = self.application_protocol_context(entry.descriptor.clone());
            let stream = tokio::select! {
                result = tokio::time::timeout(
                    mesh_llm_transport::protocol::OPEN_STREAM_TIMEOUT,
                    provider.open_stage(context.clone(), peer),
                ) => result?,
                () = context.cancelled() => anyhow::bail!("stage transport stopped during open"),
            }?;
            ensure!(
                !required || stream.is_some(),
                "required stage transport provider declined peer {}",
                peer.fmt_short()
            );
            return Ok(stream);
        }
        ensure!(
            !required,
            "peer does not advertise the required stage provider policy"
        );
        Ok(None)
    }

    pub(crate) async fn accept_application_stage_stream(
        &self,
        peer: PeerId,
        stream: ProtocolStream,
    ) -> Result<()> {
        ensure!(
            self.stage_required_peer_allowed(peer, false).await,
            "peer does not enforce the required stage provider policy"
        );
        let mut peers = self.application_protocols.peers.subscribe();
        let mut cancellation = self.application_protocols.cancelled.subscribe();
        let transfer =
            crate::network::tunnel::application_stage::accept(self.clone(), peer, stream);
        tokio::pin!(transfer);
        loop {
            ensure!(!*cancellation.borrow(), "stage transport stopped");
            ensure!(
                self.stage_required_peer_allowed(peer, false).await,
                "peer no longer enforces the required stage provider policy"
            );
            ensure!(
                peers.borrow().iter().any(|p| p.id == peer),
                "stage peer is no longer admitted"
            );
            tokio::select! {
                result = &mut transfer => return result,
                _ = cancellation.changed() => {},
                changed = peers.changed() => { changed?; },
            }
        }
    }
}
