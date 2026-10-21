//! Application protocols hosted on admitted mesh connections.

use std::{
    fmt::Debug,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use anyhow::{Result, ensure};
pub use async_trait::async_trait;
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    sync::watch,
};

pub use iroh::EndpointId as PeerId;

/// Deadline for opening an outbound application protocol stream before the open fails.
pub const OPEN_STREAM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Names owned by built-in mesh subprotocols. Application protocols may not
/// register these names; the host and gossip filtering apply the same list.
pub const BUILT_IN_PROTOCOL_NAMES: &[&str] = &["skippy-stage"];

/// Host-owned capability identifying nodes which enforce provider-only stage traffic.
pub const REQUIRED_STAGE_PROVIDER_FEATURE: &str = "stage-provider-required";

/// Select whether inter-node activation streams may use the default transport.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StageTransportPolicy {
    #[default]
    PreferProvider,
    RequireProvider,
}

impl StageTransportPolicy {
    pub fn validate(self, protocols: &[Arc<dyn MeshProtocol>]) -> Result<()> {
        validate_protocols(protocols)?;
        if self == Self::RequireProvider {
            let provider = protocols
                .iter()
                .find(|protocol| protocol.stage_transport().is_some())
                .ok_or_else(|| {
                    anyhow::anyhow!("required stage transport provider is not registered")
                })?;
            ensure!(
                provider.descriptor().features.len() < 32,
                "required stage provider must leave room for the host capability feature"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolDescriptor {
    pub name: String,
    pub major: u32,
    pub features: Vec<String>,
}

impl ProtocolDescriptor {
    pub fn new(name: impl Into<String>, major: u32) -> Self {
        Self {
            name: name.into(),
            major,
            features: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.name.is_empty()
                && self.name.len() <= 64
                && self.name.bytes().all(|c| c.is_ascii_lowercase()
                    || c.is_ascii_digit()
                    || matches!(c, b'-' | b'.')),
            "invalid application protocol name"
        );
        ensure!(
            self.major > 0,
            "application protocol major must be positive"
        );
        ensure!(
            self.features.len() <= 32
                && self.features.iter().all(|feature| !feature.is_empty()
                    && feature.len() <= 64
                    && feature
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'.' | b'_'))),
            "invalid application protocol features"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolPeer {
    pub id: PeerId,
    pub protocols: Vec<ProtocolDescriptor>,
}

impl ProtocolPeer {
    pub fn supports(&self, protocol: &ProtocolDescriptor) -> bool {
        self.protocols
            .iter()
            .any(|candidate| candidate.name == protocol.name && candidate.major == protocol.major)
    }
}

pub trait ProtocolIo: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> ProtocolIo for T {}

pub struct ProtocolStream(Box<dyn ProtocolIo>);

impl ProtocolStream {
    pub fn new(io: impl ProtocolIo + 'static) -> Self {
        Self(Box::new(io))
    }

    #[doc(hidden)]
    pub fn from_iroh(send: iroh::endpoint::SendStream, recv: iroh::endpoint::RecvStream) -> Self {
        Self::new(IrohStream { send, recv })
    }
}

impl Debug for ProtocolStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProtocolStream").finish_non_exhaustive()
    }
}

impl AsyncRead for ProtocolStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.0).poll_read(cx, buf)
    }
}

impl AsyncWrite for ProtocolStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut *self.0).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.0).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.0).poll_shutdown(cx)
    }
}

struct IrohStream {
    send: iroh::endpoint::SendStream,
    recv: iroh::endpoint::RecvStream,
}

impl AsyncRead for IrohStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        AsyncRead::poll_read(Pin::new(&mut self.recv), cx, buf)
    }
}

impl AsyncWrite for IrohStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        AsyncWrite::poll_write(Pin::new(&mut self.send), cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        AsyncWrite::poll_flush(Pin::new(&mut self.send), cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        AsyncWrite::poll_shutdown(Pin::new(&mut self.send), cx)
    }
}

#[async_trait]
pub trait ProtocolNetwork: Debug + Send + Sync {
    async fn open(&self, peer: PeerId, protocol: &ProtocolDescriptor) -> Result<ProtocolStream>;

    async fn accept_stage(&self, _peer: PeerId, _stream: ProtocolStream) -> Result<()> {
        anyhow::bail!("stage transport is unavailable")
    }
}

#[derive(Clone, Debug)]
pub struct ProtocolContext {
    local_peer: PeerId,
    descriptor: ProtocolDescriptor,
    network: Arc<dyn ProtocolNetwork>,
    peers: watch::Receiver<Vec<ProtocolPeer>>,
    cancellation: watch::Receiver<bool>,
}

impl ProtocolContext {
    #[doc(hidden)]
    pub fn new(
        local_peer: PeerId,
        descriptor: ProtocolDescriptor,
        network: Arc<dyn ProtocolNetwork>,
        peers: watch::Receiver<Vec<ProtocolPeer>>,
        cancellation: watch::Receiver<bool>,
    ) -> Self {
        Self {
            local_peer,
            descriptor,
            network,
            peers,
            cancellation,
        }
    }

    pub fn local_peer_id(&self) -> PeerId {
        self.local_peer
    }
    pub fn descriptor(&self) -> &ProtocolDescriptor {
        &self.descriptor
    }
    /// Contains a current snapshot as well as subsequent admission/capability changes.
    pub fn watch_peers(&self) -> watch::Receiver<Vec<ProtocolPeer>> {
        self.peers.clone()
    }

    pub async fn open(&self, peer: PeerId) -> Result<ProtocolStream> {
        ensure!(
            !*self.cancellation.borrow(),
            "application protocol is stopped"
        );
        tokio::select! {
            result = tokio::time::timeout(OPEN_STREAM_TIMEOUT, self.network.open(peer, &self.descriptor)) => result?,
            () = self.cancelled() => anyhow::bail!("application protocol stopped while opening stream"),
        }
    }

    /// Route a provider's authenticated stream to an authorized local stage.
    /// `peer` must come from the authenticated setup channel, not payload metadata.
    pub async fn accept_stage(&self, peer: PeerId, stream: ProtocolStream) -> Result<()> {
        ensure!(
            !*self.cancellation.borrow(),
            "application protocol is stopped"
        );
        tokio::select! {
            result = self.network.accept_stage(peer, stream) => result,
            () = self.cancelled() => anyhow::bail!("application protocol stopped during stage transport"),
        }
    }

    pub async fn cancelled(&self) {
        let mut cancellation = self.cancellation.clone();
        if *cancellation.borrow() {
            return;
        }
        while cancellation.changed().await.is_ok() {
            if *cancellation.borrow() {
                return;
            }
        }
    }
}

#[derive(Debug)]
pub struct IncomingProtocolStream {
    /// Established by the mesh connection, never taken from an application frame.
    pub peer: PeerId,
    pub stream: ProtocolStream,
}

/// Outgoing delivery measurements supplied locally by a stage transport provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageLinkObservation {
    /// Mean outgoing DATA-to-receive-acknowledgement time, including backpressure.
    pub acknowledgement_latency: std::time::Duration,
    /// Acknowledged payload bytes divided by active delivery time; not raw link speed.
    pub payload_bytes_per_second: u64,
    pub samples: u32,
    /// Age of the newest sample. Providers must also exclude stale older samples.
    pub age: std::time::Duration,
}

/// Supplies reliable, ordered duplex stage streams with half-close.
#[async_trait]
pub trait StageTransportProvider: Debug + Send + Sync {
    /// Cached readiness hint for planning; no network I/O or native probes.
    /// The default permits setup to be attempted. This does not prove reachability.
    /// Keep peers with usable active channels eligible when new capacity is exhausted.
    fn stage_available(&self, _peer: PeerId) -> bool {
        true
    }

    /// Local observations of an active outgoing path. Return `None` when closed,
    /// failed, unmeasured, or when a new open would decline to the default path.
    /// This synchronous snapshot must not perform network I/O or native probes.
    fn stage_link(&self, _peer: PeerId) -> Option<StageLinkObservation> {
        None
    }

    /// Decline with `None` before sending stage bytes. Errors fail the connection;
    /// they never trigger replay through the default transport.
    async fn open_stage(
        &self,
        context: ProtocolContext,
        peer: PeerId,
    ) -> Result<Option<ProtocolStream>>;
}

#[async_trait]
pub trait MeshProtocol: Debug + Send + Sync + 'static {
    fn descriptor(&self) -> ProtocolDescriptor;

    fn stage_transport(&self) -> Option<&dyn StageTransportProvider> {
        None
    }

    /// Runs on the embedded node's runtime and is cancelled when that node stops.
    async fn run(&self, context: ProtocolContext) -> Result<()> {
        context.cancelled().await;
        Ok(())
    }

    async fn accept(
        &self,
        context: ProtocolContext,
        incoming: IncomingProtocolStream,
    ) -> Result<()>;
}

pub fn validate_protocols(protocols: &[Arc<dyn MeshProtocol>]) -> Result<()> {
    ensure!(
        protocols.len() <= 16,
        "at most 16 application protocols may be registered"
    );
    ensure!(
        protocols
            .iter()
            .filter(|p| p.stage_transport().is_some())
            .count()
            <= 1,
        "at most one stage transport provider may be registered"
    );
    let mut keys = std::collections::HashSet::new();
    for protocol in protocols {
        let descriptor = protocol.descriptor();
        descriptor.validate()?;
        ensure!(
            !descriptor
                .features
                .iter()
                .any(|feature| feature == REQUIRED_STAGE_PROVIDER_FEATURE),
            "stage-provider-required is a host-owned capability feature"
        );
        ensure!(
            !BUILT_IN_PROTOCOL_NAMES.contains(&descriptor.name.as_str()),
            "built-in protocol name is reserved"
        );
        ensure!(
            keys.insert((descriptor.name, descriptor.major)),
            "duplicate application protocol registration"
        );
    }
    Ok(())
}
