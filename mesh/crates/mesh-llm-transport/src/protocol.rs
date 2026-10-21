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
            result = tokio::time::timeout(std::time::Duration::from_secs(10), self.network.open(peer, &self.descriptor)) => result?,
            () = self.cancelled() => anyhow::bail!("application protocol stopped while opening stream"),
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

#[async_trait]
pub trait MeshProtocol: Debug + Send + Sync + 'static {
    fn descriptor(&self) -> ProtocolDescriptor;

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
    let mut keys = std::collections::HashSet::new();
    for protocol in protocols {
        let descriptor = protocol.descriptor();
        descriptor.validate()?;
        ensure!(
            descriptor.name != "skippy-stage",
            "built-in protocol name is reserved"
        );
        ensure!(
            keys.insert((descriptor.name, descriptor.major)),
            "duplicate application protocol registration"
        );
    }
    Ok(())
}
