//! Execution-facing stage connections. Async providers are consumed directly by
//! blocking execution threads; no socket relay or background copy loop is used.

mod async_connection;
#[cfg(test)]
mod tests;

use std::{
    fmt::Debug,
    io::{self, Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};
use tokio::io::{AsyncRead, AsyncWrite};

use async_connection::AsyncConnection;

/// Node-local connection factory and bounded inbound admission queue for one stage.
/// Endpoints are interpreted by the embedding host, never by the stage codec.
pub trait StageTransport: Debug + Send + Sync {
    /// Execution has loaded its model and installed its ingress.
    fn ready(&self) {}
    /// Execution stopped accepting connections, including after startup failure.
    fn close(&self) {}
    fn connect(
        &self,
        endpoint: &str,
        deadline: Instant,
        shutdown: Option<&AtomicBool>,
    ) -> io::Result<StageConnection>;

    /// Return `WouldBlock` when no connection is pending.
    fn try_accept(&self) -> io::Result<StageConnection>;
}

#[derive(Debug)]
enum Connection {
    Tcp(TcpStream),
    Async(Arc<AsyncConnection>),
}

/// Ordered duplex I/O with shared shutdown and independent read/write progress.
#[derive(Debug)]
pub struct StageConnection(Connection);

impl StageConnection {
    #[cfg(test)]
    pub(crate) fn tcp(&self) -> Option<&TcpStream> {
        match &self.0 {
            Connection::Tcp(stream) => Some(stream),
            Connection::Async(_) => None,
        }
    }
    /// Use only from blocking execution threads while the supplied runtime runs.
    pub fn from_async<S>(stream: S, runtime: tokio::runtime::Handle) -> Self
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        Self(Connection::Async(Arc::new(AsyncConnection::new(
            stream, runtime,
        ))))
    }

    pub fn connect(address: impl ToSocketAddrs) -> io::Result<Self> {
        TcpStream::connect(address).map(Self::from)
    }

    pub fn try_clone(&self) -> io::Result<Self> {
        match &self.0 {
            Connection::Tcp(stream) => stream.try_clone().map(Self::from),
            Connection::Async(stream) => Ok(Self(Connection::Async(stream.clone()))),
        }
    }

    pub fn shutdown(&self, direction: Shutdown) -> io::Result<()> {
        match &self.0 {
            Connection::Tcp(stream) => stream.shutdown(direction),
            Connection::Async(stream) => stream.shutdown(direction),
        }
    }

    pub fn peek(&self, buffer: &mut [u8]) -> io::Result<usize> {
        match &self.0 {
            Connection::Tcp(stream) => stream.peek(buffer),
            Connection::Async(stream) => stream.peek(buffer),
        }
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        match &self.0 {
            Connection::Tcp(stream) => stream.set_read_timeout(timeout),
            Connection::Async(stream) => stream.set_read_timeout(timeout),
        }
    }

    pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
        match &self.0 {
            Connection::Tcp(stream) => stream.read_timeout(),
            Connection::Async(stream) => stream.read_timeout(),
        }
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        match &self.0 {
            Connection::Tcp(stream) => stream.set_write_timeout(timeout),
            Connection::Async(stream) => stream.set_write_timeout(timeout),
        }
    }

    pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
        match &self.0 {
            Connection::Tcp(stream) => stream.write_timeout(),
            Connection::Async(stream) => stream.write_timeout(),
        }
    }

    pub fn set_nonblocking(&self, enabled: bool) -> io::Result<()> {
        match &self.0 {
            Connection::Tcp(stream) => stream.set_nonblocking(enabled),
            Connection::Async(stream) => stream.set_nonblocking(enabled),
        }
    }

    pub fn set_nodelay(&self, enabled: bool) -> io::Result<()> {
        match &self.0 {
            Connection::Tcp(stream) => stream.set_nodelay(enabled),
            Connection::Async(_) => Ok(()),
        }
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        match &self.0 {
            Connection::Tcp(stream) => stream.local_addr(),
            Connection::Async(_) => Err(io::ErrorKind::Unsupported.into()),
        }
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        match &self.0 {
            Connection::Tcp(stream) => stream.peer_addr(),
            Connection::Async(_) => Err(io::ErrorKind::Unsupported.into()),
        }
    }
}

impl From<TcpStream> for StageConnection {
    fn from(stream: TcpStream) -> Self {
        Self(Connection::Tcp(stream))
    }
}

impl Read for &StageConnection {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match &self.0 {
            Connection::Tcp(stream) => (&*stream).read(buffer),
            Connection::Async(stream) => stream.read(buffer),
        }
    }
}

impl Write for &StageConnection {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        match &self.0 {
            Connection::Tcp(stream) => (&*stream).write(buffer),
            Connection::Async(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &self.0 {
            Connection::Tcp(stream) => (&*stream).flush(),
            Connection::Async(stream) => stream.flush(),
        }
    }
}

impl Read for StageConnection {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        (&*self).read(buffer)
    }
}

impl Write for StageConnection {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        (&*self).write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        (&*self).flush()
    }
}

/// Stage ingress supplied by a host, or a standalone TCP listener.
pub struct StageListener {
    tcp: Option<TcpListener>,
    transport: Option<Arc<dyn StageTransport>>,
}

impl Drop for StageListener {
    fn drop(&mut self) {
        if let Some(transport) = &self.transport {
            transport.close();
        }
    }
}

impl StageListener {
    pub fn bind(address: impl ToSocketAddrs) -> io::Result<Self> {
        Ok(Self {
            tcp: Some(TcpListener::bind(address)?),
            transport: None,
        })
    }

    pub fn supplied(transport: Arc<dyn StageTransport>) -> Self {
        transport.ready();
        Self {
            tcp: None,
            transport: Some(transport),
        }
    }

    pub fn accept(&self) -> io::Result<(StageConnection, Option<SocketAddr>)> {
        if let Some(transport) = &self.transport {
            return transport.try_accept().map(|stream| (stream, None));
        }
        self.tcp
            .as_ref()
            .expect("TCP listener")
            .accept()
            .map(|(stream, peer)| (stream.into(), Some(peer)))
    }

    pub fn set_nonblocking(&self, enabled: bool) -> io::Result<()> {
        match &self.tcp {
            Some(listener) => listener.set_nonblocking(enabled),
            None => Ok(()),
        }
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.tcp
            .as_ref()
            .ok_or(io::ErrorKind::Unsupported)?
            .local_addr()
    }
}
