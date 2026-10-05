//! TCP listener setup for public inference APIs.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use tokio::net::TcpListener;

/// Accept backlog for benchmark-facing serving listeners.
///
/// `TcpListener::bind` requests a backlog of 128, which a simultaneous
/// c256 connect burst can overflow before the accept task is polled again
/// (the kernel then RSTs the excess SYNs, surfacing client-side as
/// `ECONNRESET` in well under a millisecond). Request an explicit larger
/// backlog; hosts with a lower `somaxconn` clamp it as they see fit.
const SERVE_LISTEN_BACKLOG: i32 = 1024;

/// Bind a serving TCP listener with [`SERVE_LISTEN_BACKLOG`].
pub(crate) fn bind_serve_listener(bind_addr: SocketAddr) -> Result<TcpListener> {
    let domain = match bind_addr {
        SocketAddr::V4(_) => socket2::Domain::IPV4,
        SocketAddr::V6(_) => socket2::Domain::IPV6,
    };
    let socket = socket2::Socket::new(domain, socket2::Type::STREAM, Some(socket2::Protocol::TCP))
        .with_context(|| format!("create serving socket for {bind_addr}"))?;
    // Match Tokio's Unix listener semantics: a stopped server may rebind while
    // its closed connections are in TIME_WAIT. Do not enable SO_REUSEPORT or
    // Windows SO_REUSEADDR, which can permit sharing a live listener's port.
    #[cfg(unix)]
    socket
        .set_reuse_address(true)
        .context("enable serving listener address reuse")?;
    socket
        .bind(&bind_addr.into())
        .with_context(|| format!("bind serving socket to {bind_addr}"))?;
    socket
        .listen(SERVE_LISTEN_BACKLOG)
        .with_context(|| format!("listen on {bind_addr} with backlog {SERVE_LISTEN_BACKLOG}"))?;
    socket
        .set_nonblocking(true)
        .context("set serving listener nonblocking")?;
    let std_listener: std::net::TcpListener = socket.into();
    TcpListener::from_std(std_listener).context("register serving listener with tokio")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    /// Verify a completed server can immediately reuse its listening address.
    async fn serving_listener_can_rebind_after_server_closes_connection() -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let listener = bind_serve_listener("127.0.0.1:0".parse()?)?;
            let addr = listener.local_addr()?;
            let mut client = tokio::net::TcpStream::connect(addr).await?;
            let (mut accepted, _) = listener.accept().await?;
            // The server actively closes, leaving its connection in TIME_WAIT.
            accepted.shutdown().await?;
            client.read_to_end(&mut Vec::new()).await?;
            drop(accepted);
            drop(client);
            drop(listener);
            let replacement = bind_serve_listener(addr)?;
            assert_eq!(replacement.local_addr()?, addr);
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("serving listener restart timed out")?
    }

    #[tokio::test]
    /// Preserve exclusive address ownership while the first listener remains live.
    async fn serving_listener_rejects_another_live_listener() -> Result<()> {
        let listener = bind_serve_listener("127.0.0.1:0".parse()?)?;
        let error = bind_serve_listener(listener.local_addr()?)
            .expect_err("a second listener must not share the live port");
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::AddrInUse
        );
        Ok(())
    }
}
