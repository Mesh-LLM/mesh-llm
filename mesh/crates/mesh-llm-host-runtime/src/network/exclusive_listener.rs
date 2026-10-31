//! TCP listeners that no other socket can share on Windows.
//!
//! Windows lets a wildcard listener and a loopback one share a port unless the
//! socket asks for exclusive use, and loopback clients then reach the more
//! specific one. A node bound to `0.0.0.0:P` would report ready while local
//! clients talk to whatever already holds `127.0.0.1:P`. `SO_EXCLUSIVEADDRUSE`
//! makes that bind fail instead. Other platforms bind as before.

/// Bind `addr` (`host:port`) for exclusive use on Windows.
#[cfg(windows)]
pub(crate) async fn bind_exclusive_tcp_listener(
    addr: &str,
) -> std::io::Result<tokio::net::TcpListener> {
    use std::os::windows::io::AsRawSocket;
    use windows_sys::Win32::Networking::WinSock::{SO_EXCLUSIVEADDRUSE, SOL_SOCKET, setsockopt};

    let addr = tokio::net::lookup_host(addr).await?.next().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "no address to bind")
    })?;
    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    let exclusive: i32 = 1;
    // SAFETY: a valid socket handle owned by `socket`, and a pointer to an
    // i32 of the length passed, as SO_EXCLUSIVEADDRUSE expects.
    let result = unsafe {
        setsockopt(
            socket.as_raw_socket() as usize,
            SOL_SOCKET,
            SO_EXCLUSIVEADDRUSE,
            (&exclusive as *const i32).cast(),
            std::mem::size_of::<i32>() as i32,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    socket.set_nonblocking(true)?;
    tokio::net::TcpListener::from_std(socket.into())
}

/// Bind `addr` (`host:port`) with the platform's usual rules.
#[cfg(not(windows))]
pub(crate) async fn bind_exclusive_tcp_listener(
    addr: &str,
) -> std::io::Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(addr).await
}

#[cfg(test)]
mod tests {
    use super::bind_exclusive_tcp_listener;

    /// The case that leaked loopback traffic: a foreign `127.0.0.1:P`
    /// listener, then a wildcard bind on the same port.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_wildcard_bind_fails_on_a_port_held_on_loopback() {
        let foreign = std::net::TcpListener::bind("127.0.0.1:0").expect("foreign listener");
        let port = foreign.local_addr().expect("foreign addr").port();

        let result = bind_exclusive_tcp_listener(&format!("0.0.0.0:{port}")).await;

        assert!(result.is_err(), "0.0.0.0:{port} must not share a held port");
    }

    #[tokio::test]
    async fn a_free_port_binds_and_accepts() {
        let listener = bind_exclusive_tcp_listener("127.0.0.1:0")
            .await
            .expect("bind a free port");
        let addr = listener.local_addr().expect("local addr");

        let (client, accepted) =
            tokio::join!(tokio::net::TcpStream::connect(addr), listener.accept());

        client.expect("connect");
        accepted.expect("accept");
    }
}
