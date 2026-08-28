use std::{
    io,
    net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs},
    sync::atomic::{AtomicBool, Ordering},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::StageConfig;
use socket2::{Domain, Protocol, SockAddr, Socket, Type};

#[cfg(target_os = "macos")]
use std::{net::Ipv4Addr, os::fd::AsRawFd, ptr};

pub(crate) fn downstream_source_ip(config: &StageConfig) -> Result<Option<IpAddr>> {
    let bind_addr = config
        .bind_addr
        .parse::<SocketAddr>()
        .with_context(|| format!("parse stage bind_addr {}", config.bind_addr))?;
    let ip = bind_addr.ip();
    if ip.is_unspecified() {
        Ok(None)
    } else {
        Ok(Some(ip))
    }
}

pub(crate) fn resolve_downstream_endpoint(
    endpoint: &str,
    source_ip: Option<IpAddr>,
) -> Result<SocketAddr> {
    if let Ok(addr) = endpoint.parse::<SocketAddr>() {
        return Ok(addr);
    }
    let addrs = endpoint
        .to_socket_addrs()
        .with_context(|| format!("resolve downstream binary stage endpoint {endpoint}"))?
        .collect::<Vec<_>>();
    select_downstream_address(&addrs, source_ip).with_context(|| {
        format!("downstream binary stage endpoint resolved no addresses: {endpoint}")
    })
}

fn select_downstream_address(
    addrs: &[SocketAddr],
    source_ip: Option<IpAddr>,
) -> Option<SocketAddr> {
    source_ip
        .and_then(|source_ip| {
            addrs
                .iter()
                .find(|addr| addr.is_ipv4() == source_ip.is_ipv4())
        })
        .or_else(|| addrs.iter().find(|addr| addr.is_ipv4()))
        .or_else(|| addrs.first())
        .copied()
}

pub(crate) fn resolve_downstream_endpoint_cancellable(
    endpoint: &str,
    source_ip: Option<IpAddr>,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<SocketAddr> {
    if shutdown.load(Ordering::Acquire) {
        bail!("downstream endpoint resolution cancelled during shutdown");
    }
    if let Ok(addr) = endpoint.parse::<SocketAddr>() {
        return Ok(addr);
    }
    let endpoint = endpoint.to_string();
    let (tx, rx) = mpsc::sync_channel(1);
    // The system resolver has no cancellation API. Detach only that call so the
    // join-owned connection worker can still honor its deadline and shutdown;
    // the resolver exits when libc returns and drops its result if we moved on.
    thread::Builder::new()
        .name("skippy-downstream-resolver".to_string())
        .spawn(move || {
            let result = resolve_downstream_endpoint(&endpoint, source_ip);
            let _ = tx.send(result);
        })
        .context("spawn downstream endpoint resolver")?;
    wait_for_downstream_resolution(rx, deadline, shutdown)
}

fn wait_for_downstream_resolution(
    rx: mpsc::Receiver<Result<SocketAddr>>,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<SocketAddr> {
    const POLL: Duration = Duration::from_millis(50);
    loop {
        if shutdown.load(Ordering::Acquire) {
            bail!("downstream endpoint resolution cancelled during shutdown");
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("downstream endpoint resolution timed out");
        }
        match rx.recv_timeout(remaining.min(POLL)) {
            Ok(result) => return result,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!(
                    "downstream endpoint resolver stopped without a result"
                ));
            }
        }
    }
}

pub(crate) fn connect_downstream_socket(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    connect_downstream_socket_inner(downstream_addr, source_ip, timeout, None)
}

pub(crate) fn connect_downstream_socket_cancellable(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    shutdown: &AtomicBool,
) -> io::Result<TcpStream> {
    connect_downstream_socket_inner(downstream_addr, source_ip, timeout, Some(shutdown))
}

fn connect_downstream_socket_inner(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    shutdown: Option<&AtomicBool>,
) -> io::Result<TcpStream> {
    let mut errors = Vec::new();

    macro_rules! try_connect {
        ($mode:literal, $connect:expr_2021) => {{
            if shutdown.is_some_and(|shutdown| shutdown.load(Ordering::Acquire)) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "downstream connection cancelled during shutdown",
                ));
            }
            match $connect {
                Ok(stream) => return Ok(stream),
                Err(error) => {
                    eprintln!(
                        "downstream connect retry: source={source_ip:?} remote={downstream_addr} mode={} error={error}",
                        $mode
                    );
                    errors.push(format!("{} failed: {error}", $mode));
                }
            }
        }};
    }

    try_connect!(
        "route-selected",
        connect_route_selected_with_timeout(downstream_addr, source_ip, timeout)
    );
    try_connect!(
        "bound-interface",
        connect_bound_with_timeout(downstream_addr, source_ip, timeout, true)
    );
    try_connect!(
        "source-bound",
        connect_bound_with_timeout(downstream_addr, source_ip, timeout, false)
    );
    // The blocking fallbacks spawn helper threads that cannot be interrupted.
    // Keep them only on the legacy non-cancellable path so shutdown never
    // returns while a downstream acquisition helper is still running.
    if shutdown.is_none() {
        try_connect!(
            "blocking-source-bound",
            connect_blocking_with_timeout(downstream_addr, source_ip, timeout, false)
        );
        try_connect!(
            "blocking-route-selected",
            connect_route_selected_blocking_with_timeout(downstream_addr, source_ip, timeout)
        );
    }

    Err(io::Error::other(errors.join("; ")))
}

pub(super) fn connect_route_selected_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let stream = TcpStream::connect_timeout(&downstream_addr, timeout)?;
    if let Some(source_ip) = source_ip {
        let local_ip = stream.local_addr()?.ip();
        if local_ip != source_ip {
            return Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                format!("route-selected local address {local_ip} did not match {source_ip}"),
            ));
        }
    }
    eprintln!(
        "downstream connect succeeded: source={source_ip:?} remote={downstream_addr} mode=route-selected"
    );
    Ok(stream)
}

pub(super) fn connect_route_selected_blocking_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = TcpStream::connect(downstream_addr)
            .and_then(|stream| validate_route_selected_stream(stream, source_ip));
        let _ = tx.send(result);
    });
    rx.recv_timeout(timeout).map_err(|_| {
        io::Error::new(
            io::ErrorKind::TimedOut,
            "blocking route-selected fallback connect timed out",
        )
    })?
}

pub(super) fn validate_route_selected_stream(
    stream: TcpStream,
    source_ip: Option<IpAddr>,
) -> io::Result<TcpStream> {
    if let Some(source_ip) = source_ip {
        let local_ip = stream.local_addr()?.ip();
        if local_ip != source_ip {
            return Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                format!("route-selected local address {local_ip} did not match {source_ip}"),
            ));
        }
    }
    eprintln!(
        "downstream connect retry succeeded: source={source_ip:?} mode=blocking-route-selected"
    );
    Ok(stream)
}

pub(super) fn connect_bound_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let Some(source_ip) = source_ip else {
        return TcpStream::connect_timeout(&downstream_addr, timeout);
    };
    let domain = match (source_ip, downstream_addr) {
        (IpAddr::V4(_), SocketAddr::V4(_)) => Domain::IPV4,
        (IpAddr::V6(_), SocketAddr::V6(_)) => Domain::IPV6,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("source address {source_ip} cannot connect to {downstream_addr}"),
            ));
        }
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.bind(&SockAddr::from(SocketAddr::new(source_ip, 0)))?;
    if bind_interface {
        bind_socket_to_source_interface(&socket, source_ip)?;
    }
    socket.connect_timeout(&SockAddr::from(downstream_addr), timeout)?;
    Ok(socket.into())
}

pub(super) fn connect_blocking_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = tx.send(connect_bound_blocking(
            downstream_addr,
            source_ip,
            bind_interface,
        ));
    });
    rx.recv_timeout(timeout)
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "fallback connect timed out"))?
}

pub(super) fn connect_bound_blocking(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let Some(source_ip) = source_ip else {
        return TcpStream::connect(downstream_addr);
    };
    let domain = match (source_ip, downstream_addr) {
        (IpAddr::V4(_), SocketAddr::V4(_)) => Domain::IPV4,
        (IpAddr::V6(_), SocketAddr::V6(_)) => Domain::IPV6,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("source address {source_ip} cannot connect to {downstream_addr}"),
            ));
        }
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.bind(&SockAddr::from(SocketAddr::new(source_ip, 0)))?;
    if bind_interface {
        bind_socket_to_source_interface(&socket, source_ip)?;
    }
    socket.connect(&SockAddr::from(downstream_addr))?;
    Ok(socket.into())
}

#[cfg(target_os = "macos")]
pub(super) fn bind_socket_to_source_interface(
    socket: &Socket,
    source_ip: IpAddr,
) -> io::Result<()> {
    let Some(interface_index) = interface_index_for_ip(source_ip)? else {
        return Ok(());
    };
    let interface_index = interface_index as libc::c_int;
    let (level, optname) = match source_ip {
        IpAddr::V4(_) => (libc::IPPROTO_IP, 25),
        IpAddr::V6(_) => (libc::IPPROTO_IPV6, 125),
    };
    let result = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            level,
            optname,
            ptr::addr_of!(interface_index).cast(),
            std::mem::size_of_val(&interface_index) as libc::socklen_t,
        )
    };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub(super) fn bind_socket_to_source_interface(
    _socket: &Socket,
    _source_ip: IpAddr,
) -> io::Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn interface_index_for_ip(source_ip: IpAddr) -> io::Result<Option<u32>> {
    let mut addrs: *mut libc::ifaddrs = ptr::null_mut();
    let result = unsafe { libc::getifaddrs(ptr::addr_of_mut!(addrs)) };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    let mut cursor = addrs;
    while !cursor.is_null() {
        let ifaddr = unsafe { &*cursor };
        if !ifaddr.ifa_addr.is_null() && sockaddr_ip(ifaddr.ifa_addr) == Some(source_ip) {
            let index = unsafe { libc::if_nametoindex(ifaddr.ifa_name) };
            unsafe { libc::freeifaddrs(addrs) };
            if index == 0 {
                return Err(io::Error::last_os_error());
            }
            return Ok(Some(index));
        }
        cursor = ifaddr.ifa_next;
    }
    unsafe { libc::freeifaddrs(addrs) };
    Ok(None)
}

#[cfg(target_os = "macos")]
pub(super) fn sockaddr_ip(addr: *const libc::sockaddr) -> Option<IpAddr> {
    match unsafe { (*addr).sa_family as libc::c_int } {
        libc::AF_INET => {
            let addr = unsafe { &*(addr.cast::<libc::sockaddr_in>()) };
            Some(IpAddr::V4(Ipv4Addr::from(
                addr.sin_addr.s_addr.to_ne_bytes(),
            )))
        }
        libc::AF_INET6 => {
            let addr = unsafe { &*(addr.cast::<libc::sockaddr_in6>()) };
            Some(IpAddr::V6(addr.sin6_addr.s6_addr.into()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        connect_downstream_socket_cancellable, select_downstream_address,
        wait_for_downstream_resolution,
    };
    use std::{
        net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
        sync::{atomic::AtomicBool, mpsc},
        time::{Duration, Instant},
    };

    #[test]
    fn cancelled_downstream_connect_does_not_start_an_attempt() {
        let shutdown = AtomicBool::new(true);
        let error = connect_downstream_socket_cancellable(
            SocketAddr::from((Ipv4Addr::LOCALHOST, 9)),
            None,
            Duration::from_secs(30),
            &shutdown,
        )
        .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::Interrupted);
    }

    #[test]
    fn downstream_resolution_observes_its_deadline() {
        let (_tx, rx) = mpsc::sync_channel(1);
        let shutdown = AtomicBool::new(false);
        let error = wait_for_downstream_resolution(
            rx,
            Instant::now() + Duration::from_millis(10),
            &shutdown,
        )
        .unwrap_err();

        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn downstream_resolution_prefers_source_address_family() {
        let ipv4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9337);
        let ipv6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 9337);

        assert_eq!(
            select_downstream_address(&[ipv4, ipv6], Some(IpAddr::V6(Ipv6Addr::LOCALHOST))),
            Some(ipv6)
        );
        assert_eq!(
            select_downstream_address(&[ipv6, ipv4], Some(IpAddr::V4(Ipv4Addr::LOCALHOST))),
            Some(ipv4)
        );
    }

    #[test]
    fn downstream_resolution_preserves_generic_ipv4_preference() {
        let ipv4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9337);
        let ipv6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 9337);

        assert_eq!(select_downstream_address(&[ipv6, ipv4], None), Some(ipv4));
        assert_eq!(
            select_downstream_address(&[ipv4], Some(IpAddr::V6(Ipv6Addr::LOCALHOST))),
            Some(ipv4)
        );
    }
}
